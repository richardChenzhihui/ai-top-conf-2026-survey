"""
LLM-based paper classifier using MiniMax (OpenAI-compatible API).

Reads the three raw JSON files, classifies each paper into L1/L2 by having the
model read title + abstract, and ALSO generates a fluent Chinese title and a
2-3 sentence Chinese summary. Writes one classified JSON per conference with the
exact same 12-field schema the downstream scripts (visualize.py / build_excel.py)
expect, so nothing downstream needs to change.

Design (see plan):
  - Reuses the taxonomy (CATEGORIES / L2_RULES / CATEGORY_ORDER / CORE_CATEGORIES)
    from classify_keywords.py — single source of truth.
  - Batches multiple papers per request (default 10); the model returns one JSON
    object {"results":[{i, category_l1, category_l2, title_zh, summary_zh}, ...]}.
    Papers are index-keyed so a reordered / short response still maps correctly.
  - Parallel requests via ThreadPoolExecutor with exponential backoff on errors.
  - Disk-first / resumable: every batch writes its own checkpoint file under
    data/_analysis/{conf}/llm_batch_NNNN.json; reruns skip completed batches.
  - Validates category_l1 against the whitelist; merges by paper_id into the
    final {conf}_classified.json with a set-difference completeness check.

Env:
  MINIMAX_API_KEY must be set (loaded from ~/.zshrc by the caller).

Usage:
  python scripts/classify_minimax.py --conf cvpr --limit 24      # smoke test
  python scripts/classify_minimax.py --conf all                  # full run
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent))
from classify_keywords import CATEGORIES, L2_RULES, CATEGORY_ORDER, CORE_CATEGORIES  # noqa: E402

from openai import OpenAI  # noqa: E402

BASE = Path(__file__).parent.parent / "data"
OUT = BASE / "_analysis"
OUT.mkdir(exist_ok=True)

BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
DEFAULT_MODEL = "MiniMax-M2.5-highspeed"

L1_SET = set(CATEGORY_ORDER)
# l1 -> ordered list of L2 candidate labels (drop empty catch-all phrase markers)
L2_CANDIDATES = {l1: [label for label, _ in rules] for l1, rules in L2_RULES.items()}


def _canon(s):
    """Canonicalize a label so model output tolerantly matches the taxonomy:
    strip a leading list number ('14. '), drop ASCII/CJK spaces, unify parens."""
    s = (s or "").strip()
    s = re.sub(r'^\s*\d+\s*[\.、\)\:、]\s*', '', s)
    s = s.replace(' ', '').replace('　', '')
    s = s.replace('(', '（').replace(')', '）')
    return s


# canonical-form -> exact taxonomy name
_L1_CANON = {_canon(name): name for name in CATEGORY_ORDER}
# also index by the short prefix before the first paren, for looser matches
_L1_PREFIX = {}
for _name in CATEGORY_ORDER:
    _pfx = _canon(_name).split('（')[0]
    if _pfx and _pfx not in _L1_PREFIX:
        _L1_PREFIX[_pfx] = _name


def resolve_l1(raw):
    """Map a model-returned L1 string to an exact taxonomy name, else '其他'."""
    c = _canon(raw)
    if c in _L1_CANON:
        return _L1_CANON[c]
    pfx = c.split('（')[0]
    if pfx in _L1_PREFIX:
        return _L1_PREFIX[pfx]
    return "其他"

CONFS = {
    "cvpr":    (BASE / "cvpr2026_raw.json",    OUT / "cvpr_classified.json",    OUT / "cvpr"),
    "iclr":    (BASE / "iclr2026_raw.json",    OUT / "iclr_classified.json",    OUT / "iclr"),
    "neurips": (BASE / "neurips2025_raw.json", OUT / "neurips_classified.json", OUT / "neurips"),
}

# token usage tally (thread-safe)
_USAGE = {"prompt": 0, "completion": 0, "calls": 0}
_USAGE_LOCK = Lock()
_SUPPORT_JSON_MODE = {"ok": True}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_system_prompt():
    l1_lines = "\n".join(f"{i+1}. {name}" for i, name in enumerate(CATEGORY_ORDER))
    core_lines = []
    for l1 in CORE_CATEGORIES:
        cands = "、".join(L2_CANDIDATES[l1])
        core_lines.append(f"「{l1}」候选L2：{cands}")
    core_block = "\n".join(core_lines)
    return f"""你是资深 AI/机器学习/计算机视觉研究专家，擅长把论文精准归入研究方向。下面给你一批论文（标题+摘要+关键词），请逐篇分类。

【L1 一级类别】必须从下表中【原文逐字】选恰好一个（不得改写、不得自造、不要带前面的序号、括号与空格都照抄）：
{l1_lines}

【L2 二级子方向】
- 若某篇的 L1 属于以下 6 个核心类，L2 必须从该类的候选里【原文逐字】选一个；只有当所有候选都明显不贴切时，才自造一个 ≤10 字的简短中文 L2。
{core_block}
- 若 L1 不属于上述 6 个核心类：给一个 ≤10 字的简短中文 L2 描述；实在没有就留空字符串 ""。

【中文字段】
- title_zh：把英文标题翻译成流畅、专业的中文标题（保留 method/benchmark/model 等专有名词的英文原名）。
- summary_zh：2-3 句中文，覆盖 研究问题 + 方法 + 主要贡献/效果。简洁、信息密度高。

【判别要点】
- 多模态大模型(MLLM/VLM) vs 各视觉子类：以 LLM/VLM 驱动的视觉语言能力归前者；纯视觉算法（检测/分割/生成/3D等）归对应视觉类。
- 强化学习后训练(RLHF/DPO/GRPO) vs 经典强化学习与控制：前者是用于对齐/后训练大模型；后者是传统 RL/控制/博弈/MARL。
- Agent 后训练（训练/微调 LLM Agent，轨迹级 RL/SFT）与 Agent 系统（框架/工具调用/多智能体）要区分。
- 不确定且确实无法归入任何明确类别时，才用「其他」；不要滥用「其他」。

【输出】严格返回一个 JSON 对象：{{"results":[{{"i":整数下标,"category_l1":"...","category_l2":"...","title_zh":"...","summary_zh":"..."}}, ...]}}。
results 必须覆盖输入的每一篇（i 与输入一一对应）。只输出 JSON，不要任何额外文字或代码块标记。JSON 字符串内部不要用英文双引号，需强调用《》或单引号。"""


def build_user_prompt(batch):
    """batch: list of (i, paper). Includes title + truncated abstract + keywords."""
    parts = []
    for i, p in batch:
        title = (p.get("title_en") or "").strip()
        abst = (p.get("abstract") or "").strip().replace("\n", " ")
        if len(abst) > 1200:
            abst = abst[:1200] + "..."
        kws = (p.get("keywords") or "").strip()
        block = f"### 论文 i={i}\nTitle: {title}"
        if kws:
            block += f"\nKeywords: {kws}"
        block += f"\nAbstract: {abst if abst else '(无摘要，仅凭标题/关键词判断)'}"
        parts.append(block)
    return "共 {} 篇论文，请逐篇分类并返回 JSON：\n\n".format(len(batch)) + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Robust JSON parsing of the model response
# ---------------------------------------------------------------------------

def _strip_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_results(content):
    """Return list of result dicts, or raise ValueError."""
    txt = _strip_fences(content)
    obj = None
    try:
        obj = json.loads(txt)
    except Exception:
        # last resort: json_repair fixes unescaped quotes / trailing commas / etc.
        try:
            import json_repair
            obj = json_repair.loads(txt)
            if obj in ("", [], {}):
                obj = None
        except Exception:
            obj = None
    if obj is None:
        raise ValueError("could not parse JSON from model output")
    if isinstance(obj, dict):
        results = obj.get("results", obj.get("data"))
        if results is None:
            # maybe the dict IS a single result
            if "category_l1" in obj:
                results = [obj]
            else:
                raise ValueError("no 'results' key in JSON object")
    elif isinstance(obj, list):
        results = obj
    else:
        raise ValueError("unexpected JSON top-level type")
    if not isinstance(results, list):
        raise ValueError("'results' is not a list")
    return results


# ---------------------------------------------------------------------------
# Validation / normalization
# ---------------------------------------------------------------------------

def normalize_record(raw_paper, res):
    """Combine a raw paper with one model result dict into the final 12-field record."""
    l1 = resolve_l1(res.get("category_l1"))
    l2 = re.sub(r'^\s*\d+\s*[\.、\)\:]\s*', '', (res.get("category_l2") or "").strip())
    if l1 in CORE_CATEGORIES:
        cands = L2_CANDIDATES.get(l1, [])
        if not l2:
            l2 = cands[0] if cands else "其他子方向"
    title_zh = (res.get("title_zh") or "").strip()
    summary_zh = (res.get("summary_zh") or "").strip()
    return {
        **raw_paper,
        "category_l1": l1,
        "category_l2": l2,
        "title_zh": title_zh,
        "summary_zh": summary_zh,
    }


def fallback_record(raw_paper):
    return {
        **raw_paper,
        "category_l1": "其他",
        "category_l2": "",
        "title_zh": "",
        "summary_zh": "",
    }


# ---------------------------------------------------------------------------
# API call (one batch) with retries
# ---------------------------------------------------------------------------

def call_model(client, model, system_prompt, user_prompt, max_retries=5):
    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            if _SUPPORT_JSON_MODE["ok"]:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            if getattr(resp, "usage", None):
                with _USAGE_LOCK:
                    _USAGE["prompt"] += resp.usage.prompt_tokens or 0
                    _USAGE["completion"] += resp.usage.completion_tokens or 0
                    _USAGE["calls"] += 1
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "response_format" in msg and _SUPPORT_JSON_MODE["ok"]:
                # model doesn't support json_object mode; disable and retry immediately
                _SUPPORT_JSON_MODE["ok"] = False
                continue
            last_err = e
            sleep = min(2 ** attempt, 30)
            time.sleep(sleep)
    raise RuntimeError(f"call_model failed after {max_retries} retries: {last_err}")


def process_batch(client, model, system_prompt, batch, ckpt_path):
    """Classify one batch; write checkpoint; return list of final records."""
    user_prompt = build_user_prompt(batch)
    content = call_model(client, model, system_prompt, user_prompt)
    results = parse_results(content)
    by_i = {}
    for r in results:
        if isinstance(r, dict) and "i" in r:
            try:
                by_i[int(r["i"])] = r
            except (ValueError, TypeError):
                pass
    # if model ignored indices but returned right count in order, map positionally
    if not by_i and len(results) == len(batch):
        by_i = {batch[k][0]: results[k] for k in range(len(batch))}

    records = []
    for i, paper in batch:
        res = by_i.get(i)
        records.append(normalize_record(paper, res) if isinstance(res, dict) else fallback_record(paper))
    ckpt_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    return records


# ---------------------------------------------------------------------------
# Per-conference orchestration
# ---------------------------------------------------------------------------

def classify_conf(client, model, conf, batch_size, workers, limit):
    src, dst, ckpt_dir = CONFS[conf]
    if not src.exists():
        print(f"[SKIP] {src} not found")
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    papers = json.loads(src.read_text())
    if limit:
        papers = papers[:limit]
    n = len(papers)

    # build batches: list of (batch_idx, [(global_i, paper), ...])
    batches = []
    for b, start in enumerate(range(0, n, batch_size)):
        chunk = [(j, papers[j]) for j in range(start, min(start + batch_size, n))]
        batches.append((b, chunk))

    system_prompt = build_system_prompt()
    todo = []
    done = 0
    for b, chunk in batches:
        ckpt = ckpt_dir / f"llm_batch_{b:04d}.json"
        if ckpt.exists():
            done += 1
            continue
        todo.append((b, chunk, ckpt))

    print(f"\n=== {conf.upper()}: {n} papers, {len(batches)} batches "
          f"({done} cached, {len(todo)} to run), workers={workers}, batch_size={batch_size} ===")

    errors = []
    completed = 0
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_batch, client, model, system_prompt, chunk, ckpt): b
                    for b, chunk, ckpt in todo}
            for fut in as_completed(futs):
                b = futs[fut]
                try:
                    fut.result()
                    completed += 1
                except Exception as e:  # noqa: BLE001
                    errors.append((b, str(e)))
                if (completed + len(errors)) % 10 == 0 or (completed + len(errors)) == len(todo):
                    print(f"  progress: {completed + len(errors)}/{len(todo)} "
                          f"(ok={completed}, err={len(errors)})", flush=True)

    if errors:
        print(f"  [WARN] {len(errors)} batches failed: {errors[:5]}{' ...' if len(errors) > 5 else ''}")

    # ---- merge all checkpoints by paper_id ----
    by_id = {}
    for ckpt in sorted(ckpt_dir.glob("llm_batch_*.json")):
        try:
            for rec in json.loads(ckpt.read_text()):
                by_id[rec["paper_id"]] = rec
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] bad checkpoint {ckpt.name}: {e}")

    final = []
    missing = 0
    for p in papers:
        rec = by_id.get(p["paper_id"])
        if rec is None:
            rec = fallback_record(p)
            missing += 1
        final.append(rec)

    dst.write_text(json.dumps(final, ensure_ascii=False, indent=2))

    dist = Counter(r["category_l1"] for r in final)
    other = dist.get("其他", 0)
    zh_filled = sum(1 for r in final if r["title_zh"])
    print(f"  -> wrote {dst.name}: {len(final)} papers, 其他={other} ({other/len(final)*100:.1f}%), "
          f"title_zh填充={zh_filled}/{len(final)}, 缺失补其他={missing}")
    for cat, cnt in sorted(dist.items(), key=lambda x: -x[1])[:12]:
        print(f"      {cnt:4d}  {cat}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="all", choices=["all", "cvpr", "iclr", "neurips"])
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only classify first N papers (smoke test)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        sys.exit("ERROR: MINIMAX_API_KEY not set in environment")
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=90.0, max_retries=0)

    confs = ["cvpr", "iclr", "neurips"] if args.conf == "all" else [args.conf]
    t0 = time.time()
    for conf in confs:
        classify_conf(client, args.model, conf, args.batch_size, args.workers, args.limit or 0)

    dt = time.time() - t0
    print(f"\nDone in {dt:.0f}s. API calls={_USAGE['calls']}, "
          f"prompt_tokens={_USAGE['prompt']:,}, completion_tokens={_USAGE['completion']:,}, "
          f"total_tokens={_USAGE['prompt'] + _USAGE['completion']:,}")
    if _SUPPORT_JSON_MODE["ok"] is False:
        print("(note: server rejected response_format=json_object; ran without it)")


if __name__ == "__main__":
    main()
