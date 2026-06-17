"""
Build the user-friendly GitHub Pages site from the classified data.

Outputs (all under docs/):
  docs/data/papers.json   compact per-paper records for the interactive explorer
  docs/explorer.html      searchable / filterable paper browser (vanilla JS)
  docs/index.html         friendly landing page: key numbers + analysis + entries
  docs/dashboard.html     copy of survey_2026_viz.html (full Plotly dashboard)

Usage:
  python scripts/build_site.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify_keywords import CATEGORY_ORDER, CORE_CATEGORIES  # noqa: E402

BASE = Path(__file__).parent.parent
ANA = BASE / "data" / "_analysis"
DOCS = BASE / "docs"
DATA_OUT = DOCS / "data"
DATA_OUT.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/richardChenzhihui/ai-top-conf-2026-survey"
RAW_BASE = REPO_URL + "/raw/main"

CONFS = ["cvpr", "iclr", "neurips"]
CONF_LABELS = {"cvpr": "CVPR 2026", "iclr": "ICLR 2026", "neurips": "NeurIPS 2025"}
CORE_SHORT = {
    "多模态大模型（MLLM/VLM）": "MLLM/VLM",
    "强化学习后训练（RLHF/DPO/GRPO/奖励模型）": "RL 后训练",
    "Agent 系统": "Agent 系统",
    "Agent 后训练": "Agent 后训练",
    "大语言模型预训练（架构/数据/规模）": "LLM 预训练",
    "大模型 Infra（推理/训练/效率系统）": "大模型 Infra",
}


def paper_url(pid):
    """Derive a direct link to the paper from its id."""
    if pid.startswith("CVPR26_"):
        return f"https://cvpr.thecvf.com/virtual/2026/poster/{pid.split('_', 1)[1]}"
    if pid.startswith("ICLR26_"):
        return f"https://openreview.net/forum?id={pid.split('_', 1)[1]}"
    if pid.startswith("NEURIPS25_"):
        return f"https://openreview.net/forum?id={pid.split('_', 1)[1]}"
    return ""


def short_cat(c):
    return c.split("（")[0].strip()


def load():
    per_conf = {}
    for conf in CONFS:
        path = ANA / f"{conf}_classified.json"
        per_conf[conf] = json.loads(path.read_text()) if path.exists() else []
    return per_conf


# ---------------------------------------------------------------------------
# 1) compact data file for the explorer
# ---------------------------------------------------------------------------

def build_papers_json(per_conf):
    out = []
    for conf in CONFS:
        for p in per_conf[conf]:
            pid = p.get("paper_id", "")
            out.append({
                "id": pid,
                "cf": CONF_LABELS[conf],
                "tk": (p.get("track") or "").strip(),
                "l1": p.get("category_l1", "其他"),
                "l2": p.get("category_l2", ""),
                "te": p.get("title_en", ""),
                "tz": p.get("title_zh", ""),
                "sm": p.get("summary_zh", ""),
                "kw": p.get("keywords", ""),
                "u": paper_url(pid),
            })
    (DATA_OUT / "papers.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    size = (DATA_OUT / "papers.json").stat().st_size
    print(f"  docs/data/papers.json  {len(out)} papers, {size//1024} KB")
    return out


# ---------------------------------------------------------------------------
# 2) stats for the landing page
# ---------------------------------------------------------------------------

def compute_stats(per_conf):
    totals = {c: len(per_conf[c]) for c in CONFS}
    grand = sum(totals.values())
    other_rate = {}
    for c in CONFS:
        dist = Counter(p["category_l1"] for p in per_conf[c])
        other_rate[c] = dist.get("其他", 0) / max(totals[c], 1) * 100

    combined = Counter()
    for c in CONFS:
        combined.update(p["category_l1"] for p in per_conf[c])
    top_overall = [(cat, n) for cat, n in combined.most_common() if cat != "其他"][:12]

    # core-6 per conference (% of conference)
    core_pct = {}
    for cat in CORE_CATEGORIES:
        core_pct[cat] = {c: Counter(p["category_l1"] for p in per_conf[c]).get(cat, 0) / max(totals[c], 1) * 100
                         for c in CONFS}

    other_total = combined.get("其他", 0)
    return {
        "totals": totals, "grand": grand, "other_rate": other_rate,
        "top_overall": top_overall, "core_pct": core_pct,
        "other_total": other_total, "combined": combined,
    }


# ---------------------------------------------------------------------------
# 3) landing page
# ---------------------------------------------------------------------------

def build_index(stats):
    t = stats["totals"]
    bars = []
    maxv = max(n for _, n in stats["top_overall"]) if stats["top_overall"] else 1
    for cat, n in stats["top_overall"]:
        pct = n / maxv * 100
        bars.append(f"""      <div class="bar-row">
        <div class="bar-label">{short_cat(cat)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
        <div class="bar-val">{n}</div>
      </div>""")
    bars_html = "\n".join(bars)

    # core-6 table
    core_rows = []
    for cat in CORE_CATEGORIES:
        cells = "".join(f"<td>{stats['core_pct'][cat][c]:.1f}%</td>" for c in CONFS)
        core_rows.append(f"<tr><td class='cn'>{CORE_SHORT.get(cat, short_cat(cat))}</td>{cells}</tr>")
    core_html = "\n".join(core_rows)

    grand = stats["grand"]
    other_pct_overall = stats["other_total"] / grand * 100

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>顶会论文热点分析 · CVPR 2026 / ICLR 2026 / NeurIPS 2025</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: "PingFang SC","Microsoft YaHei",-apple-system,sans-serif;
         margin:0; background:#f4f6fa; color:#1c2333; line-height:1.6; }}
  a {{ color:#2E75B6; text-decoration:none; }}
  .hero {{ background:linear-gradient(135deg,#16264d 0%,#2E75B6 100%); color:#fff;
          padding:54px 24px 40px; text-align:center; }}
  .hero h1 {{ margin:0 0 10px; font-size:30px; }}
  .hero p {{ margin:0 auto; max-width:760px; opacity:.9; font-size:15px; }}
  .stat-row {{ display:flex; flex-wrap:wrap; gap:16px; justify-content:center; margin-top:28px; }}
  .stat {{ background:rgba(255,255,255,.14); border-radius:10px; padding:14px 26px; min-width:120px; }}
  .stat-n {{ font-size:30px; font-weight:700; }}
  .stat-l {{ font-size:12px; opacity:.85; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:0 20px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:18px; margin:-28px auto 8px;
           position:relative; max-width:1080px; padding:0 20px; }}
  .card {{ background:#fff; border-radius:12px; padding:22px; box-shadow:0 3px 14px rgba(20,40,80,.10);
          transition:transform .12s, box-shadow .12s; display:block; }}
  .card:hover {{ transform:translateY(-3px); box-shadow:0 8px 22px rgba(20,40,80,.16); }}
  .card .ico {{ font-size:30px; }}
  .card h3 {{ margin:8px 0 4px; font-size:17px; color:#16264d; }}
  .card p {{ margin:0; font-size:13px; color:#5a6577; }}
  .section {{ background:#fff; border-radius:12px; margin:24px 0; padding:24px 28px;
             box-shadow:0 1px 8px rgba(20,40,80,.06); }}
  .section h2 {{ margin:0 0 18px; font-size:19px; color:#16264d;
                border-left:4px solid #2E75B6; padding-left:12px; }}
  .bar-row {{ display:flex; align-items:center; gap:12px; margin:7px 0; }}
  .bar-label {{ width:200px; font-size:13px; text-align:right; color:#33405a; flex-shrink:0; }}
  .bar-track {{ flex:1; background:#eef1f6; border-radius:5px; height:18px; overflow:hidden; }}
  .bar-fill {{ background:linear-gradient(90deg,#2E75B6,#5aa0e0); height:100%; border-radius:5px; }}
  .bar-val {{ width:46px; font-size:12px; color:#5a6577; }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
  th,td {{ padding:8px 10px; text-align:center; border-bottom:1px solid #eef1f6; }}
  th {{ background:#16264d; color:#fff; font-weight:600; }}
  td.cn {{ text-align:left; font-weight:600; color:#16264d; }}
  .insights li {{ margin:6px 0; font-size:14px; }}
  .note {{ font-size:12.5px; color:#7a8499; }}
  footer {{ text-align:center; color:#9aa4b5; font-size:12px; padding:30px 20px 50px; }}
</style>
</head>
<body>
<div class="hero">
  <h1>顶会论文热点分析 · 2025–2026</h1>
  <p>对 <b>CVPR 2026 / ICLR 2026 / NeurIPS 2025 全量 {grand:,} 篇</b> 论文，用 <b>MiniMax-M2.5 大模型</b> 逐篇读「标题 + 摘要」做主题分类，并生成中文标题与摘要。点进来即可检索、可视化、下载。</p>
  <div class="stat-row">
    <div class="stat"><div class="stat-n">{t['cvpr']:,}</div><div class="stat-l">CVPR 2026</div></div>
    <div class="stat"><div class="stat-n">{t['iclr']:,}</div><div class="stat-l">ICLR 2026</div></div>
    <div class="stat"><div class="stat-n">{t['neurips']:,}</div><div class="stat-l">NeurIPS 2025</div></div>
    <div class="stat"><div class="stat-n">{grand:,}</div><div class="stat-l">总计</div></div>
    <div class="stat"><div class="stat-n">37</div><div class="stat-l">研究方向类别</div></div>
  </div>
</div>

<div class="cards">
  <a class="card" href="explorer.html"><div class="ico">🔍</div><h3>检索论文</h3><p>按关键词 / 会议 / 类别即时筛选 {grand:,} 篇论文，看中文标题与摘要，一键跳转原文。</p></a>
  <a class="card" href="dashboard.html"><div class="ico">📊</div><h3>交互式看板</h3><p>7 张交互图表：三会全景、核心方向热度、L2 子方向、Track 分布等。</p></a>
  <a class="card" href="{RAW_BASE}/survey_2026.xlsx"><div class="ico">📑</div><h3>下载 Excel</h3><p>5 个工作表的中文综述大表，可筛选、可冻结，离线分析友好。</p></a>
  <a class="card" href="{REPO_URL}"><div class="ico">💻</div><h3>源码与数据</h3><p>分类器、可视化脚本与全部结构化数据，开源可复现。</p></a>
</div>

<div class="wrap">
  <div class="section">
    <h2>📈 热门研究方向 Top 12（三会合计）</h2>
{bars_html}
  </div>

  <div class="section">
    <h2>🔥 六大核心方向 · 各会议占比</h2>
    <table>
      <tr><th>核心方向</th><th>CVPR 2026</th><th>ICLR 2026</th><th>NeurIPS 2025</th></tr>
{core_html}
    </table>
    <p class="note" style="margin-top:10px">数值为该方向论文数占该会议全部论文的百分比。</p>
  </div>

  <div class="section insights">
    <h2>🧭 一眼看懂</h2>
    <ul>
      <li>三大会议合计 <b>{grand:,}</b> 篇，经大模型逐篇分类后归入 37 个研究方向，<b>「其他」仅占 {other_pct_overall:.1f}%</b>。</li>
      <li>CVPR 以视觉方向为主，ICLR / NeurIPS 则集中在大模型、强化学习与通用机器学习理论。</li>
      <li>六大核心方向（多模态大模型、RL 后训练、Agent 系统、Agent 后训练、LLM 预训练、大模型 Infra）是当下最受关注的前沿。</li>
      <li>想快速找某个题目的论文？用上方「检索论文」搜关键词即可。</li>
    </ul>
  </div>

  <div class="section">
    <h2>🛠 方法与数据</h2>
    <p style="font-size:14px">分类由 <b>MiniMax-M2.5-highspeed</b> 完成：把每篇论文的标题、摘要、关键词喂给模型，让它从固定的 37 个一级方向中择一，并为 6 个核心方向选择细分子方向，同时产出流畅的中文标题与 2–3 句中文摘要。相比纯关键词规则，大模型能理解语义、显著降低「其他」比例并给出更准的归类。</p>
    <p class="note">数据来源：CVPR 虚拟会议站公开 JSON、OpenReview 官方 API（ICLR / NeurIPS）。仅采集公开的标题与摘要，<b>不爬取会议网站 HTML、不下载任何 PDF</b>。论文版权归原作者 / 出版方所有，本项目仅作非商业学术汇集。</p>
  </div>
</div>

<footer>开源项目 · <a href="{REPO_URL}">{REPO_URL.split('//')[1]}</a> · 数据截至 2026-06</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 4) explorer page (static, data-driven)
# ---------------------------------------------------------------------------

EXPLORER_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>论文检索 · CVPR/ICLR/NeurIPS 2025–2026</title>
<style>
  * { box-sizing:border-box; }
  body { font-family:"PingFang SC","Microsoft YaHei",-apple-system,sans-serif;
         margin:0; background:#f4f6fa; color:#1c2333; }
  a { color:#2E75B6; text-decoration:none; }
  header { background:linear-gradient(135deg,#16264d,#2E75B6); color:#fff; padding:18px 24px; }
  header h1 { margin:0; font-size:19px; display:inline-block; }
  header a.home { color:#cfe0f5; font-size:13px; margin-left:14px; }
  .controls { position:sticky; top:0; z-index:5; background:#fff; padding:14px 24px;
              box-shadow:0 2px 8px rgba(20,40,80,.08); display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  .controls input, .controls select { padding:8px 11px; border:1px solid #d3dae6; border-radius:8px;
              font-size:14px; background:#fff; }
  .controls input#q { flex:1; min-width:220px; }
  .count { font-size:13px; color:#5a6577; margin-left:auto; }
  .list { max-width:1080px; margin:18px auto; padding:0 20px; }
  .item { background:#fff; border-radius:10px; padding:16px 18px; margin-bottom:12px;
          box-shadow:0 1px 6px rgba(20,40,80,.06); }
  .item .tz { font-size:16px; font-weight:600; color:#16264d; }
  .item .te { font-size:13px; color:#7a8499; margin:2px 0 8px; }
  .tags { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }
  .tag { font-size:11.5px; padding:2px 9px; border-radius:20px; background:#eef3fb; color:#2E75B6; }
  .tag.cf { background:#e9f7ef; color:#2a9d5a; }
  .tag.tk { background:#fdf0e6; color:#d98324; }
  .sm { font-size:13.5px; color:#3c485e; }
  .item .link { font-size:13px; margin-left:8px; }
  .pager { text-align:center; margin:22px 0 50px; }
  .pager button { padding:8px 16px; margin:0 5px; border:1px solid #d3dae6; background:#fff;
          border-radius:8px; cursor:pointer; font-size:14px; }
  .pager button:disabled { opacity:.45; cursor:default; }
  .loading { text-align:center; padding:60px; color:#7a8499; }
</style>
</head>
<body>
<header>
  <h1>📚 论文检索</h1><a class="home" href="index.html">← 返回首页</a>
</header>
<div class="controls">
  <input id="q" placeholder="搜索标题 / 摘要 / 关键词（中英文均可）…">
  <select id="conf"><option value="">全部会议</option></select>
  <select id="l1"><option value="">全部一级方向</option></select>
  <select id="l2"><option value="">全部子方向</option></select>
  <span class="count" id="count">加载中…</span>
</div>
<div class="list" id="list"><div class="loading">正在加载论文数据…</div></div>
<div class="pager" id="pager"></div>

<script>
const PAGE = 50;
let DATA = [], view = [], page = 0;
const $ = id => document.getElementById(id);
const esc = s => (s||"").replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

fetch('data/papers.json').then(r => r.json()).then(d => {
  DATA = d;
  initFilters();
  apply();
}).catch(e => { $('list').innerHTML = '<div class="loading">数据加载失败：' + e + '</div>'; });

function initFilters() {
  const confs = [...new Set(DATA.map(p => p.cf))];
  confs.forEach(c => $('conf').add(new Option(c, c)));
  const l1s = [...new Set(DATA.map(p => p.l1))].sort((a,b) =>
    DATA.filter(p=>p.l1===b).length - DATA.filter(p=>p.l1===a).length);
  l1s.forEach(c => $('l1').add(new Option(c, c)));
  $('conf').addEventListener('change', () => { page = 0; apply(); });
  $('l2').addEventListener('change', () => { page = 0; apply(); });
  $('l1').addEventListener('change', () => { refreshL2(); page = 0; apply(); });
  $('q').addEventListener('input', debounce(() => { page = 0; apply(); }, 180));
}
function refreshL2() {
  const l1 = $('l1').value;
  const pool = l1 ? DATA.filter(p => p.l1===l1) : DATA;
  const l2s = [...new Set(pool.map(p => p.l2).filter(Boolean))].sort();
  $('l2').innerHTML = '<option value="">全部子方向</option>';
  l2s.forEach(c => $('l2').add(new Option(c, c)));
}
function debounce(fn, ms){ let t; return (...a)=>{ clearTimeout(t); t=setTimeout(()=>fn(...a),ms); }; }

function apply() {
  const q = $('q').value.trim().toLowerCase();
  const cf = $('conf').value, l1 = $('l1').value, l2 = $('l2').value;
  view = DATA.filter(p => {
    if (cf && p.cf !== cf) return false;
    if (l1 && p.l1 !== l1) return false;
    if (l2 && p.l2 !== l2) return false;
    if (q) {
      const hay = (p.te + ' ' + p.tz + ' ' + p.sm + ' ' + p.kw).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  $('count').textContent = '共 ' + view.length.toLocaleString() + ' 篇';
  render();
}

function render() {
  const start = page * PAGE;
  const slice = view.slice(start, start + PAGE);
  if (!slice.length) { $('list').innerHTML = '<div class="loading">没有匹配的论文，换个关键词试试。</div>'; $('pager').innerHTML=''; return; }
  $('list').innerHTML = slice.map(p => {
    const link = p.u ? `<a class="link" href="${p.u}" target="_blank" rel="noopener">原文 ↗</a>` : '';
    const tk = p.tk ? `<span class="tag tk">${esc(p.tk)}</span>` : '';
    const l2 = p.l2 ? `<span class="tag">${esc(p.l2)}</span>` : '';
    return `<div class="item">
      <div class="tz">${esc(p.tz || p.te)}${link}</div>
      <div class="te">${esc(p.te)}</div>
      <div class="tags"><span class="tag cf">${esc(p.cf)}</span><span class="tag">${esc(p.l1)}</span>${l2}${tk}</div>
      <div class="sm">${esc(p.sm)}</div>
    </div>`;
  }).join('');
  const pages = Math.ceil(view.length / PAGE);
  $('pager').innerHTML =
    `<button onclick="go(-1)" ${page<=0?'disabled':''}>← 上一页</button>` +
    `<span style="font-size:13px;color:#5a6577"> 第 ${page+1} / ${pages} 页 </span>` +
    `<button onclick="go(1)" ${page>=pages-1?'disabled':''}>下一页 →</button>`;
  window.scrollTo({top:0, behavior:'smooth'});
}
function go(d){ page += d; render(); }
</script>
</body>
</html>"""


def main():
    per_conf = load()
    if not any(per_conf.values()):
        sys.exit("ERROR: no classified data found in data/_analysis/")
    print("Building site ->", DOCS)
    build_papers_json(per_conf)
    stats = compute_stats(per_conf)
    (DOCS / "index.html").write_text(build_index(stats), encoding="utf-8")
    print("  docs/index.html")
    (DOCS / "explorer.html").write_text(EXPLORER_HTML, encoding="utf-8")
    print("  docs/explorer.html")
    viz = BASE / "survey_2026_viz.html"
    if viz.exists():
        (DOCS / "dashboard.html").write_text(viz.read_text(encoding="utf-8"), encoding="utf-8")
        print("  docs/dashboard.html (copied from survey_2026_viz.html)")
    else:
        print("  [WARN] survey_2026_viz.html not found — run visualize.py first")
    print("Done.")


if __name__ == "__main__":
    main()
