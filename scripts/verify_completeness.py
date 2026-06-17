#!/usr/bin/env python3
"""Verify completeness of fetched and classified paper data."""
import argparse, json, sys

L1_WHITELIST = [
    "多模态大模型（MLLM/VLM）",
    "强化学习后训练（RLHF/DPO/GRPO/奖励模型）",
    "Agent 系统",
    "Agent 后训练",
    "大语言模型预训练（架构/数据/规模）",
    "大模型 Infra（推理/训练/效率系统）",
    "推理与规划（CoT/数学/逻辑，非 RL 类）",
    "代码生成与编程语言处理",
    "安全与对齐（幻觉/偏见/毒性/越狱，非 RL 类）",
    "评测与基准构建",
    "图神经网络与结构化学习",
    "理论与优化",
    "对话系统与文本生成",
    "目标检测与分割",
    "生成模型（扩散/GAN/视频生成/图像编辑，非 MLLM 驱动）",
    "三维视觉（NeRF/3DGS/深度估计/点云）",
    "视频理解（动作识别/跟踪/时序建模，非 MLLM 类）",
    "底层视觉（超分/去噪/增强）",
    "人体相关（姿态/人脸/手势/运动合成）",
    "医学影像（非多模态大模型类）",
    "自动驾驶（感知/预测/规划）",
    "遥感与卫星",
    "其他",
]

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def check_raw(raw_files):
    print("=== Raw data stats ===")
    total = 0
    for f in raw_files:
        papers = load(f)
        with_abs = sum(1 for p in papers if p.get("abstract", "").strip())
        print(f"  {f}: {len(papers)} papers, {with_abs} with abstract ({100*with_abs//len(papers)}%)")
        total += len(papers)
    print(f"  TOTAL raw: {total}\n")
    return total

def check_classified(classified_path, raw_total):
    print("=== Classified data stats ===")
    papers = load(classified_path)

    # dedup check
    seen, dups = set(), 0
    for p in papers:
        pid = p.get("paper_id", "")
        if pid in seen:
            dups += 1
        seen.add(pid)

    # blank field rates
    fields = ["category_l1", "category_l2", "title_zh", "summary_zh"]
    blank = {f: sum(1 for p in papers if not str(p.get(f, "")).strip()) for f in fields}

    # L1 whitelist check
    unknown_l1 = {}
    for p in papers:
        l1 = p.get("category_l1", "").strip()
        if l1 and l1 not in L1_WHITELIST:
            unknown_l1[l1] = unknown_l1.get(l1, 0) + 1

    # L1 distribution
    from collections import Counter
    l1_dist = Counter(p.get("category_l1", "其他") for p in papers)
    conf_dist = Counter(p.get("conference", "?") for p in papers)

    coverage = 100 * len(papers) // raw_total if raw_total else 0

    print(f"  Total classified: {len(papers)} / {raw_total} raw ({coverage}%)")
    print(f"  Duplicate paper_id: {dups} (target: 0)")
    for f, n in blank.items():
        pct = 100 * n // len(papers)
        flag = " ⚠️" if pct > 1 else ""
        print(f"  Blank {f}: {n} ({pct}%){flag}")
    print(f"\n  By conference: {dict(conf_dist)}")
    print(f"\n  L1 distribution (top 10):")
    for l1, cnt in l1_dist.most_common(10):
        print(f"    {l1}: {cnt}")
    if unknown_l1:
        print(f"\n  ⚠️  Unknown L1 values ({len(unknown_l1)} types, {sum(unknown_l1.values())} papers):")
        for k, v in sorted(unknown_l1.items(), key=lambda x: -x[1])[:10]:
            print(f"    '{k}': {v}")
    else:
        print("\n  ✓ All L1 values in whitelist")

    ok = (coverage >= 95 and dups == 0 and blank["category_l1"] < len(papers) * 0.01)
    print(f"\n{'✓ PASS' if ok else '✗ FAIL'}")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",        nargs="+", help="raw JSON files")
    ap.add_argument("--classified", help="merged classified JSON file")
    args = ap.parse_args()

    raw_total = 0
    if args.raw:
        raw_total = check_raw(args.raw)
    if args.classified:
        check_classified(args.classified, raw_total)

if __name__ == "__main__":
    main()
