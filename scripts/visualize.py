"""
Generate interactive HTML visualization dashboard for the paper survey.

Usage:
  python3 scripts/visualize.py

Output:
  survey_2026_viz.html  (in project root, open in browser)
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Canonical taxonomy is defined once in classify_keywords.py — import it so the
# visualization never drifts out of sync with the classifier.
sys.path.insert(0, str(Path(__file__).parent))
from classify_keywords import CATEGORY_ORDER, CORE_CATEGORIES

BASE = Path(__file__).parent.parent
ANA  = BASE / "data" / "_analysis"
OUT  = BASE / "survey_2026_viz.html"

CONFS = ["cvpr", "iclr", "neurips"]
CONF_LABELS = {"cvpr": "CVPR 2026", "iclr": "ICLR 2026", "neurips": "NeurIPS 2025"}
CONF_COLORS = {"cvpr": "#1f77b4", "iclr": "#ff7f0e", "neurips": "#2ca02c"}

CORE_CATS = list(CORE_CATEGORIES)

CORE_SHORT = {
    "多模态大模型（MLLM/VLM）":              "MLLM/VLM",
    "强化学习后训练（RLHF/DPO/GRPO/奖励模型）": "RL后训练",
    "Agent 系统":                            "Agent系统",
    "Agent 后训练":                          "Agent后训练",
    "大语言模型预训练（架构/数据/规模）":       "LLM预训练",
    "大模型 Infra（推理/训练/效率系统）":      "大模型Infra",
}

ALL_CATS = list(CATEGORY_ORDER)

# short label = text before the first Chinese paren, with a few manual overrides
CAT_SHORT = {c: c.split("（")[0].strip() for c in ALL_CATS}
CAT_SHORT["其他"] = "其他"

# ── Load ──────────────────────────────────────────────────────────────────
def load():
    per_conf = {}
    for conf in CONFS:
        path = ANA / f"{conf}_classified.json"
        per_conf[conf] = json.loads(path.read_text()) if path.exists() else []
    return per_conf

per_conf = load()
totals = {c: len(per_conf[c]) for c in CONFS}

# ── Fig 1: 三会 L1 论文数对比（不含"其他"）─────────────────────────────────
def fig_l1_comparison():
    # sort categories by combined volume (descending) for readability
    combined = Counter()
    for conf in CONFS:
        combined.update(p["category_l1"] for p in per_conf[conf])
    cats_no_other = sorted(
        [c for c in ALL_CATS if c != "其他"],
        key=lambda c: -combined.get(c, 0),
    )
    short_labels = [CAT_SHORT[c] for c in cats_no_other]

    fig = go.Figure()
    for conf in CONFS:
        counts = Counter(p["category_l1"] for p in per_conf[conf])
        y = [counts.get(c, 0) for c in cats_no_other]
        fig.add_trace(go.Bar(
            name=CONF_LABELS[conf],
            x=short_labels,
            y=y,
            marker_color=CONF_COLORS[conf],
            text=y,
            textposition="outside",
            textfont=dict(size=9),
        ))

    fig.update_layout(
        title='三大会议 L1 类别论文数对比（不含"其他"）',
        barmode="group",
        xaxis_tickangle=-40,
        xaxis_tickfont=dict(size=10),
        yaxis_title="论文数",
        legend=dict(orientation="h", y=1.08),
        height=550,
        margin=dict(t=80, b=120),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#eeeeee")
    return fig

# ── Fig 2: 各会议 L1 占比饼图（3 合 1）──────────────────────────────────────
def fig_pie_triplet():
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]],
        subplot_titles=[CONF_LABELS[c] for c in CONFS],
    )
    # collapse small slices into "其他"
    THRESHOLD = 0.02
    for col, conf in enumerate(CONFS, 1):
        counts = Counter(p["category_l1"] for p in per_conf[conf])
        total = totals[conf]
        labels, values = [], []
        other_n = 0
        for cat in ALL_CATS:
            n = counts.get(cat, 0)
            if n / total < THRESHOLD:
                other_n += n
            else:
                labels.append(CAT_SHORT[cat])
                values.append(n)
        labels.append("其他 (<2%)")
        values.append(other_n)
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            name=CONF_LABELS[conf],
            textinfo="percent",
            hovertemplate="%{label}<br>%{value} 篇<br>%{percent}<extra></extra>",
            hole=0.35,
        ), row=1, col=col)

    fig.update_layout(
        title="各会议论文 L1 类别占比分布",
        height=480,
        legend=dict(orientation="v", x=1.01),
        paper_bgcolor="white",
        margin=dict(t=70, b=20, l=10, r=160),
    )
    return fig

# ── Fig 3: 核心 6 类 — 三会热度对比（归一化到各会议总量）───────────────────
def fig_core_normalized():
    fig = go.Figure()
    for conf in CONFS:
        counts = Counter(p["category_l1"] for p in per_conf[conf])
        total = totals[conf]
        y = [counts.get(c, 0) / total * 100 for c in CORE_CATS]
        fig.add_trace(go.Bar(
            name=CONF_LABELS[conf],
            x=[CORE_SHORT[c] for c in CORE_CATS],
            y=y,
            marker_color=CONF_COLORS[conf],
            text=[f"{v:.1f}%" for v in y],
            textposition="outside",
        ))

    fig.update_layout(
        title="核心 6 类 — 各会议论文占比（% of conference total）",
        barmode="group",
        yaxis_title="占比 (%)",
        xaxis_tickfont=dict(size=12),
        legend=dict(orientation="h", y=1.08),
        height=460,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=80, b=60),
    )
    fig.update_yaxes(gridcolor="#eeeeee")
    return fig

# ── Fig 4: 核心 6 类各自的 L2 分布（3 行 × 2 列，给 y 轴充足空间）──────────
def fig_l2_breakdown():
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[CORE_SHORT[c] for c in CORE_CATS],
        vertical_spacing=0.10,
        horizontal_spacing=0.28,   # wide gap so long y-axis labels don't collide
    )
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, cat in enumerate(CORE_CATS):
        row = idx // 2 + 1
        col = idx %  2 + 1
        color = COLORS[idx]

        l2_counts = defaultdict(int)
        for conf in CONFS:
            for p in per_conf[conf]:
                if p["category_l1"] == cat:
                    l2 = p.get("category_l2") or "未分类"
                    l2_counts[l2] += 1

        if not l2_counts:
            continue
        # keep top-10 L2 labels to avoid crowding
        sorted_l2 = sorted(l2_counts.items(), key=lambda x: x[1])[-10:]
        labels = [x[0] for x in sorted_l2]
        values = [x[1] for x in sorted_l2]

        fig.add_trace(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            text=values,
            textposition="outside",
            textfont=dict(size=10),
            marker_color=color,
            name=cat,
            showlegend=False,
        ), row=row, col=col)

        # set x-range with padding so text labels don't clip
        max_val = max(values) if values else 1
        fig.update_xaxes(range=[0, max_val * 1.25], row=row, col=col,
                         gridcolor="#eeeeee", tickfont=dict(size=9))
        fig.update_yaxes(tickfont=dict(size=10), row=row, col=col)

    fig.update_layout(
        title="核心 6 类 L2 子方向分布（三会合计，Top-10）",
        height=1100,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=80, b=40, l=200, r=80),
    )
    return fig

# ── Fig 5: CVPR track 分布（Oral/Highlight/Poster）vs L1 ──────────────────
def fig_cvpr_tracks():
    track_map = {}
    for p in per_conf["cvpr"]:
        t = (p.get("track") or "Other").strip()
        if t not in ("Oral", "Highlight", "Poster"):
            t = "Other/TBD"
        cat = CAT_SHORT[p.get("category_l1", "其他")]
        track_map.setdefault(cat, Counter())[t] += 1

    # sort by total
    sorted_cats = sorted(track_map.keys(), key=lambda c: sum(track_map[c].values()), reverse=True)
    sorted_cats = [c for c in sorted_cats if c != "其他"][:18]

    colors = {"Oral": "#d62728", "Highlight": "#ff7f0e", "Poster": "#1f77b4", "Other/TBD": "#aaaaaa"}
    fig = go.Figure()
    for track in ["Oral", "Highlight", "Poster", "Other/TBD"]:
        y = [track_map.get(c, Counter()).get(track, 0) for c in sorted_cats]
        fig.add_trace(go.Bar(
            name=track,
            x=sorted_cats,
            y=y,
            marker_color=colors[track],
        ))

    fig.update_layout(
        title="CVPR 2026 — 各 L1 类别中 Oral / Highlight / Poster 分布",
        barmode="stack",
        xaxis_tickangle=-35,
        xaxis_tickfont=dict(size=10),
        yaxis_title="论文数",
        legend=dict(orientation="h", y=1.08),
        height=520,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=80, b=130),
    )
    fig.update_yaxes(gridcolor="#eeeeee")
    return fig

# ── Fig 6: 三会热门 L2 Top-10 heatmap ────────────────────────────────────
def fig_l2_heatmap():
    # collect all L2 for core cats across conferences, exclude empty/未分类
    l2_per_conf = {c: Counter() for c in CONFS}
    for conf in CONFS:
        for p in per_conf[conf]:
            if p["category_l1"] in CORE_CATS:
                l2 = p.get("category_l2") or ""
                if l2 and l2 not in ("未分类", "其他子方向"):
                    l2_per_conf[conf][l2] += 1

    # top-20 L2 by total
    total_l2 = Counter()
    for c in CONFS:
        total_l2.update(l2_per_conf[c])
    top_l2 = [l2 for l2, _ in total_l2.most_common(20)]

    z = [[l2_per_conf[conf].get(l2, 0) for conf in CONFS] for l2 in top_l2]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[CONF_LABELS[c] for c in CONFS],
        y=top_l2,
        colorscale="Blues",
        text=z,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hoverongaps=False,
    ))
    fig.update_layout(
        title="核心类别 Top-20 L2 子方向 × 会议 热度图",
        height=580,
        paper_bgcolor="white",
        margin=dict(t=70, b=40, l=220, r=20),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig

# ── Fig 7: 生成模型 vs MLLM/VLM 跨会议趋势（bar race proxy）────────────────
def fig_generative_vs_mllm():
    gen_cat  = "生成模型（扩散/GAN/视频生成/图像编辑，非 MLLM 驱动）"
    mllm_cat = "多模态大模型（MLLM/VLM）"

    rows = []
    for conf in CONFS:
        counts = Counter(p["category_l1"] for p in per_conf[conf])
        total  = totals[conf]
        rows.append({
            "conf":   CONF_LABELS[conf],
            "生成模型（扩散/GAN）":  counts.get(gen_cat, 0),
            "MLLM/VLM":             counts.get(mllm_cat, 0),
        })

    fig = go.Figure()
    for label, color in [("生成模型（扩散/GAN）", "#ff7f0e"), ("MLLM/VLM", "#1f77b4")]:
        fig.add_trace(go.Scatter(
            x=[r["conf"] for r in rows],
            y=[r[label] for r in rows],
            mode="lines+markers+text",
            name=label,
            text=[r[label] for r in rows],
            textposition="top center",
            line=dict(width=3),
            marker=dict(size=12, color=color),
        ))

    fig.update_layout(
        title="生成模型 vs MLLM/VLM — 三会论文数趋势",
        yaxis_title="论文数",
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=70, b=40),
    )
    fig.update_yaxes(gridcolor="#eeeeee")
    return fig

# ── Assemble HTML ─────────────────────────────────────────────────────────
def build_html():
    figs = [
        ("1. 三会 L1 类别全景对比",         fig_l1_comparison()),
        ("2. 各会议类别占比（环形图）",       fig_pie_triplet()),
        ("3. 核心 6 类热度归一化对比",        fig_core_normalized()),
        ("4. 核心 6 类 L2 子方向分布",        fig_l2_breakdown()),
        ("5. CVPR 2026 Track × 类别分布",     fig_cvpr_tracks()),
        ("6. 核心类别 L2 × 会议 热度图",      fig_l2_heatmap()),
        ("7. 生成模型 vs MLLM/VLM 趋势",      fig_generative_vs_mllm()),
    ]

    html_parts = ["""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>CVPR 2026 / ICLR 2026 / NeurIPS 2025 论文热点分析</title>
<style>
  body { font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #f5f7fa; margin: 0; padding: 0; }
  .hero { background: linear-gradient(135deg, #1F3864 0%, #2E75B6 100%);
          color: white; padding: 40px 60px; }
  .hero h1 { margin: 0 0 8px; font-size: 28px; }
  .hero p  { margin: 0; opacity: 0.85; font-size: 14px; }
  .stat-row { display: flex; gap: 20px; margin-top: 24px; }
  .stat { background: rgba(255,255,255,0.15); border-radius: 8px;
          padding: 12px 24px; text-align: center; }
  .stat-n { font-size: 32px; font-weight: bold; }
  .stat-l { font-size: 12px; opacity: 0.8; margin-top: 2px; }
  .section { background: white; border-radius: 10px; margin: 24px 40px;
             padding: 20px 28px; box-shadow: 0 1px 6px rgba(0,0,0,0.07); }
  .section h2 { margin: 0 0 16px; font-size: 17px; color: #1F3864;
                border-left: 4px solid #2E75B6; padding-left: 10px; }
  footer { text-align: center; color: #aaa; font-size: 12px;
           padding: 20px 0 40px; }
</style>
</head>
<body>
<div class="hero">
  <h1>顶会论文热点分析 · 2025–2026</h1>
  <p>基于 MiniMax-M2.5 大模型逐篇分类，覆盖 CVPR 2026 / ICLR 2026 / NeurIPS 2025 全量论文</p>
  <div class="stat-row">
"""]
    for conf in CONFS:
        html_parts.append(f"""    <div class="stat">
      <div class="stat-n">{totals[conf]:,}</div>
      <div class="stat-l">{CONF_LABELS[conf]}</div>
    </div>
""")
    html_parts.append(f"""    <div class="stat">
      <div class="stat-n">{sum(totals.values()):,}</div>
      <div class="stat-l">总计</div>
    </div>
  </div>
</div>
""")

    for title, fig in figs:
        div = fig.to_html(full_html=False, include_plotlyjs="cdn" if title == figs[0][0] else False,
                          config={"displayModeBar": True, "responsive": True})
        html_parts.append(f'<div class="section"><h2>{title}</h2>{div}</div>\n')

    html_parts.append("<footer>Generated by classify_minimax.py + visualize.py · 数据来源：CVPR虚拟站、OpenReview API</footer></body></html>")
    return "".join(html_parts)

html = build_html()
OUT.write_text(html, encoding="utf-8")
print(f"Saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")
print("Open in browser to view.")
