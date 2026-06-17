# 顶会论文热点分析 · CVPR 2026 / ICLR 2026 / NeurIPS 2025

> 对三大 AI/ML 顶会 **全量 15,800 篇** 论文（标题 + 摘要），用 **MiniMax-M2.5 大模型逐篇分类**，并自动生成**中文标题 + 中文摘要**，产出可检索的论文浏览站、多维交互可视化与中文 Excel 综述表。

🔗 **在线主页（GitHub Pages）**：<https://richardchenzhihui.github.io/ai-top-conf-2026-survey/>

| 入口 | 说明 |
|---|---|
| 🔍 [检索论文](https://richardchenzhihui.github.io/ai-top-conf-2026-survey/explorer.html) | 按关键词 / 会议 / 类别即时筛选 15,800 篇，看中文标题与摘要，一键跳转原文 |
| 📊 [交互看板](https://richardchenzhihui.github.io/ai-top-conf-2026-survey/dashboard.html) | 7 张 Plotly 交互图表：三会全景、核心方向热度、L2 子方向、Track 分布等 |
| 📑 [下载 Excel](https://github.com/richardChenzhihui/ai-top-conf-2026-survey/raw/main/survey_2026.xlsx) | 5 个工作表中文综述大表（含中文标题/摘要），可筛选可冻结 |

---

## 1. 数据规模与覆盖

| 会议 | 论文数 | 已归入明确类别 | “其他”占比 |
|---|---|---|---|
| CVPR 2026 | 5,163 | 4,774 | 7.5% |
| ICLR 2026 | 5,351 | 4,953 | 7.4% |
| NeurIPS 2025 | 5,286 | 4,823 | 8.8% |
| **合计** | **15,800** | **14,550** | **7.9%** |

> 大模型逐篇判读语义后，绝大多数论文都能归入 37 个明确方向；“其他”仅占 **7.9%**（相比纯关键词规则的 ~22% 大幅下降），仅保留真正难以归类的少数论文，作为诚实残差。

## 1.1 各会议热门方向 Top 8（直接看数据）

**CVPR 2026**：三维视觉(NeRF/3DGS) 730 · 生成模型(扩散/GAN) 712 · 多模态大模型 687 · 底层视觉 224 · 视频理解 207 · 目标检测与分割 199 · 自动驾驶 184 · 具身智能与机器人 175

**ICLR 2026**：生成模型 464 · 强化学习后训练(RLHF/DPO/GRPO) 368 · 多模态大模型 344 · 理论与优化 318 · Agent 系统 268 · 评测与基准 255 · AI4Science 248 · 经典强化学习与控制 245

**NeurIPS 2025**：理论与优化 533 · 生成模型 413 · 经典强化学习与控制 316 · AI4Science 301 · 多模态大模型 284 · 强化学习后训练 262 · 图神经网络 229 · 三维视觉 210

> 一眼可见会议气质差异：CVPR 以视觉（三维/生成/多模态）为主；ICLR 偏大模型与对齐/Agent；NeurIPS 偏理论、强化学习与科学智能。

## 1.2 六大核心方向 · 各会议论文数（占比）

| 核心方向 | CVPR 2026 | ICLR 2026 | NeurIPS 2025 |
|---|---|---|---|
| 多模态大模型(MLLM/VLM) | 687 (13.3%) | 344 (6.4%) | 284 (5.4%) |
| 强化学习后训练(RLHF/DPO/GRPO) | 111 (2.1%) | 368 (6.9%) | 262 (5.0%) |
| Agent 系统 | 139 (2.7%) | 268 (5.0%) | 160 (3.0%) |
| Agent 后训练 | 24 (0.5%) | 67 (1.3%) | 28 (0.5%) |
| 大语言模型预训练 | 7 (0.1%) | 146 (2.7%) | 146 (2.8%) |
| 大模型 Infra(推理/训练/效率) | 55 (1.1%) | 213 (4.0%) | 165 (3.1%) |

> 👉 想按关键词/类别快速找论文，请用 [在线检索页](https://richardchenzhihui.github.io/ai-top-conf-2026-survey/explorer.html)。

## 2. 数据来源（仅官方 API，绝不爬虫）

| 会议 | 来源 | 方式 |
|---|---|---|
| CVPR 2026 | 虚拟会议站静态 JSON（`cvpr-2026-orals-posters.json` + `cvpr-2026-abstracts.json`） | 两次 HTTP GET |
| ICLR 2026 | OpenReview API v2（`venueid=ICLR.cc/2026/Conference`） | 分页拉取 |
| NeurIPS 2025 | OpenReview API v2（`venueid=NeurIPS.cc/2025/Conference`） | 分页拉取 |

⚠️ **数据采集红线**：① 不下载任何 PDF；② 不对会议网站做 HTML 爬虫（易被封 IP）；③ 仅使用公开官方 API。

## 3. 分类方法（大模型逐篇判读，可复现、可审计）

分类由 [`scripts/classify_minimax.py`](scripts/classify_minimax.py) 完成，调用 **MiniMax `MiniMax-M2.5-highspeed`**（OpenAI 兼容 API）。核心设计：

1. **两级标签体系**：23 个通用 L1 + 14 个补充 L1（共 37 个，覆盖三会的全部主流子领域），其中 6 个用户重点关注类带精细 L2 子方向。体系在 [`classify_keywords.py`](scripts/classify_keywords.py) 的 `CATEGORY_ORDER` / `L2_RULES` 中**唯一定义**，分类器、可视化、Excel 三处共享，永不漂移。
2. **逐篇语义判读**：把每篇的 标题 + 摘要 + 关键词 喂给大模型，让它从固定 37 个 L1 中择一；若属 6 个核心类，再从该类的候选 L2 中择一。相比关键词规则，大模型能理解语义、消歧、显著降低“其他”比例。
3. **同时产出中文**：`title_zh`（流畅中文标题，保留 method/benchmark/model 英文专名）与 `summary_zh`（2–3 句中文：问题 + 方法 + 贡献）。
4. **批量并行**：一次请求批量分类 10 篇，返回 JSON 数组；用论文下标回映射，抗乱序/抗漏条。15,800 篇约 1,580 次请求，多线程并发 + 指数退避重试。
5. **稳健工程**：① 每批结果先落盘 `data/_analysis/{conf}/llm_batch_NNNN.json`，**断点可续跑**、中断不重复计费；② `response_format=json_object` + [`json_repair`](https://github.com/mangiucugna/json_repair) 容错解析，杜绝个别非法 JSON 丢数据；③ L1 经规范化匹配体系白名单，非法值归“其他”；④ 合并时按 `paper_id` 做集合差校验，确保覆盖率 100%。

> 保留了原关键词分类器 [`classify_keywords.py`](scripts/classify_keywords.py) 作为分类体系的真源与零依赖离线兜底。

## 4. 分类体系（L1）

**★ 6 个核心关注类（带精细 L2）**：多模态大模型(MLLM/VLM)、强化学习后训练(RLHF/DPO/GRPO/奖励模型)、Agent 系统、Agent 后训练、大语言模型预训练、大模型 Infra。

**通用 NLP/ML**：推理与规划、代码生成、安全与对齐、评测与基准、图神经网络、理论与优化、对话与文本生成、经典强化学习与控制、自监督与表征学习、域泛化与 OOD、持续学习、联邦学习、隐私安全与可信 ML、优化算法与训练方法、可解释性与模型理解、AI4Science、时间序列、推荐与检索、语音与音频、神经网络架构与压缩。

**计算机视觉**：目标检测与分割、生成模型(扩散/GAN)、三维视觉(NeRF/3DGS)、视频理解、底层视觉、人体相关、医学影像、自动驾驶、遥感与卫星、具身智能与机器人(VLA)。

## 5. 产物

- 🌐 **在线浏览站 `docs/`** — 由 [`scripts/build_site.py`](scripts/build_site.py) 生成：
  - `index.html` 友好落地页（关键数字 + 简洁分析 + 各入口）；
  - `explorer.html` 论文检索页（搜索 + 会议/类别筛选 + 中文标题摘要 + 原文外链）；
  - `dashboard.html` 7 图交互看板；`data/papers.json` 精简结构化数据。
- 📑 **`survey_2026.xlsx`** — 5 个工作表：全量列表（15,800 行，含**中文标题/中文摘要**，可筛选/冻结首行）、核心热点统计、CVPR/ICLR/NeurIPS 各自分类表（含内嵌柱状图）。
- 🗂 **`data/_analysis/*_classified.json`** — 每篇论文带 `category_l1 / category_l2 / title_zh / summary_zh` 的完整结构化数据。

## 6. 一键复现

```bash
pip install -r requirements.txt
export MINIMAX_API_KEY=<你的 MiniMax API Key>     # 分类需要；其余步骤不需要

# (可选) 重新抓取原始数据 —— 仓库已附 data/*_raw.json，可跳过
python3 scripts/fetch_cvpr2026.py
python3 scripts/fetch_openreview.py --venueid "ICLR.cc/2026/Conference"    --conf iclr    --year 2026 --output data/iclr2026_raw.json
python3 scripts/fetch_openreview.py --venueid "NeurIPS.cc/2025/Conference" --conf neurips --year 2025 --output data/neurips2025_raw.json

# 1) 大模型分类（断点可续跑，输出 data/_analysis/*_classified.json）
python3 scripts/classify_minimax.py --conf all --batch-size 10 --workers 10
#    无 API Key 时可用零依赖的关键词版兜底：python3 scripts/classify_keywords.py

# 2) 生成交互可视化看板
python3 scripts/visualize.py        # -> survey_2026_viz.html

# 3) 生成中文 Excel 综述
python3 scripts/build_excel.py      # -> survey_2026.xlsx

# 4) 生成在线浏览站（检索页 + 落地页 + 看板）
python3 scripts/build_site.py       # -> docs/{index,explorer,dashboard}.html + docs/data/papers.json
```

## 7. 目录结构

```
.
├── README.md
├── LICENSE                         # MIT（含数据使用说明）
├── requirements.txt
├── docs/                           # GitHub Pages 在线站点
│   ├── index.html                  # 友好落地页
│   ├── explorer.html               # 论文检索页
│   ├── dashboard.html              # 交互可视化看板
│   └── data/papers.json            # 精简结构化数据（供检索页加载）
├── survey_2026_viz.html            # 交互看板（仓库根，便于本地打开）
├── survey_2026.xlsx                # 多 Sheet 中文 Excel 综述
├── scripts/
│   ├── classify_minimax.py         # ★ 大模型分类器（并行 + 续跑 + 中文生成）
│   ├── classify_keywords.py        # 分类体系真源 + 零依赖关键词兜底
│   ├── build_site.py               # 生成在线浏览站
│   ├── visualize.py                # Plotly 交互看板
│   ├── build_excel.py              # openpyxl 多 Sheet Excel
│   ├── fetch_cvpr2026.py           # CVPR 静态 JSON 抓取
│   ├── fetch_openreview.py         # OpenReview API 分页抓取
│   ├── split_batches.py            # raw → batch 切分（历史工具）
│   └── verify_completeness.py      # 覆盖率 / 空字段校验
└── data/
    ├── *_raw.json                  # 原始 title+abstract（官方 API 采集）
    └── _analysis/*_classified.json # 带 L1/L2 + 中文标题/摘要 的分类结果
```

## 8. 局限与说明

- 中文标题与摘要由大模型生成，绝大多数准确流畅，个别可能有细微出入，仅供快速浏览参考，严谨引用请以原文为准。
- 分类为单标签（每篇取一个 L1），交叉领域论文只归一类。
- “其他”类为体系外的诚实残差，未强行塞入。

## License

MIT（代码）。论文标题与摘要版权归原作者/出版方，仅作非商业学术研究汇集之用，采集方式见 [LICENSE](LICENSE)。
