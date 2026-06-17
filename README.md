# 顶会论文热点分析 · CVPR 2026 / ICLR 2026 / NeurIPS 2025

> 对三大 AI/ML 顶会 **全量 15,800 篇** 论文（标题 + 摘要）做规则化主题分类，并产出多维交互可视化与中文 Excel 综述表。
> **无需任何大模型 API、无需爬虫、无需下载 PDF** —— 纯 Python 关键词规则，秒级可复现。

📊 **在线交互看板（GitHub Pages）**：<https://richardchenzhihui.github.io/ai-top-conf-2026-survey/>

---

## 1. 数据规模与覆盖

| 会议 | 论文数 | 已归入明确类别 | “其他”占比 |
|---|---|---|---|
| CVPR 2026 | 5,163 | 4,186 | 18.9% |
| ICLR 2026 | 5,351 | 4,172 | 22.0% |
| NeurIPS 2025 | 5,286 | 3,916 | 25.9% |
| **合计** | **15,800** | **12,274** | **22.4%** |

> “其他”主要为学习理论 / 统计 / 小众数学等不属于本分类体系核心关注面的论文，保留为诚实的残差类，而非强行塞入。

## 2. 数据来源（仅官方 API，绝不爬虫）

| 会议 | 来源 | 方式 |
|---|---|---|
| CVPR 2026 | 虚拟会议站静态 JSON（`cvpr-2026-orals-posters.json` + `cvpr-2026-abstracts.json`） | 两次 HTTP GET |
| ICLR 2026 | OpenReview API v2（`venueid=ICLR.cc/2026/Conference`） | 分页拉取 |
| NeurIPS 2025 | OpenReview API v2（`venueid=NeurIPS.cc/2025/Conference`） | 分页拉取 |

⚠️ **数据采集红线**：① 不下载任何 PDF；② 不对会议网站做 HTML 爬虫（易被封 IP）；③ 仅使用公开官方 API。

## 3. 分类方法（规则化、可复现、可审计）

分类完全由 [`scripts/classify_keywords.py`](scripts/classify_keywords.py) 完成，**无 LLM、无随机性**，对全量 15,800 篇约 3 秒跑完。核心设计：

1. **两级标签体系**：23 个通用 L1 + 14 个补充 L1（覆盖 ICLR/NeurIPS 的通用 ML 子领域），其中 6 个用户重点关注类带精细 L2 子方向。
2. **加权打分**：标题命中 = 3 分，摘要/关键词命中 = 1 分；取最高分类别；同分按列表顺序（核心类优先）。
3. **精度加固 —— 词边界匹配**：短缩写（≤5 字符，如 `rag` / `lora` / `sam` / `mot` / `gan` / `moe`）要求词边界，避免误命中 `sto**rag**e`、`exp**lora**tion`、`**sam**ple`、`**mot**ion` 等子串。长词与短语保留子串匹配以兼容前缀/词形变化（`fine-tun` → `fine-tuning`）。
4. **连字符归一化**：`long-context` ≡ `long context`，`self-supervised` ≡ `self supervised`，消除写法差异导致的漏匹配。
5. **复合规则（Agent 后训练）**：要求标题含 `agent/agentic` **且** 出现训练动词（fine-tune / SFT / RL / 轨迹 …）**且** 含 LLM 信号，**并排除** 博弈论 / 传统 MARL，避免把“多智能体博弈”误判为“LLM Agent 后训练”。
6. **歧义覆写（OVERRIDES）**：如 medical VLM → 多模态大模型（而非医学影像）、video diffusion → 生成模型（而非视频理解）。
7. **泛匹配抑制**：评测/理论/文本生成等高频泛词类，要求标题命中才计分，避免“凡是在 benchmark 上评测”的论文都被吞进去。

> 设计目标：在“纯规则、零成本、完全可复现”的前提下，把误分类与漏分类压到尽可能低。每一条规则都对应一次真实样本审计后的修正。

## 4. 分类体系（L1）

**★ 6 个核心关注类（带精细 L2）**：多模态大模型(MLLM/VLM)、强化学习后训练(RLHF/DPO/GRPO/奖励模型)、Agent 系统、Agent 后训练、大语言模型预训练、大模型 Infra。

**通用 NLP/ML**：推理与规划、代码生成、安全与对齐、评测与基准、图神经网络、理论与优化、对话与文本生成、经典强化学习与控制、自监督与表征学习、域泛化与 OOD、持续学习、联邦学习、隐私安全与可信 ML、优化算法与训练方法、可解释性与模型理解、AI4Science、时间序列、推荐与检索、语音与音频、神经网络架构与压缩。

**计算机视觉**：目标检测与分割、生成模型(扩散/GAN)、三维视觉(NeRF/3DGS)、视频理解、底层视觉、人体相关、医学影像、自动驾驶、遥感与卫星、具身智能与机器人(VLA)。

> L1 顺序在 `classify_keywords.py` 的 `CATEGORY_ORDER` 中唯一定义，下游可视化 / Excel 脚本均 import 该列表，**永不漂移**。

## 5. 产物

- 📊 **`survey_2026_viz.html`** — 7 张交互式 Plotly 图表（三会全景对比、占比环形图、核心 6 类热度、L2 子方向分布、CVPR Track×类别、L2×会议热度图、生成模型 vs MLLM 趋势）。同步发布到 GitHub Pages。
- 📑 **`survey_2026.xlsx`** — 5 个工作表：全量列表（15,800 行，可筛选/冻结首行）、核心热点统计、CVPR/ICLR/NeurIPS 各自分类表（含内嵌柱状图）。
- 🗂 **`data/_analysis/*_classified.json`** — 每篇论文带 `category_l1` / `category_l2` 的完整结构化数据。

## 6. 一键复现

```bash
pip install -r requirements.txt

# (可选) 重新抓取原始数据 —— 仓库已附 data/*_raw.json，可跳过
python3 scripts/fetch_cvpr2026.py
python3 scripts/fetch_openreview.py --venueid "ICLR.cc/2026/Conference"    --conf iclr    --year 2026 --output data/iclr2026_raw.json
python3 scripts/fetch_openreview.py --venueid "NeurIPS.cc/2025/Conference" --conf neurips --year 2025 --output data/neurips2025_raw.json

# 1) 分类（约 3 秒，输出 data/_analysis/*_classified.json 并打印分布）
python3 scripts/classify_keywords.py

# 2) 生成交互可视化看板
python3 scripts/visualize.py        # -> survey_2026_viz.html

# 3) 生成中文 Excel 综述
python3 scripts/build_excel.py      # -> survey_2026.xlsx
```

## 7. 目录结构

```
.
├── README.md
├── LICENSE                         # MIT（含数据使用说明）
├── requirements.txt
├── docs/index.html                 # GitHub Pages 可视化看板
├── survey_2026_viz.html            # 同上（仓库根，便于本地打开）
├── survey_2026.xlsx                # 多 Sheet 中文 Excel 综述
├── scripts/
│   ├── classify_keywords.py        # ★ 核心分类器（规则 + 打分 + 词边界 + 复合规则）
│   ├── visualize.py                # Plotly 交互看板
│   ├── build_excel.py              # openpyxl 多 Sheet Excel
│   ├── fetch_cvpr2026.py           # CVPR 静态 JSON 抓取
│   ├── fetch_openreview.py         # OpenReview API 分页抓取
│   ├── split_batches.py            # raw → batch 切分（历史工具）
│   └── verify_completeness.py      # 覆盖率 / 空字段校验
└── data/
    ├── *_raw.json                  # 原始 title+abstract（官方 API 采集）
    └── _analysis/*_classified.json # 带 L1/L2 标签的分类结果
```

## 8. 局限与说明

- 关键词规则**不做翻译/摘要**，故 `title_zh` / `summary_zh` 字段留空（可后续按需补全）。
- “其他”类约 22%，主要为学习理论/统计/小众方向，属本体系外的诚实残差。
- 分类为单标签（取最高分），交叉领域论文只归一类。

## License

MIT（代码）。论文标题与摘要版权归原作者/出版方，仅作非商业学术研究汇集之用，采集方式见 [LICENSE](LICENSE)。
