"""
Keyword-based paper classifier — no LLM, no API calls.
Reads the three raw JSON files, classifies each paper into L1/L2 via rule-based
keyword matching, then writes one classified JSON per conference.

Usage:
  python3 scripts/classify_keywords.py

Output:
  data/_analysis/cvpr_classified.json
  data/_analysis/iclr_classified.json
  data/_analysis/neurips_classified.json
"""

import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent / "data"
OUT  = BASE / "_analysis"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# L1 Category definitions
# Each entry: (category_name, title_patterns, body_patterns, l2_rules)
#
# Scoring: title match = 3 pts, body (abstract+keywords) match = 1 pt
# Category with highest score wins. Ties broken by list order (higher = earlier).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Keyword matching engine (precision-hardened)
#
# Naive substring matching ("rag" in "storage") is the #1 source of false
# positives. Fix: short alphanumeric acronyms (<=5 chars, e.g. rag/lora/sam/
# mot/gan/moe/orm) are matched with word boundaries; longer tokens and
# multi-word phrases keep substring matching so prefix/inflection forms still
# match ("fine-tun" -> "fine-tuning"/"fine-tuned").
# ---------------------------------------------------------------------------
_KW_CACHE = {}

def _norm(s):
    """Lowercase and treat hyphen/underscore/slash as spaces, so 'long-context'
    matches keyword 'long context' and vice-versa. Collapses repeated spaces."""
    return re.sub(r'\s+', ' ', re.sub(r'[-_/]', ' ', (s or "").lower()))

def _compile(kw):
    pat = _KW_CACHE.get(kw)
    if pat is None:
        kw = _norm(kw)
        if re.fullmatch(r'[a-z0-9]+', kw) and len(kw) <= 5:
            # short acronym -> require non-alphanumeric boundaries on both sides
            src = r'(?<![a-z0-9])' + re.escape(kw) + r'(?![a-z0-9])'
        else:
            # longer token / phrase -> substring (allows prefix & inflection)
            src = re.escape(kw)
        pat = re.compile(src)
        _KW_CACHE[kw] = pat
    return pat

# ---------------------------------------------------------------------------
# L2 sub-rules: (l1_name) -> list of (l2_label, phrases)
# First matching rule wins; falls back to "其他子方向"
# ---------------------------------------------------------------------------
L2_RULES = {
    "多模态大模型（MLLM/VLM）": [
        ("视觉数学推理",        ["visual math", "multimodal math", "geometry reasoning", "diagram reasoning", "visual reasoning"]),
        ("视觉定位与接地（Grounding）", ["grounding", "referring expression", "visual grounding", "phrase grounding", "region caption"]),
        ("文档/图表/OCR 理解",  ["document understanding", "chart understanding", "ocr", "table understanding", "infographic", "document vqa"]),
        ("医学多模态",          ["medical vlm", "clinical vqa", "radiology vlm", "pathology vqa", "medical multimodal"]),
        ("多模态幻觉与评测",    ["hallucination", "mllm evaluation", "vlm benchmark", "multimodal benchmark", "multimodal evaluation"]),
        ("音频-视觉-语言",      ["audio-visual", "audio visual", "speech visual", "audio language", "sound visual"]),
        ("图像/视频生成控制（指令驱动）", ["instruction-guided generation", "text-to-image generation", "instruction-driven", "image editing with language", "image generation with text"]),
        ("视频理解与时序推理",  ["video llm", "video language model", "video question answering", "video vqa", "temporal reasoning", "video caption"]),
        ("图像理解与视觉推理",  []),  # catch-all for MLLM
    ],
    "强化学习后训练（RLHF/DPO/GRPO/奖励模型）": [
        ("视觉推理 RL（VLM-RL）",  ["visual rl", "vlm rl", "vision rl", "visual reinforcement", "multimodal reinforcement"]),
        ("GRPO/群体相对策略优化",  ["grpo", "group relative policy", "dapo"]),
        ("DPO 偏好学习",           ["dpo", "direct preference optimization", "preference optimization"]),
        ("过程奖励模型（PRM）",    ["process reward", "prm", "step-level reward"]),
        ("结果奖励模型（ORM）",    ["outcome reward", "orm", "outcome-level reward"]),
        ("奖励模型（RM）构建",     ["reward model", "reward modeling", "reward function"]),
        ("宪法 AI/RLAIF",          ["constitutional ai", "rlaif", "ai feedback"]),
        ("数学 RL 强化",           ["math rl", "mathematical reinforcement", "math reward"]),
        ("代码 RL 强化",           ["code rl", "code reinforcement"]),
        ("拒绝采样与 Best-of-N",   ["rejection sampling", "best-of-n", "bon", "best of n"]),
        ("RLHF/PPO 后训练",        ["rlhf", "ppo", "proximal policy"]),
    ],
    "Agent 系统": [
        ("GUI/Web/OS Agent",       ["gui agent", "web agent", "os agent", "browser agent", "computer use agent", "ui agent"]),
        ("代码 Agent",             ["code agent", "coding agent", "software agent", "programming agent"]),
        ("具身 Agent（导航/操作）", ["embodied agent", "navigation agent", "manipulation agent", "robotic agent"]),
        ("工具调用与函数调用（Tool Use）", ["tool use", "function calling", "tool call", "api call", "tool-augmented"]),
        ("多 Agent 协作框架",      ["multi-agent", "multi agent", "agent collaboration", "agent framework", "agent society"]),
        ("规划与反思机制",         ["planning agent", "reflection", "self-reflection", "chain of thought agent", "react"]),
        ("记忆与上下文管理",       ["agent memory", "long-term memory", "context management", "memory augmented agent"]),
        ("世界模型",               ["world model", "environment model"]),
    ],
    "Agent 后训练": [
        ("Agent RL 训练（轨迹级强化）", ["agent rl", "trajectory-level rl", "trajectory reinforcement", "agent reinforcement learning"]),
        ("Agent 轨迹数据构建与合成",    ["agent trajectory", "trajectory data", "agent data synthesis", "agent demonstration"]),
        ("Agent 自改进/自修正",         ["agent self-improve", "agent self-correction", "agent self-refine"]),
        ("工具学习后训练",              ["tool learning", "tool fine-tuning"]),
        ("在线 Agent 微调",             ["online agent", "online fine-tuning agent"]),
    ],
    "大语言模型预训练（架构/数据/规模）": [
        ("预训练架构设计（Transformer/SSM/混合架构/MoE）", ["mamba", "ssm", "state space model", "mixture of experts", "moe", "hybrid architecture", "transformer architecture", "linear attention"]),
        ("预训练数据工程与配比",     ["data mixture", "data curation", "data selection", "pretraining data", "web crawl"]),
        ("持续预训练与领域适配",     ["continual pretraining", "domain adaptation", "domain-adaptive pretraining"]),
        ("多语言预训练",             ["multilingual", "cross-lingual", "multilingual pretraining"]),
        ("模型规模与涌现规律",       ["scaling law", "emergent", "emergent ability", "model scale"]),
        ("分词与词表设计",           ["tokenizer", "tokenization", "vocabulary", "byte pair encoding", "bpe"]),
    ],
    "大模型 Infra（推理/训练/效率系统）": [
        ("视觉 Token 压缩（CVPR 侧）",      ["visual token compression", "visual token pruning", "token merging", "visual token reduction"]),
        ("投机解码（Speculative Decoding）", ["speculative decoding", "draft model", "speculative sampling"]),
        ("量化与稀疏化（W4/INT8/稀疏注意力）", ["quantization", "int8", "int4", "w4", "w8", "sparse attention", "pruning", "weight quantization"]),
        ("KV Cache 压缩与管理",             ["kv cache", "key-value cache", "attention cache", "cache compression"]),
        ("长上下文建模与位置外推",           ["long context", "long-context", "position extrapolation", "rope", "rotary embedding", "yarn", "alibi"]),
        ("训练并行与显存优化（TP/PP/ZeRO）", ["tensor parallel", "pipeline parallel", "zero redundancy", "zero-3", "megatron", "deepspeed", "gradient checkpointing"]),
        ("模型合并与 MoE 路由",              ["model merging", "model fusion", "moe routing", "expert routing", "load balancing"]),
        ("Prompt 压缩与上下文蒸馏",          ["prompt compression", "context compression", "context distillation", "prompt distillation"]),
    ],
}

# ---------------------------------------------------------------------------
# L1 Category matching rules
# Format: (L1_name, title_keywords, body_keywords, bonus_if_both)
# ---------------------------------------------------------------------------
CATEGORIES = [
    # ── TIER 1: LLM/AI ──────────────────────────────────────────────────────
    {
        "name": "多模态大模型（MLLM/VLM）",
        "title": [
            "vlm", "mllm", "vision-language model", "multimodal llm",
            "visual instruction", "visual reasoning", "visual question",
            "llava", "instructblip", "gpt-4v", "qwen-vl", "internvl",
            "moondream", "cogvlm", "gemini vision", "flamingo",
            "multimodal large language", "visual language model",
        ],
        "body": [
            "multimodal large language model", "vision language model",
            "visual question answering", "image-text instruction",
            "visual grounding", "multimodal reasoning",
            "image captioning with llm", "vision encoder",
        ],
    },
    {
        "name": "强化学习后训练（RLHF/DPO/GRPO/奖励模型）",
        "title": [
            "rlhf", "dpo", "grpo", "dapo", "orpo", "rlvr",
            "reward model", "preference optimization", "preference learning",
            "reinforcement learning from human feedback",
            "constitutional ai", "rlaif", "process reward", "outcome reward",
            "rejection sampling", "best-of-n", "verifiable reward",
            "outcome-supervised", "process-supervised",
        ],
        "body": [
            "rlhf", "direct preference optimization", "group relative policy",
            "reward modeling", "policy optimization for llm",
            "preference data", "human preference", "ai feedback",
            "rlvr", "verifiable reward", "verifiable reference",
        ],
    },
    {
        "name": "Agent 系统",
        "title": [
            "gui agent", "web agent", "os agent", "autonomous agent",
            "agentic", "code agent", "software agent",
            "embodied agent", "multi-agent", "multi agent",
            "tool use", "function calling", "react agent",
            "browser agent", "computer use",
        ],
        "body": [
            "autonomous agent", "tool-augmented", "multi-agent system",
            "agent framework", "agent workflow", "planning and execution",
            "function calling", "api calling", "tool use", "agent system",
        ],
    },
    {
        "name": "Agent 后训练",
        "title": [
            # explicit compound phrases
            "agent fine-tun", "agent finetuning", "agent sft",
            "agent training", "agent rl", "agent reinforcement",
            "agentic reinforcement", "agentic training", "agentic fine",
            "agent trajectory", "trajectory-level", "agent learning",
            "training llm agent", "training language agent",
            "distilling.*agent", "agent distillation",
            "agent reward", "agent policy optimization",
            "online agent", "offline agent training",
            # specific strong keywords
            "agenttuning", "fireact", "agent data synthesis",
            "agent imitation", "agent behavioral",
        ],
        "body": [
            "agent fine-tuning", "agent sft", "agent reinforcement learning",
            "trajectory-level reinforcement", "agent data synthesis",
            "agent self-improvement", "agent self-correction",
            "agentic reinforcement learning", "agentic fine-tuning",
            "training llm agents", "training language model agents",
            "agent trajectory data", "agent behavior cloning",
            "online agent fine-tuning", "agent policy optimization",
            "fine-tuning llm agents", "fine-tuning language agents",
        ],
    },
    {
        "name": "大语言模型预训练（架构/数据/规模）",
        "title": [
            "language model pretraining", "llm pretraining",
            "mamba", "state space model", "mixture of experts", "moe architecture",
            "scaling law", "tokenizer", "tokenization",
            "continual pretraining", "domain pretraining",
            "multilingual pretraining", "data curation for llm",
        ],
        "body": [
            "pretraining data", "scaling law", "mamba architecture",
            "state space model", "mixture of experts",
            "tokenization strategy", "data mixture", "model scaling",
            "continual pretraining",
        ],
    },
    {
        "name": "大模型 Infra（推理/训练/效率系统）",
        "title": [
            "speculative decoding", "kv cache", "quantization",
            "model compression", "inference acceleration",
            "long context", "flash attention", "tensor parallel",
            "pipeline parallel", "deepspeed", "megatron",
            "visual token compression", "token merging", "prompt compression",
            "parameter-efficient", "peft", "lora", "low-rank adaptation",
            "knowledge distillation", "model distillation",
            "inference optimization", "llm sparsity", "model sparsity",
            "rag", "retrieval augmented generation",
        ],
        "body": [
            "speculative decoding", "kv cache compression",
            "int8 quantization", "w4 quantization", "weight quantization",
            "inference efficiency", "training efficiency",
            "long-context modeling", "position extrapolation",
            "gradient checkpointing", "tensor parallelism",
            "parameter-efficient fine-tuning", "lora",
            "knowledge distillation for llm", "model distillation",
            "retrieval-augmented generation",
        ],
    },
    # ── TIER 2: AI/ML General ────────────────────────────────────────────────
    {
        "name": "推理与规划（CoT/数学/逻辑，非 RL 类）",
        "title": [
            "chain of thought", "chain-of-thought", "mathematical reasoning",
            "math reasoning", "logical reasoning", "theorem proving",
            "step-by-step reasoning", "commonsense reasoning",
            "symbolic reasoning",
        ],
        "body": [
            "chain of thought", "mathematical reasoning",
            "logical reasoning", "step-by-step", "theorem proving",
            "commonsense reasoning", "problem solving",
        ],
    },
    {
        "name": "代码生成与编程语言处理",
        "title": [
            "code generation", "code synthesis", "program synthesis",
            "code completion", "code repair", "bug detection", "bug fix",
            "software engineering", "program generation",
            "code review", "code search",
        ],
        "body": [
            "code generation", "program synthesis", "code completion",
            "software engineering", "code repair", "programming language model",
        ],
    },
    {
        "name": "安全与对齐（幻觉/偏见/毒性/越狱，非 RL 类）",
        "title": [
            "hallucination", "factuality", "jailbreak", "toxicity",
            "bias", "fairness", "safety alignment", "trustworthy",
            "adversarial", "red teaming", "responsible ai",
            "misinformation", "disinformation",
        ],
        "body": [
            "hallucination", "factual accuracy", "jailbreak",
            "toxic content", "social bias", "safety alignment",
            "adversarial attack", "model safety", "trustworthiness",
        ],
    },
    {
        "name": "评测与基准构建",
        "title": [
            "benchmark", "evaluation benchmark", "leaderboard",
            "evaluation framework", "evaluation dataset",
        ],
        "body": [
            "we propose a benchmark", "we introduce a benchmark",
            "evaluation benchmark", "comprehensive evaluation",
        ],
    },
    {
        "name": "图神经网络与结构化学习",
        "title": [
            "graph neural network", "gnn", "graph learning",
            "knowledge graph", "graph transformer",
            "molecular graph", "protein structure",
        ],
        "body": [
            "graph neural network", "gnn", "graph learning",
            "knowledge graph", "molecular property", "protein folding",
        ],
    },
    {
        "name": "理论与优化",
        "title": [
            "convergence", "generalization bound", "pac learning",
            "optimization theory", "learning theory",
            "loss landscape", "gradient flow", "approximation theory",
            "online learning", "regret bound", "mistake bound",
            "sample complexity", "minimax", "statistical learning",
            "uncertainty quantification", "high-dimensional statistics",
            "concentration inequality", "information-theoretic",
            "optimal transport", "kernel method",
        ],
        "body": [
            "convergence rate", "generalization bound", "pac learning",
            "learning theory", "optimization landscape", "theoretical analysis",
            "regret bound", "sample complexity", "minimax optimal",
            "statistical guarantee",
        ],
    },
    {
        "name": "对话系统与文本生成",
        "title": [
            "dialogue", "conversation", "text generation",
            "machine translation", "summarization", "question answering",
            "natural language generation", "language generation",
        ],
        "body": [
            "dialogue system", "conversational", "text generation",
            "machine translation", "text summarization", "nlp",
        ],
    },
    # ── TIER 3: Computer Vision ─────────────────────────────────────────────
    {
        "name": "目标检测与分割",
        "title": [
            "object detection", "instance segmentation", "semantic segmentation",
            "panoptic segmentation", "yolo", "detr", "detection transformer",
            "open-vocabulary detection", "open vocabulary detection",
            "grounding dino", "sam", "segment anything",
        ],
        "body": [
            "object detection", "instance segmentation", "semantic segmentation",
            "bounding box", "anchor", "detection head",
        ],
    },
    {
        "name": "生成模型（扩散/GAN/视频生成/图像编辑，非 MLLM 驱动）",
        "title": [
            "diffusion model", "stable diffusion", "text-to-image",
            "image generation", "video generation", "image synthesis",
            "image editing", "inpainting", "outpainting",
            "generative model", "gan", "flow matching",
            "consistency model", "latent diffusion",
        ],
        "body": [
            "diffusion model", "text-to-image", "image generation",
            "video generation", "generative adversarial", "image editing",
            "denoising diffusion", "score matching", "flow matching",
        ],
    },
    {
        "name": "三维视觉（NeRF/3DGS/深度估计/点云）",
        "title": [
            "nerf", "neural radiance field", "3d gaussian", "3dgs", "4dgs",
            "4d gaussian", "dynamic gaussian", "gaussian splatting",
            "depth estimation", "point cloud", "3d reconstruction",
            "novel view synthesis", "lidar", "mesh reconstruction",
            "3d scene", "stereo depth", "multiview", "multi-view stereo",
        ],
        "body": [
            "neural radiance field", "3d gaussian splatting", "point cloud",
            "depth estimation", "novel view synthesis", "3d reconstruction",
            "4d gaussian", "dynamic gaussian splatting",
        ],
    },
    {
        "name": "视频理解（动作识别/跟踪/时序建模，非 MLLM 类）",
        "title": [
            "video understanding", "action recognition", "action detection",
            "temporal action", "video classification", "optical flow",
            "video object tracking", "multi-object tracking", "mot",
            "video analysis", "video representation", "visual tracking",
            "object tracking", "single object tracking", "sot",
        ],
        "body": [
            "action recognition", "temporal modeling", "optical flow",
            "video understanding", "video classification",
            "tracking algorithm", "video representation learning",
            "object tracking", "multi-object tracking",
        ],
    },
    {
        "name": "底层视觉（超分/去噪/增强）",
        "title": [
            "super resolution", "super-resolution", "image restoration",
            "image denoising", "image enhancement", "low-light",
            "image deblurring", "deraining", "dehazing",
            "image quality", "image reconstruction",
        ],
        "body": [
            "super resolution", "image restoration", "image denoising",
            "image enhancement", "low-light enhancement", "image quality",
        ],
    },
    {
        "name": "人体相关（姿态/人脸/手势/运动合成）",
        "title": [
            "human pose", "pose estimation", "face recognition",
            "facial", "hand gesture", "gesture recognition",
            "human motion", "body pose", "skeleton",
            "person re-identification", "gait", "motion generation",
            "human body",
        ],
        "body": [
            "human pose estimation", "facial recognition", "gesture recognition",
            "human motion generation", "person re-identification", "body skeleton",
        ],
    },
    {
        "name": "医学影像（非多模态大模型类）",
        "title": [
            "medical image", "pathology", "radiology",
            "ct scan", "mri", "histology", "histopathology",
            "tumor", "lesion", "clinical", "medical segmentation",
            "medical analysis", "diagnosis",
        ],
        "body": [
            "medical image", "pathology image", "radiology",
            "clinical diagnosis", "tumor detection", "histopathology",
            "mri segmentation", "ct image",
        ],
    },
    {
        "name": "自动驾驶（感知/预测/规划）",
        "title": [
            "autonomous driving", "self-driving", "vehicle",
            "traffic", "lidar perception", "bird's eye view", "bev",
            "driving perception", "motion forecasting", "trajectory prediction",
            "lane detection", "occupancy",
        ],
        "body": [
            "autonomous driving", "self-driving car", "bird's eye view",
            "traffic prediction", "driving perception", "lidar detection",
        ],
    },
    {
        "name": "遥感与卫星",
        "title": [
            "remote sensing", "satellite", "aerial image",
            "hyperspectral", "uav", "drone image", "earth observation",
            "multispectral", "sar image",
        ],
        "body": [
            "remote sensing", "satellite image", "aerial image",
            "earth observation", "hyperspectral",
        ],
    },
    # ── TIER 2.5: General ML subfields (大量 ICLR/NeurIPS 论文落点) ──────────────
    {
        "name": "经典强化学习与控制（非 LLM）",
        "title": [
            "policy gradient", "q-learning", "actor-critic", "actor critic",
            "model-based rl", "offline rl", "offline reinforcement",
            "online reinforcement", "continuous control", "robot learning",
            "sim-to-real", "deep reinforcement learning", "markov decision",
            "multi-armed bandit", "contextual bandit", "exploration strategy",
            "inverse reinforcement", "meta reinforcement", "world model for control",
            "multi-agent reinforcement learning",
        ],
        "body": [
            "policy gradient", "q-learning", "actor-critic",
            "markov decision process", "off-policy", "on-policy",
            "value function", "reward shaping", "continuous control",
            "deep reinforcement learning",
        ],
    },
    {
        "name": "具身智能与机器人（VLA/操作/导航）",
        "title": [
            "vision-language-action", "vla", "robot manipulation",
            "robotic manipulation", "robot navigation", "visual navigation",
            "manipulation policy", "dexterous", "grasping", "object navigation",
            "humanoid", "legged locomotion", "quadruped", "robotic control",
            "bimanual", "mobile manipulation",
        ],
        "body": [
            "vision-language-action", "robot manipulation", "robotic policy",
            "manipulation task", "navigation policy", "dexterous manipulation",
            "robot learning", "embodied control",
        ],
    },
    {
        "name": "自监督与表征学习",
        "title": [
            "self-supervised", "contrastive learning", "representation learning",
            "masked autoencoder", "masked image modeling", "self-distillation",
            "metric learning", "pretext task", "instance discrimination",
            "feature representation",
        ],
        "body": [
            "self-supervised learning", "contrastive learning",
            "representation learning", "masked image modeling",
            "instance discrimination", "learned representation",
        ],
    },
    {
        "name": "域泛化与分布外鲁棒性（OOD/迁移）",
        "title": [
            "domain generalization", "domain adaptation", "out-of-distribution",
            "ood detection", "distribution shift", "test-time adaptation",
            "test-time training", "domain shift", "covariate shift",
            "transfer learning", "few-shot adaptation",
        ],
        "body": [
            "domain generalization", "domain adaptation",
            "out-of-distribution", "distribution shift",
            "test-time adaptation", "unsupervised domain adaptation",
        ],
    },
    {
        "name": "持续学习与终身学习",
        "title": [
            "continual learning", "lifelong learning", "incremental learning",
            "catastrophic forgetting", "class-incremental", "online continual",
            "task-incremental",
        ],
        "body": [
            "continual learning", "catastrophic forgetting",
            "incremental learning", "lifelong learning",
        ],
    },
    {
        "name": "联邦学习与分布式学习",
        "title": [
            "federated learning", "federated", "split learning",
            "decentralized learning", "federated unlearning",
            "federated optimization",
        ],
        "body": [
            "federated learning", "client drift", "non-iid data",
            "federated optimization", "decentralized training",
        ],
    },
    {
        "name": "隐私安全与可信机器学习",
        "title": [
            "differential privacy", "privacy-preserving", "membership inference",
            "data poisoning", "backdoor attack", "model watermarking",
            "watermark", "machine unlearning", "data protection",
            "privacy attack", "model stealing", "model extraction",
        ],
        "body": [
            "differential privacy", "privacy-preserving", "membership inference",
            "backdoor attack", "data poisoning", "watermarking",
            "machine unlearning",
        ],
    },
    {
        "name": "优化算法与训练方法",
        "title": [
            "optimizer", "sgd", "adam optimizer", "gradient descent",
            "learning rate schedule", "training dynamics", "momentum optimization",
            "second-order optimization", "adaptive optimization", "newton method",
            "sharpness-aware", "muon optimizer",
        ],
        "body": [
            "optimization algorithm", "stochastic gradient descent",
            "training dynamics", "adaptive learning rate", "optimizer",
        ],
    },
    {
        "name": "可解释性与模型理解",
        "title": [
            "interpretability", "mechanistic interpretability", "explainability",
            "explainable", "probing", "feature attribution", "saliency",
            "concept bottleneck", "circuit analysis", "understanding transformers",
            "model understanding", "sparse autoencoder",
        ],
        "body": [
            "interpretability", "mechanistic", "feature attribution",
            "saliency map", "probing classifier", "explainable ai",
        ],
    },
    {
        "name": "科学智能 AI4Science（分子/物理/生物/神经）",
        "title": [
            "molecular", "protein", "drug discovery", "quantum",
            "physics-informed", "pde", "partial differential equation",
            "neural operator", "molecular dynamics", "ai for science",
            "computational biology", "neuroscience", "eeg", "fmri",
            "single-cell", "weather forecasting", "climate", "crystal structure",
            "material discovery",
        ],
        "body": [
            "molecular property", "protein structure", "drug discovery",
            "physics-informed neural", "partial differential equation",
            "neural operator", "ai for science", "computational chemistry",
        ],
    },
    {
        "name": "时间序列与时空数据",
        "title": [
            "time series", "time-series", "temporal forecasting",
            "spatio-temporal", "spatiotemporal", "dynamical system",
            "long-term forecasting", "anomaly detection",
        ],
        "body": [
            "time series", "temporal forecasting", "spatio-temporal",
            "dynamical system", "multivariate forecasting",
        ],
    },
    {
        "name": "推荐系统与信息检索",
        "title": [
            "recommendation", "recommender", "collaborative filtering",
            "information retrieval", "learning to rank", "click-through",
            "search ranking", "dense retrieval", "sequential recommendation",
        ],
        "body": [
            "recommendation system", "recommender system",
            "collaborative filtering", "information retrieval",
            "click-through rate",
        ],
    },
    {
        "name": "语音与音频处理",
        "title": [
            "speech recognition", "speech synthesis", "text-to-speech",
            "audio generation", "music generation", "speech enhancement",
            "automatic speech", "audio classification", "speaker verification",
            "voice conversion", "sound event",
        ],
        "body": [
            "speech recognition", "text-to-speech", "audio generation",
            "music generation", "speech synthesis",
        ],
    },
    {
        "name": "神经网络架构与压缩（通用，非 LLM）",
        "title": [
            "neural architecture search", "nas", "network pruning",
            "binary neural network", "efficient architecture", "lottery ticket",
            "spiking neural network", "capsule network", "network design",
        ],
        "body": [
            "neural architecture search", "network pruning",
            "efficient network", "spiking neural network",
        ],
    },
    {
        "name": "其他",
        "title": [],
        "body": [],
    },
]

# ---------------------------------------------------------------------------
# Disambiguation overrides: if text contains these phrases, force a category
# (used AFTER scoring to resolve systematic conflicts)
# ---------------------------------------------------------------------------
OVERRIDES = [
    # Medical VLM → 多模态大模型, not 医学影像
    (["medical vlm", "clinical vlm", "medical visual language", "radiology vqa",
      "pathology vlm", "medical instruction tuning", "radiological report generation"],
     "多模态大模型（MLLM/VLM）"),
    # Video generation → 生成模型, not 视频理解
    (["video generation model", "video diffusion", "text-to-video",
      "video synthesis", "video generation via"],
     "生成模型（扩散/GAN/视频生成/图像编辑，非 MLLM 驱动）"),
]

# ── Agent 后训练: compound matching ─────────────────────────────────────────
# Papers about training/fine-tuning LLM/VLM agents.
# Requires: "agent"/"agentic" co-occurring with explicit training action verbs.
# Excludes: traditional game-theory multi-agent RL papers.
_AGENT_TERMS   = ["agent", "agentic"]
_TRAIN_VERBS   = [
    "fine-tun", "finetuning", " sft ", "imitation learning", "behavioral cloning",
    "behaviour cloning", "trajectory fine", "agent training", "training agent",
    "agentic training", "agent reward", "agent rl", "agentic rl",
    "agent reinforcement", "agentic reinforcement", "training llm agent",
    "agent policy", "distilling.*agent", "agent distill",
    "agent instruction tun", "agent data synthes",
]
# Signals that identify traditional multi-agent RL / game-theory papers (exclude)
_GAMETHEORY    = [
    "nash equilibrium", "game theory", "zero-sum game", "cooperative game",
    "stackelberg", "correlated equilibrium", "mean field game",
    "multi-player game", "opponent shaping",
    # Traditional cooperative MARL (not LLM-based)
    "cooperative multi-agent reinforcement",
    "decentralized multi-agent", "multi-agent markov",
    "communication in multi-agent",
]
# Must contain at least one LLM/VLM context signal for "Agent 后训练"
_LLM_SIGNALS   = [
    "llm", "large language model", "language model", "foundation model",
    "gpt", "claude", "gemini", "llama", "qwen", "mistral",
    "transformer", "agentic", "lm agent",
]

def _is_agent_posttrain(title, full):
    """Return True if paper is about training LLM/VLM agents (not game-theory MARL)."""
    # must have "agent" or "agentic" in title
    if not any(_norm(t) in title for t in _AGENT_TERMS):
        return False
    # must have a training verb somewhere
    if not any(_norm(v) in full for v in _TRAIN_VERBS):
        return False
    # exclude game-theory / traditional MARL papers
    if any(_norm(g) in full for g in _GAMETHEORY):
        return False
    # must have at least one LLM/foundation model signal in the full text
    if not any(_norm(s) in full for s in _LLM_SIGNALS):
        return False
    return True

# Categories where body-only match is NOT enough without title match
# (too generic to trust just abstract mentions)
REQUIRE_TITLE_OR_STRONG = {
    "评测与基准构建",          # almost every paper "evaluates on a benchmark"
    "理论与优化",              # many papers do "optimization"
    "对话系统与文本生成",      # many papers "generate text"
}

# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def _score(text, keywords):
    """Count how many keyword phrases appear in text (boundary-aware)."""
    score = 0
    for kw in keywords:
        if _compile(kw).search(text):
            score += 1
    return score


def classify_paper(paper):
    title  = _norm(paper.get("title_en"))
    abst   = _norm(paper.get("abstract"))
    kws    = _norm(paper.get("keywords"))
    body   = abst + " " + kws
    full   = title + " " + body

    # 1) Compound check: Agent 后训练 — must satisfy co-occurrence rule
    if _is_agent_posttrain(title, full):
        return "Agent 后训练", _pick_l2("Agent 后训练", full)

    # 2) Override check (exact substring on full text)
    for phrases, forced_l1 in OVERRIDES:
        if any(_norm(p) in full for p in phrases):
            l1 = forced_l1
            return l1, _pick_l2(l1, full)

    # 3) Score each category
    scores = []
    for cat in CATEGORIES:
        if not cat["title"] and not cat["body"]:  # 其他 gets default 0
            scores.append(0)
            continue
        title_hits = _score(title, cat["title"])
        body_hits  = _score(body,  cat["body"])
        score = title_hits * 3 + body_hits

        if cat["name"] in REQUIRE_TITLE_OR_STRONG and title_hits == 0:
            score = 0  # suppress generic matches in abstract

        # Agent 后训练: give extra weight to explicit training keywords in title
        if cat["name"] == "Agent 后训练" and title_hits > 0:
            score += 3

        scores.append(score)

    max_score = max(scores)
    if max_score == 0:
        l1 = "其他"
    else:
        idx = scores.index(max_score)
        l1  = CATEGORIES[idx]["name"]

    # Post-check: if scored into Agent 后训练 but is actually game-theory MARL, demote
    if l1 == "Agent 后训练":
        if any(_norm(g) in full for g in _GAMETHEORY) or not any(_norm(s) in full for s in _LLM_SIGNALS):
            l1 = "Agent 系统"

    return l1, _pick_l2(l1, full)


def _pick_l2(l1, full_text):
    rules = L2_RULES.get(l1)
    if not rules:
        return ""
    for l2_label, phrases in rules:
        if not phrases:
            return l2_label  # catch-all at end of list
        if any(_compile(p).search(full_text) for p in phrases):
            return l2_label
    return "其他子方向"


# ---------------------------------------------------------------------------
# Canonical L1 category order (single source of truth for downstream scripts)
# ---------------------------------------------------------------------------
CATEGORY_ORDER = [cat["name"] for cat in CATEGORIES]

# 6 user-priority "core" categories that carry detailed L2 sub-direction rules
CORE_CATEGORIES = list(L2_RULES.keys())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    confs = [
        ("cvpr",    BASE / "cvpr2026_raw.json",    OUT / "cvpr_classified.json"),
        ("iclr",    BASE / "iclr2026_raw.json",    OUT / "iclr_classified.json"),
        ("neurips", BASE / "neurips2025_raw.json", OUT / "neurips_classified.json"),
    ]
    from collections import Counter

    for conf, src, dst in confs:
        if not src.exists():
            print(f"[SKIP] {src} not found")
            continue

        papers = json.loads(src.read_text())
        results = []
        for p in papers:
            l1, l2 = classify_paper(p)
            results.append({
                **p,
                "category_l1": l1,
                "category_l2": l2,
                "title_zh":    "",   # keyword-based; no translation
                "summary_zh":  "",
            })

        dst.write_text(json.dumps(results, ensure_ascii=False, indent=2))

        dist = Counter(r["category_l1"] for r in results)
        other = dist.get("其他", 0)
        print(f"\n=== {conf.upper()} ({len(results)} papers, 其他={other} = {other/len(results)*100:.1f}%) ===")
        for cat, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {cnt:4d}  {cat}")

    print("\nDone. Files written to", OUT)


if __name__ == "__main__":
    main()
