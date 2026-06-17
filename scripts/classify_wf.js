export const meta = {
  name: 'classify-papers',
  description: 'Classify conference papers L1/L2 by title+abstract — NO PDF reading',
  phases: [
    { title: 'Classify', detail: 'parallel 25-paper batch agents per conference' },
  ],
}

const CATEGORIES = [
  '多模态大模型（MLLM/VLM）',
  '强化学习后训练（RLHF/DPO/GRPO/奖励模型）',
  'Agent 系统',
  'Agent 后训练',
  '大语言模型预训练（架构/数据/规模）',
  '大模型 Infra（推理/训练/效率系统）',
  '推理与规划（CoT/数学/逻辑，非 RL 类）',
  '代码生成与编程语言处理',
  '安全与对齐（幻觉/偏见/毒性/越狱，非 RL 类）',
  '评测与基准构建',
  '图神经网络与结构化学习',
  '理论与优化',
  '对话系统与文本生成',
  '目标检测与分割',
  '生成模型（扩散/GAN/视频生成/图像编辑，非 MLLM 驱动）',
  '三维视觉（NeRF/3DGS/深度估计/点云）',
  '视频理解（动作识别/跟踪/时序建模，非 MLLM 类）',
  '底层视觉（超分/去噪/增强）',
  '人体相关（姿态/人脸/手势/运动合成）',
  '医学影像（非多模态大模型类）',
  '自动驾驶（感知/预测/规划）',
  '遥感与卫星',
  '其他',
]

const CORE_L2 = {
  '多模态大模型（MLLM/VLM）': '图像理解与视觉推理,视频理解与时序推理,视觉定位与接地（Grounding）,图像/视频生成控制（指令驱动）,视觉数学推理,文档/图表/OCR 理解,医学多模态,多模态幻觉与评测,音频-视觉-语言',
  '强化学习后训练（RLHF/DPO/GRPO/奖励模型）': 'RLHF/PPO 后训练,GRPO/群体相对策略优化,DPO 偏好学习,奖励模型（RM）构建,过程奖励模型（PRM）,结果奖励模型（ORM）,宪法 AI/RLAIF,数学 RL 强化,代码 RL 强化,拒绝采样与 Best-of-N,视觉推理 RL（VLM-RL）',
  'Agent 系统': '工具调用与函数调用（Tool Use）,多 Agent 协作框架,GUI/Web/OS Agent,代码 Agent,具身 Agent（导航/操作）,规划与反思机制,记忆与上下文管理,世界模型',
  'Agent 后训练': 'Agent 轨迹数据构建与合成,Agent RL 训练（轨迹级强化）,Agent 自改进/自修正,工具学习后训练,在线 Agent 微调',
  '大语言模型预训练（架构/数据/规模）': '预训练架构设计（Transformer/SSM/混合架构/MoE）,预训练数据工程与配比,持续预训练与领域适配,多语言预训练,模型规模与涌现规律,分词与词表设计',
  '大模型 Infra（推理/训练/效率系统）': '投机解码（Speculative Decoding）,量化与稀疏化（W4/INT8/稀疏注意力）,KV Cache 压缩与管理,长上下文建模与位置外推,训练并行与显存优化（TP/PP/ZeRO）,模型合并与 MoE 路由,Prompt 压缩与上下文蒸馏,视觉 Token 压缩（CVPR 侧）',
}

// args is just the conference name string: "cvpr", "iclr", or "neurips"
const conf = args

const BASE = '/Users/chenzhihui/Desktop/Intern/literature/cvpr2026-acl2026-survey/data'
const CONF_CONFIG = {
  cvpr:    { batchDir: BASE + '/cvpr2026_batches_100',    batchCount: 52 },
  iclr:    { batchDir: BASE + '/iclr2026_batches_100',    batchCount: 54 },
  neurips: { batchDir: BASE + '/neurips2025_batches_100', batchCount: 53 },
}
const outDir = BASE + '/_analysis'

const { batchDir, batchCount } = CONF_CONFIG[conf]

const batchPaths = Array.from({ length: batchCount }, (_, i) =>
  batchDir + '/batch_' + String(i).padStart(4, '0') + '.json'
)

const catList = CATEGORIES.map((c, j) => (j + 1) + '. ' + c).join('\n')
const coreL2Block = Object.entries(CORE_L2)
  .map(([k, v]) => '  ' + k + ' 候选: ' + v)
  .join('\n')

log(conf.toUpperCase() + ': ' + batchPaths.length + ' batches to classify')
phase('Classify')

await parallel(
  batchPaths.map((batchPath, i) => () => {
    const outPath = outDir + '/' + conf + '/batch_' + String(i).padStart(4, '0') + '.json'
    return agent(
      '你是 AI/CV 研究专家，正在对三大顶会（CVPR/ICLR/NeurIPS）论文进行主题分类。\n\n请用 Read 工具读取文件：' + batchPath + '\n这是一个含 ≤25 篇论文的 JSON 数组，每篇含 title_en / abstract / keywords 字段。\n\n【绝对禁止】不要读取或下载任何 PDF 文件，也不要调用任何网络工具。只读上面这一个 JSON 文件。\n\n对每篇论文输出一个 JSON 对象，全部包在一个 JSON 数组里。\n\n━━ 可选 L1 类别（必须从此列表原文选一个，不能自造，不能缩写）━━\n' + catList + '\n\n━━ L2 规则 ━━\n若 L1 是以下核心类，L2 须从候选中选（若不合适才自造，≤10 字）：\n' + coreL2Block + '\n其他 L1 类的 L2 自由描述（≤10 字，无须从候选选）。\n\n━━ 每篇输出字段 ━━\npaper_id   : 原值原样复制\ntitle_en   : 原值原样复制\ntitle_zh   : 流畅中文标题（翻译，保留专有名词英文）\nyear       : 原值\nconference : 原值\ntrack      : 原值\nabstract   : 原值原样复制\ncategory_l1: 从上方列表选一个（原文，不能修改）\ncategory_l2: 子方向（中文，≤10 字）\nkeywords   : 若原值非空则原样复制；否则提取 5 个英文关键词逗号分隔\nsummary_zh : 2-3 句中文（研究问题 + 方法 + 核心贡献，不能只复述标题）\n\n━━ 落盘 ━━\n先用 Write 工具把 JSON 数组写入：\n' + outPath + '\n\n━━ 硬性要求 ━━\n1. JSON 字符串值内部绝对不用英文双引号 "，需要引号时用《》或单引号\n2. paper_id / title_en / abstract / year / conference / track 字段必须原样复制，不得修改\n3. 落盘完成后，最终回复【只能】是这个 JSON 数组本身，不含任何其他文字或 markdown 代码块标记',
      {
        label: conf + '-' + (i + 1) + '/' + batchPaths.length,
        phase: 'Classify',
        model: 'haiku',
      }
    )
  })
)

log('All ' + batchPaths.length + ' batches done for ' + conf.toUpperCase() + '. Results in ' + outDir + '/' + conf + '/')
return { conf: conf, batches: batchPaths.length }
