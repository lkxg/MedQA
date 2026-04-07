# MedQA 医疗大模型实验仓库

这是一个面向医疗问答与知识图谱增强的低资源深度学习实验仓库，适合硕士阶段在单卡或有限算力条件下做出可复现、可对比的实验结果。项目结合了项目代码与前期讨论中的选题思路，重点放在以下几类方向：

- 医疗领域大模型微调，尤其是 LoRA / QLoRA 这类参数高效方法
- 医疗问答评测，覆盖 MedQA、CMB、cMedQA2 等数据集
- 医疗知识图谱构建与 RotatE 嵌入表示学习

## 项目目标

本仓库的目标不是从零训练大模型，而是围绕开源模型做“低资源可落地”的医疗 NLP 实验：

- 先准备医疗指令数据与评测集
- 再基于本地 Qwen3.5 模型做 SFT/LoRA/QLoRA 微调
- 最后在多个医疗基准上评估效果并生成图表

这种路线更适合资源有限、但希望尽快产出论文和实验结果的研究场景。

## 主要内容

代码统一存放在 `src/` 目录下：

- **数据准备 (`src/data_prep/`)**
  - `prepare_data.py`：下载并预处理训练数据和评测数据
  - `build_kg_data.py`：从医疗知识源构建知识图谱
- **模型微调 (`src/train/`)**
  - `finetune_baseline.py`：基于 Qwen3.5 的 LoRA / QLoRA 微调脚本
- **模型验证与评测 (`src/eval/`)**
  - `eval_medqa.py`：评测 MedQA 中英文选择题
  - `eval_cmb.py`：评测 CMB 中文医学选择题
  - `eval_cmedqa2.py`：评测 cMedQA2 开放式问答
- **知识图谱 (`src/kg/`)**
  - `train_kg_embed.py`：使用 PyKEEN 训练 RotatE 并导出实体/关系嵌入

## 环境要求

建议使用 Python 3.10+，可以直接通过 requirements 文件安装全部依赖：

```bash
pip install -r requirements.txt
```

如果你的环境里已经有本地模型和 CUDA 版本，按需调整 `torch` 与 `bitsandbytes` 的安装方式。

本地模型目录（仓库当前默认会从如下路径加载模型）：

- `/datadisk/models/Qwen3.5-2B`
- `/datadisk/models/Qwen3.5-4B`
- `/datadisk/models/Qwen3.5-9B`

## 目录结构

```text
MedQA/
├── README.md
├── requirements.txt
├── .gitignore
├── checkpoints/           # 微调模型权重存放地
├── data/                  # 汇总所有项目数据
│   ├── medical_kg_data/   # 知识图谱三元组与实体映射
│   ├── sft_data/          # 指令微调训练验证集
│   └── test_data/         # MedQA、CMB、cMedQA2 等评测集
├── kg_embedding/          # 知识图谱表示学习的模型和权重输出
├── models/                # 基础大模型存放目录 (请勿提交到 Git)
├── outputs/               # 评测输出的 JSON 日志和图表
└── src/                   # 源代码目录
    ├── data_prep/         # 数据处理
    ├── eval/              # 模型评估
    ├── kg/                # 知识图谱训练
    └── train/             # 模型微调
```

## 数据与模型

### 训练与评测数据

`src/data_prep/prepare_data.py` 会从 Hugging Face 下载并整理：

- HuatuoGPT-SFT-v1 训练数据
- CMB 测试集
- MedQA 测试集中的中文部分

脚本默认会输出到 `data/sft_data/` 和 `data/test_data/` 中。

### 知识图谱数据

通过知识图谱脚本构建出来的数据和相应的嵌入表示分别放在 `data/medical_kg_data/` 和 `kg_embedding/` 下。

## 快速开始

### 1. 准备数据

```bash
python src/data_prep/prepare_data.py
```
运行后会自动下载和整理训练集、验证集以及后续所需的评测集。

### 2. 微调基座模型

默认使用 QLoRA 并输出模型到 `checkpoints/`，适合低显存场景：

```bash
python src/train/finetune_baseline.py
```

可以显式指定模式或修改默认参数：

```bash
python src/train/finetune_baseline.py --mode qlora
python src/train/finetune_baseline.py --mode lora
```

常用参数选项：
- `--model`：本地模型路径，默认 `/datadisk/models/Qwen3.5-4B`
- `--epochs`：训练轮数
- `--batch_size`：单卡 batch size
- `--grad_accum_steps`：梯度累积步数

### 3. 运行评测

评测脚本的输出会自动写入 `outputs/` 目录。

MedQA 评测：
```bash
python src/eval/eval_medqa.py
```
*注意：`eval_medqa.py` 当前默认读取 `data/test_data/medqa_cn_test.jsonl` 和 `data/test_data/medqa_en_test.jsonl`。如果你只运行了 `prepare_data.py` 而没有英文数据或文件名不同，需提前重命名测试集文件或在评测脚本里修改路径。*

CMB 评测：
```bash
python src/eval/eval_cmb.py
```

cMedQA2 评测：
```bash
python src/eval/eval_cmedqa2.py
```

## 输出结果

评测脚本会在 `outputs/` 下生成：
- `eval_results.json` 等原始测试指标
- `medqa_eval_comparison.png` 等对比图表

微调脚本会在 `checkpoints/` 下输出：
- `baseline_lora_medical_*/final/` (融合后的最终权重或 LoRA 权重结构)
- `training_info.json` (记录各项训练设定和时长)

知识图谱脚本（`src/kg/train_kg_embed.py`）会在 `kg_embedding/` 下生成：
- `entity_embeddings.pt`
- `relation_embeddings.pt`

## 说明与注意事项

- 数据下载与评测脚本默认开启 Hugging Face 镜像（HF_ENDPOINT），以优化国内网络环境下的访问体验。
- 仓库中的基座模型权重（`models/`）体积庞大，已被 Git 忽略，请务必本地管理，不要提交至远端。
- `finetune_baseline.py` 默认基于 QLoRA 训练设置，能够适配多数设备的显存瓶颈；如果 GPU 计算资源更强（如具有 A100 80G 等），推荐切换至常规 FP16/BF16 的 LoRA 模式从而获取更优的微调效果。
- 为了兼顾测试时效，评测程序默认对数据做 `num_samples=100` 的抽样截断来验证管线；获取正式指标时，请找到对应脚本并将此参数设定为 `None`。

## 研究定位

如果将此项目作为毕业设计或科研论文的工作底座，非常适合开展以下课题或实验：

- **聚焦高效医疗问答**：不同量级、不同参数高效微调方法（PEFT）在多模态或医疗数据集上的消融与效果探索。
- **知识图谱嵌入融合或检索**：借助现有从文本中清洗出的三元组数据，探索外部知识如何帮助模型抑制事实性幻觉（如基于 RAG 的思路）。

