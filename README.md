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

- `prepare_data.py`：下载并预处理训练数据和评测数据
- `finetune_baseline.py`：基于 Qwen3.5 的 LoRA / QLoRA 微调脚本
- `eval_medqa.py`：评测 MedQA 中英文选择题
- `eval_cmb.py`：评测 CMB 中文医学选择题
- `eval_cmedqa2.py`：评测 cMedQA2 开放式问答
- `MedicalKGBuilder.py`：从医疗知识源构建知识图谱
- `Embedding.py`：使用 PyKEEN 训练 RotatE 并导出实体/关系嵌入

## 环境要求

建议使用 Python 3.10+，并准备以下依赖：

- `torch`
- `transformers`
- `datasets`
- `peft`
- `trl`
- `bitsandbytes`
- `accelerate`
- `jieba`
- `nltk`
- `rouge-chinese`
- `bert-score`
- `tqdm`
- `matplotlib`
- `pykeen`

如果你使用的是本地模型目录，仓库当前默认会从如下路径加载模型：

- `/datadisk/models/Qwen3.5-2B`
- `/datadisk/models/Qwen3.5-4B`
- `/datadisk/models/Qwen3.5-9B`

## 目录结构

```text
MedQA/
├── Embedding.py
├── MedicalKGBuilder.py
├── eval_cmb.py
├── eval_cmedqa2.py
├── eval_medqa.py
├── finetune_baseline.py
├── prepare_data.py
├── models/
├── kg_embedding/
├── medical_kg_data/
├── sft_data/
└── test_data/
```

## 数据与模型

### 训练数据

`prepare_data.py` 会从 Hugging Face 下载并整理：

- HuatuoGPT-SFT-v1 训练数据
- CMB 测试集
- MedQA 测试集中的中文部分

脚本默认会输出到：

- `sft_data/train.jsonl`
- `sft_data/val.jsonl`
- `sft_data/cmb_test.jsonl`
- `sft_data/medqa_test.jsonl`

### 本地模型

训练和评测默认使用本地 Qwen3.5 模型目录，不建议把大模型权重提交到 Git 仓库。

### 知识图谱数据

项目中还包含医疗知识图谱相关数据：

- `medical_kg_data/`
- `kg_embedding/`

## 快速开始

### 1. 安装依赖

建议先创建虚拟环境，然后安装必要包：

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate jieba nltk rouge-chinese bert-score tqdm matplotlib pykeen
```

如果你的环境里已经有本地模型和 CUDA 版本，按需调整 `torch` 与 `bitsandbytes` 的安装方式。

### 2. 准备数据

```bash
python prepare_data.py
```

运行后会自动下载和整理训练集、验证集以及评测集。

### 3. 微调基座模型

默认是 QLoRA，适合低显存场景：

```bash
python finetune_baseline.py
```

如果显存足够，也可以显式指定模式：

```bash
python finetune_baseline.py --mode qlora
python finetune_baseline.py --mode lora
```

常用参数：

- `--model`：本地模型路径，默认 `/datadisk/models/Qwen3.5-4B`
- `--epochs`：训练轮数
- `--batch_size`：单卡 batch size
- `--grad_accum_steps`：梯度累积步数
- `--max_len`：最大序列长度
- `--lora_r`：LoRA 低秩维度
- `--resume`：从 checkpoint 恢复训练

### 4. 运行评测

MedQA：

```bash
python eval_medqa.py
```

注意：`eval_medqa.py` 当前默认读取 `test_data/medqa_cn_test.jsonl` 和 `test_data/medqa_en_test.jsonl`。如果你只运行了 `prepare_data.py`，需要先把数据整理成这两个文件名，或者按你的数据格式修改脚本里的路径。

CMB：

```bash
python eval_cmb.py
```

cMedQA2：

```bash
python eval_cmedqa2.py
```

## 输出结果

评测脚本会生成如下文件：

- `eval_results.json`
- `eval_results_cmb.json`
- `eval_results_cmedqa.json`
- `medqa_eval_comparison.png`
- `cmb_eval_comparison.png`
- `cmedqa_eval_comparison.png`

微调脚本会输出：

- `baseline_lora_medical_*/final/`
- `training_info.json`

知识图谱脚本会输出：

- `medical_kg_data/triples.tsv`
- `medical_kg_data/entity_to_id.json`
- `medical_kg_data/relation_to_id.json`
- `medical_kg_data/entity_types.json`
- `medical_kg_data/stats.json`
- `kg_embedding/entity_embeddings.pt`
- `kg_embedding/relation_embeddings.pt`

## 说明与注意事项

- `prepare_data.py` 和评测脚本默认使用 Hugging Face 镜像，适合国内网络环境。
- 仓库中的模型权重体积很大，建议本地保存，不要提交到 GitHub。
- `finetune_baseline.py` 默认基于 QLoRA，更适合有限显存；如果你有更强 GPU，可以尝试 LoRA 模式。
- 评测脚本默认只跑前 100 条样本，适合快速检查流程；如果要完整评测，可以把 `num_samples` 改为 `None`。

## 研究定位

如果把这个仓库作为论文课题的基础，它比较适合以下路线：

- 医疗问答场景下的参数高效微调
- 结合知识图谱或检索增强的医疗问答
- 小资源条件下的医疗模型评测与对比实验

这类方向的优点是：数据集和 baseline 较齐全、实验容易做、适合逐步迭代出论文结果。

## 许可

仓库当前包含本地模型与数据处理脚本，实际使用时请同时遵守相关数据集和模型的原始许可证。