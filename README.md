# MedQA-Lab: Low-Resource Medical LLM & Knowledge Graph Framework
[**English**](#english) | [**中文**](#中文)

---

This is a low-resource deep learning experiment repository oriented towards medical question-answering and knowledge graph enhancement. It is suitable for producing reproducible and comparable experimental results under single-GPU or limited computing power conditions. The project focuses on the following directions:

- Large model fine-tuning in the medical domain, especially parameter-efficient methods like LoRA / QLoRA
- Medical QA evaluation, covering datasets such as MedQA, CMB, and cMedQA2
- LLM-as-a-judge evaluation for medical correctness, safety, and response quality
- Medical knowledge graph construction and RotatE embedding representation learning

### Project Objectives

The goal of this repository is not to train large models from scratch, but rather to conduct "low-resource and practical" medical NLP experiments around open-source models:

- First, prepare medical instruction data and evaluation sets
- Second, perform SFT/LoRA/QLoRA fine-tuning based on local Qwen3.5 models
- Finally, evaluate the performance on multiple medical benchmarks and generate charts

This approach is more suitable for research scenarios with limited resources but a desire to produce papers and experimental results quickly.

### Main Content

The code is uniformly stored in the `src/` directory:

- **Data Preparation (`src/data_prep/`)**
  - `download_models.py`: Automatically download required Qwen3.5 models (New)
  - `prepare_data.py`: Download and preprocess training and evaluation data
  - `build_kg_data.py`: Construct a knowledge graph from medical knowledge sources
- **Model Fine-Tuning (`src/train/`)**
  - `finetune_baseline.py`: LoRA / QLoRA fine-tuning script based on Qwen3.5
- **Model Validation & Evaluation (`src/eval/`)**
  - `eval_medqa.py`: Evaluate MedQA Chinese and English multiple-choice questions
  - `eval_cmb.py`: Evaluate CMB Chinese medical multiple-choice questions
  - `eval_cmedqa2.py`: Evaluate cMedQA2 open-ended QA
- **Knowledge Graph (`src/kg/`)**
  - `train_kg_embed.py`: Train RotatE using PyKEEN and export entity/relation embeddings

### Environment Requirements

Python 3.10+ is recommended. You can install all dependencies directly via the requirements file:

```bash
pip install -r requirements.txt
```

If your environment already has local models and a CUDA version, adjust the installation methods of `torch` and `bitsandbytes` as needed.

Local model directory (the repository currently loads models from the following paths by default):

- `models/Qwen3.5-2B`
- `models/Qwen3.5-4B`
- `models/Qwen3.5-9B`

### Directory Structure

```text
MedQA/
├── README.md
├── requirements.txt
├── .gitignore
├── checkpoints/           # Directory for fine-tuned model weights
├── data/                  # Summary of all project data
│   ├── medical_kg_data/   # Knowledge graph triples and entity mapping
│   ├── sft_data/          # Instruction fine-tuning training and validation sets
│   └── test_data/         # Evaluation sets such as MedQA, CMB, cMedQA2, etc.
├── kg_embedding/          # Model and weight outputs of knowledge graph representation learning
├── models/                # Directory for base large models (Do not commit to Git)
├── outputs/               # Evaluated JSON logs and charts
└── src/                   # Source code directory
    ├── data_prep/         # Data processing
    ├── eval/              # Model evaluation
    ├── kg/                # Knowledge graph training
    └── train/             # Model fine-tuning
```

### Data and Models

#### Base Model Download

Because base large models (such as Qwen3.5-4B) are large in size, the project does not include the model files themselves. You need to use the following separate download script, which will by default utilize the HuggingFace domestic mirror source to download models to the `models/` directory (consistent with the default paths referenced by the code):

```bash
python src/data_prep/download_models.py
```
You can also specify downloading only the required version via the `--models Qwen3.5-4B` parameter, or modify the save path:
```bash
python src/data_prep/download_models.py --save_dir /your/custom/models --models Qwen3.5-4B
```

#### Training and Evaluation Data

`src/data_prep/prepare_data.py` will download and organize from Hugging Face:

- HuatuoGPT-SFT-v1 training data
- CMB test set
- Chinese part of the MedQA test set

The script will by default output to `data/sft_data/` and `data/test_data/`.

#### Knowledge Graph Data

The data constructed via the knowledge graph scripts and the corresponding embedding representations are placed under `data/medical_kg_data/` and `kg_embedding/`, respectively.

### Quick Start

#### 1. Prepare Data

```bash
python src/data_prep/prepare_data.py
```
After running, it will automatically download and organize the training set, validation set, and subsequent required evaluation sets.

#### 2. Fine-tune Base Model

QLoRA is used by default and the model is output to `checkpoints/`, suitable for low-VRAM scenarios:

```bash
python src/train/finetune_baseline.py
```

You can explicitly specify the mode or modify default parameters:

```bash
python src/train/finetune_baseline.py --mode qlora
python src/train/finetune_baseline.py --mode lora
```

Common parameter options:
- `--model`: Local model path, default `models/Qwen3.5-4B`
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size per GPU
- `--grad_accum_steps`: Gradient accumulation steps

#### 3. Run Evaluation

The evaluation scripts' outputs will be automatically written to the `outputs/` directory.

MedQA Evaluation:
```bash
python src/eval/eval_medqa.py
```
*Note: `eval_medqa.py` currently reads `data/test_data/medqa_cn_test.jsonl` and `data/test_data/medqa_en_test.jsonl` by default. If you only ran `prepare_data.py` without English data or with different filenames, you need to rename the test set files in advance or modify the paths in the evaluation script.*

CMB Evaluation:
```bash
python src/eval/eval_cmb.py
```

cMedQA2 Evaluation:
```bash
python src/eval/eval_cmedqa2.py
```

### Output Results

Evaluation scripts will generate under `outputs/`:
- `eval_results.json` and other raw test metrics
- `medqa_eval_comparison.png` and other comparison charts

Fine-tuning scripts will output under `checkpoints/`:
- `baseline_lora_medical_*/final/` (Merged final weights or LoRA weight structure)
- `training_info.json` (Records various training settings and duration)

Knowledge graph script (`src/kg/train_kg_embed.py`) will generate under `kg_embedding/`:
- `entity_embeddings.pt`
- `relation_embeddings.pt`

### Instructions & Notes

- Data download and evaluation scripts enable the Hugging Face mirror (HF_ENDPOINT) by default to optimize access speed in domestic network environments.
- Base model weights in the repository (`models/`) are large in size and have been ignored by Git; please be sure to manage them locally and do not submit them to remote.
- `finetune_baseline.py` defaults to QLoRA training settings, capable of adapting to VRAM bottlenecks on most devices; if GPU computing resources are stronger (such as having A100 80G, etc.), it is recommended to switch to the regular FP16/BF16 LoRA mode to achieve better fine-tuning results.
- In order to balance test efficiency, the evaluation programs default to sampling and truncating data with `num_samples=100` to verify the pipeline; when obtaining formal metrics, please find the corresponding script and set this parameter to `None`.

### Research Positioning

If taking this project as a working foundation for graduation designs or scientific research papers, it is very suitable for carrying out the following topics or experiments:

- **Focus on Efficient Medical QA**: Ablation and effect exploration of parameter-efficient fine-tuning (PEFT) methods of different scales on multimodal or medical datasets.
- **Knowledge Graph Embedding Fusion or Retrieval**: Leveraging existing triple data cleaned from text, explore how external knowledge can help models suppress factual hallucinations (such as RAG-based ideas).

---

<h2 id="中文">🇨🇳 中文说明</h2>

这是一个面向医疗问答与知识图谱增强的低资源深度学习实验仓库，适合单卡或有限算力条件下做出可复现、可对比的实验结果。项目结合了项目代码与前期讨论中的选题思路，重点放在以下几类方向：

- 医疗领域大模型微调，尤其是 LoRA / QLoRA 这类参数高效方法
- 医疗问答评测，覆盖 MedQA、CMB、cMedQA2 等数据集
- 针对医疗正确性、安全性以及回复质量的 LLM-as-a-judge 评估
- 医疗知识图谱构建与 RotatE 嵌入表示学习

### 项目目标

本仓库的目标不是从零训练大模型，而是围绕开源模型做“低资源可落地”的医疗 NLP 实验：

- 先准备医疗指令数据与评测集
- 再基于本地 Qwen3.5 模型做 SFT/LoRA/QLoRA 微调
- 最后在多个医疗基准上评估效果并生成图表

这种路线更适合资源有限、但希望尽快产出论文和实验结果的研究场景。

### 主要内容

代码统一存放在 `src/` 目录下：

- **数据准备 (`src/data_prep/`)**
  - `download_models.py`：自动下载所需的 Qwen3.5 模型 （新增）
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

### 环境要求

建议使用 Python 3.10+，可以直接通过 requirements 文件安装全部依赖：

```bash
pip install -r requirements.txt
```

如果你的环境里已经有本地模型和 CUDA 版本，按需调整 `torch` 与 `bitsandbytes` 的安装方式。

本地模型目录（仓库当前默认会从如下路径加载模型）：

- `models/Qwen3.5-2B`
- `models/Qwen3.5-4B`
- `models/Qwen3.5-9B`

### 目录结构

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

### 数据与模型

#### 基座模型下载

由于基座大模型（如 Qwen3.5-4B）体积庞大，项目中并不包含模型文件本身。你需要使用如下单独的下载脚本，它会默认利用 HuggingFace 国内镜像源下载模型到 `models/` 目录下（与代码默认引用的路径一致）：

```bash
python src/data_prep/download_models.py
```
你也可以通过 `--models Qwen3.5-4B` 参数指定只下载需要的版本，或修改保存路径：
```bash
python src/data_prep/download_models.py --save_dir /your/custom/models --models Qwen3.5-4B
```

#### 训练与评测数据

`src/data_prep/prepare_data.py` 会从 Hugging Face 下载并整理：

- HuatuoGPT-SFT-v1 训练数据
- CMB 测试集
- MedQA 测试集中的中文部分

脚本默认会输出到 `data/sft_data/` 和 `data/test_data/` 中。

#### 知识图谱数据

通过知识图谱脚本构建出来的数据和相应的嵌入表示分别放在 `data/medical_kg_data/` 和 `kg_embedding/` 下。

### 快速开始

#### 1. 准备数据

```bash
python src/data_prep/prepare_data.py
```
运行后会自动下载和整理训练集、验证集以及后续所需的评测集。

#### 2. 微调基座模型

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
- `--model`：本地模型路径，默认 `models/Qwen3.5-4B`
- `--epochs`：训练轮数
- `--batch_size`：单卡 batch size
- `--grad_accum_steps`：梯度累积步数

#### 3. 运行评测

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

### 输出结果

评测脚本会在 `outputs/` 下生成：
- `eval_results.json` 等原始测试指标
- `medqa_eval_comparison.png` 等对比图表

微调脚本会在 `checkpoints/` 下输出：
- `baseline_lora_medical_*/final/` (融合后的最终权重或 LoRA 权重结构)
- `training_info.json` (记录各项训练设定和时长)

知识图谱脚本（`src/kg/train_kg_embed.py`）会在 `kg_embedding/` 下生成：
- `entity_embeddings.pt`
- `relation_embeddings.pt`

### 说明与注意事项

- 数据下载与评测脚本默认开启 Hugging Face 镜像（HF_ENDPOINT），以优化国内网络环境下的访问体验。
- 仓库中的基座模型权重（`models/`）体积庞大，已被 Git 忽略，请务必本地管理，不要提交至远端。
- `finetune_baseline.py` 默认基于 QLoRA 训练设置，能够适配多数设备的显存瓶颈；如果 GPU 计算资源更强（如具有 A100 80G 等），推荐切换至常规 FP16/BF16 的 LoRA 模式从而获取更优的微调效果。
- 为了兼顾测试时效，评测程序默认对数据做 `num_samples=100` 的抽样截断来验证管线；获取正式指标时，请找到对应脚本并将此参数设定为 `None`。

### 研究定位

如果将此项目作为毕业设计或科研论文的工作底座，非常适合开展以下课题或实验：

- **聚焦高效医疗问答**：不同量级、不同参数高效微调方法（PEFT）在多模态或医疗数据集上的消融与效果探索。
- **知识图谱嵌入融合或检索**：借助现有从文本中清洗出的三元组数据，探索外部知识如何帮助模型抑制事实性幻觉（如基于 RAG 的思路）。
