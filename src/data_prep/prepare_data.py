"""
Step 1: 下载并预处理数据集
  - 训练集: HuatuoGPT-SFT-v1 (医疗指令微调数据)
  - 评测集: CMB (中文医学基准选择题)
  - 评测集: MedQA (医学执照考试选择题)

⚠️ 国内服务器需要 HuggingFace 镜像，脚本会自动设置。
   如果已能访问 HuggingFace，可注释掉下面的 os.environ 行。
"""
import os

# ⭐ 国内 HuggingFace 镜像 (必须在 import datasets 之前设置)
# 如果你能直接访问 huggingface.co，可以注释掉这行
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from datasets import concatenate_datasets, load_dataset

SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/sft_data"))
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 1. 下载 HuatuoGPT-SFT 训练数据
# ============================================================
print("📥 正在下载 HuatuoGPT-SFT 数据集 (约22万条医疗指令数据)...")
sft_ds = load_dataset("FreedomIntelligence/HuatuoGPT-sft-data-v1")

# 查看数据结构
print(f"  数据集大小: {sft_ds}")
sample = sft_ds["train"][0]
print(f"  字段: {list(sample.keys())}")
sample_data = sample.get('data', ['', ''])
print(f"  示例:\n    instruction: {sample_data[0][:80]}...")
print(f"    output:      {sample_data[1][:80]}...")

# 格式化为 Qwen 聊天模板所需的 JSON Lines 格式
def format_for_qwen(example):
    """将 HuatuoGPT 数据转换为 Qwen chat 格式"""
    data = example.get("data", [])
    if len(data) >= 2:
        instruction = data[0].strip()
        output = data[1].strip()
        # 清除开头的“问：”和“答：”前缀（如果需要）
        if instruction.startswith("问："):
            instruction = instruction[2:].strip()
        if output.startswith("答："):
            output = output[2:].strip()
    else:
        instruction = ""
        output = ""
    
    return {
        "messages": [
            {"role": "system", "content": "你是一个专业的医学AI助手，请根据医学知识准确回答问题。"},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
    }

print("\n🔄 格式化训练数据为 Qwen 聊天模板 (随机抽取5万条数据)...")

# ============================================================
# 1. 准备 HuatuoGPT-SFT 数据 (取5万条)
# ============================================================
print("\n🔄 格式化 HuatuoGPT-SFT 数据 (抽取5万条)...")
huatuo_subset = sft_ds["train"].shuffle(seed=42).select(range(50000))
huatuo_formatted = huatuo_subset.map(format_for_qwen, remove_columns=huatuo_subset.column_names)

# ============================================================
# 2. 准备 cMedQA2 真实问答数据 (取1万条)
# ============================================================
print("\n📥 正在下载 cMedQA2 数据集...")
# HuggingFace 上有开源的 cMedQA2 数据集（例如 fzkuji/cMedQA2）
cmedqa_ds = load_dataset("fzkuji/cMedQA2", "deduplicate_all", split="train")

def format_cmedqa_for_qwen(example):
    """将 cMedQA2 一问一答的真实问诊转换为 Qwen chat 格式"""
    instruction = example.get("question", "").strip()
    output = example.get("answer", "").strip()
    
    return {
        "messages": [
            {"role": "system", "content": "你是一个专业的医学AI助手，请根据医学知识准确回答问题。"},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
    }

print("🔄 格式化 cMedQA2 数据 (抽取1万条)...")

# 过滤掉空数据并随机抽取1万条 (因为加载时指定了 split="train"，所以此处直接使用 cmedqa_ds)
cmedqa_subset = cmedqa_ds.filter(lambda x: len(x['question']) > 0 and len(x['answer']) > 0)
cmedqa_subset = cmedqa_subset.shuffle(seed=42).select(range(10000))
cmedqa_formatted = cmedqa_subset.map(format_cmedqa_for_qwen, remove_columns=cmedqa_subset.column_names)


# ============================================================
# 3. 混合数据 (合并并打乱)
# ============================================================
print("\n🔀 正在混合 HuatuoGPT (5W) + cMedQA2 (1W) 联合数据集...")
# 将两部分数据合并
mixed_dataset = concatenate_datasets([huatuo_formatted, cmedqa_formatted])
# ⭐ 混合后一定要整体 shuffle，防止训练时局部梯度震荡
mixed_dataset = mixed_dataset.shuffle(seed=42)

# 划分训练/验证集 (依然保证 90% / 10% 的切割)
split = mixed_dataset.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
val_data = split["test"]

# 保存
train_path = os.path.join(SAVE_DIR, "train.jsonl")
val_path = os.path.join(SAVE_DIR, "val.jsonl")

with open(train_path, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"  ✅ 训练集: {len(train_data)} 条 → {train_path}")
print(f"  ✅ 验证集: {len(val_data)} 条 → {val_path}")

# ============================================================
# 3. 下载 CMB 评测数据
# ============================================================
print("\n📥 正在下载 CMB (中文医学基准) 评测数据集...")
try:
    print("尝试直接加载 CMB-test JSON 文件...")
    cmb_ds = load_dataset(
        "json",
        data_files="hf://datasets/FreedomIntelligence/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json",
        split="train"
    )
except Exception as e:
    print(f"直接加载失败: {e}，尝试使用默认配置...")
    cmb_ds = load_dataset("FreedomIntelligence/CMB", "CMB-Exam", split="test", trust_remote_code=True)

print(f"  数据集: length={len(cmb_ds)}")
sample = cmb_ds[0]
print(f"  字段: {list(sample.keys())}")
print(f"  示例: {json.dumps(sample, ensure_ascii=False, indent=2)[:500]}")

# 保存 CMB 评测数据
cmb_path = os.path.join(SAVE_DIR, "cmb_test.jsonl")

with open(cmb_path, "w", encoding="utf-8") as f:
    for item in cmb_ds:
        f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")

print(f"  ✅ CMB 测试集: {len(cmb_ds)} 条 → {cmb_path}")

# ============================================================
# 3. 下载 MedQA 评测数据 (中文部分)
# ============================================================
print("\n📥 正在下载 MedQA 数据集...")
try:
    import urllib.request
    import zipfile
    
    medqa_zip_path = "med_qa.zip"
    medqa_url = "https://hf-mirror.com/datasets/bigbio/med_qa/resolve/main/data_clean.zip"
    
    if not os.path.exists(medqa_zip_path):
        print("  正在从 HuggingFace 镜像下载 MedQA zip 文件...")
        urllib.request.urlretrieve(medqa_url, medqa_zip_path)
    
    medqa_path = os.path.join(SAVE_DIR, "medqa_test.jsonl")
    
    # 从 zip 中提取 test.jsonl
    with zipfile.ZipFile(medqa_zip_path, 'r') as z:
        # Mainland 4_options 为中文选择题数据集
        with z.open('data_clean/questions/Mainland/4_options/test.jsonl') as zf, open(medqa_path, 'wb') as f:
            f.write(zf.read())
            
    print(f"  ✅ MedQA 测试集已提取 → {medqa_path}")
    
    # 统计条目
    with open(medqa_path, "r", encoding="utf-8") as f:
        medqa_lines = sum(1 for _ in f)
    print(f"  ✅ MedQA 测试集数量: {medqa_lines} 条")
    
except Exception as e:
    print(f"  ⚠️ MedQA 下载失败 ({e})，跳过。可以后续手动下载。")

# ============================================================
# 统计
# ============================================================
print("\n" + "=" * 50)
print("📊 数据准备完成！")
print("=" * 50)
print(f"  训练数据: {train_path} ({len(train_data)} 条)")
print(f"  验证数据: {val_path} ({len(val_data)} 条)")
print(f"  CMB评测:  {cmb_path}")
print(f"\n  📁 所有数据保存在: {os.path.abspath(SAVE_DIR)}")