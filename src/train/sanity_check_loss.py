import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import os

dummy_data = [
    {"messages": [
        {"role": "system", "content": "你是医疗助手。"},
        {"role": "user", "content": "我头痛怎么办？"},
        {"role": "assistant", "content": "多喝水，吃点布洛芬。"}
    ]}
]
dataset = Dataset.from_list(dummy_data)

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/Qwen3.5-9B"))

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

# 只为了验数据结构，不开 bfloat16 或 device="auto"，避免某些机器无 GPU 时 HF Validator 拒绝
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

config = SFTConfig(
    output_dir="/tmp/sanity_check", 
    max_length=128, 
    assistant_only_loss=True,
    per_device_train_batch_size=1,
    use_cpu=True,   # 绕过报错
    bf16=False      # 绕过报错
)

trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer
)

train_dataloader = trainer.get_train_dataloader()
batch = next(iter(train_dataloader))

input_ids = batch["input_ids"][0]
labels = batch["labels"][0]

print("="*50)
print("Sanity Check: 验证 assistant_only_loss 生效")
print("="*50)

for idx, label in zip(input_ids, labels):
    token_str = tokenizer.decode([idx])
    if label == -100:
        print(f"Token: {token_str!r:<20} | Label: -100 (IGNORED)")
    else:
        print(f"Token: {token_str!r:<20} | Label: {label.item()} (CALCULATING LOSS ✓)")

print("="*50)
