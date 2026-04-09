"""
小样本测试微调脚本 (10个样本)
用于测试模型加载、数据格式、输入输出、Loss计算是否正常。
用法:
  python test_finetune.py
"""
import argparse
import json
import os
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

# ================= 新增：训练过程中生成文本的回调 =================
class PrintGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, test_query):
        self.tokenizer = tokenizer
        self.test_query = test_query

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print(f"\n[Epoch {state.epoch}] ⏳ 正在随机抽样查看模型当前的输出能力...")
        prompt = [
            {"role": "system", "content": "你是一名网络医疗问答医生。请用中文互联网医疗平台医生回复的常见风格回答问题：先给简短判断，再给1到2条处理建议。不要过度诊断，不要给复杂病理分析，不要给过多药名、剂量或详细治疗方案；若信息不足，只说“考虑”“可能”“建议检查/复查”。回答控制在2句内、100字以内，不要分点，不要标题，不要废话。"},
            {"role": "user", "content": self.test_query}
        ]
        textForGen = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        model_inputs = self.tokenizer(textForGen, return_tensors="pt").to(model.device)
        
        # 临时开启 cache 并切换到评估模式进行推理
        model.config.use_cache = False  # 避免与 gradient checkpointing 冲突
        was_training = model.training
        model.eval()
        
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs, 
                max_new_tokens=100,
                do_sample=True,         # 必须开启这个，temperature / top_p 才会生效
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        if was_training:
            model.train()
        
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, outputs)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print("\n" + "="*50)
        print(f"用户输入: {self.test_query}")
        print(f"模型当前输出:\n{response.strip()}")
        print("="*50 + "\n")

# ============================================================
# 配置
# ============================================================
MODEL_NAME = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/Qwen3.5-4B"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/sft_data"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints/test_lora_medical"))
MAX_SEQ_LEN = 512

# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="小样本测试微调")
    parser.add_argument("--mode", choices=["lora", "qlora"], default="lora")
    return parser.parse_args()


def load_data_subset(data_dir, num_samples=10):
    """仅加载少量数据用于测试"""
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    
    def read_jsonl(path, limit):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                data.append(json.loads(line.strip()))
        return data
    
    train_data = read_jsonl(train_path, num_samples)
    val_data = read_jsonl(val_path, num_samples)
    
    print(f"📂 测试模式 - 截取训练集: {len(train_data)} 条")
    print(f"📂 测试模式 - 截取验证集: {len(val_data)} 条")
    
    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def load_model_and_tokenizer(model_name, mode="lora"):
    print(f"\n🔧 加载本地模型权重: {model_name} (模式: {mode})")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    
    if mode == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["dtype"] = torch.bfloat16
        
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    if mode == "qlora":
        model = prepare_model_for_kbit_training(model)
        
    model.config.use_cache = False
    return model, tokenizer


def main():
    args = parse_args()
    print("=" * 60)
    print("🧪 医疗大模型 - 小样本微调测试")
    print("=" * 60)
    
    # 加载 10 条数据
    train_ds, val_ds = load_data_subset(DATA_DIR, num_samples=10)
    
    # 打印一条数据看看格式是否正常
    print("\n[数据格式检查] 第一条训练数据：")
    print(json.dumps(train_ds[0]["messages"], ensure_ascii=False, indent=2))
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, args.mode)
    
    # 添加 LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    # 测试前向传播，检查输入输出和Loss是否正常
    print("\n[环境测试] 前向传播测试...")
    test_text = tokenizer.apply_chat_template(train_ds[0]["messages"], tokenize=False, add_generation_prompt=False)
    test_batch = tokenizer(test_text, return_tensors="pt").to(model.device)
    inputs = {
        "input_ids": test_batch["input_ids"],
        "attention_mask": test_batch["attention_mask"],
        "labels": test_batch["input_ids"].clone()
    }
    
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"  -> 测试前向传播 Loss: {outputs.loss.item():.4f}")
        print(f"  -> Logits 形状: {outputs.logits.shape}")
    
    training_args = SFTConfig(
        max_length=MAX_SEQ_LEN,
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,              # 跑两个 epoch，样本很少所以很快
        per_device_train_batch_size=2,   # 非常小的 batch
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        # 日志 & 保存设置
        logging_steps=1,                 # 每1步打印loss，方便观察变化
        save_strategy="no",              # 测试不保存
        eval_strategy="steps",
        eval_steps=2,                    # 很快eval一次
        # 其他
        bf16=True,
        report_to="none",
        seed=42,
        assistant_only_loss=True,
        remove_unused_columns=True
    )
    
    # 找一条用户提问进行测试
    test_user_query = train_ds[0]["messages"][0]["content"] if train_ds[0]["messages"][0]["role"] == "user" else "医生您好，我最近总是头晕、乏力，这是怎么回事？"

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[PrintGenerationCallback(tokenizer, test_user_query)] # 注入自定义回调函数
    )
    
    print("\n🚀 开始小样本跑通测试...")
    t_start = time.time()
    trainer.train()
    print(f"\n✅ 测试完成! 训练耗时: {(time.time() - t_start):.1f} 秒")
    print("🎉 如果中途没有报错，并且每步打印了 loss，说明流程完全走通！")

    # ================= 新增：测试模型输出 =================
    print("\n[模型输出测试] 尝试让微调后的模型进行推理生成...")
    model.config.use_cache = False # 保守起见关闭，并设为 eval 模式
    model.eval()
    
    # 找一条用户提问进行测试
    test_user_query = train_ds[0]["messages"][0]["content"] if train_ds[0]["messages"][0]["role"] == "user" else "医生您好，我最近总是头晕、乏力，这是怎么回事？"
    
    prompt_messages = [
        {"role": "system", "content": "你是一名网络医疗问答医生。请用中文互联网医疗平台医生回复的常见风格回答问题：先给简短判断，再给1到2条处理建议。不要过度诊断，不要给复杂病理分析，不要给过多药名、剂量或详细治疗方案；若信息不足，只说“考虑”“可能”“建议检查/复查”。回答控制在2句内、100字以内，不要分点，不要标题，不要废话。"},
        {"role": "user", "content": test_user_query}
    ]
    
    textForGen = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)
    model_inputs = tokenizer(textForGen, return_tensors="pt").to(model.device)
    
    print(f"  -> 输入: {test_user_query}")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,             # 修复解码无效警告，防止输出乱码
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 截断掉输入 prompt 部分，仅保留新生成的 token
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("  -> 模型输出:")
    print("-" * 40)
    print(response.strip())
    print("-" * 40)
    
if __name__ == "__main__":
    main()
