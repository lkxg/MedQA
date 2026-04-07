"""
Step 2: 标准 LoRA/QLoRA 微调 Baseline
  基座模型: Qwen/Qwen3.5-4B
  训练数据: HuatuoGPT-SFT
  
  用法:
    python finetune_baseline.py                  # 默认 QLoRA (显存 ≤ 16GB)
    python finetune_baseline.py --mode lora      # LoRA fp16 (需要 24GB+ 显存)
    python finetune_baseline.py --mode qlora     # QLoRA 4-bit
"""
import argparse
import json
import os
import time

# ⭐ 国内 HuggingFace 镜像与进度条加速 (必须在其他包之前设置)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # (可选) 开启 Rust 极速下载
# 减少显存碎片导致的 OOM 风险
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "/datadisk/models/Qwen3.5-4B" # 基座模型
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/sft_data"))                   # 数据目录 (prepare_data.py 的输出)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints/baseline_lora_medical"))     # 模型输出目录
MAX_SEQ_LEN = 1024                        # 32GB 显存下更稳妥的默认长度

# LoRA 超参数
LORA_R = 16                               # 低秩维度
LORA_ALPHA = 32                           # 缩放因子 (通常 = 2r)
LORA_DROPOUT = 0.05                       # LoRA dropout
LORA_TARGET_MODULES = [                   # 应用 LoRA 的模块
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 训练超参数
NUM_EPOCHS = 1
BATCH_SIZE = 1                            # 单卡 32GB + 9B 模型建议从 1 起步
GRAD_ACCUM_STEPS = 8                      # 等效 batch_size = 1 × 8 = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LOGGING_STEPS = 20
SAVE_STEPS = 500


def parse_args():
    parser = argparse.ArgumentParser(description="医疗大模型 LoRA/QLoRA 微调 Baseline")
    parser.add_argument("--mode", choices=["lora", "qlora"], default="qlora",
                        help="微调模式: lora(fp16) 或 qlora(4-bit量化)")
    parser.add_argument("--model", default=MODEL_NAME, help="基座模型名称")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    return parser.parse_args()


def load_data(data_dir):
    """加载预处理好的训练/验证数据"""
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    
    def read_jsonl(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    train_data = read_jsonl(train_path)
    val_data = read_jsonl(val_path)
    
    print(f"📂 训练集: {len(train_data)} 条")
    print(f"📂 验证集: {len(val_data)} 条")
    
    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def load_model_and_tokenizer(model_name, mode="lora"):
    """加载基座模型和 tokenizer"""
    
    print(f"\n🔧 检查并加载本地模型权重: {model_name}")
    # 由于模型已移动到本地目录，直接加载即可
    
    print(f"\n🔧 开始将模型加载到显存: {model_name} (模式: {mode})")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 模型加载配置
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "attn_implementation": "sdpa",  # PyTorch 2.0+ 原生支持能大幅加速且节省显存 (等效Flash Attention)
    }
    
    if mode == "qlora":
        # 4-bit 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        print("  📦 使用 4-bit NF4 量化 (QLoRA 模式)")
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        print("  📦 使用 bfloat16 (LoRA 模式)")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # QLoRA 需要额外准备
    if mode == "qlora":
        model = prepare_model_for_kbit_training(model)
    
    # 关闭 cache (训练时不需要)
    model.config.use_cache = False
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 模型总参数量: {total_params / 1e9:.2f}B")
    
    return model, tokenizer


def apply_lora(model, lora_r=16, lora_alpha=32):
    """应用 LoRA 适配器"""
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def format_chat(example, tokenizer):
    """将 messages 格式转换为模型输入文本"""
    messages = example["messages"]
    # 使用 tokenizer 内置的 chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🏥 医疗大模型 LoRA Baseline 微调")
    print("=" * 60)
    print(f"  模型:     {args.model}")
    print(f"  模式:     {args.mode}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch_size} × {args.grad_accum_steps} = {args.batch_size * args.grad_accum_steps}")
    print(f"  LR:       {args.lr}")
    print(f"  LoRA r:   {args.lora_r}")
    print(f"  Max Len:  {args.max_len}")
    print("=" * 60)
    
    # 1. 加载数据
    train_ds, val_ds = load_data(DATA_DIR)
    
    # 2. 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model, args.mode)
    
    # 3. 格式化数据 (应用 chat template)
    print("\n🔄 应用 Qwen 聊天模板格式化数据...")
    train_ds = train_ds.map(lambda x: format_chat(x, tokenizer), num_proc=4)
    val_ds = val_ds.map(lambda x: format_chat(x, tokenizer), num_proc=4)
    
    # 打印一个样例
    print(f"\n--- 格式化后的样例 (截取前500字) ---")
    print(train_ds[0]["text"][:500])
    print("---")
    
    # 4. 应用 LoRA
    model = apply_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_r * 2)
    
    # 5. 训练配置
    output_dir = f"{OUTPUT_DIR}_{args.mode}_r{args.lora_r}"
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        
        # 精度
        bf16=True,
        
        # 日志 & 保存
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        
        # 其他
        gradient_checkpointing=True,
        report_to="none",                 # 不上传到 wandb, 如需要改为 "wandb"
        seed=42,
        dataset_text_field="text",
        # max_seq_length 针对 TRL 1.0 需配在 SFTConfig 的 init 之外或按最新参数处理
        dataloader_num_workers=4,         # 使用多进程加速数据加载
    )
    
    # 针对 trl==1.0.0 的特定参数改动：最大长度属性已改名为 max_seq_length 放于 SFTConfig 中，或者需要在 trainer 初始化传递。
    training_args.max_seq_length = args.max_len
    
    # 6. 训练
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    
    print(f"\n🚀 开始训练... 输出目录: {output_dir}")
    t_start = time.time()
    
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    t_elapsed = time.time() - t_start
    print(f"\n⏱️ 训练完成! 耗时: {t_elapsed / 60:.1f} 分钟")
    
    # 7. 保存最终模型
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"💾 模型已保存至: {final_dir}")
    
    # 8. 保存训练信息
    info = {
        "model": args.model,
        "mode": args.mode,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_r * 2,
        "epochs": args.epochs,
        "batch_size": args.batch_size * args.grad_accum_steps,
        "learning_rate": args.lr,
        "max_seq_len": args.max_len,
        "training_time_min": round(t_elapsed / 60, 1),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    with open(os.path.join(output_dir, "training_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 全部完成!")
    print(f"  下一步: python evaluate_cmb.py --model_path {final_dir}")


if __name__ == "__main__":
    main()
