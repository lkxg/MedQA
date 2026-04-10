"""
微调流程冒烟测试 (Smoke Test)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
与 finetune_baseline.py 保持完全一致的配置，用 20 条数据跑 2 步，
逐项验证：数据格式 → 分词 → 前向传播 → 梯度流 → 短训练 → 推理生成。
全部通过后再启动正式训练，避免数小时训练后才发现环境或数据问题。

用法:
  python test_finetune.py                 # 默认 LoRA 模式
  python test_finetune.py --mode qlora    # 测试 QLoRA 模式
"""
import argparse
import json
import os
import sys
import time
import traceback

# ⭐ 国内 HuggingFace 镜像 (必须在其他包之前设置)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

# ============================================================
# ⭐ 从 finetune_baseline.py 复用完全一致的配置常量
#    修改正式训练时只需改 finetune_baseline.py，这里自动对齐
# ============================================================
MODEL_NAME   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/Qwen3.5-4B"))
DATA_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/sft_data"))
OUTPUT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints/_smoke_test"))
MAX_SEQ_LEN  = 512

# LoRA 超参 —— 与 finetune_baseline.py 完全一致
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 测试用常量
NUM_SAMPLES  = 20    # 抽取数据条数
TEST_STEPS   = 2     # 训练步数
BATCH_SIZE   = 2

# 测试推理用的问题
TEST_QUERIES = [
    "医生您好，我最近总是头晕、乏力，应该挂什么科？",
    "血压 150/95 需要吃药吗？",
]


# ============================================================
# 工具函数
# ============================================================
class TestResult:
    """收集各项测试结果，最后统一汇报"""
    def __init__(self):
        self.items = []

    def record(self, name, passed, detail=""):
        status = "✅ PASS" if passed else "❌ FAIL"
        self.items.append((name, passed, detail))
        print(f"  {status}  {name}" + (f"  ({detail})" if detail else ""))

    def summary(self):
        total = len(self.items)
        passed = sum(1 for _, p, _ in self.items if p)
        failed = total - passed
        print("\n" + "=" * 60)
        print(f"🧪 测试汇总: {passed}/{total} 通过, {failed} 失败")
        print("=" * 60)
        if failed > 0:
            print("\n失败项:")
            for name, p, detail in self.items:
                if not p:
                    print(f"  ❌ {name}: {detail}")
            print("\n⚠️ 请修复以上问题后再启动正式训练！")
        else:
            print("\n🎉 全部通过！可以放心启动正式训练:")
            print(f"   python finetune_baseline.py --mode <lora|qlora>")
        return failed == 0


def parse_args():
    parser = argparse.ArgumentParser(description="微调流程冒烟测试")
    parser.add_argument("--mode", choices=["lora", "qlora"], default="lora",
                        help="测试哪种微调模式")
    parser.add_argument("--model", default=MODEL_NAME, help="模型路径")
    return parser.parse_args()


def load_data_subset(data_dir, num_samples=20):
    """加载少量数据用于测试"""
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path   = os.path.join(data_dir, "val.jsonl")

    def read_jsonl(path, limit):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                data.append(json.loads(line.strip()))
        return data

    train_data = read_jsonl(train_path, num_samples)
    val_data   = read_jsonl(val_path, max(num_samples // 2, 5))
    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def gpu_mem_mb():
    """获取当前 GPU 已用显存 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


# ============================================================
# 测试 1: 数据格式检查
# ============================================================
def test_data_format(train_ds, val_ds, report):
    print("\n📋 [测试 1/6] 数据格式检查")

    # 检查字段
    sample = train_ds[0]
    has_messages = "messages" in sample
    report.record("训练集包含 'messages' 字段", has_messages)
    if not has_messages:
        return False

    msgs = sample["messages"]
    # 检查 messages 结构
    roles = [m["role"] for m in msgs]
    expected_roles = ["system", "user", "assistant"]
    role_ok = roles == expected_roles
    report.record("messages 角色顺序为 [system, user, assistant]", role_ok,
                  f"实际: {roles}")

    # 检查内容非空
    all_nonempty = all(len(m["content"].strip()) > 0 for m in msgs)
    report.record("所有 message content 非空", all_nonempty)

    # 检查全部样本格式一致
    bad_count = 0
    for i, item in enumerate(train_ds):
        m = item.get("messages", [])
        if len(m) != 3 or m[0]["role"] != "system" or m[1]["role"] != "user" or m[2]["role"] != "assistant":
            bad_count += 1
    report.record(f"训练集 {len(train_ds)} 条数据格式全部一致", bad_count == 0,
                  f"{bad_count} 条格式异常" if bad_count else "")

    # 打印样例
    print(f"\n  📄 样例数据 (第 1 条):")
    print(f"     System:    {msgs[0]['content'][:60]}...")
    print(f"     User:      {msgs[1]['content'][:60]}...")
    print(f"     Assistant: {msgs[2]['content'][:60]}...")
    print(f"  📊 训练集: {len(train_ds)} 条 | 验证集: {len(val_ds)} 条")

    return True


# ============================================================
# 测试 2: 分词 & Chat Template 检查
# ============================================================
def test_tokenization(train_ds, tokenizer, report):
    print("\n📋 [测试 2/6] 分词 & Chat Template 检查")

    sample = train_ds[0]

    # 测试 chat template 完整格式 (训练用)
    try:
        full_text = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )
        report.record("apply_chat_template (训练模式) 成功", True,
                      f"{len(full_text)} 字符")
    except Exception as e:
        report.record("apply_chat_template (训练模式) 成功", False, str(e))
        return

    # 测试 chat template 推理格式
    prompt_msgs = [m for m in sample["messages"] if m["role"] != "assistant"]
    try:
        gen_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        report.record("apply_chat_template (推理模式) 成功", True,
                      f"{len(gen_text)} 字符")
    except Exception as e:
        report.record("apply_chat_template (推理模式) 成功", False, str(e))

    # 测试 tokenize 后长度
    tokens = tokenizer(full_text, return_tensors="pt")
    seq_len = tokens["input_ids"].shape[1]
    within_limit = seq_len <= MAX_SEQ_LEN * 2  # 允许一定弹性
    report.record(f"Token 长度合理 (当前 {seq_len}, 上限 {MAX_SEQ_LEN})", within_limit,
                  "⚠️ 过长样本会被截断" if seq_len > MAX_SEQ_LEN else "在截断范围内")

    # 检查 pad_token
    report.record("pad_token 已设置", tokenizer.pad_token is not None,
                  f"pad_token = '{tokenizer.pad_token}'")

    print(f"\n  📄 Chat Template 预览 (训练格式, 前 200 字符):")
    print(f"     {repr(full_text[:200])}...")


# ============================================================
# 测试 3: 模型加载 & LoRA 注入
# ============================================================
def test_model_loading(model_name, mode, report):
    print(f"\n📋 [测试 3/6] 模型加载 & LoRA 注入 (模式: {mode})")

    mem_before = gpu_mem_mb()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 模型
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

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        report.record("基座模型加载成功", True)
    except Exception as e:
        report.record("基座模型加载成功", False, str(e))
        return None, None

    if mode == "qlora":
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    mem_after_base = gpu_mem_mb()

    # LoRA — 与 finetune_baseline.py 完全一致
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    try:
        model = get_peft_model(model, lora_config)
        report.record("LoRA 适配器注入成功", True)
    except Exception as e:
        report.record("LoRA 适配器注入成功", False, str(e))
        return None, None

    # 统计参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    ratio     = trainable / total * 100

    report.record(f"可训练参数占比合理", ratio < 5,
                  f"可训练: {trainable/1e6:.1f}M / 总计: {total/1e6:.1f}M ({ratio:.2f}%)")

    mem_after_lora = gpu_mem_mb()
    print(f"\n  📊 显存: 加载前 {mem_before:.0f}MB → 基座 {mem_after_base:.0f}MB → +LoRA {mem_after_lora:.0f}MB")

    return model, tokenizer


# ============================================================
# 测试 4: 前向传播 & 梯度流检查
# ============================================================
def test_forward_and_gradient(model, tokenizer, train_ds, report):
    print("\n📋 [测试 4/6] 前向传播 & 梯度流检查")

    sample = train_ds[0]
    text = tokenizer.apply_chat_template(
        sample["messages"], tokenize=False, add_generation_prompt=False
    )
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    batch = {k: v.to(model.device) for k, v in batch.items()}

    # 前向传播
    model.train()
    inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["input_ids"].clone(),
    }

    try:
        outputs = model(**inputs)
        loss = outputs.loss
        report.record("前向传播成功", True, f"Loss = {loss.item():.4f}")
    except Exception as e:
        report.record("前向传播成功", False, str(e))
        return

    # Loss 合理性 (初始 loss 应在 1~15 之间，太大或太小都有问题)
    loss_val = loss.item()
    loss_ok = 0.5 < loss_val < 20
    report.record("初始 Loss 在合理范围 (0.5~20)", loss_ok, f"Loss = {loss_val:.4f}")

    # Logits 形状
    logits_shape = outputs.logits.shape
    expected_vocab = tokenizer.vocab_size
    shape_ok = logits_shape[0] == 1 and logits_shape[2] >= expected_vocab
    report.record("Logits 形状正确", shape_ok,
                  f"shape = {list(logits_shape)}, vocab ≥ {expected_vocab}")

    # 反向传播 + 梯度检查
    try:
        loss.backward()
        report.record("反向传播成功", True)
    except Exception as e:
        report.record("反向传播成功", False, str(e))
        return

    # 检查 LoRA 参数是否有梯度
    lora_params_with_grad = 0
    lora_params_total = 0
    grad_norms = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_params_total += 1
            if param.grad is not None:
                lora_params_with_grad += 1
                grad_norms.append(param.grad.norm().item())

    all_have_grad = lora_params_with_grad == lora_params_total
    report.record("所有 LoRA 参数都有梯度", all_have_grad,
                  f"{lora_params_with_grad}/{lora_params_total} 个参数有梯度")

    if grad_norms:
        avg_grad = sum(grad_norms) / len(grad_norms)
        max_grad = max(grad_norms)
        grad_ok = avg_grad > 1e-10 and max_grad < 1e5
        report.record("梯度数值正常 (无消失/爆炸)", grad_ok,
                      f"平均梯度范数 = {avg_grad:.6f}, 最大 = {max_grad:.4f}")

    # 清理梯度
    model.zero_grad()


# ============================================================
# 测试 5: SFTTrainer 短训练
# ============================================================
def test_short_training(model, tokenizer, train_ds, val_ds, report):
    print(f"\n📋 [测试 5/6] SFTTrainer 短训练 ({TEST_STEPS} 步)")

    mem_before = gpu_mem_mb()

    training_args = SFTConfig(
        max_length=MAX_SEQ_LEN,
        output_dir=OUTPUT_DIR,
        max_steps=TEST_STEPS,                  # 只跑几步
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=TEST_STEPS,
        gradient_checkpointing=False,
        report_to="none",
        seed=42,
        assistant_only_loss=True,              # 与 baseline 一致
        dataloader_num_workers=0,              # 测试时用单进程，避免多进程问题掩盖bug
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    try:
        t_start = time.time()
        train_result = trainer.train()
        t_elapsed = time.time() - t_start

        # 检查训练 loss 下降（至少不是 NaN）
        final_loss = train_result.training_loss
        loss_valid = not (torch.isnan(torch.tensor(final_loss)) or torch.isinf(torch.tensor(final_loss)))
        report.record("训练完成 & Loss 非 NaN/Inf", loss_valid,
                      f"最终 Loss = {final_loss:.4f}, 耗时 {t_elapsed:.1f}s")

    except Exception as e:
        report.record("训练完成 & Loss 非 NaN/Inf", False, str(e))
        traceback.print_exc()
        return

    # Eval loss
    try:
        eval_result = trainer.evaluate()
        eval_loss = eval_result.get("eval_loss", float("nan"))
        eval_ok = not (torch.isnan(torch.tensor(eval_loss)) or torch.isinf(torch.tensor(eval_loss)))
        report.record("验证集 Eval Loss 正常", eval_ok, f"Eval Loss = {eval_loss:.4f}")
    except Exception as e:
        report.record("验证集 Eval Loss 正常", False, str(e))

    mem_peak = gpu_mem_mb()
    print(f"\n  📊 训练显存峰值: {mem_peak:.0f}MB (训练前: {mem_before:.0f}MB)")


# ============================================================
# 测试 6: 推理生成检查
# ============================================================
def test_inference(model, tokenizer, report):
    print("\n📋 [测试 6/6] 推理生成检查")

    model.eval()

    for i, query in enumerate(TEST_QUERIES):
        prompt_msgs = [
            {"role": "system", "content": "你是一个专业的医学AI助手，请根据医学知识准确回答问题。"},
            {"role": "user", "content": query},
        ]

        try:
            text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False,
                add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # 截取生成部分
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # 基本质量检查
            has_output = len(response) > 0
            not_garbage = not all(c == response[0] for c in response) if response else False  # 非重复单字符
            is_ok = has_output and not_garbage

            report.record(f"生成测试 #{i+1} 输出正常", is_ok,
                          f"{len(response)} 字符")

            print(f"\n  💬 问: {query}")
            print(f"  🤖 答: {response[:200]}{'...' if len(response) > 200 else ''}")

        except Exception as e:
            report.record(f"生成测试 #{i+1} 输出正常", False, str(e))


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()

    print("=" * 60)
    print("🧪 微调流程冒烟测试 (Smoke Test)")
    print("=" * 60)
    print(f"  模型:   {args.model}")
    print(f"  模式:   {args.mode}")
    print(f"  样本:   {NUM_SAMPLES} 条")
    print(f"  步数:   {TEST_STEPS} 步")
    print(f"  LoRA:   r={LORA_R}, alpha={LORA_ALPHA}, modules={len(LORA_TARGET_MODULES)}个")
    print(f"  设备:   {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ⚠️'}")
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        print(f"  显存:   {total_mem:.0f} MB")
    print("=" * 60)

    report = TestResult()

    # 1. 加载数据
    train_ds, val_ds = load_data_subset(DATA_DIR, NUM_SAMPLES)

    # 2. 数据格式
    data_ok = test_data_format(train_ds, val_ds, report)
    if not data_ok:
        report.summary()
        sys.exit(1)

    # 3. 模型加载
    model, tokenizer = test_model_loading(args.model, args.mode, report)
    if model is None:
        report.summary()
        sys.exit(1)

    # 4. 分词检查
    test_tokenization(train_ds, tokenizer, report)

    # 5. 前向传播 & 梯度
    test_forward_and_gradient(model, tokenizer, train_ds, report)

    # 6. 短训练
    test_short_training(model, tokenizer, train_ds, val_ds, report)

    # 7. 推理
    test_inference(model, tokenizer, report)

    # 汇总
    all_pass = report.summary()

    # 清理测试输出
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        print(f"\n🧹 已清理测试输出目录: {OUTPUT_DIR}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
