import os
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置模型路径
MODELS = {
    "Qwen3.5-2B": "/datadisk/models/Qwen3.5-2B",
    "Qwen3.5-4B": "/datadisk/models/Qwen3.5-4B",
    "Qwen3.5-9B": "/datadisk/models/Qwen3.5-9B",
}

# 数据集配置文件路径 (假设中英文都以相同格式存在)
# 中文使用我们现有的 medqa_test.jsonl
DATASETS = {
    "MedQA-CN": "test_data/medqa_cn_test.jsonl",
    "MedQA-EN": "test_data/medqa_en_test.jsonl"
}

def extract_answer(text):
    """鲁棒地从模型输出中提取选项字母"""
    text = text.strip()
    # 完全匹配单个字母
    if text.upper() in ['A', 'B', 'C', 'D']:
        return text.upper()
    # 匹配 "答案是X" / "The answer is X" 等模式
    match = re.search(r'(?:答案|answer)\s*(?:is|为|：|:)?\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # 匹配第一个独立的 A-D
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1)
    return ""

def load_data(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}")
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def build_prompt(item, lang="CN"):
    question = item["question"]
    options = item["options"]
    if lang == "CN":
        system_prompt = "你是一个专业的医学专家AI助手。请严格根据你的专业医学知识准确回答以下单项选择题。请直接给出唯一的正确选项字母（A、B、C或D），不要输出任何解释或多余的文字。"
        prompt = f"问题: {question}\n选项:\n"
        for k, v in options.items():
            prompt += f"{k}. {v}\n"
        prompt += "请直接回答最合适的选项字母（A、B、C或D）。"
    else:
        system_prompt = "You are an expert medical AI assistant. Please accurately answer the following multiple-choice questions based strictly on your professional medical knowledge. Directly output the only correct option letter (A, B, C, or D) without any explanation or extra text."
        prompt = f"Question: {question}\nOptions:\n"
        for k, v in options.items():
            prompt += f"{k}. {v}\n"
        prompt += "Please reply with the best option letter directly (A, B, C, or D)."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return messages

def evaluate_model(model_path, dataset_path, lang="CN", num_samples=100):
    dataset = load_data(dataset_path)
    if not dataset:
        return 0.0
    
    # 为了测试速度，默认只测前num_samples条
    dataset = dataset[:num_samples] if num_samples else dataset
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 为了使用较少显存加载，可以使用 fp16 和 device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    correct = 0
    total = len(dataset)

    with torch.no_grad():
        for item in tqdm(dataset, desc=f"Evaluating {os.path.basename(model_path)} on {lang}"):
            messages = build_prompt(item, lang)
            # enable_thinking
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            # 限制生成长度
            outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # 简单匹配选项
            pred_char = extract_answer(pred_text)
            if pred_char == item.get("answer_idx", ""):
                correct += 1

    del model
    del tokenizer
    torch.cuda.empty_cache()

    accuracy = correct / total
    return accuracy

def main():
    results = {
        "Qwen3.5-2B": {"MedQA-CN": 0.0, "MedQA-EN": 0.0},
        "Qwen3.5-4B": {"MedQA-CN": 0.0, "MedQA-EN": 0.0},
        "Qwen3.5-9B": {"MedQA-CN": 0.0, "MedQA-EN": 0.0},
    }

    # 执行评估
    # num_samples=100 代表为快速看到结果，只跑前100条。如果要全量测试请设置为 None
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}, path does not exist.")
            continue
            
        for db_name, db_path in DATASETS.items():
            lang = "CN" if "CN" in db_name else "EN"
            print(f"\\n--- Running {model_name} on {db_name} ---")
            acc = evaluate_model(model_path, db_path, lang=lang, num_samples=100)
            results[model_name][db_name] = acc
            print(f"Accuracy: {acc * 100:.2f}%")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 画图
    models = list(results.keys())
    cn_accs = [results[m]["MedQA-CN"] for m in models]
    en_accs = [results[m]["MedQA-EN"] for m in models]

    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar([i - width/2 for i in x], cn_accs, width, label='MedQA-CN (Chinese)')
    rects2 = ax.bar([i + width/2 for i in x], en_accs, width, label='MedQA-EN (English)')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy of Qwen3.5 Models on MedQA')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.0)

    # 绘制数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig("medqa_eval_comparison.png")
    print("\n评估完成，结果已保存到 eval_results.json，并生成图表 medqa_eval_comparison.png")

if __name__ == "__main__":
    main()
