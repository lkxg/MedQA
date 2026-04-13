import os
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置模型路径: 专门针对 4B 模型的对比
MODELS = {
    "Qwen3.5-4B (Base)": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/Qwen3.5-4B")),
    "Qwen3.5-4B (Finetuned)": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints/baseline_lora_medical_lora_r16/final")),
}

# 中文测试集
DATASETS = {
    "MedQA-CN": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/test_data/medqa_cn_test.jsonl")),
}

def extract_answer(text):
    text = text.strip()
    if text.upper() in ['A', 'B', 'C', 'D']: return text.upper()
    
    # 1. 假设直接输出在了开头，例如 "C。HCV..." 或 "A，因为..."
    match = re.search(r'^([A-D])[^A-Za-z]', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # 2. 尝试匹配带前缀的答案表达式
    match = re.search(r'(?:答案|选项|选|answer\s+is|应选)\s*(?:为|是|：|:)?\s*([A-D])', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # 3. 兜底提取独立的A-D字母（排除 HCV, CT, DNA, B超 等医疗术语中的字母）
    letters = re.findall(r'(?<![A-Za-z])[A-D](?![A-Za-z])', text.upper())
    if letters:
        # 如果没有匹配到开头或带有答案前缀，且文本里有独立的单个大写字母，默认选第一个出现的
        return letters[0]
        
    return ""

def load_data(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
    return data

def build_prompt(item):
    question = item["question"]
    options = item["options"]
    system_prompt = "你是一个专业的医学专家AI助手。请严格根据你的专业医学知识准确回答以下单项选择题。请直接给出唯一的正确选项字母（A、B、C或D），不要输出任何解释或多余的文字。"
    prompt = f"问题: {question}\n选项:\n"
    for k, v in options.items():
        prompt += f"{k}. {v}\n"
    prompt += "请直接回答最合适的选项字母（A、B、C或D）。"
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

def evaluate_model(model_path, dataset_path, num_samples=100):
    dataset = load_data(dataset_path)
    if not dataset: return 0.0
    dataset = dataset[:num_samples] if num_samples else dataset
    
    # 动态支持 PEFT/LoRA 加载
    is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if is_peft:
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            base_model_path = json.load(f).get("base_model_name_or_path")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    correct, total = 0, len(dataset)
    with torch.no_grad():
        for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {os.path.basename(model_path)}")):
            # `build_prompt` returns dictionary containing strings for instruction but `apply_chat_template` needs a list of dicts.
            messages = build_prompt(item)
            # `apply_chat_template` requires messages format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            if isinstance(messages, tuple):
                messages = list(messages)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            # 加入聊天模板的结束符 <|im_end|> 作为停止条件，防止模型自言自语（角色扮演幻觉）
            stop_words_ids = [tokenizer.eos_token_id]
            if "<|im_end|>" in tokenizer.vocab:
                stop_words_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
                
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_words_ids
            )
            
            pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            pred_char = extract_answer(pred_text)
            
            if i < 10:
                print(f"真实答案: {item.get('answer_idx', '')} | 模型预测: {pred_char} | 模型原文: {pred_text}")
                
            if pred_char == item.get("answer_idx", ""):
                correct += 1

    del model, tokenizer
    torch.cuda.empty_cache()
    return correct / total

def main():
    results = {"Qwen3.5-4B (Base)": 0.0, "Qwen3.5-4B (Finetuned)": 0.0}
    num_samples = 100

    print("=== 开始 MedQA (4B) 评估 ===")
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"[{model_name}] 跳过，路径不存在")
            continue
        print(f"\n--- Running {model_name} on MedQA-CN ---")
        acc = evaluate_model(model_path, DATASETS["MedQA-CN"], num_samples=num_samples)
        results[model_name] = acc
        print(f"Accuracy: {acc * 100:.2f}%")

    with open(os.path.join(os.path.dirname(__file__), "../../outputs/medqa_4b_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 绘图
    models_list = list(results.keys())
    accs = [results[m] for m in models_list]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(models_list, accs, width=0.4, color='lightgreen')
    plt.ylabel('Accuracy')
    plt.title('Accuracy on MedQA-CN Test')
    plt.ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "../../outputs/medqa_4b_comparison.png"))
    print("\n评估完成，结果已保存到 medqa_4b_results.json，并生成图表 medqa_4b_comparison.png")

if __name__ == "__main__":
    main()
