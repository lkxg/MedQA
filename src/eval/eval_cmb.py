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

# CMB 数据集
CMB_DATASET = "../../data/test_data/cmb_test.jsonl"

def extract_answer(text):
    """鲁棒地从模型输出中提取选项字母，支持单选和多选提取"""
    text = text.strip()
    # 尝试匹配带前缀的答案表达式
    match = re.search(r'(?:答案|answer)\s*(?:is|are|为|：|:)?\s*([A-F\s,、及和]+)', text, re.IGNORECASE)
    if match:
        res = match.group(1)
    else:
        # 假设直接输出在了开头
        match = re.search(r'^([A-F\s,、及和]+)', text)
        if match:
            res = match.group(1)
        else:
            res = text
            
    # 从初步提取的字符串中，匹配所有的 A-F，去重并按首字母排序拼合
    letters = re.findall(r'[A-F]', res.upper())
    return "".join(sorted(list(set(letters))))

def load_data(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}")
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def build_prompt_cmb(item):
    question = item.get("question", "")
    options = item.get("option", {})
    q_type = item.get("question_type", "单项选择题")
    is_multi = "多" in q_type
    
    if is_multi:
        system_prompt = "你是一个专业的医学专家AI助手。请严格根据你的专业医学知识准确回答以下多项选择题。请直接给出所有正确选项字母（例如AB或ABC），不要输出任何解释或多余的文字。"
    else:
        system_prompt = "你是一个专业的医学专家AI助手。请严格根据你的专业医学知识准确回答以下单项选择题。请直接给出唯一的正确选项字母（如A、B、C、D、E），不要输出任何解释或多余的文字。"
    
    prompt = f"题型: {q_type}\n问题: {question}\n选项:\n"
    # 获取所有的选项键并排序
    option_keys = sorted(options.keys())
    for k in option_keys:
        prompt += f"{k}. {options[k]}\n"
    
    # 动态给出所有可用字母，明确单选还是多选
    available_letters = "、".join(option_keys)
    if is_multi:
        prompt += f"请直接回答所有正确的选项字母（可选范围：{available_letters}，多选）。"
    else:
        prompt += f"请直接回答最合适的选项字母（可选范围：{available_letters}，单选）。"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return messages

def evaluate_cmb(model_path, dataset_path, num_samples=100):
    dataset = load_data(dataset_path)
    if not dataset:
        return 0.0
    
    dataset = dataset[:num_samples] if num_samples else dataset
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
        for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {os.path.basename(model_path)} on CMB")):
            messages = build_prompt_cmb(item)
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
            
            # 提取与对比答案
            pred_char = extract_answer(pred_text)
            ground_truth = item.get("answer_idx", "")
            # 对基准答案同样进行 A-F 去重排序的大写格式化（防止如'BA'而模型答'AB'的情况错判）
            ground_truth = "".join(sorted(list(set(re.findall(r'[A-F]', ground_truth.upper())))))
            
            is_correct = pred_char == ground_truth
            if is_correct:
                correct += 1
                
            if i < 10:
                print(f"\n[题目 {i+1}] {item.get('question_type')} - {item.get('question')}")
                print(f"模型原输出: {pred_text}")
                print(f"提取出答案: {pred_char} | 真实正确答案: {ground_truth} => [{'正确' if is_correct else '错误'}]")

    del model
    del tokenizer
    torch.cuda.empty_cache()

    accuracy = correct / total
    return accuracy

def main():
    results = {
        "Qwen3.5-2B": 0.0,
        "Qwen3.5-4B": 0.0,
        "Qwen3.5-9B": 0.0,
    }

    # 为了快速测试，设置只测试前100条如果需要全量，请改为 None
    num_samples = 100

    print("=== 开始 CMB 数据集上的评估 ===")
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"[{model_name}] 路径不存在，跳过")
            continue
            
        print(f"\n--- {model_name} 评估中 ---")
        acc = evaluate_cmb(model_path, CMB_DATASET, num_samples=num_samples)
        results[model_name] = acc
        print(f"[{model_name}] CMB Accuracy: {acc * 100:.2f}%")

    # 保存 CMB 结果
    with open(os.path.join(os.path.dirname(__file__), "../../outputs/eval_results_cmb.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 绘制单独的 CMB 结果图
    models_list = list(results.keys())
    accs = [results[m] for m in models_list]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(models_list, accs, width=0.4, color='skyblue', label='CMB (Chinese)')
    
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Qwen3.5 Models on CMB Test')
    plt.ylim(0, 1.0)
    plt.legend()

    # 在柱子上标注具体准确率数值
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "../../outputs/cmb_eval_comparison.png"))
    print("\nCMB 评估完成，结果已保存到 eval_results_cmb.json，并生成图表 cmb_eval_comparison.png")

if __name__ == "__main__":
    main()