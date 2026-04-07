# 指定国内 HF 镜像和禁用在线探测
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import torch
import jieba
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
import bert_score

# 设置模型路径
MODELS = {
    "Qwen3.5-2B": "/datadisk/models/Qwen3.5-2B",
    "Qwen3.5-4B": "/datadisk/models/Qwen3.5-4B",
    "Qwen3.5-9B": "/datadisk/models/Qwen3.5-9B",
}

# cMedQA2 测试集
CMEDQA2_DATASET = "test_data/cmedqa2_test.jsonl"

def load_data(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}")
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def calc_metrics(preds, refs):
    """
    计算 BLEU-4, ROUGE-L, BERTScore
    统一使用 jieba 词级别分词，避免因分词粒度不一致导致的评估侧次偏差
    """
    # 统一提取词粒度的分词列表与空格拼接字符串
    preds_tokens = [list(jieba.cut(p)) if p.strip() else ["无"] for p in preds]
    refs_tokens =  [list(jieba.cut(r)) if r.strip() else ["无"] for r in refs]
    
    preds_jieba = [" ".join(p) for p in preds_tokens]
    refs_jieba  = [" ".join(r) for r in refs_tokens]

    # ====== 1. BLEU-4 ======
    smooth = SmoothingFunction().method1
    bleu_scores = []
    for pred_tok, ref_tok in zip(preds_tokens, refs_tokens):
        if pred_tok == ["无"]:
            bleu_scores.append(0.0)
            continue
        try:
            # 采用词粒度（jieba 分词结果）计算 BLEU
            score = sentence_bleu([ref_tok], pred_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        except Exception:
            score = 0.0
        bleu_scores.append(score)
    avg_bleu4 = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    # ====== 2. ROUGE-L ======
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(preds_jieba, refs_jieba, avg=True)
        rouge_l_f1 = rouge_scores['rouge-l']['f']
    except Exception:
        rouge_l_f1 = 0.0

    try:
        # 使用多语言模型来计算中文 BERTScore
        # rescale_with_baseline 可以做也可以不做，看具体需求
        P, R, F1 = bert_score.score(preds, refs, lang="zh", model_type="bert-base-chinese", verbose=False)
        avg_bert_score = F1.mean().item()
    except Exception as e:
        print(f"BERTScore calculation error: {e}")
        avg_bert_score = 0.0

    return {
        "BLEU-4": avg_bleu4,
        "ROUGE-L": rouge_l_f1,
        "BERTScore": avg_bert_score
    }

def evaluate_cmedqa(model_path, dataset_path, num_samples=100):
    dataset = load_data(dataset_path)
    if not dataset:
        return {}
    
    dataset = dataset[:num_samples] if num_samples else dataset
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    preds = []
    refs = []

    with torch.no_grad():
        for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {os.path.basename(model_path)} on cMedQA2")):
            messages = item.get("messages")
            # 提取 user 和 system（排除了真实的 assistant 答案），并动态注入更严厉的系统提示词以去除废话和限制篇幅
            prompt_msgs = []
            for m in messages:
                if m["role"] == "system":
                    # 覆盖原来的系统提示词
                    m_copy = m.copy()
                    m_copy["content"] = "你是一名网络医疗问答医生。请用中文互联网医疗平台医生回复的常见风格回答问题：先给简短判断，再给1到2条处理建议。不要过度诊断，不要给复杂病理分析，不要给过多药名、剂量或详细治疗方案；若信息不足，只说“考虑”“可能”“建议检查/复查”。回答控制在2句内、100字以内，不要分点，不要标题，不要废话。"
                    prompt_msgs.append(m_copy)
                elif m["role"] == "user":
                    prompt_msgs.append(m)
                    
            text = tokenizer.apply_chat_template(
                prompt_msgs, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            # 采用官方推荐的 Instruct 模式生成参数
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128,
                do_sample=False,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            ref_text = item.get("answer", "").strip()
            
            preds.append(pred_text)
            refs.append(ref_text)
            
            if i < 5:
                print(f"\n[示例 {i+1}] ========== 测试抽样 ==========")
                print(f"【问题】: {item.get('question')}")
                print(f"【预测】: {pred_text}")
                print(f"【真实】: {ref_text}")
                print("========================================")

    del model
    del tokenizer
    torch.cuda.empty_cache()

    print(f"\n开始计算指标 (共 {len(preds)} 个样本)...")
    metrics = calc_metrics(preds, refs)
    return metrics

def main():
    results = {
        "Qwen3.5-2B": {},
        "Qwen3.5-4B": {},
        "Qwen3.5-9B": {},
    }

    num_samples = 100

    print("=== 开始 cMedQA2 开放式问答评估 ===")
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"[{model_name}] 路径不存在，跳过")
            continue
            
        print(f"\n--- {model_name} 评估中 ---")
        metrics = evaluate_cmedqa(model_path, CMEDQA2_DATASET, num_samples=num_samples)
        results[model_name] = metrics
        print(f"[{model_name}] {metrics}")

    with open("eval_results_cmedqa.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 画图
    models_list = [m for m in results.keys() if results[m]]
    bleu4_scores = [results[m]["BLEU-4"] for m in models_list]
    rouge_l_scores = [results[m]["ROUGE-L"] for m in models_list]
    bert_scores = [results[m]["BERTScore"] for m in models_list]

    x = range(len(models_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width for i in x], bleu4_scores, width, label='BLEU-4')
    rects2 = ax.bar([i for i in x], rouge_l_scores, width, label='ROUGE-L')
    rects3 = ax.bar([i + width for i in x], bert_scores, width, label='BERTScore (F1)')

    ax.set_ylabel('Score')
    ax.set_title('Open-ended QA Evaluation on cMedQA2')
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend()
    ax.set_ylim(0, 1.0)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.savefig("cmedqa_eval_comparison.png")
    print("\ncMedQA 评估完成，结果已保存到 eval_results_cmedqa.json，并生成图表 cmedqa_eval_comparison.png")

if __name__ == "__main__":
    main()