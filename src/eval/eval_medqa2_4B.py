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

# 设置模型路径: 对比基座和微调版本
MODELS = {
    "Qwen3.5-4B (Base)": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/Qwen3.5-4B")),
    "Qwen3.5-4B (Finetuned)": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints/baseline_lora_medical_lora_r16/final")),
}

# cMedQA2 测试集
CMEDQA2_DATASET = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/test_data/cmedqa2_test.jsonl"))

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
    preds_tokens = [list(jieba.cut(p)) if p.strip() else ["无"] for p in preds]
    refs_tokens =  [list(jieba.cut(r)) if r.strip() else ["无"] for r in refs]
    preds_jieba = [" ".join(p) for p in preds_tokens]
    refs_jieba  = [" ".join(r) for r in refs_tokens]

    smooth = SmoothingFunction().method1
    bleu_scores = []
    for pred_tok, ref_tok in zip(preds_tokens, refs_tokens):
        if pred_tok == ["无"]:
            bleu_scores.append(0.0)
            continue
        try:
            score = sentence_bleu([ref_tok], pred_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        except Exception:
            score = 0.0
        bleu_scores.append(score)
    avg_bleu4 = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(preds_jieba, refs_jieba, avg=True)
        rouge_l_f1 = rouge_scores['rouge-l']['f']
    except Exception:
        rouge_l_f1 = 0.0

    try:
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
    
    # 支持加载 PEFT (LoRA)
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

    preds, refs = [], []
    with torch.no_grad():
        for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {os.path.basename(model_path)}")):
            prompt_msgs = []
            for m in item.get("messages"):
                if m["role"] == "system":
                    m_copy = m.copy()
                    m_copy["content"] = "你是一名专业的医疗问答医生：先给简短判断，再给1到2条处理建议。回答控制在2、3句内，不要分点，不要标题"
                    prompt_msgs.append(m_copy)
                elif m["role"] == "user":
                    prompt_msgs.append(m)
                    
            text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            # Qwen3.5 的 chat 模板是用 <|im_end|> 作为结束符
            # 需要将它加入到停止条件里，否则模型会无限生成停不下来
            stop_words_ids = [tokenizer.eos_token_id]
            if "<|im_end|>" in tokenizer.vocab:
                stop_words_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=True, 
                temperature=0.7, 
                top_p=0.85, 
                pad_token_id=tokenizer.eos_token_id, 
                eos_token_id=stop_words_ids
            )
            pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            ref_text = item.get("answer", "").strip()
            preds.append(pred_text)
            refs.append(ref_text)
            
            if i < 3:
                print(f"\n[示例 {i+1}]")
                print(f"【问题】: {item.get('question')}")
                print(f"【预测】: {pred_text}\n")

    del model, tokenizer
    torch.cuda.empty_cache()
    return calc_metrics(preds, refs)

def main():
    results = {"Qwen3.5-4B (Base)": {}, "Qwen3.5-4B (Finetuned)": {}}
    num_samples = 100

    print("=== 开始 cMedQA2 (4B) 评估 ===")
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"[{model_name}] 路径不存在，跳过")
            continue
        print(f"\n--- {model_name} ---")
        metrics = evaluate_cmedqa(model_path, CMEDQA2_DATASET, num_samples=num_samples)
        results[model_name] = metrics
        print(f"[{model_name}] {metrics}")

    with open(os.path.join(os.path.dirname(__file__), "../../outputs/cmedqa2_4b_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    models_list = [m for m in results.keys() if results[m]]
    if not models_list: return
    bleu4_scores = [results[m]["BLEU-4"] for m in models_list]
    rouge_l_scores = [results[m]["ROUGE-L"] for m in models_list]
    bert_scores = [results[m]["BERTScore"] for m in models_list]

    x, width = range(len(models_list)), 0.25
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([i - width for i in x], bleu4_scores, width, label='BLEU-4')
    ax.bar([i for i in x], rouge_l_scores, width, label='ROUGE-L')
    ax.bar([i + width for i in x], bert_scores, width, label='BERTScore (F1)')
    ax.set_title('Open-ended QA: cMedQA2 4B Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "../../outputs/cmedqa2_4b_comparison.png"))
    print("\n评估完成图表 cmedqa2_4b_comparison.png 已生成")

if __name__ == "__main__":
    main()
