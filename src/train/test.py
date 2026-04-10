import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ========== 路径配置 ==========
BASE_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/Qwen3.5-4B"))
LORA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints/baseline_lora_medical_lora_r16/final"))

print("1. 加载基座模型和 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("2. 挂载微调后的 LoRA Adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()
print(f"✅ 模型与权重加载成功，正在使用 GPU...")

# ========== 测试用例 ==========
system_prompt = "你是一名网络医疗问答医生。请用中文互联网医疗平台医生回复的常见风格回答问题：先给简短判断，再给1到2条处理建议。不要过度诊断，不要给复杂病理分析，不要给过多药名、剂量或详细治疗方案；若信息不足，只说“考虑”“可能”“建议检查/复查”。回答控制在2句内、100字以内，不要分点，不要标题，不要废话。"

test_questions = [
    "如何洗掉衣服上风油精？",
    "医生，我这两天老是头晕，还有点恶心，是不是感冒了？"
]

print("\n" + "="*50)
with torch.no_grad():
    for q in test_questions:
        print(f"🗣️ 患者: {q}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 限制最大生成长度在 128 (因为我们要求 100 字以内)
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
        
        # 去掉输入 prompt 的部分，只截取生成的答案
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"👨‍⚕️ 医生: {response}\n" + "-"*50)
