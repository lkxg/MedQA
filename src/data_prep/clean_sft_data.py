"""
清洗 HuatuoGPT-SFT 训练数据集 (v2 - 完整版)
目标: 移除 assistant 回复末尾的所有客套废话
策略: 按句子从末尾向前逐句判断是否为废话，是则删除

用法:
    python src/data/clean_sft_data.py
"""
import json
import re
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

INPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/sft_data/train.jsonl"))
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/sft_data/train_cleaned.jsonl"))

# ============================================================
# 废话关键短语（出现在句子中即判定整句为废话）
# ============================================================
GARBAGE_PHRASES = [
    # 祝福系列
    "祝您", "祝你", "祝宝宝", "祝早日", "祝身体", "祝孩子",
    
    # "希望..." 系列
    "希望我的回答", "希望我的建议", "希望我的解答",
    "希望对您有", "希望对你有",
    "希望您早日", "希望你早日",
    "希望能对您", "希望能对你", "希望能帮",
    "希望这些信息", "希望这些建议", "希望这些内容",
    "希望以上信息", "希望以上建议", "希望以上内容", "希望以上回答",
    
    # "如果您还有..." 系列
    "如果您还有其他", "如果你还有其他",
    "如果您有任何", "如果你有任何",
    "如果您有更多", "如果你有更多",
    "如果您需要更多", "如果你需要更多",
    "如果您还有疑问", "如果你还有疑问",
    "如有疑问", "如有其他",
    
    # "请随时..." 系列
    "请随时向我", "请随时与我", "请随时咨询", "请随时与医生",
    "请随时联系", "请随时提问",
    "可以随时向我", "可以随时咨询", "可以随时联系",
    
    # "欢迎..." 系列
    "欢迎随时", "欢迎继续", "欢迎您随时",
    
    # "感谢/谢谢..." 系列（仅匹配客套用法）
    "感谢您的咨询", "感谢您的信任", "感谢您的提问",
    "谢谢您的咨询", "谢谢您的信任", "谢谢您的提问",
    
    # 自曝身份
    "我是一名医疗", "作为一名AI", "作为AI",
    
    # "以上是..." 系列
    "以上是对", "以上就是", "以上是我的",
]

# 训练数据中 54000 条全部使用了同一个 system prompt，
# 清洗时直接移除所有 system 角色消息，保留基座模型的指令跟随能力。

def is_garbage_sentence(sentence):
    """判断一个句子是否为废话"""
    s = sentence.strip()
    if not s:
        return True
    for phrase in GARBAGE_PHRASES:
        if phrase in s:
            return True
    return False

def clean_assistant_text(text):
    """从末尾逐句剥离废话"""
    original = text.strip()
    if not original:
        return original
    
    # [新增优化] 正则剔除开头的 "你好，我是XXX语言模型..." 废话
    original = re.sub(
        r'^(您好|你好)[，。！]?(我是|作为)(一个|一名)?(医疗)?(语言模型|人工智能|大模型)?[a-zA-Z]*(HuatuoGPT)?[，。！\s]*', 
        '', original, flags=re.IGNORECASE
    )
    # 按中文标点、逗号、分号等分句，避免长句被整段误删
    parts = re.split(r'([。！？\n，；.,;])', original)
    
    # 重组为完整句子 (句子+标点)
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and re.match(r'^[。！？\n，；.,;]$', parts[i+1]):
            sentences.append(parts[i] + parts[i+1])
            i += 2
        else:
            sentences.append(parts[i])
            i += 1
    
    # 去掉空白句子
    sentences = [s for s in sentences if s.strip()]
    
    if not sentences:
        return original
    
    # 从末尾向前剥离废话句子
    while sentences and is_garbage_sentence(sentences[-1]):
        sentences.pop()
    
    if not sentences:
        # 全部被删了，保留原文（不能搞成空的）
        return original
    
    result = ''.join(sentences).strip()
    
    # 二次清理：末尾可能残留的逗号、空格
    result = re.sub(r'[，,\s]+$', '', result)
    
    # 补上句号（如果末尾不是标点）
    if result and result[-1] not in '。！？…》）)】]"\'':
        result += '。'
    
    return result

def clean_file(input_file, output_file):
    print(f"\n🚀 开始处理: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total = len(lines)
    modified_count = 0
    cleaned_data = []
    
    # 废话关键词（用于统计）
    stat_keywords = ['祝您', '祝你', '希望我的回答', '希望对您', '希望这些', '希望以上',
                     '如果您还有其他', '如果您有任何', '请随时向我', '请随时与我',
                     '可以随时向我', '欢迎随时', '感谢您的', '请随时咨询',
                     '希望您早日', '请随时与医生', '希望能对']
    before_counts = {k: 0 for k in stat_keywords}
    after_counts = {k: 0 for k in stat_keywords}
    
    system_removed = 0
    
    for line in lines:
        d = json.loads(line)
        modified = False
        
        # 1. 移除固定的 system prompt
        original_len = len(d['messages'])
        d['messages'] = [m for m in d['messages'] if m['role'] != 'system']
        if len(d['messages']) < original_len:
            system_removed += 1
            modified = True
        
        # 2. 清洗 assistant 回复末尾废话
        for m in d['messages']:
            if m['role'] == 'assistant':
                original = m['content']
                for k in stat_keywords:
                    if k in original:
                        before_counts[k] += 1
                
                cleaned = clean_assistant_text(original)
                
                if cleaned != original.strip():
                    m['content'] = cleaned
                    modified = True
                
                for k in stat_keywords:
                    if k in cleaned:
                        after_counts[k] += 1
        
        cleaned_data.append(d)
        if modified:
            modified_count += 1
    
    # 写入
    with open(output_file, 'w', encoding='utf-8') as f:
        for d in cleaned_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    
    # 报告
    print(f"\n{'='*55}")
    print(f" 清洗报告 ({os.path.basename(input_file)})")
    print(f"{'='*55}")
    print(f" 总样本数: {total}")
    print(f" 被修改的样本数: {modified_count} ({modified_count/total*100:.1f}%)")
    print(f" 移除 system prompt: {system_removed} 条")
    print(f"\n {'关键词':<16} {'清洗前':>7} {'清洗后':>7} {'清除率':>7}")
    print(f" {'-'*42}")
    for k in stat_keywords:
        b, a = before_counts[k], after_counts[k]
        rate = (1 - a/b)*100 if b > 0 else 0
        print(f"  {k:<14} {b:>6} {a:>6}   {rate:>5.1f}%")
    
    # 清洗前后再统计一次总命中
    hit_before = sum([1 for d in [json.loads(line) for line in lines] for m in d['messages'] if m['role'] == 'assistant' and any(k in m['content'] for k in stat_keywords)])
    hit_after = 0
    for d in cleaned_data:
        for m in d['messages']:
            if m['role'] == 'assistant':
                for k in stat_keywords:
                    if k in m['content']:
                        hit_after += 1
                        break
    print(f"\n 清洗前含废话样本: {hit_before} ({hit_before/total*100:.1f}%)")
    print(f" 清洗后含废话样本: {hit_after} ({hit_after/total*100:.1f}%)")
    print(f"\n✅ 清洗后数据已保存: {output_file}")
    
    # 示例
    print(f"\n{'='*55}")
    print(f" 清洗效果示例 ({os.path.basename(input_file)})")
    print(f"{'='*55}")
    with open(input_file, 'r', encoding='utf-8') as f:
        orig_lines = f.readlines()
    shown = 0
    for i, (ol, cd) in enumerate(zip(orig_lines, cleaned_data)):
        orig = json.loads(ol)
        for mo, mc in zip(orig['messages'], cd['messages']):
            if mo['role'] == 'assistant' and mo['content'].strip() != mc['content']:
                print(f"\n[样本 {i+1}]")
                print(f"  前: ...{mo['content'].strip()[-100:]}")
                print(f"  后: ...{mc['content'][-100:]}")
                shown += 1
                break
        if shown >= 3: # 示例减少到3条避免输出过长
            break

def main():
    val_input = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/sft_data/val.jsonl"))
    val_output = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/sft_data/val_cleaned.jsonl"))
    
    # 清洗训练集
    clean_file(INPUT_FILE, OUTPUT_FILE)
    
    # 清洗验证集
    if os.path.exists(val_input):
        clean_file(val_input, val_output)
    else:
        print(f"未找到验证集文件: {val_input}")

if __name__ == "__main__":
    main()