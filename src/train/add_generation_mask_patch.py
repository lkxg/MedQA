import json
import os
import glob

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))

for chat_template_path in glob.glob(os.path.join(models_dir, "*/chat_template.jinja")):
    print(f"Patching {chat_template_path}")
    with open(chat_template_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    # 替换 assistant 的 content 生成逻辑
    old_str1 = """        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}"""
    
    new_str1 = """        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' }}{% generation %}{{ content }}{% endgeneration %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' }}{% generation %}{{ content }}{% endgeneration %}
        {%- endif %}"""
    
    if old_str1 in template:
        template = template.replace(old_str1, new_str1)
        with open(chat_template_path, "w", encoding="utf-8") as f:
            f.write(template)
        print("Patched successfully!")
    elif "{% generation %}" in template:
        print("Already patched!")
    else:
        print("Failed to find replacement target!")

    # 还要更新 tokenizer_config.json 里的 chat_template
    tokenizer_config_path = os.path.dirname(chat_template_path) + "/tokenizer_config.json"
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "chat_template" in config:
            config["chat_template"] = template
            with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"Updated {tokenizer_config_path}")

