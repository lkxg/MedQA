"""
一键下载项目所需的 Qwen3.5 系列模型。
默认使用 huggingface 国内镜像源加速下载。
"""
import os
import argparse
from huggingface_hub import snapshot_download

# 强制设置国内 HF 镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 开启 hf-transfer 极速下载（如果安装了 hf_transfer 的话）
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MODELS = {
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B"
}

def main():
    parser = argparse.ArgumentParser(description="一键下载 MedQA 项目所需的 Qwen3.5 模型")
    parser.add_argument("--save_dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models")), 
                        help="模型的保存根目录 (默认: 项目根目录下的 models 文件夹)")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()) + ["all"], 
                        default=["all"], help="指定要下载的模型")
    args = parser.parse_args()

    targets = list(MODELS.keys()) if "all" in args.models else args.models
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"🚀 准备下载模型到: {args.save_dir}")
    print(f"📦 目标模型: {targets}")

    for model_name in targets:
        repo_id = MODELS[model_name]
        local_dir = os.path.join(args.save_dir, model_name)
        
        print(f"\n" + "="*50)
        print(f"⏳ 开始下载: {model_name} (HuggingFace ID: {repo_id})")
        print(f"📂 目标路径: {local_dir}")
        print("="*50)
        
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False, # 直接保存文件而非软链接，方便随时移动
                resume_download=True,         # 支持断点续传
                max_workers=4                 # 多线程下载
            )
            print(f"✅ {model_name} 下载完成！")
        except Exception as e:
            print(f"❌ {model_name} 下载失败，错误信息: {e}")

if __name__ == "__main__":
    main()
