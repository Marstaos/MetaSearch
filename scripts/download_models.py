import os
import sys
import argparse
import yaml
import logging
from huggingface_hub import snapshot_download
from pathlib import Path

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 模型映射表，将配置中的模型名称映射到Hugging Face上的模型ID
MODEL_MAPPING = {
    'bce-embedding-base_v1': 'maidalun1020/bce-embedding-base_v1',
    'bge-reranker-v2-m3': 'BAAI/bge-reranker-v2-m3',
    'qwen2.5-14B': 'Qwen/Qwen2.5-14B-Chat'
}

def download_model(model_name, model_type, output_dir, use_mirror=True):
    """下载模型
    
    Args:
        model_name: 模型名称
        model_type: 模型类型（embedding, reranker, llm）
        output_dir: 输出目录
        use_mirror: 是否使用镜像站
    """
    if model_name not in MODEL_MAPPING:
        logger.error(f"未知的模型名称: {model_name}")
        return False
    
    repo_id = MODEL_MAPPING[model_name]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        logger.info(f"开始下载模型: {model_name} -> {repo_id}")
        
        # 设置镜像站
        hf_endpoint = "https://hf-mirror.com" if use_mirror else None
        
        # 下载模型
        snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            endpoint=hf_endpoint
        )
        
        logger.info(f"模型 {model_name} 下载完成，保存到 {output_dir}")
        return True
    except Exception as e:
        logger.error(f"下载模型 {model_name} 失败: {e}")
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='下载项目所需的模型')
    parser.add_argument('--model_type', type=str, choices=['embedding', 'reranker', 'llm', 'all'], 
                        default='all', help='要下载的模型类型')
    parser.add_argument('--no_mirror', action='store_true', help='不使用镜像站')
    parser.add_argument('--skip_qwen', action='store_true', help='跳过下载qwen模型')
    
    args = parser.parse_args()
    use_mirror = not args.no_mirror
    
    # 确保模型目录存在
    os.makedirs('models', exist_ok=True)
    
    # 下载模型
    if args.model_type == 'all' or args.model_type == 'embedding':
        model_name = config['models']['embedding']['name']
        model_path = config['models']['embedding']['path']
        download_model(
            config['models']['embedding']['path'].split('/')[-1], 
            'embedding',
            model_path,
            use_mirror
        )
    
    if args.model_type == 'all' or args.model_type == 'reranker':
        model_name = config['models']['reranker']['name']
        model_path = config['models']['reranker']['path']
        download_model(
            config['models']['reranker']['path'].split('/')[-1], 
            'reranker',
            model_path,
            use_mirror
        )
    
    if (args.model_type == 'all' or args.model_type == 'llm') and not args.skip_qwen:
        model_name = config['models']['llm']['name']
        model_path = config['models']['llm']['path']
        download_model(
            config['models']['llm']['path'].split('/')[-1], 
            'llm',
            model_path,
            use_mirror
        )
    elif args.model_type == 'llm' and args.skip_qwen:
        logger.info("根据设置跳过下载qwen模型")
    
    logger.info("所有模型下载完成")

if __name__ == "__main__":
    main()