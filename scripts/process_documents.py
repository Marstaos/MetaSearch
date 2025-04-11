import os
import sys
import argparse
import yaml
import logging

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepsearch.preprocessing.document import process_document
from deepsearch.llm.model import LLMFactory

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='处理文档并分割成chunks')
    parser.add_argument('--file', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, help='输出目录路径')
    parser.add_argument('--chunk_size', type=int, help='chunk大小')
    parser.add_argument('--overlap_size', type=int, help='chunk重叠大小')
    parser.add_argument('--model_type', type=str, choices=['api', 'local'], default='api', help='LLM类型')
    
    args = parser.parse_args()
    
    # 初始化LLM，用于生成摘要
    llm = LLMFactory.create_llm(model_type=args.model_type)
    
    # 处理文档
    try:
        process_document(
            file_path=args.file,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            overlap_size=args.overlap_size,
            get_abstract_func=llm
        )
        logger.info("文档处理完成")
    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()