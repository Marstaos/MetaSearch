import os
import sys
import argparse
import yaml
import logging
import pickle

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepsearch.indexing.vector_index import VectorEncoder, VectorIndexBuilder
from deepsearch.indexing.tfidf_index import TFIDFIndexBuilder
from deepsearch.indexing.kg_index import KGIndexBuilder
from deepsearch.llm.model import LLMFactory

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='构建索引')
    parser.add_argument('--chunks', type=str, required=True, help='chunks文件路径')
    parser.add_argument('--force', action='store_true', help='强制重建索引')
    parser.add_argument('--skip_vector', action='store_true', help='跳过向量索引构建')
    parser.add_argument('--skip_tfidf', action='store_true', help='跳过TF-IDF索引构建')
    parser.add_argument('--skip_kg', action='store_true', help='跳过知识图谱索引构建')
    parser.add_argument('--model_type', type=str, choices=['api', 'local'], default='api', help='LLM类型')
    
    args = parser.parse_args()
    
    # 加载chunks
    try:
        with open(args.chunks, 'rb') as f:
            chunks = pickle.load(f)
        logger.info(f"加载chunks成功，共 {len(chunks)} 个")
    except Exception as e:
        logger.error(f"加载chunks失败: {e}")
        sys.exit(1)
    
    # 构建向量索引
    if not args.skip_vector:
        try:
            encoder = VectorEncoder()
            vector_builder = VectorIndexBuilder(encoder=encoder)
            vector_builder.build_index(chunks, force_rebuild=args.force)
            logger.info("向量索引构建完成")
        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
    
    # 构建TF-IDF索引
    if not args.skip_tfidf:
        try:
            tfidf_builder = TFIDFIndexBuilder()
            tfidf_builder.build_index(chunks, force_rebuild=args.force)
            logger.info("TF-IDF索引构建完成")
        except Exception as e:
            logger.error(f"构建TF-IDF索引失败: {e}")
    
    # 构建知识图谱索引
    if not args.skip_kg:
        try:
            # 初始化LLM，用于抽取实体
            llm = LLMFactory.create_llm(model_type=args.model_type)
            
            kg_builder = KGIndexBuilder(extract_entity_func=llm)
            kg_builder.build_index(chunks, force_rebuild=args.force)
            logger.info("知识图谱索引构建完成")
        except Exception as e:
            logger.error(f"构建知识图谱索引失败: {e}")

if __name__ == "__main__":
    main()