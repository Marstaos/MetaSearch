import os
import sys
import argparse
import yaml
import logging
import pickle
import time

from deepsearch.llm.model import LLMFactory
from deepsearch.indexing.vector_index import VectorEncoder, VectorRetriever
from deepsearch.indexing.tfidf_index import TFIDFRetriever
from deepsearch.indexing.kg_index import KGRetriever
from deepsearch.retrieval.ranker import ReRanker
from deepsearch.retrieval.searcher import CombinedSearcher
from deepsearch.rag.query_expander import QueryExpander
from deepsearch.rag.deep_rag import DeepRAG

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DeepSearch RAG 系统')
    parser.add_argument('--query', type=str, help='输入查询')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    parser.add_argument('--model_type', type=str, choices=['api', 'local'], default='api', help='LLM类型')
    
    args = parser.parse_args()
    
    # 加载系统组件
    logger.info("欢迎使用MetaSearch！初始化系统组件...")
    
    # 加载chunks
    chunks_path = os.path.join(config['paths']['data']['processed'], 'index_chunk.pkl')
    try:
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        logger.info(f"加载chunks成功，共 {len(chunks)} 个")
    except Exception as e:
        logger.error(f"加载chunks失败: {e}")
        sys.exit(1)
    
    # 初始化LLM
    llm = LLMFactory.create_llm(model_type=args.model_type)
    
    # 初始化检索组件
    vector_encoder = VectorEncoder()
    vector_retriever = VectorRetriever(encoder=vector_encoder)
    tfidf_retriever = TFIDFRetriever()
    kg_retriever = KGRetriever()
    
    # 初始化重排序组件
    reranker = ReRanker()
    
    # 初始化搜索器
    searcher = CombinedSearcher(
        vector_retriever=vector_retriever,
        tfidf_retriever=tfidf_retriever,
        kg_retriever=kg_retriever,
        reranker=reranker,
        chunks=chunks
    )
    
    # 初始化查询扩展器
    query_expander = QueryExpander(llm=llm, reranker=reranker)
    
    # 初始化深度RAG
    deep_rag = DeepRAG(
        searcher=searcher,
        llm=llm,
        query_expander=query_expander
    )
    
    logger.info("系统初始化完成")
    
    # 处理查询
    if args.interactive:
        # 交互模式
        print("欢迎使用MetaSearch系统！输入'quit'或'exit'退出")
        while True:
            try:
                query = input("\n请输入您的问题: ")
                if query.lower() in ['quit', 'exit']:
                    break
                
                start_time = time.time()
                answer = deep_rag.answer(query)
                end_time = time.time()
                
                print(f"\n答案:\n{answer}")
                print(f"\n[用时: {end_time - start_time:.2f}秒]")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"处理查询失败: {e}")
                print(f"处理查询时发生错误: {e}")
    else:
        # 单次查询模式
        if not args.query:
            logger.error("非交互模式下必须提供--query参数")
            sys.exit(1)
        
        try:
            answer = deep_rag.answer(args.query)
            print(answer)
        except Exception as e:
            logger.error(f"处理查询失败: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()