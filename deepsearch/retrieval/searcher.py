import os
import yaml
import logging
import time
from typing import List, Dict, Tuple, Optional, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class CombinedSearcher:
    """综合搜索器，整合多种检索方法"""
    
    def __init__(
        self, 
        vector_retriever=None, 
        tfidf_retriever=None, 
        kg_retriever=None,
        reranker=None,
        chunks=None
    ):
        """初始化综合搜索器
        
        Args:
            vector_retriever: 向量检索器
            tfidf_retriever: TF-IDF检索器
            kg_retriever: 知识图谱检索器
            reranker: 重排序器
            chunks: 文档块字典
        """
        self.vector_retriever = vector_retriever
        self.tfidf_retriever = tfidf_retriever
        self.kg_retriever = kg_retriever
        self.reranker = reranker
        self.chunks = chunks
        
        # 设置默认检索参数
        self.faiss_candidates = config['retrieval']['faiss_candidates']
        self.tfidf_candidates = config['retrieval']['tfidf_candidates']
        self.total_candidates = config['retrieval']['total_candidates']
    
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        use_vector: bool = True,
        use_tfidf: bool = True,
        use_kg: bool = False
    ) -> Tuple[str, List[int]]:
        """执行综合搜索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            use_vector: 是否使用向量检索
            use_tfidf: 是否使用TF-IDF检索
            use_kg: 是否使用知识图谱检索
            
        Returns:
            (拼接的内容字符串, chunk ID列表)
        """
        if top_k is None:
            top_k = self.total_candidates
            
        candidate_ids = []
        
        # 向量检索
        if use_vector and self.vector_retriever:
            logger.info(f"开始向量检索: {query}")
            start_time = time.time()
            vector_ids = self.vector_retriever.search(query, self.faiss_candidates)
            end_time = time.time()
            candidate_ids.extend(vector_ids)
            logger.info(f"向量检索完成，找到 {len(vector_ids)} 个结果，用时: {end_time - start_time:.2f}秒")
        
        # TF-IDF检索
        if use_tfidf and self.tfidf_retriever:
            logger.info(f"开始TF-IDF检索: {query}")
            start_time = time.time()
            tfidf_ids = self.tfidf_retriever.search(query, self.tfidf_candidates)
            end_time = time.time()
            candidate_ids.extend(tfidf_ids)
            logger.info(f"TF-IDF检索完成，找到 {len(tfidf_ids)} 个结果，用时: {end_time - start_time:.2f}秒")
        
        # 知识图谱检索
        if use_kg and self.kg_retriever:
            logger.info(f"开始知识图谱检索: {query}")
            start_time = time.time()
            kg_ids = self.kg_retriever.search(query)
            end_time = time.time()
            candidate_ids.extend(kg_ids)
            logger.info(f"知识图谱检索完成，找到 {len(kg_ids)} 个结果，用时: {end_time - start_time:.2f}秒")
        
        # 去重
        start_time = time.time()
        original_count = len(candidate_ids)
        candidate_ids = list(set(candidate_ids))
        end_time = time.time()
        logger.info(f"候选文档去重完成: 从 {original_count} 个减少到 {len(candidate_ids)} 个，用时: {end_time - start_time:.2f}秒")
        
        # 重排序
        if self.reranker and self.chunks:
            logger.info(f"开始准备重排序数据...")
            start_time = time.time()
            # 准备查询-文档对，确保内容为字符串
            pairs = []
            for id in candidate_ids:
                abstract = self.chunks[id].abstract
                if abstract is None:
                    abstract = ""
                elif not isinstance(abstract, str):
                    abstract = str(abstract)
                pairs.append([query, abstract])
            
            # 获取向量表示
            vectors = [self.vector_retriever.chunkid_vector[str(id)] for id in candidate_ids]
            end_time = time.time()
            logger.info(f"重排序数据准备完成，用时: {end_time - start_time:.2f}秒")
            
            # 计算相关性得分
            logger.info(f"开始计算相关性得分...")
            start_time = time.time()
            scores = self.reranker.compute_scores(pairs)
            end_time = time.time()
            logger.info(f"相关性得分计算完成，用时: {end_time - start_time:.2f}秒")
            
            # 应用MMR进行多样性重排序
            logger.info(f"开始MMR多样性重排序...")
            start_time = time.time()
            from deepsearch.retrieval.ranker import MMRReranker
            top_ids = MMRReranker.rerank(
                candidate_ids, 
                scores.tolist(), 
                vectors, 
                top_k=top_k
            )
            end_time = time.time()
            logger.info(f"MMR重排序完成，选出前 {len(top_ids)} 个文档，用时: {end_time - start_time:.2f}秒")
        else:
            # 如果没有重排序器，简单截取前top_k个
            logger.info(f"未使用重排序，直接选择前 {top_k} 个文档")
            top_ids = candidate_ids[:top_k]
        
        # 构建内容字符串
        logger.info(f"开始构建内容字符串...")
        start_time = time.time()
        contents = [f"知识点{i+1}:\n{self.chunks[id].content}" for i, id in enumerate(top_ids)]
        content_str = "\n\n".join(contents)
        end_time = time.time()
        logger.info(f"内容字符串构建完成，总长度: {len(content_str)}，用时: {end_time - start_time:.2f}秒")
        
        return content_str, top_ids