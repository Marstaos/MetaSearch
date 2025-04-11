import os
import yaml
import logging
from typing import Dict, List, Optional
import pickle

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class TFIDFIndexBuilder:
    """TF-IDF索引构建器"""
    
    def __init__(self, index_dir: Optional[str] = None):
        """初始化TF-IDF索引构建器
        
        Args:
            index_dir: 索引存储目录
        """
        if index_dir is None:
            index_dir = config['paths']['indexes']['tfidf']
            
        self.index_dir = index_dir
        
        # 创建索引目录
        os.makedirs(index_dir, exist_ok=True)
    
    def build_index(self, chunks: Dict, force_rebuild: bool = False) -> None:
        """构建TF-IDF索引
        
        Args:
            chunks: 文档块字典
            force_rebuild: 是否强制重建索引
        """
        from whoosh.index import create_in, exists_in
        from whoosh.fields import Schema, TEXT, ID
        
        # 检查索引是否已存在
        if exists_in(self.index_dir) and not force_rebuild:
            logger.info(f"TF-IDF索引已存在: {self.index_dir}")
            return
        
        logger.info(f"开始构建TF-IDF索引，共 {len(chunks)} 个文档块")
        
        # 定义Schema
        schema = Schema(content=TEXT, path=ID(stored=True))
        
        # 创建索引
        ix = create_in(self.index_dir, schema)
        writer = ix.writer()
        
        # 添加文档
        for chunk_id, chunk in chunks.items():
            writer.add_document(content=chunk.content, path=str(chunk_id))
        
        # 提交索引
        writer.commit()
        
        logger.info("TF-IDF索引构建完成")


class TFIDFRetriever:
    """TF-IDF检索器"""
    
    def __init__(self, index_dir: Optional[str] = None):
        """初始化TF-IDF检索器
        
        Args:
            index_dir: 索引目录
        """
        if index_dir is None:
            index_dir = config['paths']['indexes']['tfidf']
            
        self.index_dir = index_dir
    
    def search(self, query: str, num: int = 100) -> List[int]:
        """根据查询检索相关文档
        
        Args:
            query: 查询文本
            num: 返回结果数量
            
        Returns:
            相关chunk ID列表
        """
        import jieba
        from whoosh.index import open_dir
        from whoosh.qparser import QueryParser
        from whoosh.query import Or
        
        # 对查询进行分词
        query_terms = list([s for s in jieba.cut(query)])
        
        # 打开索引
        ix = open_dir(self.index_dir)
        
        with ix.searcher() as searcher:
            # 构建OR查询（包含任意查询词的文档）
            query_list = []
            for q in query_terms:
                query_list.append(QueryParser("content", ix.schema).parse(q))
            
            query_obj = Or(query_list)
            
            # 执行搜索
            results = searcher.search(query_obj, limit=num)
            
            # 提取chunk ID
            chunk_ids = [int(result["path"]) for result in results]
            
            return chunk_ids