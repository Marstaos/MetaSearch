import os
import json
import yaml
import logging
from typing import Dict, List, Optional
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class KGIndexBuilder:
    """知识图谱索引构建器"""
    
    def __init__(
        self, 
        extract_entity_func=None,
        index_dir: Optional[str] = None
    ):
        """初始化知识图谱索引构建器
        
        Args:
            extract_entity_func: 实体抽取函数
            index_dir: 索引存储目录
        """
        if index_dir is None:
            index_dir = config['paths']['indexes']['kg']
            
        self.extract_entity_func = extract_entity_func
        self.index_dir = index_dir
        
        # 创建索引目录
        os.makedirs(index_dir, exist_ok=True)
    
    def build_index(self, chunks: Dict, force_rebuild: bool = False) -> None:
        """构建知识图谱索引
        
        Args:
            chunks: 文档块字典
            force_rebuild: 是否强制重建索引
        """
        # 检查索引文件是否已存在
        index_file = os.path.join(self.index_dir, 'entity_chunks.json')
        if os.path.exists(index_file) and not force_rebuild:
            logger.info(f"知识图谱索引已存在: {index_file}")
            return
        
        if self.extract_entity_func is None:
            logger.warning("未提供实体抽取函数，跳过知识图谱索引构建")
            return
        
        logger.info(f"开始构建知识图谱索引，共 {len(chunks)} 个文档块")
        
        entity_prompt = "抽取上述句子的三元组，实体为人，地点，历史事件，主谓宾用-隔开，如刘备-兄弟-关羽，三元组之间用|隔开"
        entity_chunk = {}
        
        for chunk_id, chunk in tqdm(chunks.items()):
            if chunk.parent is None:
                continue
                
            # 对chunk的父块进行抽取三元组
            text = chunk.parent
            try:
                response = self.extract_entity_func(text + entity_prompt)
                # 处理AIMessage对象，提取content属性
                if hasattr(response, 'content'):
                    triples = response.content
                else:
                    triples = response
                triples = set(triples.split("|"))
                
                for triple in triples:
                    if not triple or triple.isspace():
                        continue
                        
                    if triple not in entity_chunk:
                        entity_chunk[triple] = []
                    entity_chunk[triple].append(chunk_id)
            except Exception as e:
                logger.error(f"抽取实体失败: {e}")
        
        # 保存知识图谱索引
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(entity_chunk, f, ensure_ascii=False, indent=2)
        
        logger.info("知识图谱索引构建完成")


class KGRetriever:
    """知识图谱检索器"""
    
    def __init__(self, index_dir: Optional[str] = None):
        """初始化知识图谱检索器
        
        Args:
            index_dir: 索引目录
        """
        if index_dir is None:
            index_dir = config['paths']['indexes']['kg']
            
        self.index_dir = index_dir
        
        # 加载索引
        self._load_index()
    
    def _load_index(self) -> None:
        """加载索引文件"""
        logger.info("加载知识图谱索引")
        
        try:
            with open(os.path.join(self.index_dir, 'entity_chunks.json'), 'r', encoding='utf-8') as f:
                self.entity_chunks = json.load(f)
            logger.info("知识图谱索引加载完成")
        except FileNotFoundError:
            logger.warning("知识图谱索引不存在，将使用空索引")
            self.entity_chunks = {}
    
    def search(self, query: str, num: int = 100) -> List[int]:
        """根据查询检索相关文档
        
        Args:
            query: 查询文本
            num: 返回结果数量
            
        Returns:
            相关chunk ID列表
        """
        # 简单匹配：如果查询包含实体，返回相应的chunk
        matched_chunks = []
        
        for entity, chunks in self.entity_chunks.items():
            # 简单字符串匹配，实际系统中应该使用更复杂的实体匹配算法
            if any(part in query for part in entity.split('-')):
                matched_chunks.extend(chunks)
        
        # 去重并限制返回数量
        matched_chunks = list(set(matched_chunks))[:num]
        
        return matched_chunks