import yaml
import logging
from typing import List, Tuple, Optional, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class QueryExpander:
    """查询扩展器，用于生成和选择子查询"""
    
    def __init__(self, llm=None, reranker=None):
        """初始化查询扩展器
        
        Args:
            llm: 大语言模型接口
            reranker: 重排序模型
        """
        self.llm = llm
        self.reranker = reranker
    
    def extend_query(
        self, 
        queries: List[str], 
        responses: List[str], 
        num: int = 10
    ) -> List[str]:
        """扩展查询集合
        
        Args:
            queries: 原始查询列表
            responses: 对应查询的回答列表
            num: 返回的扩展查询数量
            
        Returns:
            扩展后的查询列表
        """
        logger.info(f"开始扩展查询，原始查询数: {len(queries)}")
        
        all_queries_scores = []
        
        for query, response in zip(queries, responses):
            if response is None:
                continue
                
            # 生成子查询及其得分
            queries_scores = self.generate_subquery(query, response, num)
            all_queries_scores.extend(queries_scores)
        
        # 按得分降序排序
        all_queries_scores = sorted(all_queries_scores, key=lambda s: s[2], reverse=True)
        
        # 选择前num个进行合并
        next_queries = [
            self.combine_query(s[0], s[1]) 
            for s in all_queries_scores[:num]
        ]
        
        logger.info(f"查询扩展完成，新生成查询数: {len(next_queries)}")
        return next_queries
    
    def generate_subquery(
        self, 
        query: str, 
        response: str, 
        num: int
    ) -> List[Tuple[str, str, float]]:
        """根据原始查询和回答生成子查询
        
        Args:
            query: 原始查询
            response: 查询的回答
            num: 生成的子查询数量
            
        Returns:
            元组列表 [(原始查询, 子查询, 得分)]
        """
        # 使用大模型提取关键搜索词
        prompt = f"{response}\n根据上述内容，提取{2*num}个核心搜索词，进行下一步搜索，搜索词之间用|隔开，不要输出不相关内容"
        
        try:
            result = self.llm(prompt, temperature=1.9).content
            extend_query = result.split("|")
            extend_query = [s.strip() for s in extend_query if len(s.strip()) > 1]
            
            # 计算扩展查询与原始查询的相关性得分
            pairs = [[query, s] for s in extend_query]
            if self.reranker:
                scores = self.reranker.compute_scores(pairs)
                scores = [s.item() for s in scores]
            else:
                # 如果没有重排序器，使用长度作为简单分数
                scores = [len(s)/100 for s in extend_query]
            
            return list(zip([query] * len(extend_query), extend_query, scores))
        except Exception as e:
            logger.error(f"生成子查询失败: {e}")
            return []
    
    def combine_query(self, query1: str, query2: str) -> str:
        """合并两个查询
        
        Args:
            query1: 第一个查询
            query2: 第二个查询
            
        Returns:
            合并后的查询
        """
        try:
            prompt = f"将搜索词'{query1}'和搜索词'{query2}'进行合并改写成一个搜索词,只输出搜索词，不要输出无关内容"
            result = self.llm(prompt).content
            return result.strip()
        except Exception as e:
            logger.error(f"合并查询失败: {e}")
            return f"{query1} {query2}"