import yaml
import logging
from typing import List, Dict, Tuple, Optional, Any, Set
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class DeepRAG:
    """深度RAG实现类"""
    
    def __init__(
        self, 
        searcher=None, 
        llm=None, 
        query_expander=None,
        max_iterations: Optional[int] = None,
        growth_rate_threshold: Optional[float] = None,
        extend_query_num: Optional[int] = None
    ):
        """初始化深度RAG
        
        Args:
            searcher: 搜索器
            llm: 大语言模型
            query_expander: 查询扩展器
            max_iterations: 最大迭代次数
            growth_rate_threshold: 信息增长率阈值
            extend_query_num: 扩展查询数量
        """
        self.searcher = searcher
        self.llm = llm
        self.query_expander = query_expander
        
        # 使用配置文件的默认值
        self.max_iterations = max_iterations or config['deepsearch']['max_iterations']
        self.growth_rate_threshold = growth_rate_threshold or config['deepsearch']['growth_rate_threshold']
        self.extend_query_num = extend_query_num or config['deepsearch']['extend_query_num']
    
    def answer(self, query: str) -> str:
        """深度RAG问答流程
        
        Args:
            query: 用户查询
            
        Returns:
            最终回答
        """
        # 记录开始时间
        start_time = time.time()
        logger.info(f"开始处理查询: {query}")
        
        # 初始化
        sub_queries = [query]  # 初始子查询就是原始查询
        knowledge = []  # 存储收集到的知识
        exist_ids = set()  # 已检索到的chunk ID集合
        
        # 迭代深度搜索
        for i in range(self.max_iterations):
            logger.debug(f"第 {i+1} 轮迭代开始，当前子查询列表: {sub_queries}")
            
            new_ids = set()
            response_list = []
            
            # 对每个子查询执行标准RAG
            for sub_query in sub_queries:
                try:
                    logger.debug(f"处理子查询: {sub_query}")
                    response, ids = self._standard_rag(sub_query)
                    logger.debug(f"子查询处理完成，获取回答长度: {len(response)}")
                    
                    # 添加到知识库
                    knowledge.append(response)
                    
                    # 更新新发现的chunk IDs
                    new_ids.update([s for s in ids if s not in exist_ids])
                    
                    # 记录回答
                    response_list.append(response)
                except Exception as e:
                    logger.error(f"处理子查询失败: {e}")
                    response_list.append(None)
            
            # 计算信息增长率
            info_growth_rate = len(new_ids) / max(len(exist_ids), 1)
            logger.info(f"第 {i+1} 轮信息增长率: {info_growth_rate:.4f}")
            
            # 更新已发现的chunk ID集合
            exist_ids.update(new_ids)
            
            # 如果信息增长率低于阈值，结束迭代
            if info_growth_rate < self.growth_rate_threshold:
                logger.info(f"信息增长率低于阈值 {self.growth_rate_threshold}，结束深度搜索")
                break
            
            # 如果不是最后一轮，扩展查询
            if i < self.max_iterations - 1:
                logger.info(f"第 {i+1} 轮扩展前查询: {sub_queries}")
                # 使用查询扩展器生成新的子查询
                sub_queries = self.query_expander.extend_query(
                    sub_queries, 
                    response_list,
                    self.extend_query_num
                )
                logger.info(f"第 {i+1} 轮扩展后查询: {sub_queries}")
        
        # 格式化收集到的知识
        # 修改格式化知识的方式
        formatted_knowledge = []
        for i, s in enumerate(knowledge):
            # 使用searcher中的reranker计算相关性
            score = self.searcher.reranker.compute_scores([[query, s]])[0].item()
            formatted_knowledge.append(
                f"\n\n## 知识点 {i+1}\n"
                f"**来源查询**: {sub_queries[i] if i < len(sub_queries) else '综合生成'}\n"
                f"**详细内容**:\n{s}\n"
                f"**相关度评分**: {score:.2f}/1.0"  # 使用reranker的评分
            )
        knowledge_text = "".join(formatted_knowledge)
        
        # 最终提示词
        prompt = (
            f"你是一位专业的研究助理，请根据以下收集到的所有知识，撰写一份详细的报告来回答原始问题。\n"
            f"\n## 原始问题：\n{query}\n"
            f"\n## 收集到的知识：{knowledge_text}\n"
            f"\n## 报告撰写要求\n"
            f"1. 首先用1-2句话总结核心答案\n"
            f"2. 然后详细阐述，每个点都要引用具体参考文献，你需要在回答完成后，再输出你参考文献的原文\n"
            f"4. 最后给出综合分析和个人见解\n"
            f"\n请开始撰写详细报告："
        )
        final_response = self.llm(prompt).content
        
        # 记录结束时间
        end_time = time.time()
        logger.info(f"查询处理完成，用时: {end_time - start_time:.2f}秒")
        
        return final_response
    
    def _standard_rag(self, query: str) -> Tuple[str, List[int]]:
        """执行标准的RAG过程
        
        Args:
            query: 查询字符串
            
        Returns:
            (回答内容, chunk ID列表)
        """
        try:
            # 搜索相关文档
            logger.debug(f"开始搜索查询: {query}")
            content, ids = self.searcher.search(query)
            logger.debug(f"搜索完成，获取到内容类型: {type(content)}，内容长度: {len(content) if hasattr(content, '__len__') else 'N/A'}")
            logger.debug(f"获取到IDs: {ids}")
        
            # 确保content是字符串
            if not isinstance(content, str):
                logger.debug(f"内容非字符串类型，执行转换前内容示例: {str(content)[:200]}...")
                content = str(content)
                logger.debug(f"转换后内容类型: {type(content)}")
        
            # 构建提示词
            prompt = f"{content}\n\n根据上述内容回答问题：{query}，尽可能全面"
            logger.debug(f"构建的提示词长度: {len(prompt)}")
        
            # 使用大模型生成回答
            logger.debug("开始调用LLM生成回答")
            response = self.llm(prompt).content
            logger.debug("LLM回答生成完成")
        
            return response, ids
        except Exception as e:
            logger.error(f"标准RAG处理失败: {e}", exc_info=True)
            raise
