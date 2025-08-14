"""
检索执行智能体 (RetrievalAgent)

负责执行各种检索策略，与现有的CombinedSearcher深度集成
"""

import time
from typing import Dict, List, Any, Tuple, Optional
from .base_agent import BaseAgent

class RetrievalAgent(BaseAgent):
    """检索执行智能体 - 负责执行各种检索策略"""
    
    def __init__(self, name: str = "RetrievalAgent", searcher=None, **kwargs):
        super().__init__(name, **kwargs)
        self.searcher = searcher
        
        # 检索策略配置
        self.strategy_configs = {
            'vector': {'use_vector': True, 'use_tfidf': False, 'use_kg': False},
            'keyword': {'use_vector': False, 'use_tfidf': True, 'use_kg': False},
            'hybrid': {'use_vector': True, 'use_tfidf': True, 'use_kg': False},
            'kg_enhanced': {'use_vector': True, 'use_tfidf': True, 'use_kg': True},
            'multi_modal': {'use_vector': True, 'use_tfidf': True, 'use_kg': True}
        }
        
        # 记录检索历史
        self.retrieval_history = []
    
    def execute(self, task: Dict) -> Dict:
        """执行检索任务
        
        Args:
            task: 检索任务字典，包含查询、策略等信息
            
        Returns:
            检索结果字典
        """
        # 提取任务信息
        query = task.get('current_query', task.get('original_query', ''))
        plan = task.get('plan', {})
        strategy = task.get('retrieval_strategy', plan.get('retrieval_strategy', 'hybrid'))
        top_k = task.get('top_k', plan.get('info_sources_needed', 5))
        current_round = task.get('current_round', 1)
        
        self.log_step("开始检索准备", f"查询={query[:50]}..., 策略={strategy}")
        
        if not self.searcher:
            raise ValueError("检索器未初始化")
        
        # 第1步：策略验证和调整
        actual_strategy = self._validate_and_adjust_strategy(strategy, current_round)
        if actual_strategy != strategy:
            self.log_step("策略调整", f"从{strategy}调整为{actual_strategy}")
        
        # 第2步：执行检索
        search_start_time = time.time()
        content, ids, raw_chunks = self._execute_search(query, actual_strategy, top_k)
        search_time = time.time() - search_start_time
        
        self.log_step("检索执行完成", 
                     f"耗时={search_time:.2f}s, 获得{len(raw_chunks)}个chunks")
        
        # 第3步：检索结果后处理
        processed_chunks = self._post_process_chunks(raw_chunks, query)
        self.log_step("结果后处理完成", f"处理了{len(processed_chunks)}个chunks")
        
        # 第4步：记录检索历史
        self._record_retrieval(query, actual_strategy, len(raw_chunks), search_time)
        
        # 第5步：生成检索报告
        retrieval_report = self._generate_retrieval_report(
            query, actual_strategy, raw_chunks, search_time
        )
        
        return {
            'agent': self.name,
            'retrieved_content': content,
            'source_ids': ids,
            'raw_chunks': processed_chunks,
            'strategy_used': actual_strategy,
            'query_processed': query,
            'retrieval_time': search_time,
            'chunks_count': len(raw_chunks),
            'retrieval_report': retrieval_report
        }
    
    def _validate_and_adjust_strategy(self, strategy: str, current_round: int) -> str:
        """验证和调整检索策略"""
        # 检查策略是否有效
        if strategy not in self.strategy_configs:
            self.log_step("策略无效", f"未知策略{strategy}, 使用hybrid替代")
            return 'hybrid'
        
        # 根据轮次调整策略
        if current_round > 1:
            # 后续轮次可能需要不同策略
            if strategy == 'vector':
                return 'hybrid'  # 后续轮次使用混合检索
            elif strategy == 'keyword':
                return 'vector'   # 切换到语义检索
        
        # 检查searcher能力
        if not self._check_searcher_capability(strategy):
            self.log_step("searcher能力不足", f"降级到hybrid策略")
            return 'hybrid'
        
        return strategy
    
    def _check_searcher_capability(self, strategy: str) -> bool:
        """检查searcher是否支持指定策略"""
        config = self.strategy_configs[strategy]
        
        # 检查向量检索能力
        if config['use_vector'] and not self.searcher.vector_retriever:
            return False
        
        # 检查TF-IDF检索能力  
        if config['use_tfidf'] and not self.searcher.tfidf_retriever:
            return False
        
        # 检查知识图谱检索能力
        if config['use_kg'] and not self.searcher.kg_retriever:
            return False
        
        return True
    
    def _execute_search(self, query: str, strategy: str, top_k: int) -> Tuple[str, List[int], List[Dict]]:
        """执行具体的搜索操作"""
        self.log_step("执行搜索", f"策略={strategy}, top_k={top_k}")
        
        # 获取策略配置
        config = self.strategy_configs[strategy]
        
        # 调用现有的searcher
        content, ids = self.searcher.search(
            query=query,
            top_k=top_k,
            **config
        )
        
        # 构造raw_chunks（从searcher的结果重建）
        raw_chunks = self._build_raw_chunks(content, ids)
        
        return content, ids, raw_chunks
    
    def _build_raw_chunks(self, content: str, ids: List[int]) -> List[Dict]:
        """从搜索结果构建原始chunks列表"""
        raw_chunks = []
        
        # 解析content字符串，提取各个知识点
        knowledge_parts = content.split('\n\n')
        
        for i, part in enumerate(knowledge_parts):
            if i < len(ids):
                # 提取知识点内容（去掉"知识点X:"前缀）
                if part.startswith('知识点'):
                    lines = part.split('\n')
                    if len(lines) > 1:
                        chunk_content = '\n'.join(lines[1:])
                    else:
                        chunk_content = part
                else:
                    chunk_content = part
                
                # 从searcher获取原始chunk信息（如果可用）
                chunk_info = self._get_chunk_info(ids[i])
                
                raw_chunks.append({
                    'id': ids[i],
                    'content': chunk_content.strip(),
                    'source': f"chunk_{ids[i]}",
                    'original_rank': i,
                    'rerank_score': None,  # 将在evaluator中计算
                    'metadata': chunk_info
                })
        
        return raw_chunks
    
    def _get_chunk_info(self, chunk_id: int) -> Dict[str, Any]:
        """获取chunk的元信息"""
        if hasattr(self.searcher, 'chunks') and self.searcher.chunks:
            chunk = self.searcher.chunks.get(chunk_id)
            if chunk:
                return {
                    'title': getattr(chunk, 'title', ''),
                    'abstract': getattr(chunk, 'abstract', ''),
                    'source_file': getattr(chunk, 'source', ''),
                    'length': len(chunk.content) if hasattr(chunk, 'content') else 0
                }
        
        return {
            'title': '',
            'abstract': '',
            'source_file': '',
            'length': 0
        }
    
    def _post_process_chunks(self, raw_chunks: List[Dict], query: str) -> List[Dict]:
        """对检索结果进行后处理"""
        self.log_step("开始后处理", f"处理{len(raw_chunks)}个chunks")
        
        processed_chunks = []
        
        for chunk in raw_chunks:
            # 内容清理
            cleaned_content = self._clean_chunk_content(chunk['content'])
            
            # 计算基础统计
            word_count = len(cleaned_content.split())
            
            # 添加处理后的信息
            processed_chunk = {
                **chunk,
                'content': cleaned_content,
                'word_count': word_count,
                'relevance_preview': cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content,
                'processed': True
            }
            
            processed_chunks.append(processed_chunk)
        
        self.log_step("后处理完成", f"平均长度={sum(c['word_count'] for c in processed_chunks)/len(processed_chunks):.1f}词")
        
        return processed_chunks
    
    def _clean_chunk_content(self, content: str) -> str:
        """清理chunk内容"""
        # 去除多余空白
        content = ' '.join(content.split())
        
        # 去除特殊字符（保留基本标点）
        import re
        content = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()""''—-]', '', content)
        
        return content.strip()
    
    def _record_retrieval(self, query: str, strategy: str, chunks_count: int, search_time: float):
        """记录检索历史"""
        record = {
            'timestamp': time.time(),
            'query': query,
            'strategy': strategy,
            'chunks_count': chunks_count,
            'search_time': search_time
        }
        
        self.retrieval_history.append(record)
        
        # 限制历史记录大小
        if len(self.retrieval_history) > 100:
            self.retrieval_history = self.retrieval_history[-100:]
    
    def _generate_retrieval_report(self, query: str, strategy: str, 
                                  raw_chunks: List[Dict], search_time: float) -> Dict[str, Any]:
        """生成检索报告"""
        if not raw_chunks:
            return {
                'status': 'empty',
                'message': '未检索到相关内容',
                'suggestions': ['尝试修改查询词', '使用不同的检索策略']
            }
        
        # 计算统计信息
        total_content_length = sum(len(chunk['content']) for chunk in raw_chunks)
        avg_content_length = total_content_length / len(raw_chunks)
        
        # 内容质量初步评估
        quality_indicators = {
            'has_long_content': avg_content_length > 100,
            'diverse_sources': len(set(chunk['source'] for chunk in raw_chunks)) > 1,
            'reasonable_count': 3 <= len(raw_chunks) <= 10
        }
        
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        
        return {
            'status': 'success',
            'strategy_used': strategy,
            'chunks_retrieved': len(raw_chunks),
            'total_content_length': total_content_length,
            'avg_content_length': avg_content_length,
            'search_time': search_time,
            'quality_score': quality_score,
            'quality_indicators': quality_indicators,
            'retrieval_efficiency': len(raw_chunks) / max(search_time, 0.1),  # chunks per second
            'content_preview': raw_chunks[0]['content'][:150] + "..." if raw_chunks else ""
        }
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        if not self.retrieval_history:
            return {'message': '暂无检索历史'}
        
        # 策略使用统计
        strategy_count = {}
        total_time = 0
        total_chunks = 0
        
        for record in self.retrieval_history:
            strategy = record['strategy']
            strategy_count[strategy] = strategy_count.get(strategy, 0) + 1
            total_time += record['search_time']
            total_chunks += record['chunks_count']
        
        return {
            'total_retrievals': len(self.retrieval_history),
            'strategy_distribution': strategy_count,
            'avg_search_time': total_time / len(self.retrieval_history),
            'avg_chunks_per_search': total_chunks / len(self.retrieval_history),
            'most_used_strategy': max(strategy_count.keys(), key=strategy_count.get)
        }
    
    def optimize_next_strategy(self, current_results: Dict, evaluation_feedback: Dict) -> str:
        """基于当前结果和评估反馈优化下一轮策略"""
        current_strategy = current_results.get('strategy_used', 'hybrid')
        quality_score = evaluation_feedback.get('quality_score', 0.5)
        
        # 如果当前结果质量很高，保持策略
        if quality_score > 0.8:
            return current_strategy
        
        # 如果质量不高，尝试切换策略
        strategy_alternatives = {
            'vector': 'hybrid',
            'keyword': 'vector', 
            'hybrid': 'kg_enhanced'
        }
        
        return strategy_alternatives.get(current_strategy, 'hybrid')