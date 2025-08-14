"""
评估决策智能体 (EvaluatorAgent)

基于重排序结果进行质量评估和下一步决策
如用户要求：召回chunks → 重排序 → 取top-k → LLM评估决策
"""

import json
import time
from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent

class EvaluatorAgent(BaseAgent):
    """评估决策智能体 - 基于重排序结果的智能评估和决策"""
    
    def __init__(self, name: str = "EvaluatorAgent", searcher=None, **kwargs):
        super().__init__(name, **kwargs)
        self.searcher = searcher  # 需要访问reranker
        
        # 评估配置
        self.evaluation_config = {
            'top_k_for_evaluation': 5,  # 评估前K个chunks
            'quality_thresholds': {
                'high': 0.8,
                'medium': 0.5,
                'low': 0.3
            },
            'decision_rules': {
                'generate_answer': {'coverage': 4, 'depth': 4},
                'focused_search': {'coverage': 3, 'depth': 2},
                'diversified_search': {'coverage': 2, 'depth': 2}
            }
        }
        
        # 评估历史
        self.evaluation_history = []
    
    def execute(self, task: Dict) -> Dict:
        """执行评估决策任务
        
        Args:
            task: 包含查询和原始chunks的任务字典
            
        Returns:
            评估结果和决策字典
        """
        query = task.get('original_query', '')
        raw_chunks = task.get('raw_chunks', [])
        current_round = task.get('current_round', 1)
        max_rounds = task.get('max_rounds', 3)
        
        self.log_step("开始评估流程", f"输入{len(raw_chunks)}个chunks, 当前第{current_round}轮")
        
        if not raw_chunks:
            self.log_step("无检索结果", "返回默认决策")
            return self._create_empty_result(query, current_round, max_rounds)
        
        # 第1步：使用现有重排序系统对chunks排序
        rerank_start_time = time.time()
        reranked_chunks = self._rerank_chunks(query, raw_chunks)
        rerank_time = time.time() - rerank_start_time
        
        self.log_step("重排序完成", 
                     f"耗时={rerank_time:.2f}s, 最高分={reranked_chunks[0].get('rerank_score', 0):.3f}")
        
        # 第2步：选择top-k进行详细评估
        top_k = min(self.evaluation_config['top_k_for_evaluation'], len(reranked_chunks))
        top_chunks = reranked_chunks[:top_k]
        
        self.log_step("选择评估对象", f"取前{top_k}个chunks进行详细评估")
        
        # 第3步：LLM质量评估
        eval_start_time = time.time()
        evaluation_result = self._evaluate_top_chunks(query, top_chunks, current_round, max_rounds)
        eval_time = time.time() - eval_start_time
        
        self.log_step("LLM评估完成", 
                     f"耗时={eval_time:.2f}s, 整体质量={evaluation_result['overall_quality']:.2f}")
        
        # 第4步：基于评估结果做决策
        decision = self._make_decision(evaluation_result, current_round, max_rounds)
        
        self.log_step("决策完成", 
                     f"下一步={decision['next_action']}, 置信度={decision['confidence']:.2f}")
        
        # 第5步：记录评估历史
        self._record_evaluation(query, len(raw_chunks), top_k, evaluation_result, decision)
        
        return {
            'agent': self.name,
            'reranked_chunks': reranked_chunks,
            'top_chunks_used': top_chunks,
            'evaluation': evaluation_result,
            'decision': decision,
            'current_round': current_round,
            'performance': {
                'rerank_time': rerank_time,
                'eval_time': eval_time,
                'total_time': rerank_time + eval_time
            }
        }
    
    def _rerank_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """使用现有的重排序系统对chunks排序"""
        
        if not chunks:
            return []
        
        self.log_step("开始重排序", f"处理{len(chunks)}个chunks")
        
        # 提取chunk文本内容
        chunk_texts = [chunk['content'] for chunk in chunks]
        
        # 使用现有searcher的reranker进行重排序
        if hasattr(self.searcher, 'reranker') and self.searcher.reranker:
            try:
                # 构造query-document对
                query_doc_pairs = [[query, text] for text in chunk_texts]
                
                # 计算重排序分数
                rerank_scores = self.searcher.reranker.compute_scores(query_doc_pairs)
                
                # 结合原始chunks和重排序分数
                scored_chunks = []
                for i, chunk in enumerate(chunks):
                    scored_chunks.append({
                        **chunk,
                        'rerank_score': rerank_scores[i].item(),
                        'original_rank': i
                    })
                
                # 按重排序分数排序
                scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
                
                self.log_step("重排序成功", 
                             f"分数范围: {scored_chunks[-1]['rerank_score']:.3f} - {scored_chunks[0]['rerank_score']:.3f}")
                
                return scored_chunks
                
            except Exception as e:
                self.log_step("重排序失败", f"错误: {str(e)}, 使用原始顺序")
                
                # 降级处理：添加默认分数
                for i, chunk in enumerate(chunks):
                    chunk['rerank_score'] = 0.5 - i * 0.01  # 递减分数
                    chunk['original_rank'] = i
                
                return chunks
        else:
            self.log_step("重排序器不可用", "使用原始顺序")
            
            # 添加默认分数
            for i, chunk in enumerate(chunks):
                chunk['rerank_score'] = 0.5 - i * 0.01
                chunk['original_rank'] = i
            
            return chunks
    
    def _evaluate_top_chunks(self, query: str, top_chunks: List[Dict], 
                            current_round: int, max_rounds: int) -> Dict[str, Any]:
        """使用LLM评估top-k chunks的整体质量"""
        
        # 构造评估用的文本
        chunks_text = self._format_chunks_for_evaluation(top_chunks)
        
        evaluation_prompt = f"""
        你是一个信息质量评估专家。请评估以下经过重排序的top-{len(top_chunks)}文档片段是否足够回答用户查询。
        
        **用户查询**: {query}
        **当前检索轮次**: {current_round}/{max_rounds}
        
        **重排序后的文档片段**:
        {chunks_text}
        
        请从以下4个维度评估（1-5分）：
        1. **信息覆盖度**: 是否覆盖了查询的主要方面
        2. **信息深度**: 信息是否足够详细深入
        3. **信息一致性**: 不同片段间是否一致，无矛盾
        4. **信息新颖性**: 相比预期是否有足够信息量
        
        **决策标准**:
        - 如果覆盖度≥4且深度≥4: 建议"生成答案"
        - 如果覆盖度≥3但深度<4: 建议"深入检索" 
        - 如果覆盖度<3: 建议"扩展检索"
        - 如果已达最大轮次: 必须建议"生成答案"
        
        请严格按照JSON格式输出：
        {{
            "coverage_score": 分数(1-5),
            "depth_score": 分数(1-5),
            "consistency_score": 分数(1-5),
            "novelty_score": 分数(1-5),
            "overall_quality": 平均分数,
            "recommended_action": "生成答案/深入检索/扩展检索",
            "reasoning": "详细推理过程",
            "key_gaps": ["缺失的关键信息点1", "缺失点2"]
        }}
        """
        
        if self.llm:
            try:
                response = self.llm(evaluation_prompt).content
                evaluation_result = self._parse_evaluation_response(response)
                self.log_step("LLM评估成功", f"覆盖度={evaluation_result['coverage_score']}, 深度={evaluation_result['depth_score']}")
                return evaluation_result
            except Exception as e:
                self.log_step("LLM评估失败", f"错误: {str(e)}, 使用启发式评估")
        
        # 降级到启发式评估
        return self._heuristic_evaluation(top_chunks, current_round, max_rounds)
    
    def _format_chunks_for_evaluation(self, chunks: List[Dict]) -> str:
        """格式化chunks用于LLM评估"""
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            score = chunk.get('rerank_score', 0)
            content = chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content']
            
            formatted_chunks.append(
                f"\n--- Chunk {i+1} (重排序分数: {score:.3f}) ---\n{content}"
            )
        
        return "\n".join(formatted_chunks)
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """解析LLM评估响应"""
        try:
            # 提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # 验证必要字段
                required_fields = ['coverage_score', 'depth_score', 'consistency_score', 
                                 'novelty_score', 'recommended_action']
                
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"缺少必要字段: {field}")
                
                # 计算overall_quality
                if 'overall_quality' not in result:
                    result['overall_quality'] = (
                        result['coverage_score'] + result['depth_score'] + 
                        result['consistency_score'] + result['novelty_score']
                    ) / 4.0
                
                return result
            else:
                raise ValueError("未找到JSON格式")
                
        except Exception as e:
            self.log_step("JSON解析失败", str(e))
            raise e
    
    def _heuristic_evaluation(self, top_chunks: List[Dict], 
                             current_round: int, max_rounds: int) -> Dict[str, Any]:
        """启发式评估，作为LLM评估的降级方案"""
        
        # 基于重排序分数的简单启发式
        if not top_chunks:
            return self._create_minimal_evaluation()
        
        avg_rerank_score = sum(chunk.get('rerank_score', 0.5) for chunk in top_chunks) / len(top_chunks)
        total_content_length = sum(len(chunk['content']) for chunk in top_chunks)
        
        # 计算各项分数
        coverage_score = min(5, max(1, int(avg_rerank_score * 5) + 1))
        depth_score = min(5, max(1, int(total_content_length / 500) + 2))  # 基于内容长度
        consistency_score = 4  # 假设重排序后的内容一致性较好
        novelty_score = max(1, 6 - current_round)  # 轮次越高，新颖性越低
        
        overall_quality = (coverage_score + depth_score + consistency_score + novelty_score) / 4
        
        # 决策逻辑
        if overall_quality >= 4.0 or current_round >= max_rounds:
            action = "生成答案"
        elif coverage_score >= 3:
            action = "深入检索"
        else:
            action = "扩展检索"
        
        return {
            "coverage_score": coverage_score,
            "depth_score": depth_score,
            "consistency_score": consistency_score,
            "novelty_score": novelty_score,
            "overall_quality": overall_quality,
            "recommended_action": action,
            "reasoning": f"启发式评估: 重排序分数{avg_rerank_score:.3f}, 内容长度{total_content_length}",
            "key_gaps": []
        }
    
    def _make_decision(self, evaluation: Dict, current_round: int, max_rounds: int) -> Dict[str, Any]:
        """基于评估结果做最终决策"""
        
        action = evaluation['recommended_action']
        reasoning = evaluation['reasoning']
        
        # 强制规则：达到最大轮次必须生成答案
        if current_round >= max_rounds and action != "生成答案":
            action = "生成答案"
            reasoning = f"已达最大轮次({max_rounds})，强制生成答案。" + reasoning
        
        # 计算决策置信度
        confidence = self._calculate_decision_confidence(evaluation, current_round, max_rounds)
        
        # 建议下一轮策略
        next_strategy = self._suggest_next_strategy(action, evaluation)
        
        return {
            'next_action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'quality_scores': {
                'coverage': evaluation['coverage_score'],
                'depth': evaluation['depth_score'],
                'consistency': evaluation['consistency_score'],
                'novelty': evaluation['novelty_score'],
                'overall': evaluation['overall_quality']
            },
            'suggested_strategy': next_strategy,
            'key_gaps': evaluation.get('key_gaps', [])
        }
    
    def _calculate_decision_confidence(self, evaluation: Dict, current_round: int, max_rounds: int) -> float:
        """计算决策置信度"""
        base_confidence = evaluation['overall_quality'] / 5.0
        
        # 根据轮次调整置信度
        round_factor = 1.0 + (current_round - 1) * 0.1  # 后续轮次提高置信度
        
        # 根据分数一致性调整
        scores = [evaluation['coverage_score'], evaluation['depth_score'], 
                 evaluation['consistency_score'], evaluation['novelty_score']]
        score_variance = sum((s - evaluation['overall_quality'])**2 for s in scores) / len(scores)
        consistency_factor = max(0.8, 1.0 - score_variance * 0.1)
        
        final_confidence = base_confidence * round_factor * consistency_factor
        
        return max(0.3, min(1.0, final_confidence))
    
    def _suggest_next_strategy(self, action: str, evaluation: Dict) -> str:
        """为下一轮检索建议策略"""
        
        if action == "深入检索":
            return "focused_search"  # 基于当前top chunks进行更深入的相关搜索
        elif action == "扩展检索":
            return "diversified_search"  # 扩展查询角度，寻找不同方面的信息
        else:
            return "none"  # 生成答案，不需要下一轮检索
    
    def _create_empty_result(self, query: str, current_round: int, max_rounds: int) -> Dict:
        """创建空结果的默认返回"""
        return {
            'agent': self.name,
            'reranked_chunks': [],
            'top_chunks_used': [],
            'evaluation': {
                'coverage_score': 1,
                'depth_score': 1,
                'consistency_score': 1,
                'novelty_score': 1,
                'overall_quality': 1.0,
                'recommended_action': "扩展检索" if current_round < max_rounds else "生成答案",
                'reasoning': "无检索结果，建议扩展检索范围",
                'key_gaps': ["缺少相关信息"]
            },
            'decision': {
                'next_action': "扩展检索" if current_round < max_rounds else "生成答案",
                'confidence': 0.3,
                'reasoning': "无检索结果，建议扩展检索范围",
                'quality_scores': {'coverage': 1, 'depth': 1, 'consistency': 1, 'novelty': 1, 'overall': 1.0},
                'suggested_strategy': "diversified_search",
                'key_gaps': ["缺少相关信息"]
            },
            'current_round': current_round
        }
    
    def _create_minimal_evaluation(self) -> Dict[str, Any]:
        """创建最小评估结果"""
        return {
            'coverage_score': 2,
            'depth_score': 2,
            'consistency_score': 3,
            'novelty_score': 2,
            'overall_quality': 2.25,
            'recommended_action': "扩展检索",
            'reasoning': "启发式评估: 信息不足",
            'key_gaps': ["需要更多相关信息"]
        }
    
    def _record_evaluation(self, query: str, chunks_count: int, top_k: int, 
                          evaluation: Dict, decision: Dict):
        """记录评估历史"""
        record = {
            'timestamp': time.time(),
            'query': query,
            'chunks_count': chunks_count,
            'top_k_evaluated': top_k,
            'overall_quality': evaluation['overall_quality'],
            'decision': decision['next_action'],
            'confidence': decision['confidence']
        }
        
        self.evaluation_history.append(record)
        
        # 限制历史记录大小
        if len(self.evaluation_history) > 100:
            self.evaluation_history = self.evaluation_history[-100:]
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        if not self.evaluation_history:
            return {'message': '暂无评估历史'}
        
        # 决策分布统计
        decision_count = {}
        quality_scores = []
        confidences = []
        
        for record in self.evaluation_history:
            decision = record['decision']
            decision_count[decision] = decision_count.get(decision, 0) + 1
            quality_scores.append(record['overall_quality'])
            confidences.append(record['confidence'])
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'decision_distribution': decision_count,
            'avg_quality_score': sum(quality_scores) / len(quality_scores),
            'avg_confidence': sum(confidences) / len(confidences),
            'most_common_decision': max(decision_count.keys(), key=decision_count.get)
        }