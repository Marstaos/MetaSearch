"""
智能体协调器 (AgentCoordinator)

系统的核心大脑，负责协调所有Agent的工作流程
"""

import time
from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent, AgentRegistry
from .planner_agent import QueryPlannerAgent
from .retrieval_agent import RetrievalAgent
from .evaluator_agent import EvaluatorAgent
from .generator_agent import GeneratorAgent
from .logger import get_agent_logger

class AgentCoordinator:
    """智能体协调器 - 负责协调多个Agent完成复杂的RAG任务"""
    
    def __init__(self, searcher, llm, query_expander):
        """初始化协调器
        
        Args:
            searcher: 检索器实例
            llm: 大语言模型接口
            query_expander: 查询扩展器
        """
        self.searcher = searcher
        self.llm = llm
        self.query_expander = query_expander
        
        # 获取统一日志器
        self.logger = get_agent_logger()
        
        # 初始化所有Agent
        self.agents = self._initialize_agents()
        
        # 注册所有Agent
        for agent in self.agents.values():
            AgentRegistry.register(agent)
        
        # 协调器状态
        self.current_task = {}
        self.execution_history = []
        
        # 配置参数
        self.config = {
            'max_rounds': 4,           # 最大检索轮数
            'max_execution_time': 300,  # 最大执行时间（秒）
            'enable_query_expansion': True,  # 是否启用查询扩展
            'fallback_on_error': True,       # 错误时是否降级处理
        }
        
        self.logger.log_agent_start("AgentCoordinator", "智能体协调器初始化完成")
    
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """初始化所有Agent实例"""
        agents = {
            'planner': QueryPlannerAgent('QueryPlannerAgent', llm=self.llm),
            'retriever': RetrievalAgent('RetrievalAgent', searcher=self.searcher, llm=self.llm),
            'evaluator': EvaluatorAgent('EvaluatorAgent', searcher=self.searcher, llm=self.llm),
            'generator': GeneratorAgent('GeneratorAgent', llm=self.llm)
        }
        
        return agents
    
    def process_query(self, user_query: str) -> str:
        """处理用户查询的主要协调流程
        
        Args:
            user_query: 用户查询字符串
            
        Returns:
            最终答案字符串
        """
        # 记录查询开始
        query_id = self.logger.log_query_start(user_query)
        session_start_time = time.time()
        
        try:
            # 初始化任务状态
            self._initialize_task_state(user_query, query_id)
            
            # Phase 1: 查询规划
            planning_result = self._execute_planning_phase(user_query)
            
            # Phase 2: 多轮检索和评估循环
            retrieval_results = self._execute_retrieval_evaluation_loop(user_query, planning_result)
            
            # Phase 3: 答案生成
            final_answer = self._execute_generation_phase(user_query, planning_result, retrieval_results)
            
            # 记录成功完成
            total_time = time.time() - session_start_time
            self.logger.log_query_complete(query_id, total_time, final_answer[:200])
            
            # 记录执行历史
            self._record_execution_history(user_query, total_time, True, final_answer)
            
            return final_answer
            
        except Exception as e:
            # 错误处理
            total_time = time.time() - session_start_time
            self.logger.log_error("AgentCoordinator", e, f"查询处理失败: {user_query}")
            
            # 记录失败历史
            self._record_execution_history(user_query, total_time, False, str(e))
            
            if self.config['fallback_on_error']:
                return self._generate_error_fallback(user_query, str(e))
            else:
                raise e
    
    def _initialize_task_state(self, query: str, query_id: str):
        """初始化任务状态"""
        self.current_task = {
            'query_id': query_id,
            'original_query': query,
            'start_time': time.time(),
            'current_round': 0,
            'max_rounds': self.config['max_rounds'],
            'all_retrieved_content': [],
            'all_retrieval_results': [],
            'agent_interactions': [],
            'performance_metrics': {}
        }
    
    def _execute_planning_phase(self, user_query: str) -> Dict:
        """执行查询规划阶段"""
        self.logger.log_agent_step("AgentCoordinator", "开始查询规划阶段")
        
        # 执行查询规划Agent
        planning_task = {'query': user_query}
        planning_result = self.agents['planner'].run_with_monitoring(
            planning_task, 
            f"分析查询并制定检索策略"
        )
        
        if not planning_result.get('plan'):
            raise ValueError("查询规划失败，无法获得有效计划")
        
        # 更新任务状态
        self.current_task['plan'] = planning_result['plan']
        self.current_task['max_rounds'] = planning_result['plan'].get('max_rounds', self.config['max_rounds'])
        
        self.logger.log_agent_step("AgentCoordinator", "查询规划完成", 
                                  f"策略={planning_result['plan']['retrieval_strategy']}")
        
        return planning_result
    
    def _execute_retrieval_evaluation_loop(self, user_query: str, planning_result: Dict) -> List[Dict]:
        """执行多轮检索和评估循环"""
        self.logger.log_agent_step("AgentCoordinator", "开始检索评估循环")
        
        plan = planning_result['plan']
        current_queries = plan.get('sub_queries', [user_query])
        all_retrieval_results = []
        
        # 检索循环
        while (self.current_task['current_round'] < self.current_task['max_rounds'] and
               self._check_execution_time_limit()):
            
            round_num = self.current_task['current_round'] + 1
            self.current_task['current_round'] = round_num
            
            self.logger.log_agent_step("AgentCoordinator", f"开始第{round_num}轮检索")
            
            # Step 1: 执行检索
            retrieval_result = self._execute_retrieval_step(current_queries, round_num)
            
            if not retrieval_result.get('raw_chunks'):
                self.logger.log_agent_step("AgentCoordinator", f"第{round_num}轮检索无结果，尝试查询扩展")
                
                # 如果没有结果，尝试查询扩展
                if self.config['enable_query_expansion'] and round_num == 1:
                    expanded_queries = self._expand_initial_query(user_query)
                    if expanded_queries != current_queries:
                        current_queries = expanded_queries
                        continue  # 重试当前轮次
                
                # 如果还是没有结果，结束循环
                self.logger.log_agent_step("AgentCoordinator", f"第{round_num}轮无结果，结束检索")
                break
            
            # Step 2: 执行评估
            evaluation_result = self._execute_evaluation_step(retrieval_result, round_num)
            
            # Step 3: 记录本轮结果
            round_result = {
                'round': round_num,
                'queries_used': current_queries,
                'retrieval_result': retrieval_result,
                'evaluation_result': evaluation_result
            }
            all_retrieval_results.append(round_result)
            
            # 更新累积内容
            self.current_task['all_retrieved_content'].append(retrieval_result['retrieved_content'])
            
            # Step 4: 根据评估结果决定下一步
            decision = evaluation_result['decision']
            next_action = decision['next_action']
            
            self.logger.log_agent_step("AgentCoordinator", f"第{round_num}轮决策", 
                                      f"{next_action}, 置信度={decision['confidence']:.2f}")
            
            if next_action == "生成答案":
                self.logger.log_agent_step("AgentCoordinator", "评估通过，准备生成答案")
                break
            elif next_action == "深入检索":
                # 基于当前结果生成更聚焦的查询
                current_queries = self._generate_focused_queries(user_query, evaluation_result)
            elif next_action == "扩展检索":
                # 扩展查询范围
                current_queries = self._expand_queries(user_query, retrieval_result, round_num)
            else:
                self.logger.log_agent_step("AgentCoordinator", f"未知决策动作: {next_action}，结束检索")
                break
        
        self.logger.log_agent_step("AgentCoordinator", "检索评估循环完成", 
                                  f"共执行{len(all_retrieval_results)}轮")
        
        return all_retrieval_results
    
    def _execute_retrieval_step(self, queries: List[str], round_num: int) -> Dict:
        """执行检索步骤"""
        # 使用第一个查询进行检索（简化处理，实际可以合并多个查询）
        main_query = queries[0] if queries else self.current_task['original_query']
        
        retrieval_task = {
            **self.current_task,
            'current_query': main_query,
            'retrieval_strategy': self.current_task['plan']['retrieval_strategy'],
            'current_round': round_num
        }
        
        return self.agents['retriever'].run_with_monitoring(
            retrieval_task,
            f"第{round_num}轮检索: {main_query[:50]}..."
        )
    
    def _execute_evaluation_step(self, retrieval_result: Dict, round_num: int) -> Dict:
        """执行评估步骤"""
        evaluation_task = {
            **self.current_task,
            'raw_chunks': retrieval_result['raw_chunks'],
            'current_round': round_num
        }
        
        return self.agents['evaluator'].run_with_monitoring(
            evaluation_task,
            f"第{round_num}轮评估: 评估{len(retrieval_result['raw_chunks'])}个chunks"
        )
    
    def _execute_generation_phase(self, user_query: str, planning_result: Dict, 
                                 retrieval_results: List[Dict]) -> str:
        """执行答案生成阶段"""
        self.logger.log_agent_step("AgentCoordinator", "开始答案生成阶段")
        
        generation_task = {
            **self.current_task,
            'plan': planning_result['plan']
        }
        
        generation_result = self.agents['generator'].run_with_monitoring(
            generation_task,
            f"基于{len(self.current_task['all_retrieved_content'])}轮检索结果生成答案"
        )
        
        if not generation_result.get('final_answer'):
            raise ValueError("答案生成失败，无法获得有效答案")
        
        self.logger.log_agent_step("AgentCoordinator", "答案生成完成", 
                                  f"长度={len(generation_result['final_answer'])}")
        
        return generation_result['final_answer']
    
    def _generate_focused_queries(self, original_query: str, evaluation_result: Dict) -> List[str]:
        """基于评估结果生成聚焦查询"""
        try:
            top_chunks = evaluation_result.get('top_chunks_used', [])
            if not top_chunks:
                return [original_query]
            
            # 提取关键信息用于生成聚焦查询
            key_concepts = []
            for chunk in top_chunks[:3]:  # 使用前3个chunks
                content = chunk['content'][:200]  # 限制长度
                if content:
                    key_concepts.append(content)
            
            if self.llm and key_concepts:
                prompt = f"""
                基于以下关键信息，为原查询生成2-3个更聚焦的子查询：
                原查询：{original_query}
                关键信息：{' | '.join(key_concepts)}
                
                请生成更具体、更聚焦的查询，用|分隔：
                """
                
                response = self.llm(prompt).content.strip()
                focused_queries = [q.strip() for q in response.split('|') if q.strip()]
                
                if focused_queries:
                    return focused_queries[:3]  # 最多3个查询
            
            return [original_query]
            
        except Exception as e:
            self.logger.log_agent_step("AgentCoordinator", "聚焦查询生成失败", str(e))
            return [original_query]
    
    def _expand_queries(self, original_query: str, retrieval_result: Dict, round_num: int) -> List[str]:
        """扩展查询范围"""
        if not self.config['enable_query_expansion']:
            return [original_query]
        
        try:
            # 使用现有的查询扩展器
            current_content = retrieval_result.get('retrieved_content', '')
            expanded_queries = self.query_expander.extend_query(
                [original_query], 
                [current_content], 
                3
            )
            
            if expanded_queries and expanded_queries != [original_query]:
                self.logger.log_agent_step("AgentCoordinator", "查询扩展成功", 
                                          f"生成{len(expanded_queries)}个扩展查询")
                return expanded_queries
            
        except Exception as e:
            self.logger.log_agent_step("AgentCoordinator", "查询扩展失败", str(e))
        
        return [original_query]
    
    def _expand_initial_query(self, query: str) -> List[str]:
        """扩展初始查询（在没有检索结果时使用）"""
        try:
            if self.llm:
                prompt = f"""
                请为以下查询生成2-3个相关的搜索变体，以提高检索成功率：
                原查询：{query}
                
                请生成语义相关但表达不同的查询变体，用|分隔：
                """
                
                response = self.llm(prompt).content.strip()
                variants = [q.strip() for q in response.split('|') if q.strip()]
                
                if variants:
                    # 包含原查询
                    return [query] + variants[:2]
            
            return [query]
            
        except Exception as e:
            self.logger.log_agent_step("AgentCoordinator", "初始查询扩展失败", str(e))
            return [query]
    
    def _check_execution_time_limit(self) -> bool:
        """检查执行时间限制"""
        elapsed_time = time.time() - self.current_task['start_time']
        
        if elapsed_time > self.config['max_execution_time']:
            self.logger.log_agent_step("AgentCoordinator", "达到时间限制", 
                                      f"已执行{elapsed_time:.2f}秒")
            return False
        
        return True
    
    def _generate_error_fallback(self, query: str, error_msg: str) -> str:
        """生成错误降级回答"""
        return f"""
抱歉，在处理您的查询时遇到了技术问题。

查询: {query}
错误信息: {error_msg}

建议您：
1. 尝试简化或重新表述您的问题
2. 检查查询中是否包含特殊字符
3. 稍后再试

我们正在努力改进系统的稳定性，感谢您的理解。
"""
    
    def _record_execution_history(self, query: str, execution_time: float, 
                                 success: bool, result: str):
        """记录执行历史"""
        record = {
            'timestamp': time.time(),
            'query': query,
            'execution_time': execution_time,
            'success': success,
            'rounds_executed': self.current_task.get('current_round', 0),
            'result_length': len(result) if success else 0,
            'error_message': result if not success else None
        }
        
        self.execution_history.append(record)
        
        # 限制历史记录大小
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态报告"""
        # 获取所有Agent的性能报告
        agent_performance = AgentRegistry.get_performance_report()
        
        # 执行历史统计
        if self.execution_history:
            successful_executions = [r for r in self.execution_history if r['success']]
            success_rate = len(successful_executions) / len(self.execution_history)
            avg_execution_time = sum(r['execution_time'] for r in successful_executions) / max(len(successful_executions), 1)
        else:
            success_rate = 0.0
            avg_execution_time = 0.0
        
        return {
            'coordinator_status': {
                'total_queries_processed': len(self.execution_history),
                'success_rate': success_rate * 100,
                'average_execution_time': avg_execution_time,
                'current_task_active': bool(self.current_task)
            },
            'agent_performance': agent_performance,
            'configuration': self.config,
            'log_file': self.logger.get_log_path()
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(new_config)
        self.logger.log_agent_step("AgentCoordinator", "配置已更新", str(new_config))
    
    def reset_system(self):
        """重置系统状态"""
        # 重置所有Agent统计
        AgentRegistry.reset_all_stats()
        
        # 清空执行历史
        self.execution_history.clear()
        self.current_task.clear()
        
        # 记录日志总结
        self.logger.log_session_summary()
        
        self.logger.log_agent_step("AgentCoordinator", "系统已重置")
    
    def __del__(self):
        """析构函数 - 记录会话总结"""
        try:
            if hasattr(self, 'logger'):
                self.logger.log_session_summary()
        except:
            pass  # 忽略析构时的错误