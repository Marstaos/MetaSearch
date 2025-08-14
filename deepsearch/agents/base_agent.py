"""
Agent RAG 基础智能体类

定义了所有Agent的基础接口和通用功能
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from .logger import get_agent_logger

class BaseAgent(ABC):
    """所有Agent的基础类"""
    
    def __init__(self, name: str, llm=None, tools: List = None, **kwargs):
        """初始化基础Agent
        
        Args:
            name: Agent名称
            llm: 大语言模型接口
            tools: 可用工具列表
            **kwargs: 其他配置参数
        """
        self.name = name
        self.llm = llm
        self.tools = tools or []
        self.config = kwargs
        
        # 获取统一日志器
        self.logger = get_agent_logger()
        
        # Agent状态
        self.is_active = False
        self.memory = []  # Agent记忆
        self.execution_count = 0
        
        # 性能统计
        self.performance_stats = {
            'total_executions': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'last_execution_time': 0.0,
            'success_count': 0,
            'error_count': 0
        }
    
    @abstractmethod 
    def execute(self, task: Dict) -> Dict:
        """执行任务的核心方法 - 必须由子类实现
        
        Args:
            task: 任务描述字典
            
        Returns:
            执行结果字典
        """
        pass
    
    def run_with_monitoring(self, task: Dict, task_description: str = "") -> Dict:
        """带监控的执行方法
        
        Args:
            task: 任务描述字典
            task_description: 任务描述文本
            
        Returns:
            执行结果字典
        """
        # 记录开始执行
        start_time = time.time()
        self.is_active = True
        self.execution_count += 1
        
        # 日志记录
        desc = task_description or f"执行任务 #{self.execution_count}"
        self.logger.log_agent_start(self.name, desc)
        
        try:
            # 执行核心逻辑
            result = self.execute(task)
            
            # 记录成功
            execution_time = time.time() - start_time
            self.performance_stats['total_executions'] += 1
            self.performance_stats['total_execution_time'] += execution_time
            self.performance_stats['last_execution_time'] = execution_time
            self.performance_stats['average_execution_time'] = (
                self.performance_stats['total_execution_time'] / 
                self.performance_stats['total_executions']
            )
            self.performance_stats['success_count'] += 1
            
            # 添加性能信息到结果
            result['performance'] = {
                'execution_time': execution_time,
                'agent_name': self.name
            }
            
            # 记录执行结果
            self.logger.log_agent_result(self.name, result)
            
            # 添加到记忆
            self.add_to_memory({
                'timestamp': start_time,
                'task': task,
                'result': result,
                'execution_time': execution_time,
                'success': True
            })
            
            return result
            
        except Exception as e:
            # 记录错误
            execution_time = time.time() - start_time
            self.performance_stats['error_count'] += 1
            
            error_context = f"任务: {task.get('query', 'Unknown')}"
            self.logger.log_error(self.name, e, error_context)
            
            # 添加到记忆
            self.add_to_memory({
                'timestamp': start_time,
                'task': task,
                'error': str(e),
                'execution_time': execution_time,
                'success': False
            })
            
            # 返回错误结果
            return {
                'agent': self.name,
                'success': False,
                'error': str(e),
                'performance': {
                    'execution_time': execution_time,
                    'agent_name': self.name
                }
            }
            
        finally:
            self.is_active = False
    
    def add_to_memory(self, interaction: Dict):
        """添加交互记录到记忆中
        
        Args:
            interaction: 交互记录字典
        """
        self.memory.append(interaction)
        
        # 限制记忆大小，只保留最近50次交互
        if len(self.memory) > 50:
            self.memory = self.memory[-50:]
    
    def get_memory_summary(self) -> str:
        """获取记忆摘要
        
        Returns:
            记忆摘要文本
        """
        if not self.memory:
            return "无历史记录"
        
        successful = sum(1 for m in self.memory if m.get('success', False))
        failed = len(self.memory) - successful
        
        recent_tasks = [m.get('task', {}).get('query', 'Unknown')[:50] for m in self.memory[-3:]]
        
        return f"历史执行: {successful}成功, {failed}失败. 最近任务: {', '.join(recent_tasks)}"
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能总结
        
        Returns:
            性能统计字典
        """
        return {
            **self.performance_stats,
            'success_rate': (
                self.performance_stats['success_count'] / 
                max(self.performance_stats['total_executions'], 1)
            ) * 100,
            'memory_size': len(self.memory)
        }
    
    def reset_stats(self):
        """重置性能统计"""
        self.performance_stats = {
            'total_executions': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'last_execution_time': 0.0,
            'success_count': 0,
            'error_count': 0
        }
        self.memory.clear()
        self.execution_count = 0
    
    def log_step(self, step: str, details: str = ""):
        """记录执行步骤
        
        Args:
            step: 步骤描述
            details: 详细信息
        """
        self.logger.log_agent_step(self.name, step, details)
    
    def __str__(self):
        """字符串表示"""
        return f"{self.name}(executions={self.execution_count}, active={self.is_active})"
    
    def __repr__(self):
        """详细字符串表示"""
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"executions={self.execution_count}, "
                f"success_rate={self.get_performance_summary()['success_rate']:.1f}%)")


class AgentRegistry:
    """Agent注册表 - 管理所有Agent实例"""
    
    _agents = {}
    
    @classmethod
    def register(cls, agent: BaseAgent):
        """注册Agent"""
        cls._agents[agent.name] = agent
    
    @classmethod
    def get_agent(cls, name: str) -> Optional[BaseAgent]:
        """获取Agent"""
        return cls._agents.get(name)
    
    @classmethod
    def get_all_agents(cls) -> Dict[str, BaseAgent]:
        """获取所有Agent"""
        return cls._agents.copy()
    
    @classmethod
    def get_performance_report(cls) -> Dict[str, Any]:
        """获取所有Agent的性能报告"""
        report = {}
        for name, agent in cls._agents.items():
            report[name] = agent.get_performance_summary()
        return report
    
    @classmethod
    def reset_all_stats(cls):
        """重置所有Agent的统计信息"""
        for agent in cls._agents.values():
            agent.reset_stats()