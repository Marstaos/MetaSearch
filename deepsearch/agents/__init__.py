"""
MetaSearch Agent RAG 智能体系统

这个包含了基于多智能体协作的检索增强生成(RAG)系统。
主要组件包括：
- BaseAgent: 基础智能体类
- QueryPlannerAgent: 查询规划智能体
- RetrievalAgent: 检索执行智能体  
- EvaluatorAgent: 评估决策智能体
- GeneratorAgent: 答案生成智能体
- AgentCoordinator: 智能体协调器
- AgentLogger: 统一日志系统
"""

# 基础组件
from .base_agent import BaseAgent, AgentRegistry
from .logger import AgentLogger, get_agent_logger

# 专门Agent
from .planner_agent import QueryPlannerAgent
from .retrieval_agent import RetrievalAgent
from .evaluator_agent import EvaluatorAgent
from .generator_agent import GeneratorAgent

# 协调器
from .coordinator import AgentCoordinator

# 导出所有主要类
__all__ = [
    'BaseAgent',
    'AgentRegistry', 
    'AgentLogger',
    'get_agent_logger',
    'QueryPlannerAgent',
    'RetrievalAgent',
    'EvaluatorAgent',
    'GeneratorAgent',
    'AgentCoordinator'
]

__version__ = "1.0.0"
__author__ = "MetaSearch Team"