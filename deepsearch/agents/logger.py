"""
Agent RAG 统一日志系统

所有Agent的日志都写入同一个文件，便于统一查看和分析
"""

import os
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class AgentLogger:
    """Agent RAG专用日志器 - 统一日志文件"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """单例模式，确保所有Agent使用同一个日志器"""
        if cls._instance is None:
            cls._instance = super(AgentLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化日志器（只执行一次）"""
        if AgentLogger._initialized:
            return
            
        # 创建logs目录
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # 生成日志文件名：agent_rag_YYYYMMDD_HHMM.log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.log_filename = f"agent_rag_{timestamp}.log"
        self.log_path = self.log_dir / self.log_filename
        
        # 配置日志器
        self.logger = logging.getLogger("AgentRAG")
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的handlers
        self.logger.handlers.clear()
        
        # 创建文件handler
        file_handler = logging.FileHandler(self.log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 记录会话开始
        self.logger.info("=" * 80)
        self.logger.info("🚀 Agent RAG 会话开始")
        self.logger.info(f"📝 日志文件: {self.log_filename}")
        self.logger.info("=" * 80)
        
        # 统计信息
        self.session_stats = {
            'start_time': time.time(),
            'queries_processed': 0,
            'agents_activated': set(),
            'total_retrievals': 0,
            'total_evaluations': 0,
            'total_generations': 0
        }
        
        AgentLogger._initialized = True
    
    def log_query_start(self, query: str) -> str:
        """记录查询开始"""
        query_id = f"Q{int(time.time() * 1000) % 100000:05d}"  # 生成查询ID
        self.session_stats['queries_processed'] += 1
        
        self.logger.info("")
        self.logger.info("🤖" + "=" * 78)
        self.logger.info(f"🤖 查询 [{query_id}] 开始处理")
        self.logger.info(f"📝 用户查询: {query}")
        self.logger.info("🤖" + "=" * 78)
        
        return query_id
    
    def log_agent_start(self, agent_name: str, task_description: str):
        """记录Agent开始执行"""
        self.session_stats['agents_activated'].add(agent_name)
        
        # Agent图标映射
        agent_icons = {
            'QueryPlannerAgent': '📋',
            'RetrievalAgent': '🔍', 
            'EvaluatorAgent': '📊',
            'GeneratorAgent': '✍️',
            'AgentCoordinator': '🎯'
        }
        
        icon = agent_icons.get(agent_name, '🔧')
        self.logger.info(f"")
        self.logger.info(f"{icon} [{agent_name}] 开始执行")
        self.logger.info(f"   任务: {task_description}")
    
    def log_agent_step(self, agent_name: str, step: str, details: str = ""):
        """记录Agent执行步骤"""
        agent_icons = {
            'QueryPlannerAgent': '📋',
            'RetrievalAgent': '🔍', 
            'EvaluatorAgent': '📊',
            'GeneratorAgent': '✍️',
            'AgentCoordinator': '🎯'
        }
        
        icon = agent_icons.get(agent_name, '🔧')
        
        if details:
            self.logger.info(f"{icon} [{agent_name}] {step}: {details}")
        else:
            self.logger.info(f"{icon} [{agent_name}] {step}")
    
    def log_agent_result(self, agent_name: str, result: Dict[str, Any]):
        """记录Agent执行结果"""
        agent_icons = {
            'QueryPlannerAgent': '📋',
            'RetrievalAgent': '🔍', 
            'EvaluatorAgent': '📊',
            'GeneratorAgent': '✍️',
            'AgentCoordinator': '🎯'
        }
        
        icon = agent_icons.get(agent_name, '🔧')
        
        # 记录关键结果信息
        if agent_name == 'QueryPlannerAgent':
            plan = result.get('plan', {})
            self.logger.info(f"{icon} [{agent_name}] ✅ 规划完成")
            self.logger.info(f"   查询类型: {plan.get('query_type', 'unknown')}")
            self.logger.info(f"   复杂度: {plan.get('complexity', 'unknown')}")  
            self.logger.info(f"   检索策略: {plan.get('retrieval_strategy', 'unknown')}")
            self.logger.info(f"   多轮检索: {plan.get('multi_round_needed', False)}")
            
        elif agent_name == 'RetrievalAgent':
            self.session_stats['total_retrievals'] += 1
            raw_chunks = result.get('raw_chunks', [])
            strategy = result.get('strategy_used', 'unknown')
            self.logger.info(f"{icon} [{agent_name}] ✅ 检索完成")
            self.logger.info(f"   策略: {strategy}")
            self.logger.info(f"   检索到: {len(raw_chunks)} 个chunks")
            
        elif agent_name == 'EvaluatorAgent':
            self.session_stats['total_evaluations'] += 1
            decision = result.get('decision', {})
            top_chunks = result.get('top_chunks_used', [])
            self.logger.info(f"{icon} [{agent_name}] ✅ 评估完成")
            self.logger.info(f"   评估chunks: {len(top_chunks)} 个")
            self.logger.info(f"   质量得分: {decision.get('confidence', 0):.2f}")
            self.logger.info(f"   下一步行动: {decision.get('next_action', 'unknown')}")
            
        elif agent_name == 'GeneratorAgent':
            self.session_stats['total_generations'] += 1
            sources_used = result.get('sources_used', 0)
            answer_length = len(result.get('final_answer', ''))
            self.logger.info(f"{icon} [{agent_name}] ✅ 生成完成")
            self.logger.info(f"   使用轮次: {sources_used} 轮检索结果")
            self.logger.info(f"   答案长度: {answer_length} 字符")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """记录性能指标"""
        self.logger.info("📈 性能指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"   {key}: {value:.3f}")
            else:
                self.logger.info(f"   {key}: {value}")
    
    def log_error(self, agent_name: str, error: Exception, context: str = ""):
        """记录错误信息"""
        self.logger.error(f"❌ [{agent_name}] 错误发生: {str(error)}")
        if context:
            self.logger.error(f"   上下文: {context}")
    
    def log_query_complete(self, query_id: str, total_time: float, result_preview: str = ""):
        """记录查询完成"""
        self.logger.info("")
        self.logger.info(f"✅ 查询 [{query_id}] 处理完成")
        self.logger.info(f"⏱️  总耗时: {total_time:.2f} 秒")
        
        if result_preview:
            preview = result_preview[:200] + "..." if len(result_preview) > 200 else result_preview
            self.logger.info(f"📄 结果预览: {preview}")
        
        self.logger.info("🤖" + "=" * 78)
    
    def log_session_summary(self):
        """记录会话总结"""
        session_time = time.time() - self.session_stats['start_time']
        
        self.logger.info("")
        self.logger.info("📊 会话总结:")
        self.logger.info("=" * 80)
        self.logger.info(f"⏱️  总会话时间: {session_time:.2f} 秒")
        self.logger.info(f"❓ 处理查询数: {self.session_stats['queries_processed']}")
        self.logger.info(f"🤖 激活的Agent: {', '.join(sorted(self.session_stats['agents_activated']))}")
        self.logger.info(f"🔍 总检索次数: {self.session_stats['total_retrievals']}")
        self.logger.info(f"📊 总评估次数: {self.session_stats['total_evaluations']}")
        self.logger.info(f"✍️  总生成次数: {self.session_stats['total_generations']}")
        self.logger.info("=" * 80)
    
    def get_logger(self):
        """获取底层logger实例"""
        return self.logger
    
    def get_log_path(self):
        """获取日志文件路径"""
        return str(self.log_path)


# 创建全局日志实例
agent_logger = AgentLogger()

def get_agent_logger():
    """获取Agent日志器实例"""
    return agent_logger