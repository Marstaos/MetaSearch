"""
查询规划智能体 (QueryPlannerAgent)

负责分析用户查询，制定检索策略和执行计划
"""

import json
import re
from typing import Dict, List, Any
from .base_agent import BaseAgent

class QueryPlannerAgent(BaseAgent):
    """查询规划智能体 - 分析查询复杂度并制定检索策略"""
    
    def __init__(self, name: str = "QueryPlannerAgent", **kwargs):
        super().__init__(name, **kwargs)
        
        # 查询类型模式匹配
        self.query_patterns = {
            'factual': [
                r'什么是', r'谁是', r'哪里', r'何时', r'定义', r'概念',
                r'what is', r'who is', r'where', r'when', r'definition'
            ],
            'analytical': [
                r'为什么', r'如何', r'怎样', r'原因', r'分析', r'解释',
                r'why', r'how', r'analyze', r'explain', r'reason'
            ],
            'comparative': [
                r'比较', r'对比', r'区别', r'不同', r'相同', r'优缺点',
                r'compare', r'difference', r'similarity', r'vs', r'versus'
            ],
            'procedural': [
                r'步骤', r'流程', r'方法', r'如何做', r'教程',
                r'step', r'process', r'procedure', r'tutorial', r'how to'
            ],
            'opinion': [
                r'观点', r'看法', r'评价', r'建议', r'推荐',
                r'opinion', r'view', r'recommendation', r'suggest'
            ]
        }
    
    def execute(self, task: Dict) -> Dict:
        """执行查询规划任务
        
        Args:
            task: 包含查询信息的任务字典
            
        Returns:
            规划结果字典
        """
        query = task.get('query', '')
        
        self.log_step("开始分析用户查询")
        
        # 第1步：基础查询分析
        basic_analysis = self._analyze_query_basic(query)
        self.log_step("基础分析完成", f"长度={basic_analysis['length']}, 词数={basic_analysis['word_count']}")
        
        # 第2步：查询类型识别
        query_type = self._identify_query_type(query)
        self.log_step("类型识别完成", f"类型={query_type}")
        
        # 第3步：复杂度评估
        complexity = self._assess_complexity(query, basic_analysis)
        self.log_step("复杂度评估完成", f"复杂度={complexity}")
        
        # 第4步：检索策略制定
        retrieval_strategy = self._determine_retrieval_strategy(query_type, complexity)
        self.log_step("检索策略制定", f"策略={retrieval_strategy}")
        
        # 第5步：多轮检索需求判断
        multi_round_needed = self._needs_multi_round(query, complexity)
        self.log_step("多轮检索判断", f"需要多轮={multi_round_needed}")
        
        # 第6步：生成子查询（如果需要）
        sub_queries = self._generate_sub_queries(query, query_type)
        if sub_queries:
            self.log_step("子查询生成", f"生成{len(sub_queries)}个子查询")
        
        # 第7步：LLM增强分析（可选）
        enhanced_analysis = None
        if self.llm and complexity == "complex":
            enhanced_analysis = self._llm_enhanced_analysis(query, basic_analysis)
            self.log_step("LLM增强分析完成", "获得详细洞察")
        
        # 构建规划结果
        plan = {
            'query_type': query_type,
            'complexity': complexity,
            'info_sources_needed': self._estimate_sources_needed(complexity, query_type),
            'retrieval_strategy': retrieval_strategy,
            'multi_round_needed': multi_round_needed,
            'max_rounds': self._determine_max_rounds(complexity),
            'sub_queries': sub_queries,
            'basic_analysis': basic_analysis,
            'enhanced_analysis': enhanced_analysis,
            'confidence': self._calculate_confidence(query_type, complexity)
        }
        
        return {
            'agent': self.name,
            'plan': plan,
            'original_query': query
        }
    
    def _analyze_query_basic(self, query: str) -> Dict[str, Any]:
        """基础查询分析"""
        words = query.split()
        
        # 检测关键词
        key_indicators = {
            'question_words': ['什么', '谁', '哪里', '何时', '为什么', '如何', 'what', 'who', 'where', 'when', 'why', 'how'],
            'comparison_words': ['比较', '对比', '区别', '差异', 'vs', 'versus', 'compare', 'difference'],
            'process_words': ['步骤', '流程', '方法', '过程', 'step', 'process', 'procedure'],
            'technical_words': ['算法', '模型', '系统', '架构', 'algorithm', 'model', 'system', 'architecture']
        }
        
        detected_indicators = {}
        for category, indicators in key_indicators.items():
            detected_indicators[category] = any(indicator in query.lower() for indicator in indicators)
        
        return {
            'length': len(query),
            'word_count': len(words),
            'has_punctuation': any(p in query for p in '？?！!'),
            'detected_indicators': detected_indicators,
            'language': 'chinese' if any('\u4e00' <= c <= '\u9fff' for c in query) else 'english'
        }
    
    def _identify_query_type(self, query: str) -> str:
        """识别查询类型"""
        query_lower = query.lower()
        
        # 计算每种类型的匹配度
        type_scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            type_scores[query_type] = score
        
        # 返回得分最高的类型
        if max(type_scores.values()) > 0:
            return max(type_scores.keys(), key=type_scores.get)
        else:
            return 'general'  # 默认类型
    
    def _assess_complexity(self, query: str, basic_analysis: Dict) -> str:
        """评估查询复杂度"""
        complexity_score = 0
        
        # 基于长度
        if basic_analysis['word_count'] > 15:
            complexity_score += 2
        elif basic_analysis['word_count'] > 8:
            complexity_score += 1
        
        # 基于指示词
        indicators = basic_analysis['detected_indicators']
        if indicators.get('technical_words'):
            complexity_score += 2
        if indicators.get('comparison_words'):
            complexity_score += 1
        if indicators.get('process_words'):
            complexity_score += 1
        
        # 基于查询结构
        if '并且' in query or 'and' in query.lower():
            complexity_score += 1
        if query.count('？') > 1 or query.count('?') > 1:
            complexity_score += 1
        
        # 转换为复杂度等级
        if complexity_score >= 4:
            return 'complex'
        elif complexity_score >= 2:
            return 'medium'
        else:
            return 'simple'
    
    def _determine_retrieval_strategy(self, query_type: str, complexity: str) -> str:
        """确定检索策略"""
        # 策略映射表
        strategy_map = {
            ('factual', 'simple'): 'vector',
            ('factual', 'medium'): 'hybrid',
            ('factual', 'complex'): 'hybrid',
            ('analytical', 'simple'): 'hybrid',
            ('analytical', 'medium'): 'hybrid',
            ('analytical', 'complex'): 'hybrid',
            ('comparative', 'simple'): 'hybrid',
            ('comparative', 'medium'): 'hybrid',
            ('comparative', 'complex'): 'multi_modal',
            ('procedural', 'simple'): 'keyword',
            ('procedural', 'medium'): 'hybrid',
            ('procedural', 'complex'): 'hybrid'
        }
        
        return strategy_map.get((query_type, complexity), 'hybrid')
    
    def _needs_multi_round(self, query: str, complexity: str) -> bool:
        """判断是否需要多轮检索"""
        # 复杂查询通常需要多轮检索
        if complexity == 'complex':
            return True
        
        # 包含多个问题的查询
        if query.count('？') > 1 or query.count('?') > 1:
            return True
        
        # 包含连接词的查询
        if any(connector in query for connector in ['并且', '以及', '还有', 'and', 'also', 'additionally']):
            return True
        
        return False
    
    def _estimate_sources_needed(self, complexity: str, query_type: str) -> int:
        """估计需要的信息源数量"""
        base_sources = {
            'simple': 2,
            'medium': 3,
            'complex': 5
        }
        
        sources = base_sources.get(complexity, 3)
        
        # 比较类查询需要更多源
        if query_type == 'comparative':
            sources += 1
        
        return min(sources, 8)  # 最大8个源
    
    def _determine_max_rounds(self, complexity: str) -> int:
        """确定最大检索轮数"""
        rounds_map = {
            'simple': 2,
            'medium': 3,
            'complex': 4
        }
        
        return rounds_map.get(complexity, 3)
    
    def _generate_sub_queries(self, query: str, query_type: str) -> List[str]:
        """生成初始子查询"""
        sub_queries = []
        
        # 基于查询类型生成子查询
        if query_type == 'comparative':
            # 比较类查询分解
            if '比较' in query or 'compare' in query.lower():
                # 简单分解，实际可以更复杂
                sub_queries = [query]  # 暂时返回原查询
        
        elif query_type == 'analytical':
            # 分析类查询分解
            sub_queries = [query]  # 暂时返回原查询
        
        else:
            # 其他类型暂时返回原查询
            sub_queries = [query]
        
        return sub_queries
    
    def _calculate_confidence(self, query_type: str, complexity: str) -> float:
        """计算规划置信度"""
        base_confidence = 0.8
        
        # 根据查询类型调整
        type_confidence_map = {
            'factual': 0.9,
            'analytical': 0.8,
            'comparative': 0.85,
            'procedural': 0.85,
            'general': 0.7
        }
        
        type_conf = type_confidence_map.get(query_type, 0.7)
        
        # 根据复杂度调整
        complexity_penalty = {
            'simple': 0.0,
            'medium': -0.05,
            'complex': -0.1
        }
        
        final_confidence = type_conf + complexity_penalty.get(complexity, 0)
        
        return max(0.5, min(1.0, final_confidence))
    
    def _llm_enhanced_analysis(self, query: str, basic_analysis: Dict) -> Dict[str, Any]:
        """使用LLM进行增强分析"""
        if not self.llm:
            return None
        
        prompt = f"""
        请分析以下用户查询，提供详细的洞察：
        
        查询: {query}
        基础分析: {basic_analysis}
        
        请从以下维度分析：
        1. 用户意图分析
        2. 关键信息需求
        3. 可能的歧义点
        4. 最佳回答策略
        
        请以JSON格式输出结果：
        {{
            "user_intent": "用户真实意图",
            "key_info_needs": ["需求1", "需求2"],
            "potential_ambiguities": ["歧义1", "歧义2"],
            "answer_strategy": "推荐的回答策略"
        }}
        """
        
        try:
            response = self.llm(prompt).content
            # 尝试解析JSON
            enhanced = json.loads(response)
            return enhanced
        except Exception as e:
            self.log_step("LLM增强分析失败", str(e))
            return {
                'user_intent': '分析失败',
                'key_info_needs': [],
                'potential_ambiguities': [],
                'answer_strategy': '标准RAG流程'
            }