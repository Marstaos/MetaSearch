"""
答案生成智能体 (GeneratorAgent)

基于收集的多轮检索信息生成最终答案
"""

import time
from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent

class GeneratorAgent(BaseAgent):
    """答案生成智能体 - 负责基于收集的信息生成最终答案"""
    
    def __init__(self, name: str = "GeneratorAgent", **kwargs):
        super().__init__(name, **kwargs)
        
        # 生成配置
        self.generation_config = {
            'max_context_length': 4000,  # 最大上下文长度
            'answer_min_length': 200,    # 最小答案长度
            'answer_max_length': 2000,   # 最大答案长度
            'include_sources': True,     # 是否包含来源信息
            'response_format': 'detailed'  # 详细格式
        }
        
        # 答案模板
        self.answer_templates = {
            'detailed': {
                'structure': ["核心答案", "详细分析", "补充说明", "参考来源"],
                'prompt_template': self._get_detailed_template
            },
            'concise': {
                'structure': ["核心答案", "关键要点"],
                'prompt_template': self._get_concise_template
            },
            'analytical': {
                'structure': ["问题分析", "深入解答", "总结结论", "延伸思考"],
                'prompt_template': self._get_analytical_template
            }
        }
        
        # 生成历史
        self.generation_history = []
    
    def execute(self, task: Dict) -> Dict:
        """执行答案生成任务
        
        Args:
            task: 包含查询和所有检索内容的任务字典
            
        Returns:
            生成结果字典
        """
        query = task.get('original_query', '')
        all_retrieved_content = task.get('all_retrieved_content', [])
        retrieval_rounds = len(all_retrieved_content)
        plan = task.get('plan', {})
        
        self.log_step("开始生成答案", f"基于{retrieval_rounds}轮检索结果")
        
        if not all_retrieved_content:
            self.log_step("无检索内容", "生成默认回答")
            return self._generate_fallback_answer(query)
        
        # 第1步：整合所有检索内容
        integration_start_time = time.time()
        integrated_content = self._integrate_retrieval_content(all_retrieved_content, query)
        integration_time = time.time() - integration_start_time
        
        self.log_step("内容整合完成", 
                     f"耗时={integration_time:.2f}s, 整合长度={len(integrated_content)}")
        
        # 第2步：确定生成格式
        response_format = self._determine_response_format(query, plan)
        self.log_step("确定生成格式", f"格式={response_format}")
        
        # 第3步：构建生成提示词
        prompt_start_time = time.time()
        generation_prompt = self._build_generation_prompt(query, integrated_content, response_format)
        prompt_time = time.time() - prompt_start_time
        
        self.log_step("提示词构建完成", 
                     f"耗时={prompt_time:.2f}s, 提示词长度={len(generation_prompt)}")
        
        # 第4步：使用LLM生成答案
        generation_start_time = time.time()
        final_answer = self._generate_with_llm(generation_prompt, response_format)
        generation_time = time.time() - generation_start_time
        
        self.log_step("LLM生成完成", 
                     f"耗时={generation_time:.2f}s, 答案长度={len(final_answer)}")
        
        # 第5步：答案后处理
        post_process_start_time = time.time()
        processed_answer = self._post_process_answer(final_answer, query, integrated_content)
        post_process_time = time.time() - post_process_start_time
        
        self.log_step("答案后处理完成", 
                     f"耗时={post_process_time:.2f}s, 最终长度={len(processed_answer)}")
        
        # 第6步：生成答案质量评估
        quality_metrics = self._evaluate_answer_quality(processed_answer, query, all_retrieved_content)
        
        # 第7步：记录生成历史
        self._record_generation(query, retrieval_rounds, len(processed_answer), 
                               generation_time, quality_metrics)
        
        return {
            'agent': self.name,
            'final_answer': processed_answer,
            'sources_used': retrieval_rounds,
            'response_format': response_format,
            'quality_metrics': quality_metrics,
            'performance': {
                'integration_time': integration_time,
                'prompt_time': prompt_time,
                'generation_time': generation_time,
                'post_process_time': post_process_time,
                'total_time': integration_time + prompt_time + generation_time + post_process_time
            },
            'generation_metadata': {
                'content_length': len(integrated_content),
                'prompt_length': len(generation_prompt),
                'retrieval_rounds': retrieval_rounds
            }
        }
    
    def _integrate_retrieval_content(self, all_content: List[str], query: str) -> str:
        """整合所有检索轮次的内容"""
        if not all_content:
            return ""
        
        # 按轮次整合内容
        integrated_parts = []
        
        for i, content in enumerate(all_content):
            round_num = i + 1
            # 为每轮内容添加标识
            integrated_parts.append(
                f"\n=== 检索轮次 {round_num} ===\n{content}"
            )
        
        # 检查总长度，如果太长则需要截断
        full_content = "\n\n".join(integrated_parts)
        
        if len(full_content) > self.generation_config['max_context_length']:
            self.log_step("内容过长", f"从{len(full_content)}字符截断到{self.generation_config['max_context_length']}")
            # 优先保留最新轮次的内容
            truncated_parts = []
            current_length = 0
            
            for part in reversed(integrated_parts):  # 从最新轮次开始
                if current_length + len(part) <= self.generation_config['max_context_length']:
                    truncated_parts.insert(0, part)  # 插入到开头保持顺序
                    current_length += len(part)
                else:
                    # 部分包含最后一个轮次
                    remaining_length = self.generation_config['max_context_length'] - current_length
                    if remaining_length > 500:  # 只有足够长度才包含
                        truncated_part = part[:remaining_length] + "\n[内容截断...]"
                        truncated_parts.insert(0, truncated_part)
                    break
            
            full_content = "\n\n".join(truncated_parts)
        
        return full_content
    
    def _determine_response_format(self, query: str, plan: Dict) -> str:
        """确定响应格式"""
        query_type = plan.get('query_type', 'general')
        complexity = plan.get('complexity', 'medium')
        
        # 基于查询类型和复杂度选择格式
        if query_type == 'analytical' or complexity == 'complex':
            return 'analytical'
        elif query_type == 'factual' and complexity == 'simple':
            return 'concise'
        else:
            return 'detailed'  # 默认详细格式
    
    def _build_generation_prompt(self, query: str, content: str, response_format: str) -> str:
        """构建生成提示词"""
        template_func = self.answer_templates[response_format]['prompt_template']
        return template_func(query, content)
    
    def _get_detailed_template(self, query: str, content: str) -> str:
        """详细格式模板"""
        return f"""
你是一位专业的研究助理，请根据以下收集到的所有信息，为用户提供详细准确的回答。

## 用户问题
{query}

## 收集到的信息
{content}

## 回答要求
请按以下结构回答：

### 1. 核心答案
用1-2句话简洁总结核心答案。

### 2. 详细分析
详细阐述问题的各个方面，要求：
- 逻辑清晰，层次分明
- 引用具体的信息来源
- 保证信息的准确性和完整性

### 3. 补充说明
如有必要，提供相关的背景信息、注意事项或延伸内容。

### 4. 信息来源
简要说明信息的来源和可靠性。

请确保回答内容丰富、准确、有价值。开始回答：
"""
    
    def _get_concise_template(self, query: str, content: str) -> str:
        """简洁格式模板"""
        return f"""
请基于以下信息简洁准确地回答用户问题。

用户问题：{query}

参考信息：
{content}

回答要求：
1. 直接回答核心问题
2. 突出关键要点
3. 保持简洁明了
4. 确保准确性

请开始回答：
"""
    
    def _get_analytical_template(self, query: str, content: str) -> str:
        """分析格式模板"""
        return f"""
请作为专业分析师，基于以下信息对用户问题进行深入分析。

## 分析对象
{query}

## 信息基础
{content}

## 分析框架
请按以下结构进行分析：

### 1. 问题分析
- 问题的核心要素
- 涉及的关键概念
- 分析的切入角度

### 2. 深入解答
- 基于信息的详细分析
- 不同角度的阐述
- 逻辑推理过程

### 3. 总结结论
- 综合性结论
- 关键要点总结

### 4. 延伸思考
- 相关问题的思考
- 进一步探讨的方向

请进行专业分析：
"""
    
    def _generate_with_llm(self, prompt: str, response_format: str) -> str:
        """使用LLM生成答案"""
        if not self.llm:
            return self._generate_fallback_answer_text("LLM不可用，无法生成详细答案")
        
        try:
            # 根据格式调整生成参数
            generation_params = {
                'detailed': {'temperature': 0.7, 'max_tokens': 2000},
                'concise': {'temperature': 0.3, 'max_tokens': 800},
                'analytical': {'temperature': 0.8, 'max_tokens': 2500}
            }
            
            params = generation_params.get(response_format, {'temperature': 0.7, 'max_tokens': 1500})
            
            # 调用LLM
            response = self.llm(prompt, **params).content
            
            return response.strip()
            
        except Exception as e:
            self.log_step("LLM生成失败", f"错误: {str(e)}")
            return self._generate_fallback_answer_text(f"生成过程中出现错误: {str(e)}")
    
    def _post_process_answer(self, answer: str, query: str, content: str) -> str:
        """答案后处理"""
        # 1. 基础清理
        processed = answer.strip()
        
        # 2. 长度检查
        min_len = self.generation_config['answer_min_length']
        max_len = self.generation_config['answer_max_length']
        
        if len(processed) < min_len:
            self.log_step("答案过短", f"长度{len(processed)} < {min_len}, 添加补充内容")
            processed += "\n\n注：基于可用信息提供的回答，如需更详细信息请提供更多相关材料。"
        
        if len(processed) > max_len:
            self.log_step("答案过长", f"长度{len(processed)} > {max_len}, 截断处理")
            processed = processed[:max_len] + "\n\n[答案因长度限制被截断]"
        
        # 3. 格式优化
        processed = self._optimize_formatting(processed)
        
        # 4. 添加元信息（如果配置要求）
        if self.generation_config['include_sources']:
            source_info = self._generate_source_info(content)
            if source_info:
                processed += f"\n\n---\n**信息来源**: {source_info}"
        
        return processed
    
    def _optimize_formatting(self, text: str) -> str:
        """优化文本格式"""
        # 标准化换行
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 处理多余空行
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 处理列表格式
        text = re.sub(r'\n(\d+)\.', r'\n\n\1.', text)
        text = re.sub(r'\n-', r'\n\n-', text)
        
        return text.strip()
    
    def _generate_source_info(self, content: str) -> str:
        """生成来源信息"""
        # 简单统计信息来源数量
        round_count = content.count("=== 检索轮次")
        if round_count > 0:
            return f"基于{round_count}轮信息检索结果整合"
        else:
            return "基于检索到的相关信息"
    
    def _evaluate_answer_quality(self, answer: str, query: str, all_content: List[str]) -> Dict[str, Any]:
        """评估答案质量"""
        metrics = {}
        
        # 基础指标
        metrics['answer_length'] = len(answer)
        metrics['word_count'] = len(answer.split())
        
        # 结构化程度
        metrics['has_structure'] = any(marker in answer for marker in ['###', '**', '1.', '2.'])
        
        # 内容丰富度（基于检索轮次）
        metrics['content_richness'] = min(1.0, len(all_content) / 3.0)
        
        # 信息覆盖度（简单启发式）
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        metrics['query_coverage'] = len(query_words.intersection(answer_words)) / len(query_words)
        
        # 综合质量分数
        structure_score = 1.0 if metrics['has_structure'] else 0.7
        length_score = 1.0 if 200 <= metrics['answer_length'] <= 2000 else 0.8
        coverage_score = metrics['query_coverage']
        richness_score = metrics['content_richness']
        
        metrics['overall_quality'] = (structure_score + length_score + coverage_score + richness_score) / 4.0
        
        return metrics
    
    def _generate_fallback_answer(self, query: str) -> Dict:
        """生成降级答案"""
        fallback_text = self._generate_fallback_answer_text(query)
        
        return {
            'agent': self.name,
            'final_answer': fallback_text,
            'sources_used': 0,
            'response_format': 'fallback',
            'quality_metrics': {
                'answer_length': len(fallback_text),
                'overall_quality': 0.3,
                'is_fallback': True
            },
            'performance': {'total_time': 0.0}
        }
    
    def _generate_fallback_answer_text(self, context: str) -> str:
        """生成降级答案文本"""
        return f"""
抱歉，由于检索到的相关信息有限，无法提供详细的回答。

针对您的问题，建议您：
1. 尝试使用更具体或不同的关键词重新搜索
2. 将复杂问题分解为更简单的子问题
3. 从权威来源获取更多相关信息

如果您能提供更多背景信息或调整查询方式，我将能够为您提供更好的帮助。

问题上下文: {context[:200]}...
"""
    
    def _record_generation(self, query: str, rounds: int, answer_length: int, 
                          generation_time: float, quality_metrics: Dict):
        """记录生成历史"""
        record = {
            'timestamp': time.time(),
            'query': query,
            'retrieval_rounds': rounds,
            'answer_length': answer_length,
            'generation_time': generation_time,
            'quality_score': quality_metrics.get('overall_quality', 0.5)
        }
        
        self.generation_history.append(record)
        
        # 限制历史记录大小
        if len(self.generation_history) > 100:
            self.generation_history = self.generation_history[-100:]
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """获取生成统计信息"""
        if not self.generation_history:
            return {'message': '暂无生成历史'}
        
        # 统计信息
        total_generations = len(self.generation_history)
        avg_answer_length = sum(r['answer_length'] for r in self.generation_history) / total_generations
        avg_generation_time = sum(r['generation_time'] for r in self.generation_history) / total_generations
        avg_quality = sum(r['quality_score'] for r in self.generation_history) / total_generations
        
        # 轮次分布
        rounds_distribution = {}
        for record in self.generation_history:
            rounds = record['retrieval_rounds']
            rounds_distribution[rounds] = rounds_distribution.get(rounds, 0) + 1
        
        return {
            'total_generations': total_generations,
            'avg_answer_length': avg_answer_length,
            'avg_generation_time': avg_generation_time,
            'avg_quality_score': avg_quality,
            'rounds_distribution': rounds_distribution,
            'efficiency': avg_answer_length / max(avg_generation_time, 0.1)  # chars per second
        }