"""
Agent RAG 系统演示脚本

展示Multi-Agent协作的RAG系统工作流程
"""

import sys
import os
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_demo_components():
    """创建演示用的组件"""
    
    class DemoLLM:
        """演示用的LLM模拟类"""
        def __init__(self):
            self.responses = {
                'planning': {
                    'query_type': 'factual',
                    'complexity': 'medium', 
                    'retrieval_strategy': 'hybrid',
                    'multi_round_needed': True,
                    'max_rounds': 3
                }
            }
        
        def __call__(self, prompt, **kwargs):
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            
            # 根据提示词类型返回不同响应
            if '评估以下经过重排序' in prompt:
                return MockResponse("""
                {
                    "coverage_score": 4,
                    "depth_score": 3,
                    "consistency_score": 4,
                    "novelty_score": 3,
                    "overall_quality": 3.5,
                    "recommended_action": "深入检索",
                    "reasoning": "信息覆盖度良好，但深度不够，建议进行深入检索获取更详细信息",
                    "key_gaps": ["缺少具体实现细节", "需要更多技术原理"]
                }
                """)
            elif '根据以下收集到的所有知识' in prompt:
                return MockResponse("""
                ### 1. 核心答案
                基于Multi-Agent协作的RAG系统通过四大专门智能体协同工作，实现了智能化的信息检索和生成过程。

                ### 2. 详细分析
                **智能体协作机制**：
                - QueryPlannerAgent负责分析查询复杂度和制定检索策略
                - RetrievalAgent执行多种检索策略（向量、关键词、混合检索）
                - EvaluatorAgent基于重排序结果进行质量评估和决策
                - GeneratorAgent整合多轮检索结果生成高质量答案

                **技术优势**：
                1. 自适应检索：根据查询复杂度动态调整策略
                2. 智能决策：基于质量评估自主决定下一步行动
                3. 模块化设计：每个Agent专注于特定任务，便于维护和扩展

                ### 3. 补充说明
                该系统支持传统模式和Agent模式的无缝切换，确保向后兼容性的同时提供最新的AI协作能力。

                ### 4. 信息来源
                基于3轮Agent协作检索的综合信息整合。
                """)
            elif '将搜索词' in prompt and '进行合并改写' in prompt:
                return MockResponse("机器学习注意力机制详细原理")
            elif '提取' in prompt and '核心搜索词' in prompt:
                return MockResponse("注意力权重计算|多头注意力|Transformer架构|自注意力机制")
            else:
                return MockResponse("这是一个关于查询的详细回答，展示了Agent RAG系统的强大能力。")
    
    class DemoSearcher:
        """演示用的搜索器"""
        def __init__(self):
            self.vector_retriever = True
            self.tfidf_retriever = True
            self.kg_retriever = None
            
            # 模拟reranker
            class MockReranker:
                def compute_scores(self, pairs):
                    import torch
                    # 返回模拟的重排序分数
                    scores = []
                    for pair in pairs:
                        # 简单的长度启发式评分
                        score = min(0.95, len(pair[1]) / 200 + 0.3)
                        scores.append(score)
                    return torch.tensor(scores)
            
            self.reranker = MockReranker()
        
        def search(self, query, top_k=5, **kwargs):
            # 模拟搜索结果
            mock_results = [
                f"知识点1:\n关于 '{query}' 的详细解释和原理分析。这部分内容涵盖了基础概念和核心要点...",
                f"知识点2:\n深入探讨 '{query}' 的技术实现和应用场景。包含了最新的研究进展和实践经验...", 
                f"知识点3:\n'{query}' 的相关案例研究和对比分析。提供了多角度的理解和实际应用的参考..."
            ]
            
            content = "\n\n".join(mock_results[:top_k])
            ids = list(range(1, top_k + 1))
            
            return content, ids
    
    class DemoQueryExpander:
        """演示用的查询扩展器"""
        def __init__(self, llm):
            self.llm = llm
            self.reranker = DemoSearcher().reranker
        
        def extend_query(self, queries, responses, num):
            # 模拟查询扩展
            base_query = queries[0] if queries else "默认查询"
            expanded = []
            
            for i in range(min(num, 3)):
                expanded.append(f"{base_query} 相关主题{i+1}")
            
            return expanded
    
    return DemoLLM(), DemoSearcher(), DemoQueryExpander

def demo_agent_rag():
    """演示Agent RAG系统"""
    print("🚀 MetaSearch Agent RAG 系统演示")
    print("=" * 60)
    print()
    
    try:
        # 创建演示组件
        print("📦 初始化系统组件...")
        demo_llm, demo_searcher, demo_expander = create_demo_components()
        demo_expander.llm = demo_llm
        
        # 导入Agent RAG系统
        from deepsearch.rag.deep_rag import create_agent_rag
        
        # 创建Agent RAG实例
        print("🤖 创建Agent RAG系统...")
        agent_rag = create_agent_rag(
            searcher=demo_searcher,
            llm=demo_llm,
            query_expander=demo_expander
        )
        
        print(f"✅ Agent RAG系统初始化完成")
        print(f"📍 当前模式: {agent_rag.get_system_status()['current_mode']}")
        print(f"📝 日志文件: {agent_rag.get_log_path()}")
        print()
        
        # 演示查询
        demo_query = "机器学习中的注意力机制是如何工作的？"
        print(f"❓ 演示查询: {demo_query}")
        print()
        
        print("🔄 开始Agent协作流程...")
        print("-" * 60)
        
        # 执行查询
        start_time = time.time()
        result = agent_rag.answer(demo_query)
        end_time = time.time()
        
        print("-" * 60)
        print("✅ Agent协作完成!")
        print(f"⏱️  总耗时: {end_time - start_time:.2f} 秒")
        print()
        
        # 显示结果
        print("📄 生成的答案:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        print()
        
        # 显示系统状态
        print("📊 系统状态报告:")
        status = agent_rag.get_system_status()
        if 'agent_system' in status:
            agent_status = status['agent_system']
            coord_status = agent_status['coordinator_status']
            
            print(f"🎯 协调器状态:")
            print(f"   - 处理查询数: {coord_status['total_queries_processed']}")
            print(f"   - 成功率: {coord_status['success_rate']:.1f}%")
            print(f"   - 平均耗时: {coord_status['average_execution_time']:.2f}秒")
            
            print(f"🤖 Agent性能:")
            for agent_name, perf in agent_status['agent_performance'].items():
                if perf['total_executions'] > 0:
                    print(f"   - {agent_name}: {perf['total_executions']}次执行, 成功率 {perf['success_rate']:.1f}%")
        
        print()
        print("🎉 演示完成! 查看日志文件获取详细执行过程。")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = demo_agent_rag()
    
    if success:
        print("\n🌟 Agent RAG系统运行正常!")
        print("📚 特性亮点:")
        print("   • 🤖 Multi-Agent智能体协作")
        print("   • 🧠 基于重排序的智能评估")  
        print("   • 📊 详细的日志监控系统")
        print("   • 🔄 自适应检索策略")
        print("   • ✨ 向后兼容传统模式")
    else:
        print("\n💥 系统出现问题，请检查配置和依赖。")

if __name__ == "__main__":
    main()