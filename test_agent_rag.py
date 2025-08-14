"""
Agent RAG 系统测试脚本
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_agent_system():
    """测试Agent系统基础功能"""
    print("🧪 开始测试Agent RAG系统...")
    
    try:
        # 测试导入
        print("📦 测试模块导入...")
        from deepsearch.agents import (
            BaseAgent, AgentRegistry, get_agent_logger,
            QueryPlannerAgent, RetrievalAgent, EvaluatorAgent, GeneratorAgent,
            AgentCoordinator
        )
        print("✅ 所有Agent模块导入成功")
        
        # 测试日志系统
        print("\n📝 测试日志系统...")
        logger = get_agent_logger()
        query_id = logger.log_query_start("测试查询")
        logger.log_agent_start("TestAgent", "测试Agent功能")
        logger.log_agent_step("TestAgent", "执行测试步骤")
        logger.log_query_complete(query_id, 0.5, "测试完成")
        print(f"✅ 日志系统正常，日志文件: {logger.get_log_path()}")
        
        # 测试基础Agent
        print("\n🤖 测试基础Agent...")
        
        class MockLLM:
            def __call__(self, prompt, **kwargs):
                class MockResponse:
                    content = "Mock LLM response"
                return MockResponse()
        
        class MockSearcher:
            def __init__(self):
                self.reranker = None
            def search(self, query, **kwargs):
                return "Mock content", [1, 2, 3]
        
        mock_llm = MockLLM()
        mock_searcher = MockSearcher()
        
        # 测试QueryPlannerAgent
        planner = QueryPlannerAgent(llm=mock_llm)
        planner_result = planner.run_with_monitoring(
            {'query': '什么是机器学习？'}, 
            "测试查询规划"
        )
        
        if planner_result.get('plan'):
            print("✅ QueryPlannerAgent 测试成功")
        else:
            print("❌ QueryPlannerAgent 测试失败")
        
        # 测试RetrievalAgent
        retriever = RetrievalAgent(searcher=mock_searcher, llm=mock_llm)
        retrieval_result = retriever.run_with_monitoring(
            {
                'original_query': '测试查询',
                'plan': {'retrieval_strategy': 'hybrid', 'info_sources_needed': 3}
            },
            "测试检索"
        )
        
        if retrieval_result.get('raw_chunks'):
            print("✅ RetrievalAgent 测试成功")
        else:
            print("❌ RetrievalAgent 测试失败")
        
        print("\n🎯 基础Agent测试完成")
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_deeprag_integration():
    """测试DeepRAG集成"""
    print("\n🔄 测试DeepRAG集成...")
    
    try:
        from deepsearch.rag.deep_rag import DeepRAG, create_agent_rag
        
        class MockLLM:
            def __call__(self, prompt, **kwargs):
                class MockResponse:
                    content = f"这是对查询的回答: {prompt[:100]}..."
                return MockResponse()
        
        class MockSearcher:
            def __init__(self):
                self.reranker = None
                self.vector_retriever = True
                self.tfidf_retriever = True
                self.kg_retriever = None
                
            def search(self, query, **kwargs):
                return f"检索内容: {query}", [1, 2, 3]
        
        class MockQueryExpander:
            def extend_query(self, queries, responses, num):
                return [f"扩展查询: {q}" for q in queries[:num]]
        
        mock_llm = MockLLM()
        mock_searcher = MockSearcher()
        mock_expander = MockQueryExpander()
        
        # 测试Agent模式
        print("🤖 测试Agent模式...")
        agent_rag = create_agent_rag(mock_searcher, mock_llm, mock_expander)
        
        print("📊 系统状态:")
        status = agent_rag.get_system_status()
        print(f"  - 当前模式: {status['current_mode']}")
        
        if agent_rag.get_log_path():
            print(f"  - 日志文件: {agent_rag.get_log_path()}")
        
        print("✅ DeepRAG集成测试成功")
        
    except Exception as e:
        print(f"❌ DeepRAG集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 MetaSearch Agent RAG 系统测试")
    print("=" * 50)
    
    success = True
    
    # 测试Agent系统
    success &= test_agent_system()
    
    # 测试DeepRAG集成
    success &= test_deeprag_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！Agent RAG系统准备就绪")
    else:
        print("💥 部分测试失败，请检查错误信息")
    
    return success

if __name__ == "__main__":
    main()