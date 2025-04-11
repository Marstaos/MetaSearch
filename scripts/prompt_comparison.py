import sys
import os
import yaml
import logging
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepsearch.llm.model import LLMFactory

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('d:\Playground\MetaSearch\config\config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def generate_report(query: str, knowledge_text: str, model_type: str = 'api') -> str:
    """直接使用最终prompt生成回答"""
    # 初始化LLM
    llm = LLMFactory.create_llm(model_type=model_type)
    
    # 构建提示词
    prompt = (
        f"你是一位专业的研究助理，请根据以下收集到的所有知识，撰写一份详细的报告来回答原始问题。\n"
        f"\n## 原始问题：\n{query}\n"
        f"\n## 收集到的知识：{knowledge_text}\n"
        f"\n## 报告撰写要求\n"
        f"1. 首先用1-2句话总结核心答案\n"
        f"2. 然后详细阐述，每个点都要引用具体参考文献\n"
        f"3. 回答完成后输出参考文献原文\n"
        f"4. 最后给出综合分析和个人见解\n"
        f"\n请开始撰写详细报告："
    )
    
    start_time = time.time()
    response = llm(prompt).content
    end_time = time.time()
    
    logger.info(f"生成报告完成，用时: {end_time - start_time:.2f}秒")
    return response

if __name__ == "__main__":
    # 测试用例
    test_query = "请详细介绍明朝的内阁制度"
    test_knowledge = """
    ## 知识点 1
    **来源查询**: 明朝内阁制度
    **详细内容**: 
    明朝内阁创立于永乐年间，最初是皇帝的秘书机构...
    **相关度评分**: 0.92/1.0
    
    ## 知识点 2
    **来源查询**: 明朝政治制度
    **详细内容**: 
    内阁大学士最初只是五品官， later权力逐渐增大...
    **相关度评分**: 0.85/1.0
    """
    
    # 生成报告
    report = generate_report(test_query, test_knowledge)
    print("\n生成的报告:\n")
    print(report)