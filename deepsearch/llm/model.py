import os
import yaml
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class LLMFactory:
    """LLM工厂类，用于创建不同的大语言模型接口"""
    
    @staticmethod
    def create_llm(model_type=None, api_key=None):
        """创建LLM接口
        
        Args:
            model_type: 模型类型('api'或'local')
            api_key: API密钥(如果使用API)
            
        Returns:
            LLM接口实例
        """
        # 使用配置文件默认值
        if model_type is None:
            model_type = "api"  # 默认使用API
        
        if model_type == "api":
            # 使用LangChain的API接口
            if api_key is None:
                api_key = config['models']['llm']['api_key']
                
            return ChatOpenAI(
                model=config['models']['llm']['model_name'],
                base_url=config['models']['llm']['api_endpoint'],
                api_key=api_key
            )
        elif model_type == "local":
            # 使用本地模型
            return LocalLLM(config['models']['llm']['path'])
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


class LocalLLM:
    """本地LLM封装类"""
    
    def __init__(self, model_path):
        """初始化本地模型
        
        Args:
            model_path: 模型路径
        """
        logger.info(f"正在加载本地模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("本地模型加载完成")
    
    def __call__(self, prompt):
        """调用模型生成回答
        
        Args:
            prompt: 输入提示词
            
        Returns:
            包含content属性的响应对象
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 包装为与LangChain兼容的格式
        class Response:
            def __init__(self, content):
                self.content = content
                
        return Response(response)