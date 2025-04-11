import os
import json
import faiss
import numpy as np
import pickle
import yaml
import torch
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class VectorEncoder:
    """文本向量编码器"""
    
    def __init__(self, model_path=None):
        """初始化向量编码器
        
        Args:
            model_path: 预训练模型路径
        """
        if model_path is None:
            model_path = config['models']['embedding']['path']
            
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            logger.error(f"模型路径不存在: {model_path}")
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
        logger.info(f"加载向量模型: {model_path}")
        
        # 根据模型类型自动选择合适的tokenizer和model
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # 确保CUDA可用并设置设备
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.warning("未检测到可用的GPU，将使用CPU")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
            
            logger.info(f"向量模型加载完成，使用设备: {self.device}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def encode(self, sentence: str) -> List[float]:
        """将文本编码为向量
        
        Args:
            sentence: 输入文本
            
        Returns:
            归一化后的文本向量
        """
        with torch.no_grad():
            # 确保输入数据在正确的设备上
            inputs = self.tokenizer([sentence], padding=True, truncation=True, 
                                  max_length=512, return_tensors='pt')
            # 将所有输入张量移动到相同设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 获取向量表示
            outputs = self.model(**inputs)
            vector = outputs[1].cpu().tolist()[0]  # 确保结果返回到CPU
            vector = self._normalize(vector)
            return vector
    
    def _normalize(self, vector: List[float]) -> List[float]:
        """对向量进行L2归一化
        
        Args:
            vector: 输入向量
            
        Returns:
            归一化后的向量
        """
        ss = sum([s**2 for s in vector]) ** 0.5
        return [round(s/ss, 5) for s in vector]


class VectorIndexBuilder:
    """向量索引构建器"""
    
    def __init__(
        self, 
        encoder: Optional[VectorEncoder] = None,
        index_dir: Optional[str] = None
    ):
        """初始化向量索引构建器
        
        Args:
            encoder: 向量编码器
            index_dir: 索引存储目录
        """
        if encoder is None:
            encoder = VectorEncoder()
        
        if index_dir is None:
            index_dir = config['paths']['indexes']['vector']
            
        self.encoder = encoder
        self.index_dir = index_dir
        
        # 创建索引目录
        os.makedirs(index_dir, exist_ok=True)
    
    def build_index(self, chunks: Dict, force_rebuild: bool = False) -> None:
        """构建向量索引
        
        Args:
            chunks: 文档块字典
            force_rebuild: 是否强制重建索引
        """
        # 检查索引文件是否已存在
        index_file = os.path.join(self.index_dir, 'chunk_vector.pkl')
        if os.path.exists(index_file) and not force_rebuild:
            logger.info(f"向量索引已存在: {index_file}")
            return
        
        logger.info(f"开始构建向量索引，共 {len(chunks)} 个文档块")
        
        # 初始化
        chunkid_vector = {}
        faissid_chunkid = {}
        id_vector = []
        
        # 为每个chunk生成向量
        for i, (chunk_id, chunk) in enumerate(tqdm(chunks.items())):
            vector = self.encoder.encode(chunk.content)
            chunkid_vector[chunk_id] = vector
            faissid_chunkid[len(id_vector)] = chunk_id
            id_vector.append(vector)
        
        # 创建FAISS索引
        logger.info("创建FAISS索引")
        dimension = len(id_vector[0])  # 向量维度
        index = faiss.IndexFlatL2(dimension)
        id_vector = np.array(id_vector).astype('float32')
        index.add(id_vector)
        
        # 保存索引文件
        logger.info("保存索引文件")
        with open(os.path.join(self.index_dir, 'chunk_vector.pkl'), 'wb') as f:
            pickle.dump(index, f)
        
        with open(os.path.join(self.index_dir, 'faissid_chunkid.json'), 'w', encoding='utf-8') as f:
            json.dump(faissid_chunkid, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(self.index_dir, 'chunkid_vector.json'), 'w', encoding='utf-8') as f:
            json.dump(chunkid_vector, f, ensure_ascii=False, indent=2)
        
        logger.info("向量索引构建完成")


class VectorRetriever:
    """向量检索器"""
    
    def __init__(
        self, 
        encoder: Optional[VectorEncoder] = None,
        index_dir: Optional[str] = None
    ):
        """初始化向量检索器
        
        Args:
            encoder: 向量编码器
            index_dir: 索引目录
        """
        if encoder is None:
            encoder = VectorEncoder()
            
        if index_dir is None:
            index_dir = config['paths']['indexes']['vector']
            
        self.encoder = encoder
        self.index_dir = index_dir
        
        # 加载索引
        self._load_index()
    
    def _load_index(self) -> None:
        """加载索引文件"""
        logger.info("加载向量索引")
        
        with open(os.path.join(self.index_dir, 'chunk_vector.pkl'), 'rb') as f:
            self.index = pickle.load(f)
        
        with open(os.path.join(self.index_dir, 'faissid_chunkid.json'), 'r', encoding='utf-8') as f:
            self.faissid_chunkid = json.load(f)
        
        with open(os.path.join(self.index_dir, 'chunkid_vector.json'), 'r', encoding='utf-8') as f:
            self.chunkid_vector = json.load(f)
        
        logger.info("向量索引加载完成")
    
    def search(self, query: str, num: int = 100) -> List[int]:
        """根据查询检索相关文档
        
        Args:
            query: 查询文本
            num: 返回结果数量
            
        Returns:
            相关chunk ID列表
        """
        vector = self.encoder.encode(query)
        vector = np.array([vector]).astype('float32')
        
        D, I = self.index.search(vector, num)
        
        D = D[0]  # 距离
        I = I[0]  # 索引
        
        chunk_ids = []
        for d, i in zip(D, I):
            chunk_ids.append(int(self.faissid_chunkid[str(i)]))
        
        return chunk_ids