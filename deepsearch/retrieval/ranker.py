import os
import yaml
import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class ReRanker:
    """检索结果重排序类"""
    
    def __init__(self, model_path: Optional[str] = None):
        """初始化重排序模型
        
        Args:
            model_path: 模型路径
        """
        if model_path is None:
            model_path = config['models']['reranker']['path']
            
        logger.info(f"加载重排序模型: {model_path}")
        
        # 导入必要的库
        from modelscope import AutoModelForSequenceClassification, AutoTokenizer
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        logger.info(f"重排序模型加载完成，使用设备: {self.device}")
    
    def compute_scores(self, data: List[List[str]]) -> torch.Tensor:
        """计算查询-文档对的相关性得分
        
        Args:
            data: 查询-文档对列表，每对为[query, doc]
            
        Returns:
            相关性得分张量
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                data, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=100
            ).to(self.device)
            
            scores = torch.sigmoid(self.model(**inputs, return_dict=True).logits.view(-1).float())
            
            return scores


class MMRReranker:
    """最大边际相关性重排序"""
    
    @staticmethod
    def rerank(
        items: List[int], 
        scores: List[float], 
        vectors: List[List[float]], 
        top_k: int = 20, 
        lambda_param: float = 0.6
    ) -> List[int]:
        """使用MMR算法重排序结果
        
        Args:
            items: 项目ID列表
            scores: 相关性得分列表
            vectors: 向量表示列表
            top_k: 返回结果数量
            lambda_param: 多样性与相关性平衡参数
            
        Returns:
            重排序后的ID列表
        """
        # 将列表转为numpy数组以提高效率
        vectors = np.array(vectors)
        items_scores = list(zip(items, scores))
        
        selected = []
        selected_indices = []
        
        # 保存已选择向量
        selected_vectors = []
        
        # 重排序过程
        for _ in range(min(top_k, len(items_scores))):
            best_score = -float('inf')
            best_idx = -1
            
            for i, (item, relevance) in enumerate(items_scores):
                if i in selected_indices:
                    continue
                    
                # 计算多样性成分
                if not selected_vectors:
                    diversity = 1.0
                else:
                    # 计算与已选择向量的最大余弦相似度
                    similarities = np.dot(vectors[i].reshape(1, -1), np.array(selected_vectors).T)
                    max_similarity = np.max(similarities) if similarities.size > 0 else 0
                    diversity = 1.0 - max_similarity
                
                # 计算MMR得分
                mmr_score = lambda_param * relevance + (1.0 - lambda_param) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx == -1:
                break
                
            # 添加选中的项目
            selected.append(items_scores[best_idx][0])
            selected_indices.append(best_idx)
            selected_vectors.append(vectors[best_idx])
        
        return selected


class ClusterDiversifier:
    """聚类多样化器"""
    
    @staticmethod
    def diversify(
        queries: List[str], 
        vectors: Optional[List[List[float]]] = None, 
        n_clusters: int = 5
    ) -> List[str]:
        """使用KMeans聚类获取多样化的查询
        
        Args:
            queries: 查询列表
            vectors: 查询向量表示（如果没有则使用查询列表索引）
            n_clusters: 聚类数量
            
        Returns:
            多样化的查询列表
        """
        if not queries:
            return []
            
        if vectors is None:
            # 如果没有提供向量，则使用查询列表的索引作为简单向量化表示
            vectors = [[i] for i in range(len(queries))]
        
        vectors = np.array(vectors)
        
        # 聚类数不能超过样本数
        n_clusters = min(n_clusters, len(queries))
        
        # 使用KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(vectors)
        
        # 获取聚类标签
        labels = kmeans.labels_
        
        # 获取聚类中心
        centroids = kmeans.cluster_centers_
        
        # 从每个簇中选择距离中心最近的点
        result = []
        for i in range(n_clusters):
            cluster_points = [j for j, label in enumerate(labels) if label == i]
            
            if not cluster_points:
                continue
                
            # 计算每个点到簇中心的距离
            distances = []
            for idx in cluster_points:
                dist = np.sqrt(np.sum((vectors[idx] - centroids[i]) ** 2))
                distances.append((idx, dist))
            
            # 选择最近的点
            closest_idx = min(distances, key=lambda x: x[1])[0]
            result.append(queries[closest_idx])
        
        return result