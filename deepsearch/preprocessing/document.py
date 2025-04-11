import re
import json
import os
import pickle
from typing import List, Dict, Optional
import yaml
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class Chunk:
    """文档块类，表示文档的一个片段"""
    
    def __init__(
        self, 
        content: str, 
        chunk_id: int, 
        parent: Optional[str] = None, 
        abstract: Optional[str] = None, 
        id_list: Optional[List[int]] = None
    ):
        """初始化文档块"""
        self.content = content
        self.chunk_id = chunk_id
        self.parent = parent
        self.abstract = abstract
        self.id_list = id_list if id_list is not None else []
    
    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id}, len={len(self.content)}, abstract={self.abstract})"


def split_sentences(text: str) -> List[str]:
    """将文本分割成句子"""
    pattern = r'([.!?。！？])'
    parts = re.split(pattern, text)
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            sentences.append(parts[i] + parts[i + 1])
            i += 2
        else:
            if parts[i].strip():
                sentences.append(parts[i])
            i += 1
    return sentences


def get_parent_chunk(
    chunk_info: Dict[int, Chunk], 
    id_data: Dict[int, str], 
    width: int = 2
) -> Dict[int, Chunk]:
    """为每个chunk构建父块（包含更广泛上下文的块）"""
    chunk_num = len(chunk_info)
    chunks = list(chunk_info.values())
    chunks.sort(key=lambda x: x.chunk_id)  # 确保按ID排序
    
    for i in range(width, chunk_num - width):
        # 获取周围chunks中包含的所有句子ID
        ids = list(set([
            k for j in range(i - width, i + width + 1) 
            for k in chunks[j].id_list
        ]))
        ids.sort()  # 按顺序排列ID
        
        # 构建父块内容
        parent = ""
        for id in ids:
            parent += id_data[id]
        
        # 更新chunk的parent字段
        chunks[i].parent = parent
    
    # 转回字典格式
    return {chunk.chunk_id: chunk for chunk in chunks}


def process_document(
    file_path: str, 
    output_dir: Optional[str] = None,
    chunk_size: Optional[int] = None,
    overlap_size: Optional[int] = None,
    get_abstract_func = None,
    save_interval: int = 100  # 每处理多少个chunk保存一次中间结果
) -> Dict[int, Chunk]:
    """处理文档，分割成chunks"""
    # 使用配置文件中的默认值
    if output_dir is None:
        output_dir = config['paths']['data']['processed']
    if chunk_size is None:
        chunk_size = config['processing']['chunk_size']
    if overlap_size is None:
        overlap_size = config['processing']['overlap_size']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否有中间结果可以恢复
    temp_file = os.path.join(output_dir, 'temp_chunk.pkl')
    if os.path.exists(temp_file):
        try:
            with open(temp_file, 'rb') as f:
                temp_data = pickle.load(f)
                chunk_info = temp_data.get('chunk_info', {})
                last_id = temp_data.get('last_id', 0)
                chunk_id = temp_data.get('chunk_id', 0)
                logger.info(f"从中间结果恢复，已处理到ID: {last_id}，已生成 {len(chunk_info)} 个chunks")
        except Exception as e:
            logger.error(f"恢复中间结果失败: {e}")
            chunk_info = {}
            last_id = 0
            chunk_id = 0
    else:
        chunk_info = {}
        last_id = 0
        chunk_id = 0
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 过滤空行
    data = [line.strip() for line in lines if len(line.strip()) > 0]
    
    # 分句
    sentences = [ss for s in data for ss in split_sentences(s)]
    
    # 构建句子ID字典
    id_data = {i: s for i, s in enumerate(sentences)}
    
    # 保存句子ID字典
    with open(os.path.join(output_dir, 'id_data.json'), 'w', encoding='utf-8') as f:
        json.dump(id_data, f, ensure_ascii=False, indent=2)
    
    # 构建chunks
    para = ""
    id_list = []
    id = last_id
    
    logger.info(f"开始处理文档: {file_path}")
    logger.info(f"总句子数: {len(sentences)}")
    
    # 记录已处理的最后一个ID，防止重复处理
    last_processed_id = last_id - 1
    
    try:
        while id < len(id_data):
            # 防止重复处理同一个ID
            if id <= last_processed_id:
                id = last_processed_id + 1
                if id >= len(id_data):
                    break
            
            para += id_data[id]
            id_list.append(id)
            last_processed_id = id
            id += 1
            
            if len(para) >= chunk_size or id >= len(id_data):
                if para:  # 确保有内容
                    # 获取摘要
                    abstract = "默认摘要"
                    if get_abstract_func:
                        try:
                            abstract = get_abstract_func(f"{para}。提取上述句子的简短摘要，不超过50字")
                            logger.info(f"Chunk {chunk_id} 摘要: {abstract}")
                        except Exception as e:
                            logger.error(f"获取摘要失败: {e}")
                    
                    # 创建chunk
                    chunk = Chunk(para, chunk_id, None, abstract, id_list.copy())
                    chunk_info[chunk_id] = chunk
                    
                    # 定期保存中间结果
                    if chunk_id % save_interval == 0 and chunk_id > 0:
                        temp_data = {
                            'chunk_info': chunk_info,
                            'last_id': last_processed_id,
                            'chunk_id': chunk_id
                        }
                        with open(temp_file, 'wb') as f:
                            pickle.dump(temp_data, f)
                        logger.info(f"已保存中间结果，当前处理到ID: {last_processed_id}，已生成 {len(chunk_info)} 个chunks")
                    
                    # 回退指定数量的token，确保chunks之间有重叠
                    token = ""
                    chunk_id += 1
                    
                    # 修改回退逻辑，确保不会无限循环
                    if id_list and id < len(id_data):
                        back_id = None
                        for i in reversed(id_list[:-1]):  # 排除最后一个ID，防止原地踏步
                            token += id_data[i]
                            if len(token) >= overlap_size:
                                back_id = i
                                break
                        
                        if back_id is not None and back_id > last_processed_id - len(id_list):
                            id = back_id  # 只有当回退不会导致重复处理大量内容时才回退
                    
                    para = ""
                    id_list = []
        
        # 构建父块
        chunk_info = get_parent_chunk(chunk_info, id_data)
        
        # 保存最终结果
        with open(os.path.join(output_dir, 'index_chunk.pkl'), 'wb') as f:
            pickle.dump(chunk_info, f)
        
        # 处理完成后删除临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        logger.info(f"文档处理完成，共生成 {len(chunk_info)} 个chunks")
        return chunk_info
    
    except Exception as e:
        # 发生异常时保存中间结果
        logger.error(f"处理文档时发生错误: {e}")
        temp_data = {
            'chunk_info': chunk_info,
            'last_id': last_processed_id,
            'chunk_id': chunk_id
        }
        with open(temp_file, 'wb') as f:
            pickle.dump(temp_data, f)
        logger.info(f"已保存中间结果，当前处理到ID: {last_processed_id}，已生成 {len(chunk_info)} 个chunks")
        raise