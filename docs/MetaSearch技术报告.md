# MetaSearch系统技术报告

## 1. 系统概述

MetaSearch是一个基于深度迭代检索的RAG（检索增强生成）系统，通过多轮检索不断深入探索相关信息，结合多种检索方式，实现更加全面和深入的知识探索。

Features:
  **深度迭代检索**：不同于传统RAG的单次检索，通过多轮检索不断深入探索相关信息
  **多模态检索融合**：结合向量检索、关键词检索和知识图谱检索，覆盖更广泛的相关内容
  **智能查询扩展**：使用大模型动态生成子查询，实现知识探索的广度和深度
  **基于信息增长的自适应搜索**：根据新发现信息的比例，动态决定是否继续搜索
  **多样性重排序**：使用MMR算法在相关性和多样性之间取得平衡

## 2. 系统架构

MetaSearch系统由以下几个核心模块组成：

1. **文档处理模块**：负责将原始文档分割成适当大小的文本块（chunks）
2. **索引构建模块**：构建向量索引、TF-IDF索引和知识图谱索引
3. **检索模块**：结合多种检索方式，获取与查询相关的文档
4. **查询扩展模块**：基于已检索到的信息，生成新的子查询
5. **深度RAG模块**：协调整个迭代检索过程，生成最终回答

## 3. 核心技术原理

### 3.1 文档处理

文档处理模块将原始文档分割成固定大小的文本块（chunks），每个chunk包含以下信息：
- 内容（content）
- 唯一ID（chunk_id）
- 父块（parent）：包含更广泛上下文的块
- 摘要（abstract）：使用LLM生成的内容摘要

配置文件中定义了chunk的大小和重叠部分的大小：

```yaml
processing:
  chunk_size: 512
  overlap_size: 30
```

### 3.2 多模态索引构建

系统支持三种类型的索引：

1. **向量索引**：使用预训练的语言模型（如BCE-Embedding）将文本转换为向量，并使用FAISS构建高效的向量检索索引
2. **TF-IDF索引**：基于词频-逆文档频率，适合关键词匹配
3. **知识图谱索引**：提取文本中的实体和关系，构建知识图谱

### 3.3 查询扩展机制

查询扩展是MetaSearch系统的亮点之一。它的步骤如下：

1. 对每个回答使用LLM提取关键搜索词
2. 计算这些搜索词与原始查询的相关性得分
3. 将所有候选子查询放入同一个池子中
4. 按得分降序排序，选择得分最高的几个
5. 将原始查询与选出的子查询合并，生成新的查询

这种设计从全局角度选择最有价值的子查询，而不是为每个回答单独生成固定数量的子查询。

### 3.4 深度迭代检索

深度迭代检索是整个系统的核心流程，它通过以下步骤工作：

1. 从用户的原始查询开始
2. 对每个查询执行标准RAG，获取回答
3. 计算信息增长率（新发现的chunk数量与已有chunk数量的比值）
4. 如果信息增长率低于阈值，结束迭代
5. 否则，使用查询扩展器生成新的子查询，进入下一轮迭代
6. 最终，使用所有收集到的知识生成综合回答

## 4. 流程示例

让我们通过一个具体例子来说明整个系统的工作流程：

假设用户输入查询："明朝的内阁制度"

### 第一轮迭代

1. **初始化**：
   ```
   sub_queries = ["明朝的内阁制度"]  # 只有一个原始查询
   knowledge = []  # 空知识库
   exist_ids = set()  # 空ID集合
   ```

2. **执行标准RAG**：
   - 处理查询"明朝的内阁制度"
   - 假设获得回答: "明朝内阁制度起源于永乐年间..."
   - 假设获得文档IDs: [101, 102, 103, 104, 105]
   - 添加到知识库: knowledge = ["明朝内阁制度起源于永乐年间..."]
   - 记录回答: response_list = ["明朝内阁制度起源于永乐年间..."]
   - 新发现的IDs: new_ids = {101, 102, 103, 104, 105}

3. **计算信息增长率**：
   - info_growth_rate = 5 / 1 = 5.0 (高于阈值0.1)

4. **扩展查询**：
   - 使用查询扩展器从回答中提取关键词
   - 假设生成的候选子查询有: ["内阁首辅", "内阁权力", "张居正改革", "明朝政治体制", "内阁与皇权"]
   - 计算每个子查询与原始查询的相关性得分
   - 选择得分最高的3个: ["明朝内阁首辅", "明朝内阁制度演变", "明朝内阁与皇权关系"]

### 第二轮迭代

1. **执行标准RAG**：
   - 处理子查询1: "明朝内阁首辅"
     - 获得回答: "明朝内阁首辅是..."
     - 获得文档IDs: [201, 202, 103, 104]
   - 处理子查询2: "明朝内阁制度演变"
     - 获得回答: "明朝内阁制度经历了..."
     - 获得文档IDs: [301, 302, 303]
   - 处理子查询3: "明朝内阁与皇权关系"
     - 获得回答: "明朝内阁与皇权..."
     - 获得文档IDs: [401, 402, 103]
   
   - 更新知识库: knowledge = ["明朝内阁制度起源于...", "明朝内阁首辅是...", "明朝内阁制度经历了...", "明朝内阁与皇权..."]
   - 新发现的IDs: new_ids = {201, 202, 301, 302, 303, 401, 402} (排除已有的103, 104)

2. **计算信息增长率**：
   - info_growth_rate = 7 / 5 = 1.4 (高于阈值0.1)

3. **扩展查询**：
   - 对所有第二轮的回答提取关键词
   - 从所有候选子查询中选择得分最高的3个
   - 假设新的子查询为: ["张居正改革与内阁权力", "明朝内阁与六部的关系", "明朝后期内阁的衰落"]

### 第三轮迭代

以此类推，系统会继续迭代，直到信息增长率低于阈值或达到最大迭代次数。

### 最终回答生成

1. **格式化知识**：
   - 为每个知识点添加来源查询和相关度评分
   - 使用重排序模型计算每个知识点与原始查询的相关性

2. **生成最终回答**：
   - 构建提示词，包含原始问题和所有收集到的知识
   - 使用LLM生成综合回答，包括核心答案、详细阐述和个人见解

## 5. 关键代码解析

### 5.1 查询扩展器

查询扩展器负责生成新的子查询：

```python:d:\Playground\MetaSearch\deepsearch\rag\query_expander.py
def extend_query(
    self, 
    queries: List[str], 
    responses: List[str], 
    num: int = 10
) -> List[str]:
    """扩展查询集合"""
    logger.info(f"开始扩展查询，原始查询数: {len(queries)}")
    
    all_queries_scores = []
    
    for query, response in zip(queries, responses):
        if response is None:
            continue
            
        # 生成子查询及其得分
        queries_scores = self.generate_subquery(query, response, num)
        all_queries_scores.extend(queries_scores)
    
    # 按得分降序排序
    all_queries_scores = sorted(all_queries_scores, key=lambda s: s[2], reverse=True)
    
    # 选择前num个进行合并
    next_queries = [
        self.combine_query(s[0], s[1]) 
        for s in all_queries_scores[:num]
    ]
    
    logger.info(f"查询扩展完成，新生成查询数: {len(next_queries)}")
    return next_queries
```

### 5.2 深度RAG流程

深度RAG模块协调整个迭代检索过程：

```python:d:\Playground\MetaSearch\deepsearch\rag\deep_rag.py
def answer(self, query: str) -> str:
    """深度RAG问答流程"""
    # 初始化
    sub_queries = [query]  # 初始子查询就是原始查询
    knowledge = []  # 存储收集到的知识
    exist_ids = set()  # 已检索到的chunk ID集合
    
    # 迭代深度搜索
    for i in range(self.max_iterations):
        new_ids = set()
        response_list = []
        
        # 对每个子查询执行标准RAG
        for sub_query in sub_queries:
            response, ids = self._standard_rag(sub_query)
            knowledge.append(response)
            new_ids.update([s for s in ids if s not in exist_ids])
            response_list.append(response)
        
        # 计算信息增长率
        info_growth_rate = len(new_ids) / max(len(exist_ids), 1)
        
        # 更新已发现的chunk ID集合
        exist_ids.update(new_ids)
        
        # 如果信息增长率低于阈值，结束迭代
        if info_growth_rate < self.growth_rate_threshold:
            break
        
        # 如果不是最后一轮，扩展查询
        if i < self.max_iterations - 1:
            sub_queries = self.query_expander.extend_query(
                sub_queries, 
                response_list,
                self.extend_query_num
            )
    
    # 生成最终回答
    # ...
```

## 6. 系统配置

系统的主要配置参数在`config/config.yaml`文件中定义：

```yaml:d:\Playground\MetaSearch\config\config.yaml
# 深度搜索参数
deepsearch:
  max_iterations: 5
  growth_rate_threshold: 0.1
  extend_query_num: 10
```

- `max_iterations`: 最大迭代次数
- `growth_rate_threshold`: 信息增长率阈值，低于此值时停止迭代
- `extend_query_num`: 每轮生成的子查询数量