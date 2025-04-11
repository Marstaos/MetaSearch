# MetaSearch系统
这是一个开源deepsearch实现深度迭代检索RAG的教学项目！

MetaSearch是一个基于深度迭代检索的RAG（检索增强生成）系统，通过多轮检索不断深入探索相关信息，结合多种检索方式，实现更加全面和深入的知识探索。
## 项目特点

1. **教学导向**：代码结构清晰，注释完整，非常适合用来学习现代、先进的RAG系统的开发实践
2. **规范实现**：本项目严格遵循大模型项目开发的最佳实践：
   - 模块化设计
   - 清晰的配置文件管理
   - 完善的日志系统
   - 类型注解和文档字符串
3. **生产级质量**：虽然是教学项目，但代码很规范，有助于培养良好的代码习惯
4. **可扩展架构**：各组件松耦合，便于二次开发和功能扩展
## Features

- **深度迭代检索**：不同于传统RAG的单次检索，通过多轮检索不断深入探索相关信息
- **多模态检索融合**：结合向量检索、关键词检索和知识图谱检索，覆盖更广泛的相关内容
- **智能查询扩展**：使用大模型动态生成子查询，实现知识探索的广度和深度
- **基于信息增长的自适应搜索**：根据新发现信息的比例，动态决定是否继续搜索
- **多样性重排序**：使用MMR算法在相关性和多样性之间取得平衡

## 环境搭建

### 1. 安装Conda

如果您尚未安装Conda，请先从[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装Miniconda。

### 2. 创建Conda环境

```bash
# 创建名为metasearch的新环境，使用Python 3.10
conda create -n metasearch python=3.10
# 激活环境
conda activate metasearch
```

### 3. 克隆项目

```bash
# 克隆项目仓库
git clone https://github.com/marstaos/MetaSearch.git
# 进入项目目录
cd MetaSearch
```

### 4. 安装依赖

```bash
# 安装项目依赖
pip install -r requirements.txt
```

如需使用GPU加速，请确保已安装适合您CUDA版本的PyTorch：

```bash
# 对于CUDA 11.8
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## 下载模型

MetaSearch系统需要以下预训练模型：

1. 向量编码模型（用于文本向量化）
2. 重排序模型（用于结果重排序）
3. 大语言模型（用于生成回答）

您可以使用我们提供的脚本自动下载这些模型：

```bash
# （建议选项）跳过下载Qwen大模型（仅下载embedding和reranker）
python scripts/download_models.py --all --skip_qwen

# 或者单独下载特定模型
python scripts/download_models.py --embedding
python scripts/download_models.py --reranker
python scripts/download_models.py --llm

# 全量下载
python scripts/download_models.py --all

# 不使用镜像站下载（默认使用国内镜像加速）
python scripts/download_models.py --all --no_mirror
```

## 配置系统

编辑`config/config.yaml`文件，根据您的需求配置系统：

```yaml
# 模型配置
models:
  embedding:
    name: bert-base
    path: models/embedding/bce-embedding-base_v1
  reranker:
    name: bge-reranker
    path: models/reranker/bge-reranker-v2-m3
  llm:
    name: qwen
    path: models/llm/qwen2.5-14B
    api_endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: sk-xxxx  # 替换为您的API密钥
    model_name: qwen2.5-14b-instruct-1m
```

## 使用方法

### 1. 处理文档

首先，处理文档并创建索引：

```bash
# 处理文档
python scripts/process_documents.py --file data/raw/your_document.txt

# 构建索引
python scripts/build_indexes.py --chunks data/processed/index_chunk.pkl
```
这两行命令的效果是：
1. 文档处理 ：
   - 将原始文档分割成固定大小的文本块(chunks)
   - 为每个chunk生成唯一ID和摘要
   - 保存处理结果到 data/processed/index_chunk.pkl
2. 索引构建 ：

   - 基于处理后的chunks构建三种索引：
     - 向量索引（FAISS格式）
     - TF-IDF索引（关键词检索）
     - 知识图谱索引（实体关系图）
   - 所有索引文件保存在 indexes/ 目录下

### 2. 运行系统

您可以在交互模式下运行系统：

```bash
# 交互模式
python app.py --interactive
```

或者直接提供查询：

```bash
# 单次查询
python app.py --query "介绍下明朝的内阁首辅"
```

## 高级配置

### 深度搜索参数

您可以在`config/config.yaml`中调整深度搜索参数：

```yaml
# 深度搜索参数
deepsearch:
  max_iterations: 5        # 最大迭代次数
  growth_rate_threshold: 0.1  # 信息增长率阈值
  extend_query_num: 10     # 每轮生成的子查询数量
```

### 检索参数

```yaml
# 检索参数
retrieval:
  faiss_candidates: 100    # 向量检索候选数量
  tfidf_candidates: 100    # TF-IDF检索候选数量
  total_candidates: 50     # 最终返回的候选数量
```

### 处理参数

```yaml
# 处理参数
processing:
  chunk_size: 512          # 文档块大小
  overlap_size: 30         # 重叠大小
```

## 项目结构

```
MetaSearch/
├── config/           # 配置文件
├── data/             # 数据文件
│   ├── raw/          # 原始文档
│   └── processed/    # 处理后的文档
├── indexes/          # 索引文件
│   ├── vector/       # 向量索引
│   ├── tfidf/        # TF-IDF索引
│   └── kg/           # 知识图谱索引
├── models/           # 预训练模型
│   ├── embedding/    # 向量编码模型
│   ├── reranker/     # 重排序模型
│   └── llm/          # 大语言模型
├── deepsearch/       # 核心代码
│   ├── indexing/     # 索引相关代码
│   ├── llm/          # LLM接口
│   ├── preprocessing/# 预处理代码
│   ├── rag/          # RAG实现
│   ├── retrieval/    # 检索相关代码
│   └── utils/        # 工具函数
├── scripts/          # 脚本工具
├── app.py            # 应用入口
├── requirements.txt  # 依赖列表
└── README.md         # 项目说明
```

## 贡献

欢迎提交问题和拉取请求！