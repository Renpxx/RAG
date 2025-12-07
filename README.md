
# 新增：技术文档RAG问答 + 全流程评测
为满足“Building a RAG-based Q&A System for Technical Documentation with Comprehensive Evaluation”的需求，项目新增了一套面向技术文档的高阶流程。核心特性如下：

1. **多源文档摄取**：`input/` 目录可同时放置 PDF、Markdown/纯文本、CSV/JSON 文件，此外还可在 `input/urls.txt`（或任意 `.url/.urls/.link/.links` 文件）中列出需要抓取的网页，系统会自动清洗成文本。
2. **统一向量构建脚本**：运行 `python build_vectorstore.py` 即可将所有资料分块、向量化并写入 Chroma 持久化数据库，过程中会自动去重、过滤过短片段，并额外导出 chunk 清单供 BM25 稀疏检索使用。
3. **上下文召回增强**：推理流程默认开启多查询扩展、BM25+向量混合检索与 LLM 压缩重排，可通过环境变量灵活开关，以在召回率与延迟之间找到平衡。
4. **FastAPI 推理服务**：`python main.py` 会加载最新向量库并暴露 `/v1/chat/completions`，模型会根据用户问题动态检索片段、拼接 prompt 后作答，并在结尾输出 “来源: 片段X, 片段Y” 形式的引用。
5. **自动化评测**：`python evaluate_rag.py --samples 8` 会自动生成评测问答对、调用 RAG 推理并借助 LLM 进行 QA 评分，结果写入 `evaluation_results.json`，用于观察覆盖率与答案质量。

> 说明：为保持目录简洁，历史脚本（如 `apiTest.py`、`vectorSaveTest.py`、`mainMemory.py`、`mainReranker.py` 等）以及配套的 `tools/`、`test/`、`other/` 目录已被移除，下文相关描述仅作原理参考。

## 0.1 环境变量速查
| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `RAG_API_TYPE` | 选择 openai/oneapi，仅用于日志标识 | `oneapi` |
| `RAG_CHAT_BASE_URL` / `RAG_CHAT_API_KEY` / `RAG_CHAT_MODEL` | 对话模型配置 | —— |
| `RAG_EMBEDDING_BASE_URL` / `RAG_EMBEDDING_API_KEY` / `RAG_EMBEDDING_MODEL` | 向量模型配置，未显式设置时沿用对话配置 | —— |
| `RAG_VECTOR_DIR` / `RAG_COLLECTION` | Chroma 持久化目录与集合名 | `chromaDB` / `technical_docs` |
| `RAG_INPUT_DIR` | 原始资料目录 | `input` |
| `RAG_URL_MANIFEST` | 额外 URL 清单文件路径，可选 | 自动检测 `input/urls.txt` |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | 文档切分策略 | `1200` / `200` |
| `RAG_RETRIEVAL_K` | 每次检索返回的片段数量 | `5` |
| `RAG_MIN_CHUNK_CHARS` | 过滤过短片段的最小字符数 | `200` |
| `RAG_MULTI_QUERY_VARIATIONS` | 单次查询生成的额外改写数量（0 关闭） | `2` |
| `RAG_ENABLE_COMPRESSION` | 是否启用 LLM 压缩/重排序 | `true` |
| `RAG_ENABLE_BM25` | 是否开启 BM25 稀疏检索 | `true` |
| `RAG_HYBRID_SEARCH_RATIO` | 稠密/稀疏检索权重 (0~1) | `0.5` |
| `RAG_RERANK_TOP_K` | 最多保留用于重排序的候选数量 | `10` |
| `RAG_PROMPT_TEMPLATE` | Prompt 模板路径 | `prompt_template.txt` |
| `RAG_EVAL_SAMPLES` / `RAG_EVAL_OUTPUT` | 评测问答数量及输出文件 | `5` / `evaluation_results.json` |
| `LANGCHAIN_API_KEY` / `LANGCHAIN_TRACING_V2` | 启用 LangSmith 时使用 | 空 / `false` |

> 提示：运行脚本前可在 shell 中 `export RAG_CHAT_API_KEY=xxx`，或使用 `.env`/系统密钥管理方案，避免将密钥写入代码。

## 0.2 流程总览
1. **准备资料**：把 PDF、TXT、MD、CSV、JSON 放入 `input/`；若需采集网页，在 `input/urls.txt` 中逐行写入 `https://example.com/docs` 即可。
2. **灌库**：执行 `python build_vectorstore.py`（默认会清空旧库，可通过 `--no-reset` 追加）。脚本会自动调用 `DocumentIngestor` 完成解析 → 分块 → 向量化。
3. **启动服务**：执行 `python main.py`，FastAPI 会初始化模型、加载 Chroma，并暴露 OpenAI 风格接口。可通过 `python test_local_api.py --question "..."` 本地验证接口，也可使用任意 OpenAI SDK。
4. **综合评测**：执行 `python evaluate_rag.py --samples 10` 可生成评测报告，内容包含自动构造的问题、标准答案、RAG 回答、LLM 评分与推理说明，便于快速定位薄弱环节。

下文原有章节仍保留，可继续参考原视频/脚本完成更细粒度的学习。

# 1、基础概念
## 1.1 RAG定义及技术方案架构
### （1）RAG定义
RAG:Retrieval Augmented Generation(检索增强生成):通过使用检索的方法来增强生成模型的能力       
核心思想:人找知识，会查资料；LLM找知识，会查向量数据库        
主要目标:补充LLM固有局限性，LLM的知识不是实时的，LLM可能不知道私有领域业务知识          
场景类比:可以把RAG的过程想象成为开卷考试。让LLM先翻书查找相关答案，再回答问题              
### （2）技术方案架构
离线步骤:文档加载->文档切分->向量化->灌入向量数据库           
在线步骤:获取用户问题->用户问题向量化->检索向量数据库->将检索结果和用户问题填入prompt模版->用最终的prompt调用LLM->由LLM生成回复             
### （3）几个关键概念：
向量数据库的意义是快速的检索             
向量数据库本身不生成向量，向量是由Embedding模型产生的             
向量数据库与传统的关系型数据库是互补的，不是替代关系，在实际应用中根据实际需求经常同时使用               

## 1.2 LangChain
### （1）LangChain定义
LangChain是一个用于开发由大型语言模型(LLM)驱动的应用程序的框架，官方网址：https://python.langchain.com/v0.2/docs/introduction/          
### （2）LCEL定义
LCEL(LangChain Expression Language),原来叫chain，是一种申明式语言，可轻松组合不同的调用顺序构成chain            
其特点包括流支持、异步支持、优化的并行执行、重试和回退、访问中间结果、输入和输出模式、无缝LangSmith跟踪集成、无缝LangServe部署集成            
### （3）LangSmith
LangSmith是一个用于构建生产级LLM应用程序的平台。通过它，您可以密切监控和评估您的应用程序，官方网址：https://docs.smith.langchain.com/          

## 1.3 Chroma
向量数据库，专门为向量检索设计的中间件          


# 2、前期准备工作
## 2.1 anaconda、pycharm 安装   
anaconda:提供python虚拟环境，官网下载对应系统版本的安装包安装即可                             
pycharm:提供集成开发环境，官网下载社区版本安装包安装即可                           
可参考如下视频进行安装【大模型应用开发基础】集成开发环境搭建Anaconda+PyCharm                                        
https://www.bilibili.com/video/BV1q9HxeEEtT/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                                            

## 2.2 OneAPI安装、部署、创建渠道和令牌 
### （1）OneAPI是什么
官方介绍：是OpenAI接口的管理、分发系统             
支持 Azure、Anthropic Claude、Google PaLM 2 & Gemini、智谱 ChatGLM、百度文心一言、讯飞星火认知、阿里通义千问、360 智脑以及腾讯混元             
### (2)安装、部署
使用官方提供的release软件包进行安装部署 ，详情参考如下链接中的手动部署：                  
https://github.com/songquanpeng/one-api                  
下载OneAPI可执行文件one-api并上传到服务器中然后，执行如下命令后台运行             
nohup ./one-api --port 3000 --log-dir ./logs > output.log 2>&1 &               
运行成功后，浏览器打开如下地址进入one-api页面，默认账号密码为：root 123456                 
http://IP:3000/              
### (3)创建渠道和令牌
创建渠道：大模型类型(通义千问)、APIKey(通义千问申请的真实有效的APIKey)             
创建令牌：创建OneAPI的APIKey，后续代码中直接调用此APIKey              

## 2.3 openai使用方案            
国内无法直接访问，可以使用代理的方式，具体代理方案自己选择                   
可以参考这期视频:                   
【GraphRAG最新版本0.3.0对比实战评测】使用gpt-4o-mini和qwen-plus分别构建近2万字文本知识索引+本地/全局检索对比测试                   
https://www.bilibili.com/video/BV1maHxeYEB1/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                     
https://youtu.be/iXfsJrXCEwA                                               

## 2.4 langsmith配置         
直接在langsmith官网设置页中申请APIKey(这里可以选择使用也可以不使用)             
https://smith.langchain.com/o/93f0b841-d320-5df9-a9a0-25be027a4c09/settings                  

