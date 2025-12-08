
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
