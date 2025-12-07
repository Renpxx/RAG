"""
Shared helpers for building the RAG pipeline components.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_config import RAGSettings


logger = logging.getLogger(__name__)


def create_llm_clients(settings: RAGSettings) -> Tuple[ChatOpenAI, OpenAIEmbeddings]:
    """Instantiate the chat and embedding clients based on the selected provider."""
    if not settings.chat_api_key:
        raise ValueError("Missing RAG_CHAT_API_KEY environment variable.")
    if not settings.embedding_api_key:
        raise ValueError("Missing RAG_EMBEDDING_API_KEY environment variable.")

    model = ChatOpenAI(
        base_url=settings.chat_base_url or None,
        api_key=settings.chat_api_key,
        model=settings.chat_model,
    )
    embeddings = OpenAIEmbeddings(
        base_url=settings.embedding_base_url or None,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
    )
    return model, embeddings


def connect_vectorstore(
    settings: RAGSettings, embeddings: OpenAIEmbeddings
) -> Chroma:
    """Connect to an existing persistent Chroma collection."""
    persist_dir = Path(settings.persist_directory)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store directory '{persist_dir}' not found. "
            "Please run build_vectorstore.py first."
        )
    return Chroma(
        persist_directory=str(persist_dir),
        collection_name=settings.collection_name,
        embedding_function=embeddings,
    )


def persist_vectorstore(
    documents: List[Document],
    settings: RAGSettings,
    embeddings: OpenAIEmbeddings,
    reset: bool,
) -> Chroma:
    """Create (or refresh) the persistent Chroma store from prepared documents."""
    persist_path = Path(settings.persist_directory)
    if reset and persist_path.exists():
        shutil.rmtree(persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=str(persist_path),
    )
    persist_method = getattr(vectorstore, "persist", None)
    if callable(persist_method):
        persist_method()
    else:
        client = getattr(vectorstore, "client", None) or getattr(vectorstore, "_client", None)
        flush = getattr(client, "persist", None)
        if callable(flush):
            flush()
    logger.info(
        "Persisted %s document chunk(s) into collection '%s'.",
        len(documents),
        settings.collection_name,
    )
    return vectorstore


def load_prompt_template(path: str) -> ChatPromptTemplate:
    prompt_file = Path(path)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template '{path}' not found.")
    template = prompt_file.read_text(encoding="utf-8")
    return ChatPromptTemplate.from_template(template)


def _format_documents(docs: List[Document]) -> str:
    sections = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        sections.append(f"[片段{idx}] 来源: {source}\n{doc.page_content.strip()}")
    return "\n\n".join(sections)


def _load_chunk_manifest(settings: RAGSettings) -> Optional[List[Document]]:
    chunk_manifest = (
        Path(settings.persist_directory) / f"{settings.collection_name}_chunks.json"
    )
    if not chunk_manifest.exists():
        logger.warning("Chunk manifest %s not found; BM25 retriever disabled.", chunk_manifest)
        return None
    try:
        data = json.loads(chunk_manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to parse chunk manifest %s: %s", chunk_manifest, exc)
        return None
    documents = [
        Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
        for item in data
        if item.get("page_content")
    ]
    return documents


def _maybe_build_bm25_retriever(settings: RAGSettings) -> Optional[BM25Retriever]:
    documents = _load_chunk_manifest(settings)
    if not documents:
        return None
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = max(settings.retriever_k, settings.rerank_top_k)
    return retriever


def _apply_multi_query(
    retriever, model: ChatOpenAI, variations: int
):
    if variations <= 0:
        return retriever
    mq_prompt = PromptTemplate.from_template(
        (
            "You are a search assistant helping users retrieve technical documentation.\n"
            "Generate {num_queries} diverse alternative questions that stay semantically "
            "close to the original. Return each query on a new line.\n"
            "Original question: {question}"
        )
    ).partial(num_queries=str(variations))
    return MultiQueryRetriever.from_llm(
        llm=model,
        retriever=retriever,
        prompt=mq_prompt,
    )


def _apply_compression(retriever, model: ChatOpenAI, enabled: bool):
    if not enabled:
        return retriever
    compressor = LLMChainExtractor.from_llm(model)
    return ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=compressor,
    )


def build_rag_chain(
    model: ChatOpenAI, vectorstore: Chroma, settings: RAGSettings
):
    """
    Build the runnable RAG chain (retriever -> prompt -> LLM).
    Returns both the compiled chain and the underlying prompt template.
    """

    prompt = load_prompt_template(settings.prompt_template_path)
    initial_k = settings.retriever_k * (2 if settings.enable_compression else 1)
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": initial_k},
        search_type="similarity",
    )
    retriever = base_retriever
    if settings.enable_bm25:
        bm25 = _maybe_build_bm25_retriever(settings)
        if bm25:
            ratio = min(max(settings.hybrid_search_ratio, 0.0), 1.0)
            retriever = EnsembleRetriever(
                retrievers=[base_retriever, bm25],
                weights=[1.0 - ratio, ratio],
            )
    retriever = _apply_multi_query(
        retriever, model, settings.multi_query_variations
    )
    retriever = _apply_compression(
        retriever, model, settings.enable_compression
    )

    def _format_with_limit(docs):
        rerank_cap = settings.rerank_top_k if settings.rerank_top_k > 0 else len(docs)
        trimmed = docs[:rerank_cap]
        final_cap = settings.retriever_k if settings.retriever_k > 0 else len(trimmed)
        return _format_documents(trimmed[:final_cap])

    formatter = RunnableLambda(_format_with_limit)
    chain = (
        {
            "query": RunnablePassthrough(),
            "context": retriever | formatter,
        }
        | prompt
        | model
    )
    return chain, prompt
