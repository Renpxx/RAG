# -*- coding: utf-8 -*-
"""
FastAPI entry-point for the technical-document RAG service.

The server expects that `build_vectorstore.py` has been executed beforehand
so that the Chroma collection contains the processed technical documentation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_config import RAGSettings, load_settings
from rag_pipeline import build_rag_chain, connect_vectorstore, create_llm_clients


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None


settings: Optional[RAGSettings] = None
model = None
embeddings = None
vectorstore = None
prompt = None
chain = None


def format_response(response: str) -> str:
    """
    Improve readability by normalizing paragraphs and keeping fenced code blocks.
    """
    paragraphs = [block.strip() for block in response.split("\n\n") if block.strip()]
    formatted = []
    for block in paragraphs:
        if "```" in block:
            formatted.append(block)
        else:
            formatted.append(block.replace(". ", ".\n"))
    return "\n\n".join(formatted)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings, model, embeddings, vectorstore, prompt, chain
    try:
        settings = load_settings()
        if settings.langsmith_api_key:
            # Defer to env variables consumed by LangSmith.
            import os

            os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
            os.environ["LANGCHAIN_TRACING_V2"] = (
                "true" if settings.langsmith_tracing else "false"
            )

        model, embeddings = create_llm_clients(settings)
        vectorstore = connect_vectorstore(settings, embeddings)
        chain, prompt = build_rag_chain(model, vectorstore, settings)
        logger.info(
            "Initialized RAG service with collection '%s'.",
            settings.collection_name,
        )
        yield
    except Exception as exc:  # pragma: no cover - initialization errors need surfacing
        logger.exception("Failed to initialize the RAG service: %s", exc)
        raise
    finally:
        logger.info("Shutting down RAG service.")


app = FastAPI(lifespan=lifespan)


def _chain_invoke(query: str) -> str:
    result = chain.invoke(query)
    if hasattr(result, "content"):
        return str(result.content)
    if isinstance(result, dict) and "content" in result:
        return str(result["content"])
    return str(result)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not all([model, embeddings, vectorstore, prompt, chain]):
        raise HTTPException(status_code=503, detail="RAG service is not ready.")

    try:
        query_prompt = request.messages[-1].content
        logger.info("Received query: %s", query_prompt)
        response_text = format_response(_chain_invoke(query_prompt))
        logger.info("Response ready.")

        if request.stream:
            async def generate_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                for line in response_text.split("\n"):
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": line + "\n"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"{json.dumps(chunk, ensure_ascii=False)}\n"
                    await asyncio.sleep(0.2)
                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"{json.dumps(final_chunk)}\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        response = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ]
        )
        return JSONResponse(content=response.model_dump())

    except Exception as exc:
        logger.exception("Error while processing completion request: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
