"""
Utility script to build the Chroma vector store from the raw input sources.
"""

from __future__ import annotations

import argparse
import logging

import json
from pathlib import Path

from document_processing import DocumentIngestor, chunk_documents
from rag_config import load_settings
from rag_pipeline import create_llm_clients, persist_vectorstore


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the technical-document vector store."
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Append to the existing Chroma collection instead of recreating it.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    settings = load_settings()
    ingestor = DocumentIngestor(settings.input_dir, settings.url_manifest)
    raw_docs = ingestor.load()
    if not raw_docs:
        raise RuntimeError("No documents found under the input directory.")

    chunks = chunk_documents(
        raw_docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_chars=settings.min_chunk_chars,
    )
    _, embeddings = create_llm_clients(settings)
    persist_vectorstore(
        chunks,
        settings=settings,
        embeddings=embeddings,
        reset=not args.no_reset,
    )
    chunk_dump_path = (
        Path(settings.persist_directory) / f"{settings.collection_name}_chunks.json"
    )
    chunk_dump_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dump_path.write_text(
        json.dumps(
            [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in chunks
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved chunk manifest to %s", chunk_dump_path)
    logger.info("Vector store updated successfully.")


if __name__ == "__main__":
    main()
