"""
Utilities for loading heterogeneous technical documentation inputs.

Supports PDFs, plain-text/Markdown/CSV/JSON files, and URL manifests.
"""

from __future__ import annotations

import json
import logging
import hashlib
import re
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text


logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Loads files from the input directory and normalizes them into Documents."""

    TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".json",
        ".csv",
        ".yml",
        ".yaml",
    }
    URL_EXTENSIONS = {".url", ".urls", ".link", ".links"}

    def __init__(self, input_dir: str, url_manifest: Optional[str] = None):
        self.input_dir = Path(input_dir)
        self.url_manifest = Path(url_manifest) if url_manifest else None

    def load(self) -> List[Document]:
        """Load every supported document and return the aggregated list."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory '{self.input_dir}' does not exist.")

        documents: List[Document] = []
        for file_path in sorted(self.input_dir.rglob("*")):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                doc = self._load_pdf(file_path)
                if doc:
                    documents.append(doc)
            elif suffix in self.TEXT_EXTENSIONS:
                doc = self._load_text_like(file_path)
                if doc:
                    documents.append(doc)
            elif suffix in self.URL_EXTENSIONS:
                documents.extend(self._load_urls_from_manifest(file_path))

        if self.url_manifest:
            documents.extend(self._load_urls_from_manifest(self.url_manifest))

        logger.info("Loaded %s source document(s).", len(documents))
        return documents

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    def _load_pdf(self, path: Path) -> Optional[Document]:
        try:
            text = extract_text(str(path)).strip()
            if not text:
                logger.warning("PDF %s produced no extractable text.", path)
                return None
            return Document(
                page_content=text,
                metadata={"source": str(path), "type": "pdf"},
            )
        except Exception as exc:
            logger.error("Failed to parse PDF %s: %s", path, exc)
            return None

    def _load_text_like(self, path: Path) -> Optional[Document]:
        try:
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as fp:
                    data = json.load(fp)
                text = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                with path.open("r", encoding="utf-8") as fp:
                    text = fp.read()
            normalized = self._normalize_text(text)
            if not normalized:
                logger.warning("Text file %s is empty after normalization.", path)
                return None
            return Document(
                page_content=normalized,
                metadata={"source": str(path), "type": "text"},
            )
        except Exception as exc:
            logger.error("Failed to parse text file %s: %s", path, exc)
            return None

    def _load_urls_from_manifest(self, path: Path) -> List[Document]:
        documents: List[Document] = []
        if not path.exists():
            logger.warning("URL manifest %s does not exist.", path)
            return documents

        with path.open("r", encoding="utf-8") as fp:
            urls = [line.strip() for line in fp if line.strip() and not line.startswith("#")]

        for url in urls:
            doc = self._fetch_url(url)
            if doc:
                documents.append(doc)
        return documents

    def _fetch_url(self, url: str) -> Optional[Document]:
        logger.info("Fetching url source: %s", url)
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except Exception as exc:
            logger.error("Failed to download %s: %s", url, exc)
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        text = soup.get_text(separator="\n")
        normalized = self._normalize_text(text)
        if not normalized:
            logger.warning("URL %s returned empty content.", url)
            return None
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        return Document(
            page_content=normalized,
            metadata={"source": url, "type": "url", "title": title},
        )


def _normalize_chunk_text(text: str) -> str:
    """Collapse repeated whitespace and trim the chunk content."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int = 0,
) -> List[Document]:
    """Split documents into overlapping chunks for retrieval."""
    docs = list(documents)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    cleaned_chunks: List[Document] = []
    seen_hashes = set()

    for chunk in chunks:
        normalized = _normalize_chunk_text(chunk.page_content)
        if not normalized:
            continue
        if len(normalized) < min_chunk_chars:
            continue
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        metadata = dict(chunk.metadata or {})
        metadata["chunk_index"] = len(cleaned_chunks)
        cleaned_chunks.append(Document(page_content=normalized, metadata=metadata))

    logger.info(
        "Chunked %s document(s) into %s unique segments (min chars=%s).",
        len(docs),
        len(cleaned_chunks),
        min_chunk_chars,
    )
    return cleaned_chunks
