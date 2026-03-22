"""
loader.py
---------
Handles NCERT chapter loading from PDF (URL or local) or web pages.
Splits text into chunks for embedding.
"""

import os
import requests
import tempfile
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_pdf(url: str) -> str:
    """Download a PDF from a URL to a temp file and return the path."""
    print(f"[Loader] Downloading PDF from: {url}")
    response = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(response.content)
    tmp.close()
    print(f"[Loader] Saved to temp file: {tmp.name}")
    return tmp.name


def _load_pdf(path: str) -> List[Document]:
    """Load a PDF and return a list of Document objects."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    print(f"[Loader] Loaded {len(docs)} pages from PDF.")
    return docs


def _load_web(url: str) -> List[Document]:
    """Load a web page and return Documents."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    print(f"[Loader] Loaded {len(docs)} documents from web URL.")
    return docs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_chapter(
    source: str,
    source_type: str = "auto",
) -> List[Document]:
    """
    Load an NCERT chapter from a PDF URL, local PDF path, or web URL.

    Parameters
    ----------
    source      : URL or local file path to the chapter resource.
    source_type : 'pdf_url' | 'pdf_local' | 'web' | 'auto'
                  'auto' infers from the source string.

    Returns
    -------
    List[Document] – raw (unsplit) page/section documents.
    """
    if source_type == "auto":
        if source.startswith("http") and source.endswith(".pdf"):
            source_type = "pdf_url"
        elif source.startswith("http"):
            source_type = "web"
        elif Path(source).exists():
            source_type = "pdf_local"
        else:
            raise ValueError(
                f"Cannot infer source_type for: {source}. "
                "Provide source_type explicitly."
            )

    if source_type == "pdf_url":
        local_path = _download_pdf(source)
        docs = _load_pdf(local_path)
        os.unlink(local_path)          # clean up temp file
    elif source_type == "pdf_local":
        docs = _load_pdf(source)
    elif source_type == "web":
        docs = _load_web(source)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    return docs


def split_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split raw documents into overlapping chunks using
    RecursiveCharacterTextSplitter.

    Returns
    -------
    List[Document] – chunked documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[Loader] Split into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


def load_and_split(
    source: str,
    source_type: str = "auto",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Convenience wrapper: load then split.
    """
    docs = load_chapter(source, source_type)
    return split_documents(docs, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test with a small public NCERT PDF
    test_url = "https://ncert.nic.in/textbook/pdf/leph101.pdf"
    chunks = load_and_split(test_url)
    print(f"First chunk preview:\n{chunks[0].page_content[:300]}")