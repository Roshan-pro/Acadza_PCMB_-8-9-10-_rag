"""
rag.py
------
Builds a FAISS vector store from document chunks using free HuggingFace
embeddings, and exposes a RAG retriever that grounds LLM responses in the
actual chapter content.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Embedding model – free, runs locally, no API key needed
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K = 3          # number of chunks to retrieve per query
FAISS_INDEX_PATH = "faiss_index"


def build_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Initialise (and optionally download) the HuggingFace embedding model.
    The model is cached locally by sentence-transformers after first download.
    """
    print(f"[RAG] Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("[RAG] Embedding model ready.")
    return embeddings


def build_vector_store(
    chunks: List[Document],
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    persist_path: Optional[str] = None,
) -> FAISS:
    """
    Embed all chunks and store them in a FAISS vector store.

    Parameters
    ----------
    chunks       : Chunked Document objects from loader.py
    embeddings   : Pre-built HuggingFaceEmbeddings (created if None)
    persist_path : If given, save the FAISS index to disk for reuse

    Returns
    -------
    FAISS vector store
    """
    if embeddings is None:
        embeddings = build_embeddings()

    print(f"[RAG] Building FAISS index from {len(chunks)} chunks…")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("[RAG] FAISS index built.")

    if persist_path:
        vector_store.save_local(persist_path)
        print(f"[RAG] FAISS index saved to: {persist_path}")

    return vector_store


def load_vector_store(
    persist_path: str,
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    if embeddings is None:
        embeddings = build_embeddings()
    vector_store = FAISS.load_local(
        persist_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[RAG] Loaded FAISS index from: {persist_path}")
    return vector_store


def get_retriever(
    vector_store: FAISS,
    k: int = RETRIEVAL_K,
):
    """
    Return a LangChain retriever backed by the FAISS vector store.

    Parameters
    ----------
    vector_store : The FAISS index
    k            : Number of relevant chunks to retrieve per query

    Returns
    -------
    LangChain BaseRetriever
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever


def retrieve_context(
    retriever,
    query: str,
) -> str:
    """
    Run a similarity search and return retrieved text joined as a single string.

    Parameters
    ----------
    retriever : LangChain retriever from get_retriever()
    query     : The search query (e.g. a subconcept name)

    Returns
    -------
    str – concatenated context passages
    """
    docs = retriever.invoke(query)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return context


# ---------------------------------------------------------------------------
# Pipeline builder convenience function
# ---------------------------------------------------------------------------

def build_rag_pipeline(
    chunks: List[Document],
    persist_path: Optional[str] = None,
    k: int = RETRIEVAL_K,
):
    """
    Full pipeline: embeddings → FAISS → retriever.

    Returns
    -------
    (retriever, vector_store)
    """
    embeddings = build_embeddings()
    vector_store = build_vector_store(chunks, embeddings, persist_path)
    retriever = get_retriever(vector_store, k=k)
    return retriever, vector_store


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from langchain_core.documents import Document as D

    sample_chunks = [
        D(page_content="Electric charges and their properties are fundamental to electrostatics."),
        D(page_content="Coulomb's law describes the force between two point charges."),
        D(page_content="The electric field is defined as force per unit positive test charge."),
    ]
    retriever, _ = build_rag_pipeline(sample_chunks)
    ctx = retrieve_context(retriever, "Coulomb's law")
    print("Retrieved context:\n", ctx)