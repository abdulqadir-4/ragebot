"""Persistence and retrieval over a Chroma vector store, plus keyword search.

Retrieval uses cosine similarity explicitly (rather than Chroma's raw L2
default) so relevance scores land in a well-understood 0-1 range, which is
what makes `filter_by_relevance` below a meaningful, tunable cutoff.

Two fixes for prospectus-style documents:

* `build_vectorstore` REPLACES the collection. Previously every click on
  "Build knowledge base" appended the same chunks again (the old database held
  every prospectus chunk 3 times, plus chunks from unrelated PDFs uploaded in
  earlier sessions), so top-k was filled with duplicates and stale documents.
* `hybrid_search` merges the vector results with BM25 keyword results.
  Embeddings are weak at exact tokens such as "SF-6", "Rs.", programme codes
  and people's names; keyword search is strong at exactly those.
"""

import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .llm import get_embeddings

COLLECTION_NAME = "documents"
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def build_vectorstore(chunks: list[Document], persist_dir: str | Path) -> Chroma:
    old = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(),
        collection_metadata=COLLECTION_METADATA,
    )
    old.delete_collection()
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        ids=[c.metadata["chunk_id"] for c in chunks],
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
        collection_metadata=COLLECTION_METADATA,
    )


def load_vectorstore(persist_dir: str | Path) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(),
        collection_metadata=COLLECTION_METADATA,
    )


def filter_by_relevance(
    results: list[tuple[Document, float]], threshold: float | None
) -> list[Document]:
    """Drop retrieved chunks whose relevance score falls below `threshold`.
    `threshold=None` disables filtering and returns everything."""
    if threshold is None:
        return [doc for doc, _ in results]
    return [doc for doc, score in results if score >= threshold]


def search_relevant(
    vectorstore: Chroma, query: str, k: int, score_threshold: float | None
) -> list[Document]:
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return filter_by_relevance(results, score_threshold)


_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_STOP = {
    "the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "are",
    "what", "which", "who", "how", "much", "many", "tell", "me", "about", "please",
    "do", "does", "can", "i", "my", "at", "by", "with", "from", "be", "it", "this",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class KeywordIndex:
    """BM25 over the same chunks that are in the vector store."""

    def __init__(self, chunks: list[Document]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.page_content) or ["_"] for c in chunks])

    def search(self, query: str, k: int) -> list[Document]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k] if scores[i] > 0]


def hybrid_search(
    vectorstore: Chroma,
    keyword_index: KeywordIndex | None,
    query: str,
    k: int,
    score_threshold: float | None,
) -> list[Document]:
    """Reciprocal-rank fusion of vector and keyword results, best first."""
    vector_hits = search_relevant(vectorstore, query, k, score_threshold)
    keyword_hits = keyword_index.search(query, k) if keyword_index else []

    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for hits in (vector_hits, keyword_hits):
        for rank, doc in enumerate(hits):
            key = doc.metadata.get("chunk_id") or doc.page_content
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ordered[:k]]
