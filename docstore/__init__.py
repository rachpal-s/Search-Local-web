"""docstore — uploaded-document ingestion, retrieval and chat persistence.

Modules:
    corpus     record types + content-addressed IDs (replaces app.core.corpus)
    store      SQLite: conversations, messages, documents, chunks, vectors
    ingest     per-upload I1..I5 pipeline (extract/classify/chunk/enrich/embed)
    retrieve   hybrid dense+lexical retrieval with RRF fusion

Dropped in unchanged from the Enterprise RAG project (one import rewrite each):
    chunker.py, extractors.py, pdf_extractor.py
"""
from docstore import corpus, store  # noqa: F401

__all__ = ["corpus", "store", "ingest", "retrieve"]
