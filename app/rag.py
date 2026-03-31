"""Compatibility wrapper for the active RAG implementation.

Keep imports stable for callers that still use ``app.rag`` while routing all
behavior to ``app.rag_impl.service`` so there is only one source of truth.
"""

from app.rag_impl.service import ask_chatbot, process_document

__all__ = ["process_document", "ask_chatbot"]

