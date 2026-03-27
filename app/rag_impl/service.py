from __future__ import annotations

import datetime
import os
import uuid

from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.db import docs_collection, qdrant_client
from app.rag_impl import cache
from app.rag_impl.loaders import load_documents


load_dotenv()

# Core RAG config
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "bpt_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "7"))

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-005")

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

if not PROJECT_ID:
    raise RuntimeError(
        "Env GOOGLE_CLOUD_PROJECT belum diset. Isi di file .env (lihat .env.example) agar Vertex AI bisa digunakan."
    )

# Redis cache
RAG_CACHE_ENABLED = os.getenv("RAG_CACHE_ENABLED", "1").strip() not in ("0", "false", "False", "no", "NO")
RAG_CACHE_TTL_SECONDS = int(os.getenv("RAG_CACHE_TTL_SECONDS", "3600"))
RAG_CACHE_PREFIX = os.getenv("RAG_CACHE_PREFIX", "rag:cache:v1").strip() or "rag:cache:v1"
RAG_DATA_VERSION_KEY = os.getenv("RAG_DATA_VERSION_KEY", "rag:data_version").strip() or "rag:data_version"

# Semantic cache (Qdrant index + Redis value)
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "1").strip() not in ("0", "false", "False", "no", "NO")
SEMANTIC_CACHE_COLLECTION = os.getenv("SEMANTIC_CACHE_COLLECTION", "rag_query_cache").strip() or "rag_query_cache"
SEMANTIC_CACHE_LIMIT = int(os.getenv("SEMANTIC_CACHE_LIMIT", "5"))
SEMANTIC_CACHE_MIN_SCORE = float(os.getenv("SEMANTIC_CACHE_MIN_SCORE", "0.92"))
SEMANTIC_CACHE_MIN_TOKEN_COVERAGE = float(os.getenv("SEMANTIC_CACHE_MIN_TOKEN_COVERAGE", "0.65"))

# Support / escalation contacts (used when info not found in documents)
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "bpt@komdigi.go.id").strip() or "bpt@komdigi.go.id"
SUPPORT_WHATSAPP = os.getenv("SUPPORT_WHATSAPP", "").strip()
SUPPORT_TICKETING_URL = os.getenv("SUPPORT_TICKETING_URL", "").strip()


def _fallback_message() -> str:
    parts = [
        "Maaf, informasi tersebut tidak ditemukan di dokumen resmi kami.",
    ]
    if SUPPORT_WHATSAPP:
        parts.append(f"Silakan hubungi WhatsApp: {SUPPORT_WHATSAPP}.")
    if SUPPORT_TICKETING_URL:
        parts.append(f"Atau buat tiket melalui: {SUPPORT_TICKETING_URL}.")
    if SUPPORT_EMAIL:
        parts.append(f"Email: {SUPPORT_EMAIL}.")
    return " ".join(parts).strip()


def _init_docs_collection():
    from qdrant_client.http.models import Distance, VectorParams

    try:
        qdrant_client.get_collection(COLLECTION_NAME)
    except Exception:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )


_init_docs_collection()
cache.ensure_semantic_collection(SEMANTIC_CACHE_ENABLED, SEMANTIC_CACHE_COLLECTION, vector_size=768)


gemini_model = ChatVertexAI(
    model_name=LLM_MODEL,
    project=PROJECT_ID,
    location=LOCATION,
    temperature=LLM_TEMPERATURE,
)
embeddings = VertexAIEmbeddings(
    model_name=EMBEDDING_MODEL,
    project=PROJECT_ID,
    location=LOCATION,
)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)


def process_document(file_path: str, filename: str, uploader: str = "admin"):
    docs = load_documents(file_path, filename)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)

    doc_id = str(uuid.uuid4())
    for split in splits:
        split.metadata["doc_id"] = doc_id
        split.metadata["filename"] = filename

    vector_store.add_documents(splits)

    docs_collection.insert_one(
        {
            "doc_id": doc_id,
            "filename": filename,
            "uploader": uploader,
            "status": "ready",
            "uploaded_at": datetime.datetime.utcnow().isoformat(),
        }
    )

    cache.bump_data_version(RAG_CACHE_ENABLED, RAG_DATA_VERSION_KEY)
    return doc_id


def ask_chatbot(query: str):
    data_version = cache.get_data_version(RAG_CACHE_ENABLED, RAG_DATA_VERSION_KEY)

    fingerprint = {
        "data_version": data_version,
        "query": cache.normalize(query),
        "collection": COLLECTION_NAME,
        "retriever_k": RETRIEVER_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "embedding_model": EMBEDDING_MODEL,
        "project": PROJECT_ID,
        "location": LOCATION,
        "support_email": SUPPORT_EMAIL,
        "support_whatsapp": SUPPORT_WHATSAPP,
        "support_ticketing_url": SUPPORT_TICKETING_URL,
    }
    exact_key = cache.exact_cache_key(RAG_CACHE_PREFIX, fingerprint)

    hit = cache.try_get_exact(RAG_CACHE_ENABLED, exact_key)
    if hit:
        return hit

    semantic_filters = {
        "doc_collection": COLLECTION_NAME,
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }
    semantic_hit = cache.semantic_lookup(
        enabled=SEMANTIC_CACHE_ENABLED,
        collection_name=SEMANTIC_CACHE_COLLECTION,
        limit=SEMANTIC_CACHE_LIMIT,
        min_score=SEMANTIC_CACHE_MIN_SCORE,
        min_token_coverage=SEMANTIC_CACHE_MIN_TOKEN_COVERAGE,
        query=query,
        data_version=data_version,
        filters=semantic_filters,
        embeddings=embeddings,
        redis_enabled=RAG_CACHE_ENABLED,
    )
    if semantic_hit:
        return semantic_hit

    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("filename", "Unknown") for doc in docs]))

    template = """Anda adalah Asisten Resmi BPT Komdigi.
Tugas Anda adalah memberikan jawaban yang ramah, sopan, dan akurat dari pertanyaan pengguna.
Gunakan HANYA teks referensi (Konteks) berikut untuk menjawab.
Jika jawabannya tidak ada di referensi, jawab dengan kalimat berikut (tanpa menambah informasi lain): "{fallback}"

Referensi (Konteks):
{context}

Pertanyaan Pengguna: {question}

Jawaban Anda:"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | gemini_model
    response = chain.invoke({"context": context, "question": query, "fallback": _fallback_message()})

    answer = (response.content or "").strip()
    fallback = _fallback_message()
    if "tidak ditemukan" in answer.lower() and fallback not in answer:
        answer = fallback

    result = {"answer": answer, "sources": sources}

    cache.store_exact(RAG_CACHE_ENABLED, exact_key, RAG_CACHE_TTL_SECONDS, result)
    cache.semantic_store(
        enabled=SEMANTIC_CACHE_ENABLED,
        collection_name=SEMANTIC_CACHE_COLLECTION,
        query=query,
        data_version=data_version,
        embeddings=embeddings,
        filters=semantic_filters,
        redis_key=exact_key,
        ttl_seconds=RAG_CACHE_TTL_SECONDS,
    )

    return result

