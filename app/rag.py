import os
import uuid
import datetime
import json
import hashlib
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from app.db import qdrant_client, docs_collection, get_redis_client

# Load environment variables
load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "bpt_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "7"))

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-005")

# Redis cache config (response cache)
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

# Initialize Vertex AI & Embeddings
# Pastikan GOOGLE_CLOUD_PROJECT dan GOOGLE_CLOUD_REGION ada di .env
# Jika menggunakan Service Account, pastikan GOOGLE_APPLICATION_CREDENTIALS tertunjuk ke file JSON
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

if not PROJECT_ID:
    raise RuntimeError(
        "Env GOOGLE_CLOUD_PROJECT belum diset. "
        "Isi di file .env (lihat .env.example) agar Vertex AI bisa digunakan."
    )

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

def init_qdrant():
    from qdrant_client.http.models import Distance, VectorParams
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
    except Exception:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

# Jalankan inisialisasi koleksi saat modul dimuat
init_qdrant()

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)


def _init_semantic_cache_collection():
    if not SEMANTIC_CACHE_ENABLED:
        return
    from qdrant_client.http.models import Distance, VectorParams
    try:
        qdrant_client.get_collection(SEMANTIC_CACHE_COLLECTION)
    except Exception:
        qdrant_client.create_collection(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )


_init_semantic_cache_collection()


_STOPWORDS = {
    "yang", "dan", "atau", "di", "ke", "dari", "untuk", "pada", "dengan", "tanpa", "apa", "bagaimana", "kenapa",
    "kapan", "dimana", "berapa", "apakah", "saya", "kami", "kamu", "anda", "itu", "ini", "dalam", "agar", "tolong",
    "the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were", "be", "been",
}
_MONTHS = {
    "januari", "februari", "maret", "april", "mei", "juni", "juli", "agustus", "september", "oktober", "november", "desember",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def _tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _extract_entities(text: str) -> set[str]:
    t = text.lower()
    years = set(re.findall(r"\b20\d{2}\b", t))
    numbers = set(re.findall(r"\b\d+\b", t))
    months = {m for m in _MONTHS if m in t}
    return years.union(numbers).union(months)


def _semantic_cache_lookup(query: str, redis_client, data_version: str):
    if not (SEMANTIC_CACHE_ENABLED and redis_client):
        return None

    normalized_query = _normalize_query(query)
    try:
        query_vec = embeddings.embed_query(normalized_query)
    except Exception:
        return None

    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    q_filter = Filter(
        must=[
            FieldCondition(key="data_version", match=MatchValue(value=data_version)),
            FieldCondition(key="doc_collection", match=MatchValue(value=COLLECTION_NAME)),
            FieldCondition(key="llm_model", match=MatchValue(value=LLM_MODEL)),
            FieldCondition(key="embedding_model", match=MatchValue(value=EMBEDDING_MODEL)),
        ]
    )

    try:
        hits = qdrant_client.search(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            query_vector=query_vec,
            limit=SEMANTIC_CACHE_LIMIT,
            with_payload=True,
            query_filter=q_filter,
        )
    except Exception:
        return None

    new_tokens = _tokens(normalized_query)
    new_entities = _extract_entities(normalized_query)

    for hit in hits:
        score = getattr(hit, "score", 0) or 0
        if score < SEMANTIC_CACHE_MIN_SCORE:
            continue

        payload = getattr(hit, "payload", None) or {}
        cached_query = payload.get("normalized_query", "") or ""
        cached_tokens = _tokens(cached_query)

        if new_tokens:
            coverage = len(new_tokens.intersection(cached_tokens)) / max(len(new_tokens), 1)
            if coverage < SEMANTIC_CACHE_MIN_TOKEN_COVERAGE:
                continue

        if new_entities:
            cached_entities = _extract_entities(cached_query)
            if not new_entities.issubset(cached_entities):
                continue

        redis_key = payload.get("redis_key")
        if not redis_key:
            continue

        try:
            cached_value = redis_client.get(redis_key)
        except Exception:
            cached_value = None

        if cached_value:
            try:
                return json.loads(cached_value)
            except Exception:
                return None

        # Redis key expired/invalid -> cleanup Qdrant point (best-effort)
        try:
            from qdrant_client.http.models import PointIdsList
            qdrant_client.delete(
                collection_name=SEMANTIC_CACHE_COLLECTION,
                points_selector=PointIdsList(points=[hit.id]),
            )
        except Exception:
            pass

    return None


def _semantic_cache_store(query: str, redis_key: str, data_version: str):
    if not SEMANTIC_CACHE_ENABLED:
        return

    normalized_query = _normalize_query(query)
    try:
        query_vec = embeddings.embed_query(normalized_query)
    except Exception:
        return

    payload = {
        "redis_key": redis_key,
        "normalized_query": normalized_query,
        "data_version": data_version,
        "doc_collection": COLLECTION_NAME,
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "retriever_k": RETRIEVER_K,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "ttl_seconds": RAG_CACHE_TTL_SECONDS,
    }

    try:
        from qdrant_client.http.models import PointStruct
        qdrant_client.upsert(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            points=[PointStruct(id=str(uuid.uuid4()), vector=query_vec, payload=payload)],
        )
    except Exception:
        pass


def _get_data_version() -> str:
    redis_client = get_redis_client()
    if not (RAG_CACHE_ENABLED and redis_client):
        return "0"
    try:
        version = redis_client.get(RAG_DATA_VERSION_KEY)
        return version or "0"
    except Exception:
        return "0"


def _bump_data_version():
    redis_client = get_redis_client()
    if not (RAG_CACHE_ENABLED and redis_client):
        return
    try:
        redis_client.incr(RAG_DATA_VERSION_KEY)
    except Exception:
        pass


def _cache_key_for_query(query: str, data_version: str | None = None) -> str:
    normalized_query = _normalize_query(query)
    fingerprint = {
        "data_version": data_version if data_version is not None else _get_data_version(),
        "query": normalized_query,
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
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{RAG_CACHE_PREFIX}:{digest}"


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


def process_document(file_path: str, filename: str, uploader: str = "admin"):
    """Fungsi untuk mengekstrak, memotong, dan menyimpan dokumen ke Vector DB (Qdrant)"""
    # 1. Load Document
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        # Menangani error PDF korup atau header tidak valid
        if "invalid pdf header" in str(e).lower() or "eof marker not found" in str(e).lower():
            raise ValueError(f"File '{filename}' bukan format PDF yang valid atau rusak. Pastikan file dimulai dengan teks '%PDF-'.")
        raise e
    
    # 2. Split Document (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)
    
    # Tambahkan metadata untuk pelacakan sumber
    doc_id = str(uuid.uuid4())
    for split in splits:
        split.metadata["doc_id"] = doc_id
        split.metadata["filename"] = filename

    # 3. Store in Qdrant Vector DB
    vector_store.add_documents(splits)
    
    # 4. Simpan metadata administratif ke MongoDB
    docs_collection.insert_one({
        "doc_id": doc_id,
        "filename": filename,
        "uploader": uploader,
        "status": "ready",
        "uploaded_at": datetime.datetime.utcnow().isoformat()
    })

    # Bump data version supaya cache query otomatis pindah ke versi baru
    _bump_data_version()
    
    return doc_id

def ask_chatbot(query: str):
    """Fungsi untuk menjalankan Semantic Search dan sintesis jawaban AI"""
    redis_client = get_redis_client() if RAG_CACHE_ENABLED else None
    cache_key = None
    data_version = "0"
    if redis_client:
        data_version = _get_data_version()
        cache_key = _cache_key_for_query(query, data_version=data_version)
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            cache_key = None

        # Semantic cache (paraphrase/kemiripan) via Qdrant
        semantic_hit = _semantic_cache_lookup(query, redis_client, data_version=data_version)
        if semantic_hit:
            return semantic_hit

    # 1. Retrieval: Cari potongan teks paling relevan dari Qdrant
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    docs = retriever.invoke(query)
    
    # Menggabungkan potongan teks untuk dijadikan konteks bagi LLM
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("filename", "Unknown") for doc in docs]))
    
    # 2. Prompt Template (Anti-Halusinasi)
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
    
    # 3. Dekonstruksi dan panggil LLM Gemini
    response = chain.invoke({"context": context, "question": query, "fallback": _fallback_message()})
    
    answer = response.content or ""
    fallback = _fallback_message()
    if "tidak ditemukan" in answer.lower() and fallback not in answer:
        answer = fallback

    result = {
        "answer": answer,
        "sources": sources
    }

    if redis_client and cache_key:
        try:
            redis_client.setex(cache_key, RAG_CACHE_TTL_SECONDS, json.dumps(result, ensure_ascii=False))
        except Exception:
            pass

        # Simpan embedding query ke Qdrant semantic-cache, menunjuk ke redis cache_key
        _semantic_cache_store(query, cache_key, data_version=data_version)

    return result
