import os
import uuid
import datetime
import json
import hashlib
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


def _cache_key_for_query(query: str) -> str:
    normalized_query = " ".join(query.strip().lower().split())
    fingerprint = {
        "data_version": _get_data_version(),
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
    if redis_client:
        cache_key = _cache_key_for_query(query)
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            cache_key = None

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

    return result
