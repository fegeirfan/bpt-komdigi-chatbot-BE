import os
import uuid
import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from app.db import qdrant_client, docs_collection

# Load environment variables
load_dotenv()

COLLECTION_NAME = "bpt_docs"

# Initialize Vertex AI & Embeddings
# Pastikan GOOGLE_CLOUD_PROJECT dan GOOGLE_CLOUD_REGION ada di .env
# Jika menggunakan Service Account, pastikan GOOGLE_APPLICATION_CREDENTIALS tertunjuk ke file JSON
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

gemini_model = ChatVertexAI(
    model_name="gemini-1.5-flash", 
    project=project_id, 
    location=location,
    temperature=0.3
)
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-004",
    project=project_id,
    location=location
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    # Tambahkan metadata untuk pelacakan sumber
    doc_id = str(uuid.uuid4())
    for split in splits:
        split.metadata["doc_id"] = doc_id
        split.metadata["filename"] = filename

    # 3. Store in Qdrant Vector DB
    QdrantVectorStore.from_documents(
        splits, 
        embeddings, 
        url=os.getenv("QDRANT_URL", "http://localhost:6333"), 
        collection_name=COLLECTION_NAME,
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    # 4. Simpan metadata administratif ke MongoDB
    docs_collection.insert_one({
        "doc_id": doc_id,
        "filename": filename,
        "uploader": uploader,
        "status": "ready",
        "uploaded_at": datetime.datetime.utcnow().isoformat()
    })
    
    return doc_id

def ask_chatbot(query: str):
    """Fungsi untuk menjalankan Semantic Search dan sintesis jawaban AI"""
    # 1. Retrieval: Cari potongan teks paling relevan dari Qdrant
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    retriever = qdrant.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    
    # Menggabungkan potongan teks untuk dijadikan konteks bagi LLM
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("filename", "Unknown") for doc in docs]))
    
    # 2. Prompt Template (Anti-Halusinasi)
    template = """Anda adalah Asisten Resmi BPT Komdigi.
Tugas Anda adalah memberikan jawaban yang ramah, sopan, dan akurat dari pertanyaan pengguna.
Gunakan HANYA teks referensi (Konteks) berikut untuk menjawab. Jika jawabannya tidak ada di referensi, katakan "Maaf, informasi tersebut tidak ditemukan di dokumen resmi kami. Silakan hubungi bpt@komdigi.go.id".

Referensi (Konteks):
{context}

Pertanyaan Pengguna: {question}

Jawaban Anda:"""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | gemini_model
    
    # 3. Dekonstruksi dan panggil LLM Gemini
    response = chain.invoke({"context": context, "question": query})
    
    return {
        "answer": response.content,
        "sources": sources
    }