import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from app.db import qdrant_client, docs_collection
import uuid
import datetime

load_dotenv()
COLLECTION_NAME = "bpt_docs"

# Initialize Gemini & Embeddings
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.3
    )
embeddings = GoogleGenerativeAIEmbeddings(
    model = "text-embedding-004"
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

init_qdrant()

def process_document(file_path: str, filename: str, uploader: str = "admin"):
    # 1. Load Document
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        if "invalid pdf header" in str(e).lower() or "eof marker not found" in str(e).lower():
            raise ValueError(f"File '{filename}' bukan format PDF yang valid atau rusak. Pastikan file dimulai dengan teks '%PDF-'.")
        raise e
    
    # 2. Split Document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    # Add metadata
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
    
    # 4. Save metadata to MongoDB
    docs_collection.insert_one({
        "doc_id": doc_id,
        "filename": filename,
        "uploader": uploader,
        "status": "ready",
        "uploaded_at": datetime.datetime.utcnow().isoformat()
    })
    
    return doc_id

def ask_chatbot(query: str):
    # 1. Retrieval: Semantic Search in Qdrant
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    retriever = qdrant.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("filename", "Unknown") for doc in docs]))
    
    # 2. Prompt Template
    template = """Anda adalah Asisten Resmi BPT Komdigi.
Tugas Anda adalah memberikan jawaban yang ramah dan akurat dari pertanyaan pengguna.
Gunakan HANYA teks referensi berikut untuk menjawab. Jika jawabannya tidak ada di referensi, katakan "Maaf, informasi tersebut tidak ditemukan di dokumen kami. Silakan hubungi bpt@komdigi.go.id".

Referensi (Konteks):
{context}

Pertanyaan Pengguna: {question}

Jawaban Anda:"""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | gemini_model
    
    # 3. Generate Answer
    response = chain.invoke({"context": context, "question": query})
    
    return {
        "answer": response.content,
        "sources": sources
    }
