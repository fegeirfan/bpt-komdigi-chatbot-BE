from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import datetime
from dotenv import load_dotenv
from pathlib import Path

from app.schemas import ChatRequest, ChatResponse
from app.rag import process_document, ask_chatbot
from app.db import docs_collection, chats_collection

load_dotenv()

app = FastAPI(title="BPT Komdigi RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti domain frontend kalau production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "BPT Komdigi Backend API is running."}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Hanya mendukung file PDF.")

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        doc_id = process_document(str(file_path), safe_filename)

        return {"status": "success", "doc_id": doc_id, "filename": safe_filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal upload/proses file: {str(e)}")


# Backward compatible alias (kalau Swagger/FE sudah terlanjur pakai ini)
@app.post("/upload-doc/")
async def upload_doc_alias(file: UploadFile = File(...)):
    return await upload_file(file)


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        result = ask_chatbot(request.query)

        chats_collection.insert_one(
            {
                "session_id": request.session_id,
                "query": request.query,
                "answer": result.get("answer"),
                "sources": result.get("sources", []),
                "timestamp": datetime.datetime.utcnow(),
            }
        )

        return ChatResponse(
            answer=result.get("answer"),
            sources=result.get("sources", []),
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


# Backward compatible alias (kalau Swagger/FE sudah terlanjur pakai ini)
@app.post("/ask/")
def ask_alias(request: ChatRequest):
    return chat_endpoint(request)


@app.get("/api/documents")
def get_documents():
    try:
        docs = list(docs_collection.find({}, {"_id": 0}))
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal ambil dokumen: {str(e)}")

