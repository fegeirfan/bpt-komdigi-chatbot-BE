import datetime
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.gcp_credentials import ensure_google_application_credentials
from app.schemas import ChatRequest, ChatResponse

load_dotenv()
ensure_google_application_credentials()

from app.db import chats_collection, docs_collection  # noqa: E402
from app.rag import ask_chatbot, process_document  # noqa: E402

app = FastAPI(title="BPT Komdigi RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti domain frontend kalau production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "BPT Komdigi Backend API is running."}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_path = None
    try:
        ext = Path(file.filename).suffix.lower()
        allowed_exts = {".pdf", ".doc", ".docx", ".csv", ".xlsx", ".png", ".jpg", ".jpeg"}
        if ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail="Format tidak didukung. Gunakan PDF/DOC/DOCX/CSV/XLSX/PNG/JPG.",
            )

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        doc_id = process_document(temp_path, safe_filename)

        return {"status": "success", "doc_id": doc_id, "filename": safe_filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal upload/proses file: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


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


@app.get("/api/documents")
def get_documents():
    try:
        docs = list(docs_collection.find({}, {"_id": 0}))
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal ambil dokumen: {str(e)}")
