import datetime
import logging
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from app.gcp_credentials import ensure_google_application_credentials
from app.schemas import ChatRequest, ChatResponse

load_dotenv()
ensure_google_application_credentials()

from app.db import chats_collection, docs_collection  # noqa: E402

logger = logging.getLogger("app")

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/ping")
def debug_ping():
    return {"ok": True}


@app.get("/debug/env")
def debug_env():
    def present(name: str) -> bool:
        return bool((os.getenv(name) or "").strip())

    return {
        "port": os.getenv("PORT"),
        "debug_request_log": os.getenv("DEBUG_REQUEST_LOG"),
        "google_cloud_project_set": present("GOOGLE_CLOUD_PROJECT"),
        "google_cloud_region_set": present("GOOGLE_CLOUD_REGION"),
        "google_application_credentials_set": present("GOOGLE_APPLICATION_CREDENTIALS"),
        "google_application_credentials_json_set": present("GOOGLE_APPLICATION_CREDENTIALS_JSON"),
        "qdrant_url_set": present("QDRANT_URL"),
        "mongodb_uri_set": present("MONGODB_URI"),
        "redis_url_set": present("REDIS_URL"),
        "ocr_service_url_set": present("OCR_SERVICE_URL"),
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.middleware("http")
async def request_logger(request: Request, call_next):
    enabled = (os.getenv("DEBUG_REQUEST_LOG") or "0").strip() in ("1", "true", "True", "yes", "YES")
    if not enabled:
        return await call_next(request)

    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        dur_ms = int((time.perf_counter() - start) * 1000)
        status = getattr(response, "status_code", "?")
        logger.info("REQ %s %s -> %s (%sms)", request.method, request.url.path, status, dur_ms)


def get_rag():
    from app.rag import ask_chatbot, process_document

    return ask_chatbot, process_document


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

        ask_chatbot, process_document = get_rag()
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
        ask_chatbot, process_document = get_rag()
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
