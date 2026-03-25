from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from dotenv import load_dotenv
load_dotenv()
from app.schemas import ChatRequest, ChatResponse, DocumentResponse
from app.rag import process_document, ask_chatbot
from app.db import docs_collection, chats_collection
import datetime

app = FastAPI(title="BPT Komdigi RAG API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "BPT Komdigi Backend API is running."}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya mendukung file PDF saat ini.")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Process document synchronously (for demo purposes)
    try:
        doc_id = process_document(file_path, file.filename)
        return {"status": "success", "doc_id": doc_id, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        result = ask_chatbot(request.query)
        
        # Save chat to MongoDB
        chats_collection.insert_one({
            "session_id": request.session_id,
            "query": request.query,
            "answer": result["answer"],
            "sources": result["sources"],
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
def get_documents():
    docs = list(docs_collection.find({}, {"_id": 0}))
    return {"documents": docs}