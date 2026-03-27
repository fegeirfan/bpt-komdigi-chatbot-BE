# Dokumentasi RAG (Knowledge Base + Chat)

Dokumentasi ini menjelaskan pipeline **RAG (Retrieval-Augmented Generation)** pada backend: bagaimana dokumen dimuat menjadi knowledge base, di-*embedding* ke Qdrant, lalu dipakai menjawab pertanyaan dengan Gemini (Vertex AI) dengan mekanisme cache (Redis + semantic cache).

---

## Struktur Modul (Modular)

RAG dibuat modular agar file tidak “gemuk”:

- `app/rag.py`: facade/public API (ekspor `process_document` dan `ask_chatbot`) agar import lama tidak rusak.
- `app/rag_impl/service.py`: implementasi utama RAG (ingest + chat + caching).
- `app/rag_impl/loaders.py`: loader dokumen (PDF/DOCX/CSV/XLSX + OCR gambar).
- `app/rag_impl/cache.py`: cache layer (Redis exact cache + semantic cache via Qdrant).

---

## Endpoint Terkait

- `POST /api/upload`: upload dokumen untuk knowledge base.
- `POST /api/chat`: tanya jawab (RAG).
- `GET /api/documents`: daftar metadata dokumen (MongoDB).

Catatan: file upload **tidak disimpan permanen**. Di `app/main.py`, file ditulis ke **temp file** hanya untuk keperluan parsing, lalu dihapus setelah proses selesai.

---

## Alur Upload & Ingest Knowledge Base

Fungsi: `process_document(file_path, filename, uploader="admin")` (di `app/rag_impl/service.py`).

1. Load dokumen sesuai ekstensi (lihat bagian “Format Dokumen”).
2. Chunking dengan `RecursiveCharacterTextSplitter` (default `CHUNK_SIZE=500`, `CHUNK_OVERLAP=150`).
3. Embedding + simpan vektor ke Qdrant collection `QDRANT_COLLECTION` (default `bpt_docs`).
4. Simpan metadata administratif ke MongoDB koleksi `documents`.
5. Bump versi data (`RAG_DATA_VERSION_KEY`) agar cache jawaban otomatis “berpindah versi” saat ada dokumen baru.

---

## Alur Tanya Jawab (RAG)

Fungsi: `ask_chatbot(query)` (di `app/rag_impl/service.py`).

Urutannya:

1. **Exact cache (Redis)**: jika pertanyaan sama persis (setelah normalisasi), jawaban diambil dari Redis.
2. **Semantic cache (Qdrant index -> Redis value)**: jika tidak sama persis, sistem mencari query yang mirip di koleksi Qdrant `SEMANTIC_CACHE_COLLECTION`, lalu mengambil jawaban dari Redis.
   - Ada guardrail sederhana: similarity threshold + token coverage + entity check (angka/tahun/bulan) supaya tidak salah reuse jawaban.
3. **RAG normal**: retrieval dari Qdrant (`RETRIEVER_K`, default 7) + prompt anti-halusinasi + panggil Gemini (Vertex AI).
4. **Simpan hasil**: hasil disimpan ke Redis (TTL) dan embedding query disimpan ke semantic-cache collection (Qdrant).

Jika informasi tidak ditemukan di konteks dokumen, sistem akan mengembalikan pesan fallback (kontak support) dan tidak mengarang jawaban.

---

## Format Dokumen yang Didukung (Knowledge Base)

Loader ada di `app/rag_impl/loaders.py`.

- PDF: `*.pdf` (PyPDFLoader)
- Word:
  - `*.docx` (Docx2txtLoader)
  - `*.doc` (best-effort; jika environment belum siap akan diminta konversi ke `.docx` atau install dependency loader tambahan)
- CSV: `*.csv`
- Excel: `*.xlsx` (dibaca per sheet dan diserialisasi jadi teks)
- Gambar (OCR): `*.png`, `*.jpg`, `*.jpeg` (EasyOCR; default bahasa `id` + `en`, GPU dimatikan)

---

## Konfigurasi `.env` (Ringkas)

### Vertex AI (Gemini)
- `GOOGLE_CLOUD_PROJECT` (wajib)
- `GOOGLE_CLOUD_REGION` (default `us-central1`)
- `GOOGLE_APPLICATION_CREDENTIALS` (opsional; dipakai jika autentikasi via service account JSON)
- `LLM_MODEL` (default `gemini-2.5-flash`)
- `LLM_TEMPERATURE` (default `0.2`)
- `EMBEDDING_MODEL` (default `text-embedding-005`)

### Qdrant
- `QDRANT_URL`, `QDRANT_API_KEY` (opsional)
- `QDRANT_COLLECTION` (default `bpt_docs`)

### Chunking/Retrieval
- `CHUNK_SIZE` (default `500`)
- `CHUNK_OVERLAP` (default `150`)
- `RETRIEVER_K` (default `7`)

### Redis Cache (Exact + Versioning)
- `REDIS_URL`
- `RAG_CACHE_ENABLED` (default `1`)
- `RAG_CACHE_TTL_SECONDS` (default `3600`)
- `RAG_CACHE_PREFIX` (default `rag:cache:v1`)
- `RAG_DATA_VERSION_KEY` (default `rag:data_version`)

### Semantic Cache (Qdrant index + Redis value)
- `SEMANTIC_CACHE_ENABLED` (default `1`)
- `SEMANTIC_CACHE_COLLECTION` (default `rag_query_cache`)
- `SEMANTIC_CACHE_LIMIT` (default `5`)
- `SEMANTIC_CACHE_MIN_SCORE` (default `0.92`)
- `SEMANTIC_CACHE_MIN_TOKEN_COVERAGE` (default `0.65`)

### Kontak Fallback
- `SUPPORT_EMAIL` (default `bpt@komdigi.go.id`)
- `SUPPORT_WHATSAPP`, `SUPPORT_TICKETING_URL` (opsional)

---

## Teknologi yang Digunakan

- FastAPI: API server
- LangChain: orkestrasi retrieval + LLM
- Google Vertex AI: LLM & embedding
- Qdrant: vector database (dokumen + semantic cache index)
- Redis: cache jawaban (TTL + versioning)
- MongoDB: metadata dokumen + chat log

