# Penjelasan Struktur Folder `backend/app`

Folder ini berisi logika utama *Backend* yang dibangun menggunakan FastAPI + LangChain, terintegrasi dengan **Google Vertex AI (Gemini)** untuk LLM & embedding, serta menggunakan **Qdrant** (vector database) dan **MongoDB** (metadata & chat log). Berikut rincian tiap file sesuai kode saat ini:

### 1. File Utama Aplikasi

*   **`main.py`**
    Pintu masuk utama (*entry point*) aplikasi FastAPI. File ini mengatur:
    *   Konfigurasi **CORS** (agar Frontend React bisa memanggil API ini).
    *   **Endpoints API**:
        *   `POST /api/upload`: Menerima file PDF, memprosesnya (temporary file) untuk ekstraksi teks + embedding, lalu menyimpan vektor ke Qdrant.
        *   `POST /api/chat`: Menerima pertanyaan pengguna, menjalankan pencarian dokumen relevan, dan mengembalikan jawaban dari AI Gemini.
        *   `GET /api/documents`: Menarik daftar metadata dokumen (nama file, tanggal upload) dari MongoDB.
    *   **Validasi Upload**:
        *   Hanya menerima file ber-ekstensi `.pdf`.
        *   Nama file di-*prefix* timestamp (`YYYYMMDDHHMMSS_`) untuk menghindari overwrite.
    *   **Penyimpanan Chat**:
        *   Setiap request `/api/chat` disimpan ke koleksi `chats` di MongoDB (termasuk `session_id`, `sources`, dan timestamp UTC).

*   **`rag.py`**
    Mesin inti AI yang menerapkan metode **RAG (Retrieval-Augmented Generation)**. File ini berisi logika:
    *   **Ekstraksi & Chunking**: Membaca file PDF (`PyPDFLoader`) dan memecahnya menjadi potongan kecil (`RecursiveCharacterTextSplitter`).
        *   Ada validasi error PDF rusak/tidak valid (mis. `invalid pdf header`, `EOF marker not found`) dengan pesan yang lebih jelas.
    *   **Embedding (Vertex AI)**: Mengubah teks menjadi vektor menggunakan `VertexAIEmbeddings` (default model: `text-embedding-005`, bisa dioverride via env `EMBEDDING_MODEL`).
    *   **LLM (Vertex AI / Gemini)**: Menjawab menggunakan `ChatVertexAI` (default model: `gemini-2.5-flash`, temperature default `0.2`, bisa dioverride via env `LLM_MODEL` dan `LLM_TEMPERATURE`).
    *   **Vector DB (Qdrant)**:
        *   Koleksi Qdrant di-*init* saat modul dimuat (`COLLECTION_NAME = "bpt_docs"`). Jika belum ada, dibuat dengan ukuran vektor `768` dan `COSINE` distance.
        *   Penyimpanan dan retrieval dilakukan via `QdrantVectorStore(...)` yang memakai `qdrant_client` dari `db.py` (konfigurasinya mengambil `QDRANT_URL`/`QDRANT_API_KEY` dari `.env`).
    *   **Metadata Dokumen**:
        *   Setiap chunk diberi metadata `doc_id` (UUID) dan `filename` untuk pelacakan sumber.
        *   Metadata administratif disimpan ke MongoDB koleksi `documents` (`doc_id`, `filename`, `uploader`, `status`, `uploaded_at`).
    *   **Chat Logic (Anti-Halusinasi)**:
        *   Mengambil top-k dokumen relevan (default `k=3`, bisa dioverride via env `RETRIEVER_K`) dari Qdrant.
        *   Menggabungkan konteks dan pertanyaan ke *prompt* ketat: jika jawaban tidak ada di konteks, bot diminta menjawab bahwa informasi tidak ditemukan dan mengarahkan ke email resmi.

### 2. Konfigurasi & Data

*   **`db.py`**
    Mengatur koneksi ke dua jenis database:
    *   **MongoDB**: Menggunakan `pymongo` untuk menyimpan data administratif/klasik (metadata dokumen dan riwayat pesan chat).
    *   **Qdrant**: Menggunakan `qdrant-client` untuk menyimpan data vektor teks agar bisa dicari secara cerdas (*Semantic Search*).
    *   Dilengkapi *fallback* nilai bawaan jika variabel di `.env` tidak ditemukan (mis. `mongodb://localhost:27017` dan `http://localhost:6333`).
    *   Environment yang digunakan:
        *   `MONGODB_URI`, `MONGODB_DB_NAME`
        *   `QDRANT_URL`, `QDRANT_API_KEY`

*   **`schemas.py`**
    Berisi definisi struktur data menggunakan **Pydantic**. Ini memastikan data yang dikirim oleh pengguna (seperti JSON pertanyaan) dan data yang dikirim balik oleh server memiliki format yang tepat dan aman. Contoh: `ChatRequest`, `ChatResponse`.

*   **`__init__.py`**
    File kosong yang menandakan bahwa folder `app` adalah sebuah paket (*package*) Python. Tanpa file ini, perintah `import app.main` mungkin akan mengalami galat.

---

### Folder Tambahan (Struktur Lanjutan)

*   **`uploads/`**: Tidak dipakai lagi (upload diproses via file temporary dan tidak disimpan permanen).
*   **(Opsional di masa depan) `controllers/`, `models/`, `routes/`**: Belum ada di struktur saat ini. Biasanya dipakai jika proyek mulai besar dan `main.py` perlu dipisah menjadi modul-modul.

---

### Catatan Konfigurasi Vertex AI (Gemini)

Untuk menjalankan integrasi Vertex AI di `rag.py`, pastikan environment berikut tersedia:

*   `GOOGLE_CLOUD_PROJECT`
*   `GOOGLE_CLOUD_REGION` (default: `us-central1`)
*   Jika memakai service account: `GOOGLE_APPLICATION_CREDENTIALS` mengarah ke file JSON kredensial
