# Penjelasan Struktur Folder `backend/app`

Folder ini berisi logika utama *Backend* yang dibangun menggunakan FastAPI, LangChain, dan terintegrasi dengan AI Gemini. Berikut adalah rincian setiap file:

### 1. File Utama Aplikasi

*   **`main.py`**
    Pintu masuk utama (*entry point*) aplikasi FastAPI. File ini mengatur:
    *   Konfigurasi **CORS** (agar Frontend React bisa memanggil API ini).
    *   **Endpoints API**:
        *   `POST /api/upload`: Menerima file PDF, menyimpannya di folder `uploads/`, lalu memicu proses ekstraksi teks dan embedding.
        *   `POST /api/chat`: Menerima pertanyaan pengguna, menjalankan pencarian dokumen relevan, dan mengembalikan jawaban dari AI Gemini.
        *   `GET /api/documents`: Menarik daftar metadata dokumen (nama file, tanggal upload) dari MongoDB.

*   **`rag.py`**
    Mesin inti AI yang menerapkan metode **RAG (Retrieval-Augmented Generation)**. File ini berisi logika:
    *   **Ekstraksi & Chunking**: Membaca file PDF (`PyPDFLoader`) dan memecahnya menjadi potongan kecil (`RecursiveCharacterTextSplitter`).
    *   **Embedding**: Mengubah teks menjadi angka vektor menggunakan model `models/embedding-001` (Gemini).
    *   **Vector DB (Qdrant)**: Mengirim dan menarik data vektor dari database Qdrant.
    *   **Chat Logic**: Menggabungkan konteks dokumen dengan pertanyaan pengguna ke dalam *System Prompt* yang ketat agar AI tidak mengarang jawaban.

### 2. Konfigurasi & Data

*   **`db.py`**
    Mengatur koneksi ke dua jenis database:
    *   **MongoDB**: Menggunakan `pymongo` untuk menyimpan data administratif/klasik (metadata dokumen dan riwayat pesan chat).
    *   **Qdrant**: Menggunakan `qdrant-client` untuk menyimpan data vektor teks agar bisa dicari secara cerdas (*Semantic Search*).
    *   Dilengkapi dengan sistem *fallback* (nilai bawaan) jika variabel di file `.env` tidak ditemukan.

*   **`schemas.py`**
    Berisi definisi struktur data menggunakan **Pydantic**. Ini memastikan data yang dikirim oleh pengguna (seperti JSON pertanyaan) dan data yang dikirim balik oleh server memiliki format yang tepat dan aman. Contoh: `ChatRequest`, `ChatResponse`.

*   **`__init__.py`**
    File kosong yang menandakan bahwa folder `app` adalah sebuah paket (*package*) Python. Tanpa file ini, perintah `import app.main` mungkin akan mengalami galat.

---

### Folder Tambahan (Struktur Lanjutan)

*   **`uploads/`**: Folder tempat menyimpan sementara file fisik PDF yang diunggah sebelum diproses.
*   **`controllers/`, `models/`, `routes/`**: Folder yang disiapkan untuk pengembangan skala besar di masa depan jika `main.py` sudah terlalu panjang. Saat ini logika masih disatukan untuk kemudahan uji coba awal.
