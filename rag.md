# Dokumentasi `rag.py` - Mesin Utama Chatbot AI

File `rag.py` adalah jantung dari sistem chatbot BPT Komdigi. File ini mengelola alur kerja **RAG (Retrieval-Augmented Generation)**, mulai dari pemrosesan dokumen mentah hingga pembuatan jawaban cerdas oleh AI.

---

## 🚀 Fungsi Utama

### 1. Inisialisasi & Koneksi
File ini secara otomatis menyiapkan koneksi ke layanan berikut saat dijalankan:
- **Google Vertex AI**: Menggunakan `gemini-2.5-flash` sebagai otak generator dan `text-embedding-005` untuk konversi teks ke vektor.
- **Qdrant Vector DB**: Inisialisasi koleksi `bpt_docs` untuk menyimpan memori dokumen.
- **Redis**: Digunakan sebagai sistem cache untuk mempercepat respon pertanyaan yang berulang.

### 2. `process_document(file_path, filename)`
Fungsi ini dipanggil saat Admin mengunggah PDF.
- **Ekstraksi**: Menggunakan `PyPDFLoader` untuk menarik teks dari file PDF.
- **Chunking (Pemotongan)**: Teks dipecah menjadi bagian kecil (default: 500 karakter dengan 150 karakter overlap) menggunakan `RecursiveCharacterTextSplitter`. Ini memastikan AI tidak kebingungan membaca teks yang terlalu panjang.
- **Embedding & Ingest**: Mengirim potongan teks ke Qdrant untuk disimpan dalam bentuk vektor.
- **Lacak Versi**: Setiap ada dokumen baru, sistem akan melakukan *bump* pada versi data agar cache lama otomatis kedaluwarsa.

### 3. `ask_chatbot(query)`
Fungsi inti saat pengguna bertanya di Chat Widget.
- **Cache Check**: 
    1. Mencari di Redis untuk jawaban yang *identik*.
    2. Jika tidak ada, mencari di **Semantic Cache** (Qdrant) untuk pertanyaan yang *maknanya mirip* (paraphrase).
- **Retrieval (Pencarian)**: Mengambil 7 potongan teks (`RETRIEVER_K=7`) yang paling relevan dari database Qdrant.
- **Generation (Sintesis)**: Mengirim teks rujukan tersebut ke Gemini AI bersamaan dengan instruksi khusus (*Prompt Template*).
- **Anti-Hallucination**: Jika jawaban tidak ditemukan di dokumen, sistem dipaksa memberikan pesan bantuan standar (Email/WA) daripada mengarang jawaban palsu.

---

## 🛠 Fitur Lanjutan

### 🧠 Semantic Caching
Fitur ini sangat menghemat biaya API. Jika ada dua pengguna bertanya:
1. *"Berapa biaya kursus?"*
2. *"Harga pelatihan berapa ya?"*
Sistem akan mengenali bahwa keduanya **bermakna sama** dan akan memberikan jawaban yang sudah pernah dibuat sebelumnya tanpa memanggil AI Gemini lagi.

### 🛡 Mekanisme Fallback
Sistem dilengkapi dengan `_fallback_message()` yang memberikan rujukan ke:
- Email: `bpt@komdigi.go.id`
- Kontak WhatsApp & Link Ticketing (jika diatur di `.env`).

### 📦 Teknologi yang Digunakan
- **LangChain**: Framework orkestrasi AI.
- **Google Vertex AI**: Model bahasa skala besar (LLM).
- **Qdrant**: Database vektor untuk memori dokumen.
- **Redis**: Kecepatan akses data (caching).
- **MongoDB**: Penyimpanan metadata administratif.

---

## ⚙️ Variabel Konfigurasi Utama (.env)
- `LLM_MODEL`: Versi Gemini (Contoh: `gemini-2.5-flash`).
- `RETRIEVER_K`: Jumlah potongan dokumen yang dibaca AI (Default: 7).
- `SEMANTIC_CACHE_MIN_SCORE`: Tingkat kemiripan minimal untuk menggunakan cache (Default: 0.92).
- `CHUNK_SIZE`: Panjang potongan paragraf dokumen.
