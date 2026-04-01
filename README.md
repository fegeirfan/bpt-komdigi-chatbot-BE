# bpt-komdigi-chatbot-BE

## OCR (EasyOCR) dipisah ke service terpisah

- Service OCR ada di ocr/main.py (run di port misalnya 8001).
- Backend utama akan pakai OCR service jika env OCR_SERVICE_URL diisi (lihat .env.example).

## GCP (Vertex AI)

- Untuk deploy, bisa pakai env GOOGLE_APPLICATION_CREDENTIALS_JSON (isi JSON service account) tanpa file path.

