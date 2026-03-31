# OCR Service (EasyOCR)

Service FastAPI khusus OCR gambar berbasis EasyOCR (dipisah dari backend utama).

## Run (local)

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r ocr/requirements.txt
uvicorn ocr.main:app --host 0.0.0.0 --port 8001
```

## Deploy (Railway)

- Buat service baru di Railway dari repo ini.
- Set **Dockerfile path** ke `ocr/Dockerfile` (atau set **Root Directory** ke `ocr` lalu rename/atur Dockerfile sesuai kebutuhan).
- Railway akan inject `$PORT`, image akan start pakai `uvicorn ocr.main:app`.

## Env

- `OCR_LANGS` (default: `id,en`) contoh: `en` atau `id,en`
- `OCR_GPU` (default: `0`) set `1` untuk enable GPU (jika torch/cuda siap)

## API

- `GET /health`
- `POST /ocr/file` (multipart `file`)
- `POST /ocr/base64` (JSON `{ "image_base64": "...", "langs": ["id","en"] }`)
