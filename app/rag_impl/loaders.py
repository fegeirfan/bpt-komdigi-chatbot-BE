from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.documents import Document


# =========================
# MAIN LOADER
# =========================
def load_documents(file_path: str, filename: str) -> List[Document]:
    """
    Load file ke LangChain Document.
    Support:
    - PDF
    - DOCX
    - DOC (best effort)
    - CSV
    - XLSX
    - IMAGE (via OCR service)
    """
    ext = Path(filename).suffix.lower()

    # ===== PDF =====
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path).load()

    # ===== DOCX =====
    if ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path).load()

    # ===== DOC (optional) =====
    if ext == ".doc":
        try:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        except Exception as e:
            raise ValueError(
                "File .doc butuh dependency tambahan. "
                "Gunakan .docx atau install 'unstructured'."
            ) from e

        return UnstructuredWordDocumentLoader(file_path).load()

    # ===== CSV =====
    if ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        return CSVLoader(file_path, encoding="utf-8").load()

    # ===== EXCEL =====
    if ext == ".xlsx":
        import pandas as pd

        xl = pd.ExcelFile(file_path)
        docs: List[Document] = []

        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name=sheet_name)
            text = df.to_csv(index=False)

            docs.append(
                Document(
                    page_content=f"[SHEET: {sheet_name}]\n{text}",
                    metadata={
                        "source": filename,
                        "sheet": sheet_name,
                        "type": "excel"
                    },
                )
            )
        return docs

    # ===== IMAGE =====
    if ext in (".png", ".jpg", ".jpeg"):
        return _load_image_with_ocr_service(file_path, filename)

    raise ValueError(f"Format file '{ext}' belum didukung.")


# =========================
# OCR HANDLER
# =========================
def _load_image_with_ocr_service(file_path: str, filename: str) -> List[Document]:
    service_url = (os.getenv("OCR_SERVICE_URL") or "").strip()

    if not service_url:
        raise ValueError(
            "OCR_SERVICE_URL belum diset. "
            "Contoh: https://ocrbe-production.up.railway.app"
        )

    if not service_url.startswith("http"):
        raise ValueError("OCR_SERVICE_URL tidak valid (harus http/https)")

    return _load_image_via_ocr_service(file_path, filename, service_url)


# =========================
# OCR BASE64 CALL
# =========================
def _load_image_via_ocr_service(
    file_path: str,
    filename: str,
    service_url: str
) -> List[Document]:

    langs_raw = (os.getenv("OCR_LANGS") or "id,en").strip()
    langs = [x.strip() for x in langs_raw.split(",") if x.strip()]

    # ===== READ FILE =====
    try:
        with open(file_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        raise ValueError(f"Gagal membaca file '{filename}': {str(e)}") from e

    # ===== PREPARE REQUEST =====
    payload = {
        "image_base64": image_base64,
        "filename": filename,
        "langs": langs,
        "paragraph": False
    }

    body = json.dumps(payload).encode("utf-8")

    url = service_url.rstrip("/") + "/ocr/base64"

    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")

    # ===== CALL OCR SERVICE =====
    try:
        with urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw or "{}")

    except HTTPError as e:
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)

        raise ValueError(f"OCR service error ({e.code}): {detail}") from e

    except URLError as e:
        raise ValueError(f"Tidak bisa connect ke OCR service: {str(e)}") from e

    except Exception as e:
        raise ValueError(f"OCR gagal: {str(e)}") from e

    # ===== PARSE RESULT =====
    text = (data.get("text") or "").strip()

    if not text:
        text = "[OCR tidak menemukan teks]"

    return [
        Document(
            page_content=text,
            metadata={
                "source": filename,
                "type": "image",
                "ocr": "base64-service"
            }
        )
    ]