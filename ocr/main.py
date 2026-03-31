import base64
import io
import os
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field


app = FastAPI(title="EasyOCR Service", version="0.1.0")


def _env_langs() -> List[str]:
    raw = os.getenv("OCR_LANGS", "id,en").strip()
    langs = [x.strip() for x in raw.split(",") if x.strip()]
    return langs or ["en"]


def _env_gpu() -> bool:
    return os.getenv("OCR_GPU", "0").strip() in ("1", "true", "True", "yes", "YES")


@lru_cache(maxsize=16)
def _get_reader(langs_key: str, gpu: bool):
    try:
        import easyocr
    except Exception as e:
        raise RuntimeError(
            "EasyOCR belum terpasang/siap. Install dependency easyocr (dan torch) pada service OCR ini."
        ) from e

    langs = [x for x in langs_key.split(",") if x]
    return easyocr.Reader(langs, gpu=gpu)


def _read_image_bytes(image_bytes: bytes):
    try:
        from PIL import Image
        import numpy as np
    except Exception as e:
        raise RuntimeError("Dependency PIL/numpy belum siap. Install pillow + numpy.") from e

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError("File bukan gambar yang valid (PNG/JPG).") from e

    return np.array(img)


class OcrBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64 image content (no data: prefix required).")
    filename: Optional[str] = None
    langs: Optional[List[str]] = None
    detail: int = 0
    paragraph: bool = False


class OcrResponse(BaseModel):
    text: str
    lines: List[str]
    langs: List[str]
    detail: int
    paragraph: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr/base64", response_model=OcrResponse)
def ocr_base64(payload: OcrBase64Request):
    langs = payload.langs or _env_langs()
    gpu = _env_gpu()

    try:
        image_bytes = base64.b64decode(payload.image_base64, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 tidak valid: {str(e)}")

    try:
        image = _read_image_bytes(image_bytes)
        reader = _get_reader(",".join(langs), gpu)
        lines = reader.readtext(image, detail=0, paragraph=payload.paragraph)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR gagal: {str(e)}")

    clean_lines = [str(x).strip() for x in lines if str(x).strip()]
    text = "\n".join(clean_lines).strip()
    return OcrResponse(
        text=text,
        lines=clean_lines,
        langs=langs,
        detail=payload.detail,
        paragraph=payload.paragraph,
    )


@app.post("/ocr/file", response_model=OcrResponse)
async def ocr_file(
    file: UploadFile = File(...),
    paragraph: bool = False,
    detail: int = 0,
):
    langs = _env_langs()
    gpu = _env_gpu()

    try:
        image_bytes = await file.read()
        image = _read_image_bytes(image_bytes)
        reader = _get_reader(",".join(langs), gpu)
        lines = reader.readtext(image, detail=0, paragraph=paragraph)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR gagal: {str(e)}")

    clean_lines = [str(x).strip() for x in lines if str(x).strip()]
    text = "\n".join(clean_lines).strip()
    return OcrResponse(text=text, lines=clean_lines, langs=langs, detail=detail, paragraph=paragraph)

