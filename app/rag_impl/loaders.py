from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document


def load_documents(file_path: str, filename: str) -> List[Document]:
    """
    Load a file into LangChain Documents based on its extension.
    Supports: pdf, docx, doc (best-effort), csv, xlsx, images (via EasyOCR).
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(file_path).load()

    if ext in (".docx",):
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(file_path).load()

    if ext in (".doc",):
        # Best-effort. Depending on environment, this may require extra system deps.
        try:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        except Exception as e:
            raise ValueError(
                "File .doc membutuhkan loader tambahan. "
                "Coba konversi ke .docx atau install dependency 'unstructured' + kebutuhan sistemnya."
            ) from e

        return UnstructuredWordDocumentLoader(file_path).load()

    if ext == ".csv":
        from langchain_community.document_loaders import CSVLoader

        return CSVLoader(file_path, encoding="utf-8").load()

    if ext in (".xlsx",):
        import pandas as pd

        xl = pd.ExcelFile(file_path)
        docs: List[Document] = []
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name=sheet_name)
            # Serialize as CSV-like text (readable and stable)
            text = df.to_csv(index=False)
            docs.append(
                Document(
                    page_content=f"[SHEET: {sheet_name}]\n{text}",
                    metadata={"source": filename, "sheet": sheet_name},
                )
            )
        return docs

    if ext in (".png", ".jpg", ".jpeg"):
        return _load_image_with_easyocr(file_path, filename)

    raise ValueError(f"Format file '{ext}' belum didukung. Gunakan PDF/DOCX/CSV/XLSX/PNG/JPG.")


def _load_image_with_easyocr(file_path: str, filename: str) -> List[Document]:
    try:
        import easyocr
    except Exception as e:
        raise ValueError(
            "EasyOCR belum terpasang/siap. Install dependency easyocr (dan torch) untuk OCR gambar."
        ) from e

    # Lazy singleton reader (init-nya berat)
    if not hasattr(_load_image_with_easyocr, "_reader"):
        langs = ["id", "en"]
        _load_image_with_easyocr._reader = easyocr.Reader(langs, gpu=False)  # type: ignore[attr-defined]

    reader = _load_image_with_easyocr._reader  # type: ignore[attr-defined]
    lines = reader.readtext(file_path, detail=0)
    text = "\n".join([str(x).strip() for x in lines if str(x).strip()])
    if not text.strip():
        text = ""
    return [Document(page_content=text, metadata={"source": filename, "type": "image"})]

