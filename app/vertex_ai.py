from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence

from langchain_core.embeddings import Embeddings


def _chunked(items: Sequence[str], size: int) -> Iterable[List[str]]:
    if size <= 0:
        size = 32
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def _normalize_text(text: str) -> str:
    # Vertex embedding endpoint can reject empty strings.
    if text is None:
        return " "
    s = str(text)
    return s if s.strip() else " "


def init_vertex_ai(project: str, location: str) -> None:
    import vertexai

    vertexai.init(project=project, location=location)


def _hint_auth() -> str:
    return (
        "Pastikan env GOOGLE_CLOUD_PROJECT dan GOOGLE_CLOUD_REGION terisi, dan kredensial GCP tersedia. "
        "Jika deploy di Railway, gunakan GOOGLE_APPLICATION_CREDENTIALS_JSON (lihat .env.example)."
    )


class VertexAITextEmbeddings(Embeddings):
    def __init__(
        self,
        *,
        model_name: str,
        project: str,
        location: str,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._project = project
        self._location = location

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from vertexai.language_models import TextEmbeddingInput

        if not texts:
            return []

        vectors: List[List[float]] = []
        normalized = [_normalize_text(t) for t in texts]

        for batch in _chunked(normalized, self._batch_size):
            try:
                model = _get_text_embedding_model(
                    project=self._project,
                    location=self._location,
                    model_name=self._model_name,
                )
                inputs = [TextEmbeddingInput(text=b) for b in batch]
                results = model.get_embeddings(inputs)
                vectors.extend([r.values for r in results])
            except Exception as e:
                raise RuntimeError(f"Gagal membuat embedding via Vertex AI. {_hint_auth()}") from e

        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@dataclass(frozen=True)
class VertexAIGenerativeText:
    model_name: str
    project: str
    location: str
    temperature: float = 0.2
    max_output_tokens: int = 1024

    def __post_init__(self) -> None:
        # Lazy init is done in generate(); keep __post_init__ empty to avoid import-time crashes.
        return None

    def generate(self, prompt: str) -> str:
        try:
            init_vertex_ai(self.project, self.location)
            from vertexai.preview.generative_models import GenerationConfig, GenerativeModel

            model = GenerativeModel(self.model_name)
            cfg = GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            resp = model.generate_content(prompt, generation_config=cfg)
            text: Optional[str] = getattr(resp, "text", None)
            return (text or "").strip()
        except Exception as e:
            raise RuntimeError(f"Gagal generate jawaban via Vertex AI. {_hint_auth()}") from e


@lru_cache(maxsize=8)
def _get_text_embedding_model(*, project: str, location: str, model_name: str):
    init_vertex_ai(project, location)
    from vertexai.language_models import TextEmbeddingModel

    return TextEmbeddingModel.from_pretrained(model_name)
