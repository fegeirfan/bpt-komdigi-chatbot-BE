from __future__ import annotations

import datetime
import hashlib
import json
import re
import uuid
from typing import Any, Dict, Optional

from app.db import get_redis_client, qdrant_client


_STOPWORDS = {
    "yang", "dan", "atau", "di", "ke", "dari", "untuk", "pada", "dengan", "tanpa", "apa", "bagaimana", "kenapa",
    "kapan", "dimana", "berapa", "apakah", "saya", "kami", "kamu", "anda", "itu", "ini", "dalam", "agar", "tolong",
    "the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were", "be", "been",
}
_MONTHS = {
    "januari", "februari", "maret", "april", "mei", "juni", "juli", "agustus", "september", "oktober", "november", "desember",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}


def redis_client():
    return get_redis_client()


def get_data_version(enabled: bool, version_key: str) -> str:
    r = redis_client()
    if not (enabled and r):
        return "0"
    try:
        v = r.get(version_key)
        return v or "0"
    except Exception:
        return "0"


def bump_data_version(enabled: bool, version_key: str):
    r = redis_client()
    if not (enabled and r):
        return
    try:
        r.incr(version_key)
    except Exception:
        pass


def exact_cache_key(prefix: str, fingerprint: Dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def try_get_exact(enabled: bool, key: str) -> Optional[dict]:
    r = redis_client()
    if not (enabled and r):
        return None
    try:
        raw = r.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def store_exact(enabled: bool, key: str, ttl_seconds: int, value: dict):
    r = redis_client()
    if not (enabled and r):
        return
    try:
        r.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False))
    except Exception:
        pass


def ensure_semantic_collection(enabled: bool, collection_name: str, vector_size: int = 768):
    if not enabled:
        return
    from qdrant_client.http.models import Distance, VectorParams

    try:
        qdrant_client.get_collection(collection_name)
    except Exception:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def semantic_lookup(
    enabled: bool,
    collection_name: str,
    limit: int,
    min_score: float,
    min_token_coverage: float,
    query: str,
    data_version: str,
    filters: Dict[str, Any],
    embeddings,
    redis_enabled: bool,
) -> Optional[dict]:
    r = redis_client()
    if not (enabled and r and redis_enabled):
        return None

    normalized = normalize(query)
    try:
        qvec = embeddings.embed_query(normalized)
    except Exception:
        return None

    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    must = [FieldCondition(key="data_version", match=MatchValue(value=data_version))]
    for k, v in filters.items():
        must.append(FieldCondition(key=k, match=MatchValue(value=v)))
    q_filter = Filter(must=must)

    try:
        hits = qdrant_client.search(
            collection_name=collection_name,
            query_vector=qvec,
            limit=limit,
            with_payload=True,
            query_filter=q_filter,
        )
    except Exception:
        return None

    new_tokens = tokens(normalized)
    new_entities = extract_entities(normalized)

    for hit in hits:
        score = getattr(hit, "score", 0) or 0
        if score < min_score:
            continue

        payload = getattr(hit, "payload", None) or {}
        cached_query = payload.get("normalized_query", "") or ""

        if new_tokens:
            cov = len(new_tokens.intersection(tokens(cached_query))) / max(len(new_tokens), 1)
            if cov < min_token_coverage:
                continue

        if new_entities:
            if not new_entities.issubset(extract_entities(cached_query)):
                continue

        redis_key = payload.get("redis_key")
        if not redis_key:
            continue

        try:
            raw = r.get(redis_key)
        except Exception:
            raw = None
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                return None

        # Cleanup dangling point (best-effort)
        try:
            from qdrant_client.http.models import PointIdsList

            qdrant_client.delete(collection_name=collection_name, points_selector=PointIdsList(points=[hit.id]))
        except Exception:
            pass

    return None


def semantic_store(
    enabled: bool,
    collection_name: str,
    query: str,
    data_version: str,
    embeddings,
    filters: Dict[str, Any],
    redis_key: str,
    ttl_seconds: int,
):
    if not enabled:
        return

    normalized = normalize(query)
    try:
        qvec = embeddings.embed_query(normalized)
    except Exception:
        return

    payload: Dict[str, Any] = {
        "redis_key": redis_key,
        "normalized_query": normalized,
        "data_version": data_version,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "ttl_seconds": ttl_seconds,
    }
    payload.update(filters)

    try:
        from qdrant_client.http.models import PointStruct

        qdrant_client.upsert(
            collection_name=collection_name,
            points=[PointStruct(id=str(uuid.uuid4()), vector=qvec, payload=payload)],
        )
    except Exception:
        pass


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def tokens(text: str) -> set[str]:
    ts = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {t for t in ts if t not in _STOPWORDS and len(t) > 1}


def extract_entities(text: str) -> set[str]:
    t = text.lower()
    years = set(re.findall(r"\b20\d{2}\b", t))
    numbers = set(re.findall(r"\b\d+\b", t))
    months = {m for m in _MONTHS if m in t}
    return years.union(numbers).union(months)

