"""Persist child vectors in Chroma and resolve hits through parent records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = (
    PROJECT_ROOT
    / "解析试验"
    / "父子检索"
    / "结果"
    / "20261006-qwen-child-128"
)
CHILDREN_PATH = RESULT_DIR / "children.jsonl"
PARENTS_PATH = RESULT_DIR / "parents.jsonl"
COLLECTION_NAME = "ieee_access_child_chunks"
DISTANCE_METRIC = "cosine"
EMBED_BATCH_SIZE = 20
QUERY_TOP_K = 10

SPLIT_SCRIPT_DIR = PROJECT_ROOT / "解析试验" / "语义分割" / "scripts"
sys.path.insert(0, str(SPLIT_SCRIPT_DIR))

from split_documents import _load_embedding_model, load_env_file  # noqa: E402


def chroma_dir_for_project(project_root: Path) -> Path:
    """Use an ASCII-only workspace sibling; Chroma 1.5.9 HNSW fails on Unicode paths on Windows."""
    return Path(project_root).parents[1] / "chroma_ieee_submission_assistant"


CHROMA_DIR = chroma_dir_for_project(PROJECT_ROOT)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
    return records


def _chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int)):
            normalized[key] = value
        elif isinstance(value, float):
            if math.isfinite(value):
                normalized[key] = value
        else:
            normalized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalized


def _source_fingerprint(children_path: Path) -> str:
    digest = hashlib.sha256()
    with children_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _embedding_identity(embed_model: Any) -> str:
    return str(
        getattr(embed_model, "model_name", None)
        or getattr(embed_model, "model", None)
        or type(embed_model).__qualname__
    )


def _get_client(persist_dir: Path) -> Any:
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as error:
        raise RuntimeError(
            "Chroma is not installed in this Python environment. "
            "Update the project Conda environment from environment.yml first."
        ) from error

    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def build_persistent_index(
    children_path: Path,
    persist_dir: Path,
    embed_model: Any,
    *,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Upsert all child records and their vectors; refuses a different completed index."""
    children_path = Path(children_path)
    persist_dir = Path(persist_dir)
    child_records = read_jsonl(children_path)
    if not child_records:
        raise ValueError(f"No child records found in {children_path}")

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []
    for record in child_records:
        child_id = str(record.get("chunk_id", "")).strip()
        text = str(record.get("text", ""))
        metadata = record.get("metadata") or {}
        parent_id = str(metadata.get("parent_chunk_id", "")).strip()
        if not child_id or not text.strip() or not parent_id:
            raise ValueError("Every child needs a non-empty chunk_id, text, and parent_chunk_id")
        ids.append(child_id)
        documents.append(text)
        metadatas.append(_chroma_metadata(metadata))
    if len(ids) != len(set(ids)):
        raise ValueError("Child chunk_id values must be unique")

    manifest_path = persist_dir / "index_manifest.json"
    manifest = {
        "children_sha256": _source_fingerprint(children_path),
        "embedding_model": _embedding_identity(embed_model),
        "collection_name": collection_name,
        "distance_metric": DISTANCE_METRIC,
        "child_count": len(child_records),
    }
    previous_manifest = _load_manifest(manifest_path)
    if previous_manifest is not None and previous_manifest != manifest:
        raise ValueError(
            "This Chroma directory belongs to a different child dataset or embedding model; "
            "choose a new persistence directory instead of mixing indexes."
        )

    client = _get_client(persist_dir)
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": DISTANCE_METRIC},
            embedding_function=None,
        )
        if (collection.metadata or {}).get("hnsw:space") != DISTANCE_METRIC:
            raise ValueError(
                f"Collection {collection_name!r} must use the {DISTANCE_METRIC!r} HNSW space"
            )
        expected_ids = set(ids)
        existing_ids = set(collection.get().get("ids", []))
        unexpected_ids = existing_ids - expected_ids
        if unexpected_ids:
            raise ValueError(
                "The Chroma collection contains child IDs absent from this source file; "
                "use a new persistence directory to avoid stale records."
            )

        if previous_manifest == manifest and collection.count() == len(ids):
            return len(ids)

        for start in range(0, len(ids), EMBED_BATCH_SIZE):
            batch_ids = ids[start : start + EMBED_BATCH_SIZE]
            batch_documents = documents[start : start + EMBED_BATCH_SIZE]
            batch_vectors = embed_model.get_text_embedding_batch(batch_documents)
            if len(batch_vectors) != len(batch_ids):
                raise ValueError("Embedding service returned a different number of vectors than inputs")
            vectors = [[float(value) for value in vector] for vector in batch_vectors]
            if any(not vector or not all(math.isfinite(value) for value in vector) for vector in vectors):
                raise ValueError("Embedding service returned an empty or non-finite vector")
            collection.upsert(
                ids=batch_ids,
                embeddings=vectors,
                documents=batch_documents,
                metadatas=metadatas[start : start + EMBED_BATCH_SIZE],
            )

        if collection.count() != len(ids):
            raise RuntimeError(
                f"Chroma indexed {collection.count()} records; expected {len(ids)}"
            )
        _write_manifest(manifest_path, manifest)
        return len(ids)
    finally:
        client.close()


def retrieve_parent_chunks(
    query: str,
    parents_by_id: dict[str, dict[str, Any]],
    persist_dir: Path,
    embed_model: Any,
    *,
    collection_name: str = COLLECTION_NAME,
    top_k: int = QUERY_TOP_K,
) -> list[dict[str, Any]]:
    """Search persisted child vectors and return distinct parent records in rank order."""
    child_hits = retrieve_dense_child_hits(
        query,
        persist_dir,
        embed_model,
        collection_name=collection_name,
        top_k=top_k,
    )
    parents: list[dict[str, Any]] = []
    seen_parent_ids: set[str] = set()
    for child_hit in child_hits:
        child_id = child_hit["child_id"]
        metadata = child_hit["metadata"]
        parent_id = str(metadata.get("parent_chunk_id", ""))
        if not parent_id:
            raise ValueError(f"Retrieved child {child_id!r} has no parent_chunk_id metadata")
        if parent_id in seen_parent_ids:
            continue
        parent_record = parents_by_id.get(parent_id)
        if parent_record is None:
            raise ValueError(f"Retrieved child references missing parent {parent_id!r}")
        seen_parent_ids.add(parent_id)
        parents.append(
            {
                "parent_chunk_id": parent_id,
                "text": parent_record["text"],
                "metadata": parent_record.get("metadata", {}),
                "distance": child_hit["distance"],
                "score": child_hit["score"],
                "matched_child_id": child_id,
            }
        )
    return parents


def retrieve_dense_child_hits(
    query: str,
    persist_dir: Path,
    embed_model: Any,
    *,
    collection_name: str = COLLECTION_NAME,
    top_k: int = QUERY_TOP_K,
) -> list[dict[str, Any]]:
    """Return ranked child hits from Chroma without expanding them to parent context."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    client = _get_client(Path(persist_dir))
    try:
        try:
            collection = client.get_collection(name=collection_name, embedding_function=None)
        except Exception as error:
            raise ValueError(f"Chroma collection {collection_name!r} does not exist") from error
        count = collection.count()
        if count == 0:
            return []

        result = collection.query(
            query_embeddings=[embed_model.get_query_embedding(query)],
            n_results=min(top_k, count),
            include=["metadatas", "distances"],
        )
        child_ids = result.get("ids", [[]])[0]
        metadata_rows = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {
                "child_id": child_id,
                "metadata": metadata or {},
                "distance": float(distance),
                "score": 1.0 - float(distance),
            }
            for child_id, metadata, distance in zip(child_ids, metadata_rows, distances)
        ]
    finally:
        client.close()


def main() -> None:
    if not CHILDREN_PATH.is_file() or not PARENTS_PATH.is_file():
        raise FileNotFoundError("Expected parents.jsonl and children.jsonl in the existing trial output")

    load_env_file(PROJECT_ROOT / ".env")
    embedding = _load_embedding_model()
    indexed = build_persistent_index(CHILDREN_PATH, CHROMA_DIR, embedding)
    parent_records = read_jsonl(PARENTS_PATH)
    parents_by_id = {str(record["chunk_id"]): record for record in parent_records}

    # Reopen the persistent collection and exercise one end-to-end parent lookup.
    query = "IEEE Access submission guidelines for authors"
    hits = retrieve_parent_chunks(query, parents_by_id, CHROMA_DIR, embedding)
    print(f"Persisted {indexed} child vectors in {CHROMA_DIR}")
    print(f"Reopened Chroma and resolved {len(hits)} parent chunks for: {query}")
    for rank, hit in enumerate(hits, start=1):
        title = hit["metadata"].get("title", "")
        print(f"{rank}. {title} [{hit['parent_chunk_id']}] distance={hit['distance']:.4f}")


if __name__ == "__main__":
    main()
