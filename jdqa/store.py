"""索引文件的读写与校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from jdqa.chunker import Chunk
from jdqa.config import Settings


class StoreError(RuntimeError):
    """索引不存在、格式非法或写入失败。"""


@dataclass(frozen=True)
class IndexEntry:
    id: str
    source: str
    section: str
    text: str
    char_count: int
    vector: list[float]


@dataclass(frozen=True)
class IndexData:
    model: str
    dim: int
    created_at: str
    chunks: list[IndexEntry]


def save_index(
    settings: Settings, chunks: list[Chunk], vectors: list[list[float]]
) -> Path:
    """校验后原子写入 index.json，返回索引路径。"""
    if len(chunks) != len(vectors):
        raise StoreError(
            f"文本块与向量数量不一致：{len(chunks)} 个块，{len(vectors)} 条向量"
        )

    entries: list[dict[str, object]] = []
    for chunk, vector in zip(chunks, vectors):
        if len(vector) != settings.embedding_dim:
            raise StoreError(
                f"文本块 {chunk.id} 的向量维度为 {len(vector)}，"
                f"期望 {settings.embedding_dim}"
            )
        entries.append(
            {
                "id": chunk.id,
                "source": chunk.source,
                "section": chunk.section,
                "text": chunk.text,
                "char_count": chunk.char_count,
                "vector": vector,
            }
        )

    payload = {
        "model": settings.embedding_model,
        "dim": settings.embedding_dim,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunks": entries,
    }

    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        temp_path = settings.index_file.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temp_path.replace(settings.index_file)
    except OSError as exc:
        raise StoreError(f"写入索引失败：{exc}") from exc

    return settings.index_file


def load_index(settings: Settings) -> IndexData:
    """读取索引并校验模型、维度与字段完整性。"""
    path = settings.index_file
    if not path.exists():
        raise StoreError(f"索引不存在：{path}。请先运行 python main.py build")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise StoreError(f"无法读取索引文件：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise StoreError(f"索引文件不是合法 JSON：{exc}") from exc

    if not isinstance(raw, dict):
        raise StoreError("索引文件顶层必须是 JSON 对象")

    model = raw.get("model")
    dim = raw.get("dim")
    created_at = raw.get("created_at", "")
    chunks_raw = raw.get("chunks")

    if model != settings.embedding_model:
        raise StoreError(
            f"索引模型为 {model!r}，当前配置为 {settings.embedding_model!r}，请重建索引"
        )
    if dim != settings.embedding_dim:
        raise StoreError(
            f"索引维度为 {dim!r}，当前配置为 {settings.embedding_dim}，请重建索引"
        )
    if not isinstance(chunks_raw, list) or not chunks_raw:
        raise StoreError("索引文件没有有效的 chunks")

    entries: list[IndexEntry] = []
    for position, item in enumerate(chunks_raw, start=1):
        if not isinstance(item, dict):
            raise StoreError(f"索引第 {position} 个文本块格式错误")
        try:
            vector = item["vector"]
            entry = IndexEntry(
                id=str(item["id"]),
                source=str(item["source"]),
                section=str(item["section"]),
                text=str(item["text"]),
                char_count=int(item["char_count"]),
                vector=[float(value) for value in vector],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreError(f"索引第 {position} 个文本块字段不合法") from exc

        if len(entry.vector) != settings.embedding_dim:
            raise StoreError(
                f"索引第 {position} 个文本块向量维度为 {len(entry.vector)}，"
                f"期望 {settings.embedding_dim}"
            )
        entries.append(entry)

    return IndexData(
        model=str(model),
        dim=int(dim),
        created_at=str(created_at),
        chunks=entries,
    )
