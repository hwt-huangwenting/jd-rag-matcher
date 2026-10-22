"""读取知识库中的 Markdown / TXT 文档。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jdqa.config import PROJECT_ROOT

SUPPORTED_SUFFIXES = {".md", ".txt"}


class LoaderError(RuntimeError):
    """知识库目录或文档不合法。"""


@dataclass(frozen=True)
class Document:
    """一篇待切块的原始文档。"""

    source: str
    text: str


def load_documents(knowledge_dir: Path) -> list[Document]:
    """读取目录下所有受支持的文本文件，按路径排序保证结果稳定。"""
    if not knowledge_dir.is_dir():
        raise LoaderError(f"知识库目录不存在：{knowledge_dir}")

    paths = sorted(
        path
        for path in knowledge_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise LoaderError(f"知识库目录中没有 .md 或 .txt 文件：{knowledge_dir}")

    documents: list[Document] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise LoaderError(f"文件不是有效的 UTF-8 文本：{path}") from exc
        except OSError as exc:
            raise LoaderError(f"无法读取文件：{path}（{exc}）") from exc

        if not text.strip():
            continue

        try:
            source = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            source = path.as_posix()
        documents.append(Document(source=source, text=text))

    if not documents:
        raise LoaderError(f"知识库中没有非空文档：{knowledge_dir}")
    return documents
