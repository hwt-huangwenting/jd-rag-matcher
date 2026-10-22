"""把文档切成可检索的文本块。

Markdown 按标题层级切分，TXT 先按空行分段再合并；
超长文本按句子边界继续切分，并保留字符重叠。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from jdqa.loader import Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_BLANK_LINE_RE = re.compile(r"\n\s*\n")
_SENTENCE_END = "。！？；!?;\n"


@dataclass(frozen=True)
class Chunk:
    """带来源信息的文本块。"""

    id: str
    source: str
    section: str
    text: str
    char_count: int


def chunk_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """按文档类型切块，并生成稳定 ID。"""
    chunks: list[Chunk] = []
    for document in documents:
        suffix = Path(document.source).suffix.lower()
        if suffix == ".md":
            sections = _markdown_sections(document.text)
        else:
            sections = _text_sections(document.text, chunk_size, chunk_overlap)

        index = 0
        for section, body in sections:
            for piece in _split_text(body, chunk_size, chunk_overlap):
                chunks.append(_make_chunk(document.source, section, index, piece))
                index += 1
    return chunks


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """按 ATX 标题收集章节，section 使用标题路径。"""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    current_section = "文档开头"
    in_code_block = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            buffer.append(line)
            continue

        match = None if in_code_block else _HEADING_RE.match(line)
        if match is None:
            buffer.append(line)
            continue

        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_section, body))

        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        # 一级标题视为文档标题，不放入引用路径，避免来源标注过长
        section_titles = [item for heading_level, item in stack if heading_level > 1]
        current_section = " > ".join(section_titles) if section_titles else title
        buffer = []

    body = "\n".join(buffer).strip()
    if body:
        sections.append((current_section, body))
    return sections


def _text_sections(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[tuple[str, str]]:
    """TXT 先按空行分段，短段落合并，超长段落单独切分。"""
    paragraphs = [item.strip() for item in _BLANK_LINE_RE.split(text) if item.strip()]
    pieces: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.extend(_split_text(paragraph, chunk_size, chunk_overlap))
            continue

        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            pieces.append(buffer)
            buffer = paragraph

    if buffer:
        pieces.append(buffer)
    return [("正文", piece) for piece in pieces]


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """按字符窗口切分，尽量在句子边界处断开。"""
    normalized = _normalize(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    pieces: list[str] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            boundary = _last_sentence_end(
                normalized, start + chunk_size // 2, end
            )
            if boundary is not None:
                end = boundary

        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= length:
            break

        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return pieces


def _last_sentence_end(text: str, low: int, high: int) -> int | None:
    """在 [low, high) 内从后向前找句子结束符，返回切分位置。"""
    for index in range(high - 1, low - 1, -1):
        if text[index] in _SENTENCE_END:
            return index + 1
    return None


def _normalize(text: str) -> str:
    """统一换行并压缩多余空行。"""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _make_chunk(source: str, section: str, index: int, text: str) -> Chunk:
    digest = hashlib.sha1(f"{source}|{section}|{index}".encode("utf-8")).hexdigest()
    return Chunk(
        id=digest[:12],
        source=source,
        section=section,
        text=text,
        char_count=len(text),
    )
