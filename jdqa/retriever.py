"""手写余弦相似度与 top-k 检索。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jdqa.config import Settings
from jdqa.embedder import embed_texts
from jdqa.store import IndexEntry, load_index


class RetrievalError(RuntimeError):
    """检索参数或索引内容不合法。"""


@dataclass(frozen=True)
class SearchResult:
    entry: IndexEntry
    score: float


def retrieve(
    question: str,
    settings: Settings,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[SearchResult]:
    """返回按余弦相似度降序排列且通过阈值过滤的结果。"""
    question = question.strip()
    if not question:
        raise RetrievalError("问题不能为空")

    effective_top_k = settings.top_k if top_k is None else top_k
    effective_min_score = settings.min_score if min_score is None else min_score
    if effective_top_k < 1:
        raise RetrievalError(f"top-k 必须大于 0，当前值：{effective_top_k}")
    if not 0.0 <= effective_min_score <= 1.0:
        raise RetrievalError(f"相似度阈值必须在 [0, 1] 之间，当前值：{effective_min_score}")

    index = load_index(settings)
    query_vector = embed_texts([question], settings)[0]
    if len(query_vector) != index.dim:
        raise RetrievalError(
            f"查询向量维度为 {len(query_vector)}，索引维度为 {index.dim}"
        )

    scores = _cosine_scores(query_vector, index.chunks)
    order = np.argsort(-scores, kind="stable")

    results: list[SearchResult] = []
    for position in order:
        score = float(scores[int(position)])
        if score < effective_min_score:
            break
        results.append(SearchResult(entry=index.chunks[int(position)], score=score))
        if len(results) >= effective_top_k:
            break
    return results


def _cosine_scores(query: list[float], entries: list[IndexEntry]) -> np.ndarray:
    """公式：dot(q, d) / (||q|| * ||d||)，整体用 numpy 向量化计算。"""
    matrix = np.asarray([entry.vector for entry in entries], dtype=np.float64)
    query_vector = np.asarray(query, dtype=np.float64)

    dot_products = matrix @ query_vector
    denominators = (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector) + 1e-8
    )
    return np.clip(dot_products / denominators, -1.0, 1.0)
