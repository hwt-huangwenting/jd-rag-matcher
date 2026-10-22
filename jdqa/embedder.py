"""调用智谱 embedding-3 生成文本向量。"""

from __future__ import annotations

import time

import requests

from jdqa.config import Settings

_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_BATCH_INTERVAL_SECONDS = 0.2


class EmbeddingError(RuntimeError):
    """Embedding 请求失败或响应不合法。"""


def embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    """分批生成向量，返回顺序与输入文本一致。"""
    if not texts:
        return []

    vectors: list[list[float]] = []
    total_batches = (len(texts) + settings.embed_batch_size - 1) // settings.embed_batch_size

    for batch_number, start in enumerate(
        range(0, len(texts), settings.embed_batch_size), start=1
    ):
        batch = texts[start : start + settings.embed_batch_size]
        vectors.extend(_embed_batch(batch, settings))
        if batch_number < total_batches:
            time.sleep(_BATCH_INTERVAL_SECONDS)

    return vectors


def _embed_batch(batch: list[str], settings: Settings) -> list[list[float]]:
    url = f"{settings.base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.embedding_model,
        "input": batch,
        "dimensions": settings.embedding_dim,
    }

    body = _post_with_retry(url, headers, payload, settings.request_timeout)
    data = body.get("data")
    if not isinstance(data, list) or len(data) != len(batch):
        raise EmbeddingError(
            f"Embedding 返回数量不匹配：期望 {len(batch)} 条，实际 "
            f"{len(data) if isinstance(data, list) else '未知'}"
        )

    ordered = sorted(data, key=lambda item: item.get("index", 0))
    vectors: list[list[float]] = []
    for item in ordered:
        vector = item.get("embedding")
        if not isinstance(vector, list):
            raise EmbeddingError("Embedding 响应中缺少 embedding 字段")
        if len(vector) != settings.embedding_dim:
            raise EmbeddingError(
                f"向量维度不匹配：期望 {settings.embedding_dim}，实际 {len(vector)}"
            )
        try:
            vectors.append([float(value) for value in vector])
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding 向量中包含非数字值") from exc
    return vectors


def _post_with_retry(
    url: str, headers: dict[str, str], payload: dict[str, object], timeout: int
) -> dict[str, object]:
    last_error = "未知错误"

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.Timeout:
            last_error = f"请求超时（{timeout}s）"
        except requests.RequestException as exc:
            raise EmbeddingError(f"Embedding 网络请求失败：{exc}") from exc
        else:
            if response.status_code == 200:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise EmbeddingError("Embedding 接口返回了非 JSON 响应") from exc
                if not isinstance(body, dict):
                    raise EmbeddingError("Embedding 接口响应格式不正确")
                return body

            last_error = _describe_http_error(response)
            if response.status_code not in _RETRY_STATUS:
                raise EmbeddingError(f"Embedding 请求失败（{last_error}）")

        if attempt < _MAX_ATTEMPTS:
            time.sleep(2 ** (attempt - 1))

    raise EmbeddingError(f"Embedding 请求重试 {_MAX_ATTEMPTS} 次后仍失败（{last_error}）")


def _describe_http_error(response: requests.Response) -> str:
    """尽量提取接口错误信息，但限制长度避免刷屏。"""
    message = ""
    try:
        body = response.json()
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", ""))
        if not message:
            message = str(body.get("message", ""))
    except ValueError:
        message = response.text[:200]

    message = message.strip()
    return f"HTTP {response.status_code}: {message[:200] or '无错误详情'}"
