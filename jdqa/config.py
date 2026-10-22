"""加载并校验运行配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
PLACEHOLDER_KEY = "your_api_key_here"


class ConfigError(RuntimeError):
    """配置缺失或不合法。"""


def _read_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是整数，当前值：{raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} 不能小于 {minimum}，当前值：{value}")
    return value


def _read_float(name: str, default: float, low: float, high: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是数字，当前值：{raw!r}") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name} 必须在 [{low}, {high}] 之间，当前值：{value}")
    return value


@dataclass(frozen=True)
class Settings:
    """一次运行所需的全部配置。"""

    api_key: str
    base_url: str
    chat_model: str
    embedding_model: str
    embedding_dim: int
    embed_batch_size: int
    top_k: int
    min_score: float
    chunk_size: int
    chunk_overlap: int
    max_tokens: int
    request_timeout: int
    knowledge_dir: Path
    data_dir: Path
    index_file: Path

    @property
    def masked_api_key(self) -> str:
        """返回可安全打印的 Key 形式。"""
        if not self.api_key:
            return "（未设置）"
        if len(self.api_key) <= 8:
            return "*" * len(self.api_key)
        return f"{self.api_key[:4]}{'*' * 8}{self.api_key[-4:]}"


def load_settings(require_api_key: bool = True) -> Settings:
    """读取 .env 与环境变量并做基础校验。"""
    load_dotenv(ENV_FILE)

    api_key = os.getenv("ZHIPU_API_KEY", "").strip()
    if api_key == PLACEHOLDER_KEY:
        api_key = ""
    if require_api_key and not api_key:
        raise ConfigError(
            "未找到 ZHIPU_API_KEY。请将 .env.example 复制为 .env，"
            "并填写你的智谱 API Key。"
        )

    embedding_dim = _read_int("EMBEDDING_DIM", 1024)
    if embedding_dim != 1024:
        raise ConfigError(
            f"EMBEDDING_DIM 必须为 1024（embedding-3 的约定维度），当前值：{embedding_dim}"
        )

    chunk_size = _read_int("CHUNK_SIZE", 500)
    chunk_overlap = _read_int("CHUNK_OVERLAP", 80, minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigError(
            f"CHUNK_OVERLAP 必须小于 CHUNK_SIZE，当前值：{chunk_overlap} >= {chunk_size}"
        )

    data_dir = PROJECT_ROOT / "data"

    return Settings(
        api_key=api_key,
        base_url=os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").strip(),
        chat_model=os.getenv("CHAT_MODEL", "glm-4-flash").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "embedding-3").strip(),
        embedding_dim=embedding_dim,
        embed_batch_size=_read_int("EMBED_BATCH_SIZE", 16),
        top_k=_read_int("TOP_K", 6),
        min_score=_read_float("MIN_SCORE", 0.52, 0.0, 1.0),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_tokens=_read_int("MAX_TOKENS", 800),
        request_timeout=_read_int("REQUEST_TIMEOUT", 60),
        knowledge_dir=PROJECT_ROOT / "knowledge",
        data_dir=data_dir,
        index_file=data_dir / "index.json",
    )
