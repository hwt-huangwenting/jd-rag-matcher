"""零依赖本地 Web 界面：标准库 http.server 提供页面与问答接口。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from jdqa.chunker import chunk_documents
from jdqa.config import ConfigError, Settings, load_settings
from jdqa.embedder import EmbeddingError, embed_texts
from jdqa.generator import GenerationError, generate_answer
from jdqa.loader import LoaderError, load_documents
from jdqa.retriever import RetrievalError, retrieve
from jdqa.store import StoreError, load_index, save_index

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DIR = PROJECT_ROOT / "web"
MAX_BODY_BYTES = 8 * 1024
MAX_UPLOAD_BYTES = 3 * 1024 * 1024
MAX_FILE_CHARS = 200_000
MAX_UPLOAD_FILES = 10
MAX_QUESTION_CHARS = 300
MAX_TOP_K = 20
ALLOWED_SUFFIXES = {".md", ".txt"}
_CITATION_RE = re.compile(r"\[(\d+)\]")
_REBUILD_LOCK = threading.Lock()


def _safe_filename(raw: object, position: int) -> str:
    """只保留文件名本身，拒绝目录穿越与非文本扩展名。"""
    name = Path(str(raw or "")).name.strip()
    if not name or name in {".", ".."}:
        raise ValueError(f"第 {position} 个文件缺少合法文件名")
    if "\x00" in name:
        raise ValueError(f"第 {position} 个文件名包含非法字符")
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"第 {position} 个文件仅支持 .md 或 .txt")
    return name


def _atomic_write_text(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(target)


def _clear_jd_files(jd_dir: Path) -> None:
    if not jd_dir.exists():
        return
    for item in jd_dir.iterdir():
        if item.is_file() and item.suffix.lower() in ALLOWED_SUFFIXES:
            item.unlink()


class AppServer(ThreadingHTTPServer):
    """把配置挂在 server 上，供每个请求处理器读取。"""

    def __init__(self, address: tuple[str, int], settings: Settings) -> None:
        self.settings = settings
        super().__init__(address, AppHandler)


class AppHandler(BaseHTTPRequestHandler):
    server_version = "JDMatchWeb/0.1"

    @property
    def settings(self) -> Settings:
        return self.server.settings  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_index()
        elif path == "/api/config":
            self._send_json(200, self._config_payload())
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._send_json(404, {"error": "接口不存在"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        path = urlparse(self.path).path
        if path == "/api/ask":
            self._handle_ask()
        elif path == "/api/upload":
            self._handle_upload()
        elif path == "/api/rebuild":
            self._handle_rebuild()
        else:
            self._send_json(404, {"error": "接口不存在"})

    def _handle_ask(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        question = str(payload.get("question", "")).strip()
        if not question:
            self._send_json(400, {"error": "问题不能为空"})
            return
        if len(question) > MAX_QUESTION_CHARS:
            self._send_json(
                400, {"error": f"问题过长，请控制在 {MAX_QUESTION_CHARS} 字以内"}
            )
            return

        settings = self.settings
        try:
            top_k = int(payload.get("top_k", settings.top_k))
            min_score = float(payload.get("min_score", settings.min_score))
        except (TypeError, ValueError):
            self._send_json(400, {"error": "top_k 与 min_score 必须是数字"})
            return
        if not 1 <= top_k <= MAX_TOP_K:
            self._send_json(400, {"error": f"top_k 必须在 1 到 {MAX_TOP_K} 之间"})
            return
        if not 0.0 <= min_score <= 1.0:
            self._send_json(400, {"error": "min_score 必须在 0 到 1 之间"})
            return

        started = time.perf_counter()
        try:
            results = retrieve(question, settings, top_k=top_k, min_score=min_score)
        except (RetrievalError, StoreError, EmbeddingError) as exc:
            self._send_json(500, {"error": str(exc)})
            return

        if not results:
            self._send_json(
                200,
                {
                    "question": question,
                    "answer": "",
                    "retrieved": 0,
                    "cited": [],
                    "results": [],
                    "message": "知识库中没有找到足够相关的依据，可尝试降低相似度阈值。",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                },
            )
            return

        try:
            answer = generate_answer(question, results, settings)
        except GenerationError as exc:
            self._send_json(
                502,
                {
                    "error": str(exc),
                    "question": question,
                    "retrieved": len(results),
                    "results": self._results_payload(results),
                },
            )
            return

        cited = sorted(
            {
                int(number)
                for number in _CITATION_RE.findall(answer)
                if 1 <= int(number) <= len(results)
            }
        )
        self._send_json(
            200,
            {
                "question": question,
                "answer": answer,
                "retrieved": len(results),
                "cited": cited,
                "results": self._results_payload(results),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            },
        )

    def _handle_upload(self) -> None:
        if not self._is_local_client():
            self._send_json(403, {"error": "上传接口仅允许本机访问"})
            return

        try:
            payload = self._read_json_body(MAX_UPLOAD_BYTES)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        kind = str(payload.get("kind", ""))
        files = payload.get("files")
        if kind not in {"resume", "jd"}:
            self._send_json(400, {"error": "kind 必须是 resume 或 jd"})
            return
        if not isinstance(files, list) or not files:
            self._send_json(400, {"error": "files 不能为空"})
            return
        if len(files) > MAX_UPLOAD_FILES:
            self._send_json(400, {"error": f"单次最多上传 {MAX_UPLOAD_FILES} 个文件"})
            return
        if kind == "resume" and len(files) != 1:
            self._send_json(400, {"error": "简历一次只能上传 1 个文件"})
            return

        settings = self.settings
        saved: list[str] = []
        try:
            prepared: list[tuple[Path, str]] = []
            for position, item in enumerate(files, start=1):
                if not isinstance(item, dict):
                    raise ValueError(f"第 {position} 个文件格式错误")
                content = item.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"第 {position} 个文件内容为空")
                if len(content) > MAX_FILE_CHARS:
                    raise ValueError(
                        f"第 {position} 个文件超过 {MAX_FILE_CHARS} 字符上限"
                    )

                if kind == "resume":
                    target = settings.knowledge_dir / "resume.md"
                else:
                    target = (
                        settings.knowledge_dir
                        / "jd"
                        / _safe_filename(item.get("name"), position)
                    )
                prepared.append((target, content))

            if kind == "jd" and payload.get("replace_jd") is True:
                _clear_jd_files(settings.knowledge_dir / "jd")

            for target, content in prepared:
                _atomic_write_text(target, content)
                saved.append(target.relative_to(PROJECT_ROOT).as_posix())
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except OSError as exc:
            self._send_json(500, {"error": f"写入知识库失败：{exc}"})
            return

        self._send_json(
            200,
            {
                "saved": saved,
                "requires_rebuild": True,
                "message": "文件已保存，需重建索引后生效。",
            },
        )

    def _handle_rebuild(self) -> None:
        if not self._is_local_client():
            self._send_json(403, {"error": "重建接口仅允许本机访问"})
            return
        if not _REBUILD_LOCK.acquire(blocking=False):
            self._send_json(409, {"error": "已有重建任务在进行，请稍后再试"})
            return

        settings = self.settings
        started = time.perf_counter()
        try:
            documents = load_documents(settings.knowledge_dir)
            if not documents:
                raise LoaderError("knowledge 目录下没有可用的 .md 或 .txt 文档")
            chunks = chunk_documents(
                documents, settings.chunk_size, settings.chunk_overlap
            )
            if not chunks:
                raise LoaderError("切块结果为空")
            vectors = embed_texts([chunk.text for chunk in chunks], settings)
            save_index(settings, chunks, vectors)
        except (LoaderError, StoreError) as exc:
            self._send_json(500, {"error": str(exc)})
            return
        except EmbeddingError as exc:
            self._send_json(502, {"error": f"重建索引失败：{exc}"})
            return
        finally:
            _REBUILD_LOCK.release()

        self._send_json(
            200,
            {
                "documents": len(documents),
                "chunks": len(chunks),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            },
        )

    def _is_local_client(self) -> bool:
        return self.client_address[0] in {"127.0.0.1", "::1"}

    def _send_index(self) -> None:
        index_path = WEB_DIR / "index.html"
        try:
            body = index_path.read_bytes()
        except OSError as exc:
            self._send_json(500, {"error": f"页面文件读取失败：{exc}"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _config_payload(self) -> dict[str, object]:
        settings = self.settings
        payload: dict[str, object] = {
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "top_k": settings.top_k,
            "min_score": settings.min_score,
            "index_ready": False,
            "chunks": 0,
            "index_created_at": "",
        }
        try:
            index = load_index(settings)
        except StoreError as exc:
            payload["index_error"] = str(exc)
        else:
            payload["index_ready"] = True
            payload["chunks"] = len(index.chunks)
            payload["index_created_at"] = index.created_at
        return payload

    def _read_json_body(self, max_bytes: int = MAX_BODY_BYTES) -> dict[str, object]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length 不合法") from exc
        if length <= 0:
            raise ValueError("请求体为空")
        if length > max_bytes:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体必须是 UTF-8 编码的 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    @staticmethod
    def _results_payload(results) -> list[dict[str, object]]:
        return [
            {
                "number": number,
                "source": result.entry.source,
                "section": result.entry.section,
                "score": round(result.score, 4),
                "text": result.entry.text,
            }
            for number, result in enumerate(results, start=1)
        ]

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"[web] {self.address_string()} - {format % args}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="web.py", description="启动 JD 匹配问答器的本地 Web 界面。"
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(require_api_key=True)
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    if not settings.index_file.exists():
        print(
            f"[索引错误] 索引不存在：{settings.index_file}。"
            "请先运行 python main.py build。",
            file=sys.stderr,
        )
        return 2

    try:
        server = AppServer((args.host, args.port), settings)
    except OSError as exc:
        print(f"[启动失败] 无法监听 {args.host}:{args.port}：{exc}", file=sys.stderr)
        return 2

    print(f"服务已启动：http://{args.host}:{args.port}")
    print(
        f"模型：{settings.chat_model}  向量模型：{settings.embedding_model}"
        f"  top-k：{settings.top_k}  阈值：{settings.min_score}"
    )
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
