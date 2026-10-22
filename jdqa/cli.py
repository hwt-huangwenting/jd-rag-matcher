"""命令行入口定义。"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Iterable

from jdqa.chunker import Chunk, chunk_documents
from jdqa.config import ENV_FILE, ConfigError, Settings, load_settings
from jdqa.embedder import EmbeddingError, embed_texts
from jdqa.generator import GenerationError, generate_answer
from jdqa.loader import LoaderError, load_documents
from jdqa.retriever import RetrievalError, SearchResult, retrieve
from jdqa.store import StoreError, save_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jd-rag-matcher",
        description="JD 匹配问答器：检索本地简历与招聘 JD，生成带引用来源的回答。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="构建向量索引")
    build.add_argument("--force", action="store_true", help="覆盖已有索引")
    build.add_argument("--dry-run", action="store_true", help="只切块，不调用 Embedding")
    build.add_argument("--show-chunks", action="store_true", help="打印切块结果")

    ask = subparsers.add_parser("ask", help="提问并生成带引用的回答")
    ask.add_argument("question", help="要提问的问题")
    ask.add_argument("--top-k", type=int, default=None, help="检索条数，默认读取 TOP_K")
    ask.add_argument(
        "--min-score", type=float, default=None, help="相似度阈值，默认读取 MIN_SCORE"
    )
    ask.add_argument("--show-context", action="store_true", help="打印命中的文本块")

    subparsers.add_parser("chat", help="进入交互式问答")

    subparsers.add_parser("config", help="打印当前配置（API Key 已掩码）")

    return parser


def _print_config(settings: Settings) -> None:
    env_status = f"{ENV_FILE}（{'存在' if ENV_FILE.exists() else '不存在'}）"
    items: list[tuple[str, object]] = [
        ("ZHIPU_API_KEY", settings.masked_api_key),
        ("ZHIPU_BASE_URL", settings.base_url),
        ("CHAT_MODEL", settings.chat_model),
        ("EMBEDDING_MODEL", settings.embedding_model),
        ("EMBEDDING_DIM", settings.embedding_dim),
        ("EMBED_BATCH_SIZE", settings.embed_batch_size),
        ("TOP_K", settings.top_k),
        ("MIN_SCORE", settings.min_score),
        ("CHUNK_SIZE", settings.chunk_size),
        ("CHUNK_OVERLAP", settings.chunk_overlap),
        ("MAX_TOKENS", settings.max_tokens),
        ("REQUEST_TIMEOUT", settings.request_timeout),
        ("KNOWLEDGE_DIR", settings.knowledge_dir),
        ("DATA_DIR", settings.data_dir),
        ("INDEX_FILE", settings.index_file),
        ("ENV_FILE", env_status),
    ]
    width = max(len(name) for name, _ in items)
    for name, value in items:
        print(f"{name:<{width}}  {value}")


def _run_build(settings: Settings, args: argparse.Namespace) -> int:
    if not args.dry_run and settings.index_file.exists() and not args.force:
        print(f"[索引已存在] {settings.index_file}", file=sys.stderr)
        print("如需重建，请加 --force。", file=sys.stderr)
        return 1

    try:
        documents = load_documents(settings.knowledge_dir)
    except LoaderError as exc:
        print(f"[知识库错误] {exc}", file=sys.stderr)
        return 2

    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        print("[知识库错误] 切块结果为空。", file=sys.stderr)
        return 2

    print(f"文档数：{len(documents)}  切块数：{len(chunks)}")
    for document in documents:
        count = sum(1 for chunk in chunks if chunk.source == document.source)
        print(f"  {document.source}：{count} 块")

    if args.show_chunks:
        _print_chunks(chunks)

    if args.dry_run:
        print("dry-run 完成：未调用 Embedding，未写入索引。")
        return 0

    print(
        f"正在生成向量：{len(chunks)} 个文本块"
        f"（batch={settings.embed_batch_size}）..."
    )
    started = time.perf_counter()
    try:
        vectors = embed_texts([chunk.text for chunk in chunks], settings)
    except EmbeddingError as exc:
        print(f"[Embedding 错误] {exc}", file=sys.stderr)
        return 2

    try:
        index_path = save_index(settings, chunks, vectors)
    except StoreError as exc:
        print(f"[索引错误] {exc}", file=sys.stderr)
        return 2

    elapsed = time.perf_counter() - started
    print(f"索引已写入：{index_path}")
    print(f"向量维度：{settings.embedding_dim}  耗时：{elapsed:.2f}s")
    return 0


def _run_ask(settings: Settings, args: argparse.Namespace) -> int:
    question = args.question.strip()
    if not question:
        print("[输入错误] 问题不能为空。", file=sys.stderr)
        return 2

    top_k = args.top_k if args.top_k is not None else settings.top_k
    min_score = args.min_score if args.min_score is not None else settings.min_score

    try:
        results = retrieve(question, settings, top_k=top_k, min_score=min_score)
    except RetrievalError as exc:
        print(f"[检索错误] {exc}", file=sys.stderr)
        return 2
    except StoreError as exc:
        print(f"[索引错误] {exc}", file=sys.stderr)
        return 2
    except EmbeddingError as exc:
        print(f"[Embedding 错误] {exc}", file=sys.stderr)
        return 2

    if not results:
        print(f"知识库中没有找到足够相关的依据（top-k={top_k}，阈值={min_score}）。")
        return 0

    print(f"检索到 {len(results)} 个文本块（top-k={top_k}，阈值={min_score}）")
    try:
        answer = generate_answer(question, results, settings)
    except GenerationError as exc:
        print(f"[生成错误] {exc}", file=sys.stderr)
        print("\n已检索到的来源：")
        _print_citations(results)
        return 2

    print("\n回答\n----")
    print(answer)

    print("\n引用来源\n----")
    cited_numbers = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    in_range_numbers = sorted(
        number for number in cited_numbers if 1 <= number <= len(results)
    )
    if in_range_numbers:
        _print_citations(results, in_range_numbers)
    else:
        print("（回答未标注引用，以下为本次检索结果）")
        _print_citations(results)

    invalid_numbers = sorted(
        number for number in cited_numbers if number < 1 or number > len(results)
    )
    if invalid_numbers:
        print(f"\n[提示] 回答包含无法映射的引用编号：{invalid_numbers}", file=sys.stderr)

    if args.show_context:
        print("\n命中的文本块\n----")
        _print_contexts(results)
    return 0


def _print_citations(
    results: list[SearchResult], numbers: Iterable[int] | None = None
) -> None:
    selected = range(1, len(results) + 1) if numbers is None else sorted(set(numbers))
    for number in selected:
        result = results[number - 1]
        print(
            f"[{number}] {result.entry.source} · {result.entry.section}"
            f"（相似度 {result.score:.4f}）"
        )


def _print_contexts(results: list[SearchResult]) -> None:
    for number, result in enumerate(results, start=1):
        print(
            f"\n[{number}] {result.entry.source} · {result.entry.section}"
            f"（相似度 {result.score:.4f}）"
        )
        print(result.entry.text)


def _run_chat(settings: Settings, args: argparse.Namespace) -> int:
    print("进入连续问答模式，每轮独立检索；输入 exit 退出。")
    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if question.lower() in {"exit", "quit", "q"}:
            return 0
        if not question:
            print("问题不能为空。")
            continue

        turn_args = argparse.Namespace(
            question=question, top_k=None, min_score=None, show_context=False
        )
        code = _run_ask(settings, turn_args)
        if code != 0:
            print("本轮失败，可继续提问或输入 exit 退出。")


def _print_chunks(chunks: list[Chunk]) -> None:
    for number, chunk in enumerate(chunks, start=1):
        print(
            f"\n--- [{number}/{len(chunks)}] {chunk.id}  "
            f"{chunk.source} · {chunk.section}  ({chunk.char_count} 字符)"
        )
        print(chunk.text)


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    needs_api_key = args.command in {"ask", "chat"} or (
        args.command == "build" and not args.dry_run
    )
    try:
        settings = load_settings(require_api_key=needs_api_key)
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    if args.command == "config":
        _print_config(settings)
        return 0

    if args.command == "build":
        return _run_build(settings, args)

    if args.command == "ask":
        return _run_ask(settings, args)

    if args.command == "chat":
        return _run_chat(settings, args)

    return 1
