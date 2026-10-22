"""确定性评测：8 条可回答 + 2 条应拒答。"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from jdqa.config import ConfigError, Settings, load_settings
from jdqa.embedder import EmbeddingError
from jdqa.generator import GenerationError, generate_answer
from jdqa.retriever import RetrievalError, SearchResult, retrieve
from jdqa.store import StoreError, load_index

CASES_PATH = Path(__file__).resolve().parent / "eval_cases.json"
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
REFUSAL_MARKERS = (
    "没有相关依据",
    "知识库中没有",
    "资料不足",
    "无法回答",
    "暂无相关",
    "没有找到相关",
)


class CaseFileError(RuntimeError):
    """评测用例文件缺失或格式不合法。"""


@dataclass(frozen=True)
class Case:
    question: str
    case_type: str
    expected_sources: tuple[str, ...]


@dataclass(frozen=True)
class CaseResult:
    passed: bool
    reason: str = ""


def load_cases() -> list[Case]:
    try:
        raw = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CaseFileError(f"找不到评测用例文件：{CASES_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise CaseFileError(f"评测用例不是合法 JSON：{exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise CaseFileError("评测用例必须是非空数组")

    cases: list[Case] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise CaseFileError(f"第 {index} 条用例必须是对象")

        question = item.get("question")
        case_type = item.get("type")
        expected = item.get("expected_sources", [])

        if not isinstance(question, str) or not question.strip():
            raise CaseFileError(f"第 {index} 条用例缺少有效的 question")
        if case_type not in {"answerable", "refusal"}:
            raise CaseFileError(
                f"第 {index} 条用例的 type 必须是 answerable 或 refusal"
            )
        if not isinstance(expected, list) or not all(
            isinstance(source, str) and source.strip() for source in expected
        ):
            raise CaseFileError(f"第 {index} 条用例的 expected_sources 格式不正确")
        if case_type == "answerable" and not expected:
            raise CaseFileError(f"第 {index} 条可回答用例缺少 expected_sources")

        cases.append(
            Case(
                question=question.strip(),
                case_type=case_type,
                expected_sources=tuple(expected),
            )
        )
    return cases


def evaluate(case: Case, settings: Settings) -> CaseResult:
    try:
        results = retrieve(case.question, settings)
    except (RetrievalError, StoreError, EmbeddingError) as exc:
        return CaseResult(False, f"检索执行错误：{exc}")

    if case.case_type == "refusal":
        return _evaluate_refusal(case, results, settings)

    if not results:
        return CaseResult(
            False,
            f"检索未命中（top-k={settings.top_k}，阈值={settings.min_score}）",
        )

    try:
        answer = generate_answer(case.question, results, settings)
    except GenerationError as exc:
        return CaseResult(False, f"生成执行错误：{exc}")

    citations = _extract_citations(answer)
    if not citations:
        return CaseResult(False, "回答未包含引用")

    out_of_range = sorted(
        {number for number in citations if number < 1 or number > len(results)}
    )
    if out_of_range:
        return CaseResult(
            False,
            f"引用编号越界：{_format_numbers(out_of_range)}（本次检索 {len(results)} 条）",
        )

    cited_sources = {results[number - 1].entry.source for number in citations}
    expected_sources = set(case.expected_sources)
    if not expected_sources.issubset(cited_sources):
        return CaseResult(
            False,
            "引用来源错误（期望 "
            f"{', '.join(sorted(expected_sources))}；实际 "
            f"{', '.join(sorted(cited_sources))}）",
        )
    return CaseResult(True)


def _evaluate_refusal(
    case: Case, results: list[SearchResult], settings: Settings
) -> CaseResult:
    if not results:
        return CaseResult(True)

    try:
        answer = generate_answer(case.question, results, settings)
    except GenerationError as exc:
        return CaseResult(False, f"生成执行错误：{exc}")

    citations = _extract_citations(answer)
    if citations:
        return CaseResult(
            False,
            f"该拒答未拒答（回答包含引用 {_format_numbers(citations)}）",
        )
    if any(marker in answer for marker in REFUSAL_MARKERS):
        return CaseResult(True)
    return CaseResult(False, "该拒答未拒答（回答未明确说明无依据）")


def _extract_citations(answer: str) -> list[int]:
    return [int(value) for value in CITATION_PATTERN.findall(answer)]


def _format_numbers(numbers: list[int]) -> str:
    return " ".join(f"[{number}]" for number in sorted(set(numbers))) or "无"


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    try:
        cases = load_cases()
    except CaseFileError as exc:
        print(f"[评测用例错误] {exc}", file=sys.stderr)
        return 2

    try:
        load_index(settings)
    except StoreError as exc:
        print(f"[索引错误] {exc}", file=sys.stderr)
        return 2

    answerable = sum(1 for case in cases if case.case_type == "answerable")
    refusal = len(cases) - answerable
    print(f"评测用例：{len(cases)} 条（可回答 {answerable}，应拒答 {refusal}）\n")

    passed = 0
    for number, case in enumerate(cases, start=1):
        result = evaluate(case, settings)
        passed += int(result.passed)
        label = "可回答" if case.case_type == "answerable" else "应拒答"
        status = "通过" if result.passed else "失败"
        print(f"[{number:02d}] {status} | {label} | {case.question}")
        if not result.passed:
            print(f"     原因：{result.reason}")

    rate = passed / len(cases) * 100
    print(f"\n通过 {passed}/{len(cases)}，通过率 {rate:.1f}%")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
