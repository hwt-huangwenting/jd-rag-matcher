"""调用 glm-4-flash 生成带引用编号的回答。"""

from __future__ import annotations

import re
import time

import requests

from jdqa.config import Settings
from jdqa.retriever import SearchResult

_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_CITATION_RE = re.compile(r"\[\d+\]")
_REFUSAL_MARKERS = (
    "没有相关依据",
    "知识库中没有",
    "资料不足",
    "无法回答",
    "未提及",
    "没有提到",
)

_SYSTEM_PROMPT = """你是「JD 匹配问答器」的回答生成器。你只能依据用户消息中提供的资料作答，不得使用资料之外的知识，不得编造候选人的经历或岗位要求。

必须遵守：
1. 每条关键结论后必须紧跟方括号编号标注依据，例如 [1]、[2]；回答正文至少包含一个引用，编号只能来自资料中已有的编号。不要只在末尾列来源而不在结论后标注。
2. 如果资料不足以回答问题，直接说明「知识库中没有相关依据」，不要猜测或补充外部信息；但只要资料包含回答问题所需的事实，就应基于资料作答，不要因为资料没有逐字重复问题的措辞而拒答。
3. 当问题涉及「是否匹配」时，严格按三段输出：
   - 岗位要求：只能摘录路径含 `jd/` 的资料，不得混入简历内容；
   - 候选人证据：只能摘录 `resume.md`；
   - 匹配结论与差距：说明匹配点和不足。
4. 来源角色必须匹配：岗位要求只能引用路径含 `jd/` 的资料；候选人的经历、技能和项目只能引用 `resume.md`。不要把简历内容写成岗位要求，也不要把 JD 内容写成候选人经历。
5. 回答使用简洁中文，不要逐段复述资料。"""


class GenerationError(RuntimeError):
    """对话模型请求失败或响应不合法。"""


def generate_answer(
    question: str, results: list[SearchResult], settings: Settings
) -> str:
    """把检索结果编号后交给对话模型，返回模型生成的回答正文。"""
    if not results:
        raise GenerationError("没有可用于回答的检索结果")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(question, results)},
    ]
    content = _extract_content(_post_chat(messages, settings))

    if _needs_citation_retry(content):
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "上一个回答没有标注引用。请重新回答原问题：在每条关键结论后"
                    "标注 [n]，编号只能来自资料；如果资料确实不足，请直接说明"
                    "「知识库中没有相关依据」。"
                ),
            },
        ]
        retry_content = _extract_content(_post_chat(retry_messages, settings))
        if _CITATION_RE.search(retry_content):
            content = retry_content

    return content


def _extract_content(body: dict[str, object]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GenerationError("对话模型响应中没有 choices")

    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise GenerationError("对话模型返回了空回答")
    return content.strip()


def _needs_citation_retry(content: str) -> bool:
    """资料存在时，非拒答回答必须带引用；否则重试一次。"""
    if _CITATION_RE.search(content):
        return False
    return not any(marker in content for marker in _REFUSAL_MARKERS)


def _build_user_prompt(question: str, results: list[SearchResult]) -> str:
    blocks: list[str] = []
    for number, result in enumerate(results, start=1):
        blocks.append(
            f"[{number}] 来源：{result.entry.source} · {result.entry.section}\n"
            f"内容：\n{result.entry.text}"
        )
    materials = "\n\n".join(blocks)
    return f"资料：\n\n{materials}\n\n问题：{question.strip()}"


def _post_chat(messages: list[dict[str, str]], settings: Settings) -> dict[str, object]:
    url = f"{settings.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.chat_model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": settings.max_tokens,
    }
    return _post_with_retry(url, headers, payload, settings.request_timeout)


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
            raise GenerationError(f"对话模型网络请求失败：{exc}") from exc
        else:
            if response.status_code == 200:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise GenerationError("对话模型返回了非 JSON 响应") from exc
                if not isinstance(body, dict):
                    raise GenerationError("对话模型响应格式不正确")
                return body

            last_error = _describe_http_error(response)
            if response.status_code not in _RETRY_STATUS:
                raise GenerationError(f"对话模型请求失败（{last_error}）")

        if attempt < _MAX_ATTEMPTS:
            time.sleep(2 ** (attempt - 1))

    raise GenerationError(
        f"对话模型请求重试 {_MAX_ATTEMPTS} 次后仍失败（{last_error}）"
    )


def _describe_http_error(response: requests.Response) -> str:
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
