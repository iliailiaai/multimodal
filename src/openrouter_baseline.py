"""Адаптер OpenRouter для baseline с исходными инструментами OmniGAIA."""

import asyncio
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import httpx
from service_access import ServiceAccessError, check_service_access

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
API_BASE = "https://openrouter.ai/api/v1"
MODEL = "qwen/qwen3.8-omni-flash"
JUDGE = "deepseek/deepseek-v3.2"


def load_env_file(path):
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def redact_media(value):
    if isinstance(value, list):
        return [redact_media(x) for x in value]
    if isinstance(value, dict):
        return {k: (f"<media:{len(v)} characters>" if isinstance(v, str) and
                    ((k == "data" and len(v) > 1000) or v.startswith("data:"))
                    else redact_media(v)) for k, v in value.items()}
    return value


def inline_to_openrouter(part):
    data = part["inlineData"]
    mime, encoded = data["mimeType"], data["data"]
    if mime.startswith("audio/"):
        audio_format = {"audio/mpeg": "mp3", "audio/x-wav": "wav"}.get(mime, mime.split("/")[1])
        return {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}}
    kind = "video_url" if mime.startswith("video/") else "image_url"
    return {"type": kind, kind: {"url": f"data:{mime};base64,{encoded}"}}


def media_relative_path(media):
    relative = Path(media["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Недопустимый путь медиа.")
    if len(relative.parts) == 1:
        directory = {"image": "images", "audio": "audios", "video": "videos"}[media["type"]]
        relative = Path(directory) / relative
    relative = Path(*relative.parts[-2:])
    if relative.parts[0] not in {"videos", "audios", "images"}:
        raise ValueError("Неизвестная директория медиа.")
    return relative


def extract_answer(output):
    matches = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)
    return matches[-1].strip() if matches else " ".join(output.split()[-20:])


def normalize_answer(value):
    return " ".join(str(value or "").lower().split())


def parse_judgement(content):
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)
    label = content.strip().strip(".\"'` \n").casefold()
    if label == "correct":
        return True
    if label == "incorrect":
        return False
    raise ValueError("Судья не вернул однозначный Correct/Incorrect.")


class OpenRouterClient:
    def __init__(self, key, request_log=None, timeout=600):
        self.key = key
        self.request_log = Path(request_log) if request_log else None
        self.http = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self.service_error = None
        self.free_lock = asyncio.Lock()
        self.free_next_at = 0.0
        self.caption_ledger = None

    async def close(self):
        await self.http.aclose()

    async def complete(self, model, messages, *, tools=None, tool_choice="auto",
                       max_tokens=20480, judge=False, item_id=None):
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
                   "provider": {"require_parameters": True, "allow_fallbacks": False}}
        if judge:
            payload.update(temperature=0, reasoning={"enabled": False})
        else:
            payload.update(temperature=0.6, top_p=0.95)
            if not model.startswith("nvidia/nemotron-3-nano-omni-"):
                payload["top_k"] = 20
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice)
        if model.startswith("nvidia/nemotron-3-nano-omni-"):
            payload.pop("tool_choice", None)
            if tool_choice == "none":
                payload.pop("tools", None)
        start = time.monotonic()
        for attempt in range(4):
            if self.service_error:
                raise self.service_error
            try:
                if model.endswith(":free"):
                    async with self.free_lock:
                        await asyncio.sleep(max(0, self.free_next_at - time.monotonic()))
                        self.free_next_at = time.monotonic() + 3.1
                caption_request = (self.caption_ledger.before_send(item_id, messages, attempt + 1)
                                   if self.caption_ledger is not None and not judge else None)
                response = await self.http.post(
                    API_BASE + "/chat/completions", json=payload,
                    headers={"Authorization": "Bearer " + self.key},
                )
                check_service_access("OpenRouter", response.status_code)
                if model.endswith(":free") and response.status_code == 429:
                    if "day" in response.text.lower() or attempt == 3:
                        raise ServiceAccessError("OpenRouter free quota", 429)
                retryable = response.status_code in {429, 500, 502, 503, 504} or (
                    response.status_code == 400 and "Download multimodal file timed out" in response.text
                )
                if retryable and attempt < 3:
                    LOGGER.warning("Задача %s: временный сбой OpenRouter HTTP %s, повтор %s/3",
                                   item_id, response.status_code, attempt + 1)
                    await asyncio.sleep(min(2 ** (attempt + 1), 20))
                    continue
                if response.status_code != 200:
                    detail = response.text.replace(self.key, "<redacted>")[:1200]
                    raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {detail}")
                body = response.json()
                if caption_request is not None:
                    self.caption_ledger.received(caption_request, body)
                error = body.get("error") or {}
                if isinstance(error, dict) and error.get("code") in {429, 500, 502, 503, 504} and attempt < 3:
                    LOGGER.warning("Задача %s: временная ошибка провайдера внутри HTTP 200, повтор %s/3", item_id, attempt + 1)
                    await asyncio.sleep(10 * (attempt + 1))
                    continue
                if body.get("error") or not body.get("choices"):
                    raise RuntimeError("OpenRouter: " + str(body.get("error", "empty choices"))[:1200])
                choice = body["choices"][0]
                record = {"item_id": item_id, "id": body.get("id"), "model": body.get("model"),
                          "provider": body.get("provider"), "judge": judge,
                          "usage": body.get("usage", {}), "finish_reason": choice.get("finish_reason"),
                          "attempt": attempt + 1,
                          "elapsed_seconds": round(time.monotonic() - start, 3)}
                if self.request_log:
                    self.request_log.parent.mkdir(parents=True, exist_ok=True)
                    with self.request_log.open("a") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                # OpenRouter может вернуть ошибку провайдера внутри HTTP 200.
                # Не добавляем такой ответ в диалог: повторяем исходный запрос.
                if choice.get("finish_reason") == "error" or choice.get("error"):
                    if attempt < 3:
                        LOGGER.warning("Задача %s: ошибка провайдера, повтор %s/3", item_id, attempt + 1)
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    detail = str(choice.get("error") or "finish_reason=error").replace(self.key, "<redacted>")
                    raise RuntimeError("Ошибка провайдера OpenRouter после 4 попыток: " + detail[:1000])
                return choice["message"], record
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** (attempt + 1))


class MediaPreparation:
    def __init__(self, revision, cache_limit_mb=512):
        from run_base_agent import GeminiBaseAgent
        self.revision = revision
        self.cache = ROOT / "cache" / "openrouter_media"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.cache_limit = cache_limit_mb * 1024**2
        self.lock = asyncio.Lock()
        self.preprocessor = GeminiBaseAgent("unused", "media-only", "http://localhost", ffmpeg_timeout=180)
        self.http = httpx.AsyncClient(follow_redirects=True, timeout=180)

    async def close(self):
        await self.preprocessor.close()
        await self.http.aclose()

    def prune_cache(self):
        files = sorted(self.cache.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        total = 0
        for path in files:
            total += path.stat().st_size
            if total > self.cache_limit:
                path.unlink()

    async def prepare(self, item):
        result = []
        async with self.lock:
            for media in item["omni_modal_input"]:
                relative = media_relative_path(media)
                cache_key = hashlib.sha256(f"gemini-preprocess-v1:{self.revision}:{relative}:{media['type']}".encode()).hexdigest()
                cache_file = self.cache / (cache_key + ".json")
                if cache_file.exists():
                    result.extend(json.loads(cache_file.read_text()))
                    cache_file.touch()
                    continue
                url = f"https://huggingface.co/datasets/RUC-NLPIR/OmniGAIA/resolve/{self.revision}/{relative.as_posix()}"
                with tempfile.TemporaryDirectory(prefix="raw-", dir=self.cache) as temporary:
                    path = Path(temporary) / relative.name
                    async with self.http.stream("GET", url) as response:
                        response.raise_for_status()
                        size = int(response.headers.get("content-length", 0))
                        if size + 400 * 1024**2 > shutil.disk_usage(self.cache).free:
                            raise RuntimeError("Недостаточно места для исходного медиафайла.")
                        with path.open("wb") as handle:
                            async for chunk in response.aiter_bytes(1024**2):
                                if shutil.disk_usage(self.cache).free < 300 * 1024**2:
                                    raise RuntimeError("Остановлена загрузка: мало свободного места.")
                                handle.write(chunk)
                    LOGGER.info("Загружен %s: %.1f МБ", relative, path.stat().st_size / 1024**2)
                    if media["type"] == "video":
                        video = await asyncio.to_thread(self.preprocessor._compress_video, str(path), max_frames=128)
                        audio = await self.preprocessor._file_to_base64_part(str(path), input_type="audio")
                        if not video or not audio:
                            raise RuntimeError(f"Не удалось подготовить видео и звук: {relative}")
                        parts = [inline_to_openrouter(video), inline_to_openrouter(audio)]
                    else:
                        part = await self.preprocessor._file_to_base64_part(str(path), input_type=media["type"])
                        if not part:
                            raise RuntimeError(f"Не удалось подготовить медиа: {relative}")
                        parts = [inline_to_openrouter(part)]
                atomic_json(cache_file, parts)
                self.prune_cache()
                result.extend(parts)
        return result


class BaselineAgent:
    def __init__(self, client, model=MODEL, max_turns=50, max_tokens=20480):
        from run_base_agent import SYSTEM_PROMPT, OPENAI_TOOLS_SCHEMA
        from tools.web_tools import web_search, page_browser
        from tools.isolated_code_executor import isolated_code_executor
        self.client, self.model = client, model
        self.max_turns, self.max_tokens = max_turns, max_tokens
        self.system_prompt, self.schemas = SYSTEM_PROMPT, OPENAI_TOOLS_SCHEMA
        self.functions = {"web_search": web_search, "page_browser": page_browser,
                          "code_executor": isolated_code_executor}

    async def run(self, question, parts, item_id=None):
        from run_base_agent import FORCED_FINAL_ANSWER_PROMPT
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": question}] + parts}]
        calls, requests = [], []
        for turn in range(self.max_turns + 1):
            forced = turn == self.max_turns
            if forced:
                messages.append({"role": "user", "content": FORCED_FINAL_ANSWER_PROMPT})
            message, usage = await self.client.complete(
                self.model, messages, tools=self.schemas, tool_choice="none" if forced else "auto",
                max_tokens=self.max_tokens, item_id=item_id,
            )
            requests.append(usage)
            assistant = {k: v for k, v in message.items()
                         if k in {"role", "content", "tool_calls", "reasoning", "reasoning_details"} and v is not None}
            assistant["role"] = "assistant"
            messages.append(assistant)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                output = message.get("content") or ""
                if not output:
                    raise RuntimeError("Модель вернула пустой ответ без вызовов инструментов.")
                return {"output": output, "messages": redact_media(messages),
                        "tool_calls": calls, "requests": requests, "forced_final": forced}
            if forced:
                raise RuntimeError("Модель вызвала инструменты после принудительного завершения.")
            for call in tool_calls:
                name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    if name not in self.functions:
                        raise ValueError(f"Инструмент {name} не доступен в baseline.")
                    if name == "page_browser" and isinstance(arguments.get("urls"), str):
                        arguments["urls"] = [arguments["urls"]]
                    result = await self.functions[name](**arguments)
                except (ValueError, TypeError) as exc:
                    result = {"error": str(exc)}
                calls.append({"turn": turn + 1, "name": name, "arguments": call["function"]["arguments"], "result": result})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
                LOGGER.info("Задача %s, шаг %s: %s", item_id, turn + 1, name)


async def evaluate(client, question, output, answer, judge=JUDGE, item_id=None):
    predicted = extract_answer(output)
    em = int(bool(predicted) and normalize_answer(predicted) == normalize_answer(answer))
    if em or not predicted:
        return {"predicted_answer": predicted, "em_score": em, "llm_equal": em,
                "judge_response": "Exact match" if em else "Empty prediction", "judge_error": None}
    prompt = f"""Please determine if the model correctly predicted the answer.
Question: {question}
Model Predicted Answer: {predicted}
Labeled Answer: {answer}
Return 'Correct' if the model's prediction is completely accurate, otherwise return 'Incorrect'. Provide only this single word response."""
    last_error = None
    for _ in range(3):
        try:
            message, _ = await client.complete(judge, [{"role": "user", "content": prompt}],
                                               judge=True, max_tokens=128, item_id=item_id)
            label = parse_judgement(message.get("content"))
            return {"predicted_answer": predicted, "em_score": em, "llm_equal": int(label),
                    "judge_response": message.get("content"), "judge_error": None}
        except ServiceAccessError:
            raise
        except (ValueError, RuntimeError) as exc:
            last_error = str(exc)
    return {"predicted_answer": predicted, "em_score": em, "llm_equal": None,
            "judge_response": None, "judge_error": last_error}


def summarize(results, expected_count):
    completed = [r for r in results if r.get("status") == "completed"]
    judged = [r for r in completed if r.get("llm_equal") is not None]
    correct = sum(r["llm_equal"] for r in judged)
    by_level = {}
    for level in ("Easy", "Medium", "Hard"):
        subset = [r for r in judged if r.get("Level") == level]
        by_level[level] = {"count": len(subset), "correct": sum(r["llm_equal"] for r in subset)}
    return {"expected_count": expected_count, "completed": len(completed), "judged": len(judged),
            "errors": len(results) - len(completed), "correct": correct,
            "pass_at_1": correct / expected_count if len(judged) == expected_count and expected_count else None,
            "partial_accuracy": correct / len(judged) if judged else None,
            "by_level": by_level,
            "tool_calls": dict(Counter(c["name"] for r in completed for c in r.get("tool_calls", [])))}
