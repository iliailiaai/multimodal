"""Запуск baseline OmniGAIA через OpenRouter без активного восприятия."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time

from openrouter_baseline import (
    API_BASE, MODEL, JUDGE, ROOT, OpenRouterClient, MediaPreparation, BaselineAgent,
    atomic_json, evaluate, load_env_file, redact_media, summarize, media_relative_path,
)
from jina_access import headers as reader_headers, required_keys
from service_access import ServiceAccessError


def request_usage(path):
    usage = {"requests": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            data = record.get("usage", {})
            usage["requests"] += 1
            usage["cost_usd"] += data.get("cost", 0) or 0
            usage["input_tokens"] += data.get("prompt_tokens", 0) or 0
            usage["output_tokens"] += data.get("completion_tokens", 0) or 0
    return usage


async def catalog(client):
    response = await client.http.get(API_BASE + "/models")
    response.raise_for_status()
    return {m["id"]: m for m in response.json()["data"]}


async def dataset_revision(client, items):
    response = await client.http.get("https://huggingface.co/api/datasets/RUC-NLPIR/OmniGAIA")
    response.raise_for_status()
    body = response.json()
    available = {f["rfilename"] for f in body["siblings"]}
    required = {media_relative_path(m).as_posix() for i in items for m in i["omni_modal_input"]}
    missing = required - available
    if missing:
        raise RuntimeError("На Hugging Face отсутствуют медиа: " + ", ".join(sorted(missing)))
    return body["sha"]


async def validate_web_tools(client):
    # Проверяем доступ отдельно от заданий: ошибки авторизации не должны
    # превращаться в якобы низкое качество проверяемой модели.
    response = await client.http.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": os.environ["SERPER_API_KEY"]},
        json={"q": "OmniGAIA RUC NLPIR", "num": 1},
    )
    response.raise_for_status()
    if not response.json().get("organic"):
        raise RuntimeError("Serper не вернул результаты проверочного поиска.")
    response = await client.http.get(
        "https://r.jina.ai/https://example.com",
        # У проверочной страницы бывает чужой снимок в общем кэше Jina.
        # Проверяем реальную загрузку; настройки инструмента не меняем.
        headers={**reader_headers(), "X-No-Cache": "true"},
    )
    response.raise_for_status()
    if "Example Domain" not in response.text:
        raise RuntimeError("Jina не вернула содержимое проверочной страницы.")


async def preflight(args, client, items, models):
    from tools.isolated_code_executor import isolated_code_executor
    result = {"model": args.model, "judge": args.judge, "active_perception": False,
              "missing_web_keys": [k for k in required_keys() if not os.environ.get(k)]}
    sandbox = await isolated_code_executor("print(2 + 2)")
    if sandbox.get("stdout") != "4" or sandbox.get("error"):
        raise RuntimeError(f"Проверка Python не прошла: {sandbox}")
    result["python_sandbox"] = sandbox
    agent = BaselineAgent(client, args.model, max_turns=3, max_tokens=4096)
    tool_probe = await agent.run(
        "Use code_executor to calculate 2 + 2 and print the result. Then return <answer>4</answer>. Do not search the web.",
        [], "preflight-tools",
    )
    if not any(c["name"] == "code_executor" for c in tool_probe["tool_calls"]):
        raise RuntimeError("Модель не выполнила проверочный вызов code_executor.")
    result["tool_probe"] = tool_probe
    result["judge_correct"] = await evaluate(client, "What is 2 + 2?", "<answer>four</answer>", "4", args.judge, "preflight-judge")
    result["judge_incorrect"] = await evaluate(client, "What is 2 + 2?", "<answer>5</answer>", "4", args.judge, "preflight-judge")
    if result["judge_correct"]["llm_equal"] != 1 or result["judge_incorrect"]["llm_equal"] != 0:
        raise RuntimeError("Проверка модели-судьи не прошла.")
    revision = getattr(args, "dataset_revision", None) or await dataset_revision(client, items)
    result["dataset_revision"] = revision
    atomic_json(args.run_dir / "preflight.json", result)
    media = MediaPreparation(revision)
    try:
        result["media_probes"] = []
        for item in items[:2]:
            parts = await media.prepare(item)
            message, record = await client.complete(
                args.model, [{"role": "user", "content": [
                    {"type": "text", "text": "Briefly describe what is visible and what is audible in these inputs. Do not answer any benchmark question. Use at most 120 words."},
                    *parts,
                ]}], max_tokens=4096, item_id=f"preflight-media-{item['id']}",
            )
            if not message.get("content"):
                raise RuntimeError("Пустой ответ на проверку медиа.")
            result["media_probes"].append({"id": item["id"], "input_types": [p["type"] for p in parts],
                                            "response": message, "request": record})
            atomic_json(args.run_dir / "preflight.json", result)
    finally:
        await media.close()
    if not result["missing_web_keys"]:
        await validate_web_tools(client)
        result["web_tools"] = "ok"
    atomic_json(args.run_dir / "preflight.json", result)
    print(json.dumps(redact_media(result), ensure_ascii=False, indent=2))


async def run(args, client, items, models):
    missing = [k for k in required_keys() if not os.environ.get(k)]
    if missing:
        raise RuntimeError("Для baseline нужны: " + ", ".join(missing))
    await validate_web_tools(client)
    manifest_path = args.run_dir / "manifest.json"
    item_ids = [i["id"] for i in items]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = (args.model, args.judge, item_ids, args.max_turns, args.max_tokens)
        actual = (manifest["model"], manifest["judge"], manifest["item_ids"], manifest["max_turns"], manifest["max_tokens"])
        if expected != actual:
            raise RuntimeError("Параметры отличаются от сохранённого запуска. Укажите другую --run-dir.")
        revision = manifest["dataset_revision"]
    else:
        revision = getattr(args, "dataset_revision", None) or await dataset_revision(client, items)
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "model": args.model, "judge": args.judge, "model_info": models[args.model],
            "judge_info": models[args.judge], "item_ids": item_ids,
            "dataset_revision": revision, "input_sha256": hashlib.sha256(args.input_file.read_bytes()).hexdigest(),
            "active_perception": False, "tools": ["web_search", "page_browser", "code_executor"],
            "media_preprocessing": "GeminiBaseAgent: max 128 uniformly sampled video frames; 2 FPS output; smart_resize min 224^2 max 512^2 pixels; original images; separate mono 16kHz MP3 audio, first 30 minutes",
            "agent_protocol": "OpenRouter native function calling; original baseline system prompt and tool schemas",
            "model_api_adaptations": (["top_k omitted: unsupported", "tool_choice omitted: provider default; tools removed on forced final"] if args.model.startswith("nvidia/nemotron-3-nano-omni-") else []),
            "evaluation_protocol": "upstream answer extraction and normalization; DeepSeek-V3.2; strict Correct/Incorrect parsing; judge failures excluded from final score until resolved",
            "max_turns": args.max_turns, "max_tokens": args.max_tokens,
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": None if args.model.startswith("nvidia/nemotron-3-nano-omni-") else 20},
            "jina_reader_mode": os.environ.get("JINA_READER_MODE", "authenticated"),
            "concurrency": args.concurrency,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((ROOT / "src").rglob("*.py"))},
            "requirements_sha256": hashlib.sha256((ROOT / "requirements-openrouter.lock").read_bytes()).hexdigest(),
        }
        imported_path = args.run_dir / "imported_results.json"
        if imported_path.exists():
            imported = json.loads(imported_path.read_text())
            for field, value in imported["protocol"].items():
                if manifest.get(field) != value:
                    raise RuntimeError(f"Протокол импортированных результатов отличается: {field}")
            manifest["imported_results"] = imported
        atomic_json(manifest_path, manifest)
    media = MediaPreparation(revision)
    agent = BaselineAgent(client, args.model, args.max_turns, args.max_tokens)
    semaphore = asyncio.Semaphore(args.concurrency)
    results = {}
    stop_reason = None

    def publish(state="running"):
        ordered = [results[i["id"]] for i in items if i["id"] in results]
        stats = summarize(ordered, len(items))
        stats.update(recorded=len(results), pending=len(items) - len(results))
        usage = request_usage(args.run_dir / "requests.jsonl")
        atomic_json(args.run_dir / "metrics.json", stats)
        atomic_json(args.run_dir / "status.json", {
            "state": state, "updated_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(), "stop_reason": stop_reason,
            "metrics": stats, "openrouter_usage": usage,
            "imported_count": len(manifest.get("imported_results", {}).get("item_ids", [])),
        })
        return ordered, stats

    async def process(item):
        nonlocal stop_reason
        path = args.run_dir / "items" / f"{item['id']:03d}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get("status") == "completed" and previous.get("llm_equal") is not None:
                results[item["id"]] = previous
                return
        else:
            previous = None
        async with semaphore:
            if stop_reason:
                return
            start = time.monotonic()
            entry = {"id": item["id"], "question": item["question"], "Level": item["Level"],
                     "category": item.get("category"), "task_type": item.get("task_type"), "answer": item["answer"]}
            try:
                if previous and previous.get("output"):
                    entry.update(previous)
                else:
                    logging.info("Начата задача %s (%s)", item["id"], item["Level"])
                    parts = await media.prepare(item)
                    generated = await asyncio.wait_for(
                        agent.run(item["question"], parts, item["id"]), timeout=args.item_timeout,
                    )
                    entry.update(generated, status="generated")
                    atomic_json(path, entry)
                evaluation = await evaluate(client, item["question"], entry["output"], item["answer"], args.judge, item["id"])
                entry.update(evaluation, status="completed", elapsed_seconds=round(time.monotonic() - start, 2))
            except ServiceAccessError as exc:
                stop_reason = str(exc)
                client.service_error = exc
                entry.update(status="blocked", error=str(exc), elapsed_seconds=round(time.monotonic() - start, 2))
                logging.error("Остановка прогона: %s", exc)
            except Exception as exc:
                entry.update(status="error", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=round(time.monotonic() - start, 2))
                logging.exception("Ошибка задачи %s", item["id"])
            atomic_json(path, entry)
            results[item["id"]] = entry
            _, stats = publish("stopping" if stop_reason else "running")
            logging.info("Прогресс: %s/%s; правильных %s; ошибок %s", len(results), len(items), stats["correct"], stats["errors"])

    publish()
    try:
        await asyncio.gather(*(process(item) for item in items))
    finally:
        from tools.web_tools import save_web_cache
        save_web_cache()
        await media.close()
    ordered, stats = publish("blocked" if stop_reason else "finished")
    atomic_json(args.run_dir / "results.json", ordered)
    atomic_json(args.run_dir / "report.json", {
        "model": args.model, "judge": args.judge, "active_perception": False,
        "state": "blocked" if stop_reason else ("completed_with_errors" if stats["errors"] else "completed"),
        "stop_reason": stop_reason, "metrics": stats,
        "openrouter_usage": request_usage(args.run_dir / "requests.jsonl"),
        "score_note_ru": "Технические ошибки учитываются отдельно. Итоговый pass_at_1 появится только после оценки всех выбранных задач.",
    })
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stop_reason:
        raise RuntimeError("Прогон остановлен, результаты сохранены: " + stop_reason)
    if stats["errors"] or stats["judged"] != len(items):
        raise RuntimeError("Прогон содержит ошибки; результаты сохранены. Повторный запуск продолжит обработку.")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT.parent / ".omnigaia-secrets.env")
    parser.add_argument("--input-file", type=Path, default=ROOT / "data/test_metadata.json")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--judge", default=JUDGE)
    parser.add_argument("--preflight", action="store_true", help="Проверить API, инструменты и передачу медиа без оценки бенчмарка")
    parser.add_argument("--ids", help="Номера задач через запятую; без параметра используются все 360")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=20480)
    parser.add_argument("--item-timeout", type=int, default=36000)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/openrouter_qwen38_baseline")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    load_env_file(args.env_file)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        parser.error("Нужен OPENROUTER_API_KEY в окружении или --env-file.")
    if min(args.concurrency, args.max_turns, args.max_tokens, args.item_timeout) < 1:
        parser.error("Лимиты должны быть положительными.")
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items должен быть положительным.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    items = json.loads(args.input_file.read_text())
    if args.ids:
        ids = {int(i) for i in args.ids.split(",")}
        selected = [i for i in items if i["id"] in ids]
        if len(selected) != len(ids):
            parser.error("Указан отсутствующий ID задачи.")
        items = selected
    if args.max_items:
        items = items[:args.max_items]
    client = OpenRouterClient(key, args.run_dir / "requests.jsonl")
    try:
        models = await catalog(client)
        for model in (args.model, args.judge):
            if model not in models:
                raise RuntimeError(f"Модель {model} отсутствует в каталоге OpenRouter.")
        if args.preflight:
            await preflight(args, client, items, models)
        else:
            await run(args, client, items, models)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
