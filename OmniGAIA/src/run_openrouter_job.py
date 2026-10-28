"""Долгий запуск OmniGAIA: блокировка дублей, состояние процесса и итоговый расход."""

import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx

from openrouter_baseline import API_BASE, ROOT, atomic_json, load_env_file
from run_openrouter_baseline import request_usage
from jina_access import anonymous


def now():
    return datetime.now(timezone.utc).isoformat()


async def billing_snapshot():
    result = {"timestamp": now()}
    endpoints = {
        "openrouter": (API_BASE + "/key", os.environ["OPENROUTER_API_KEY"]),
    }
    if not anonymous():
        endpoints["jina"] = ("https://dash.jina.ai/api/v1/authorization", os.environ["JINA_API_KEY"])
    else:
        result["jina"] = {"mode": "anonymous", "rpm_limit": 20}
    async with httpx.AsyncClient(timeout=30) as client:
        for service, (url, key) in endpoints.items():
            try:
                response = await client.get(url, headers={"Authorization": "Bearer " + key})
                response.raise_for_status()
                body = response.json()
                data = body.get("data", body)
                if service == "openrouter":
                    result[service] = {k: data.get(k) for k in ("usage", "usage_daily", "limit_remaining")}
                else:
                    result[service] = {"wallet": {k: data.get("wallet", {}).get(k) for k in
                                                  ("trial_balance", "regular_balance", "total_balance")},
                                       "has_payment_method": bool(data.get("payment_method"))}
            except Exception as exc:
                result[service] = {"error_type": type(exc).__name__}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--env-file", type=Path, default=ROOT.parent / ".omnigaia-secrets.env")
    parser.add_argument("--model")
    parser.add_argument("--image-mode", choices=("native", "image_captioning"), default="native")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--dataset-revision")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    load_env_file(args.env_file)
    lock = (run_dir / "job.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Этот прогон уже работает.")
    started_at = now()
    attempt = {"started_at": started_at, "pid": os.getpid()}
    before = asyncio.run(billing_snapshot())
    if not (run_dir / "billing_before.json").exists():
        atomic_json(run_dir / "billing_before.json", before)
    command = [sys.executable, "-u", str(ROOT / "src/run_openrouter_baseline.py"),
               "--run-dir", str(run_dir), "--env-file", str(args.env_file.resolve()),
               "--concurrency", str(args.concurrency)]
    for flag, value in (("--model", args.model), ("--input-file", args.input_file), ("--dataset-revision", args.dataset_revision), ("--image-mode", args.image_mode)):
        if value is not None:
            command.extend([flag, str(value)])
    with (run_dir / "run.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        attempt["runner_pid"] = process.pid

        def stop(signum, frame):
            if process.poll() is None:
                process.send_signal(signum)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        print(json.dumps({**attempt, "run_dir": str(run_dir)}, ensure_ascii=False), flush=True)
        while True:
            code = process.poll()
            atomic_json(run_dir / "process.json", {
                **attempt, "updated_at": now(), "running": code is None, "return_code": code,
                "openrouter_usage": request_usage(run_dir / "requests.jsonl"),
            })
            if code is not None:
                break
            time.sleep(15)
    after = asyncio.run(billing_snapshot())
    atomic_json(run_dir / "billing_after.json", after)
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"state": "failed_before_results"}
    imported_path = run_dir / "imported_results.json"
    imported = json.loads(imported_path.read_text()) if imported_path.exists() else {}
    usage = request_usage(run_dir / "requests.jsonl")
    report.update(
        started_at=started_at, finished_at=now(), return_code=process.returncode,
        openrouter_usage=usage,
        imported_results_count=len(imported.get("item_ids", [])),
        imported_openrouter_cost_usd=imported.get("request_cost_usd", 0),
        new_openrouter_cost_usd=usage["cost_usd"] - imported.get("request_cost_usd", 0),
        billing_note_ru="Стоимость по журналу запросов включает импортированные ответы пилота. Новые расходы указаны отдельно. Баланс ключа может включать сторонние параллельные обращения.",
    )
    atomic_json(report_path, report)
    print(json.dumps({"state": report["state"], "return_code": process.returncode,
                      "new_openrouter_cost_usd": report["new_openrouter_cost_usd"]}, ensure_ascii=False), flush=True)
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
