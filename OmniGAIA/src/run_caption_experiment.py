"""Smoke на пяти задачах, проверка протокола и продолжение на фиксированных 130."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from caption_metrics import summarize_caption_tokens
from caption_protocol import protocol
from openrouter_baseline import ROOT, atomic_json
from run_openrouter_baseline import request_usage

MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
REVISION = "adf542cf70735e061b8d73300cdbae9c847b9bc3"
SMOKE_IDS = [8, 17, 109, 124, 128]
FIXED_SHA256 = "f57628c0b0858e750c21af9152dbe671af6a401288d6e4c711dbca9a3df5e77d"


def audit_smoke(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest["item_ids"] != SMOKE_IDS or manifest.get("caption_protocol") != protocol()
            or manifest.get("model") != MODEL or manifest.get("dataset_revision") != REVISION):
        raise ValueError("Smoke: ID или протокол отличаются от зафиксированных")
    results = [json.loads((root / "items" / f"{i:03d}.json").read_text()) for i in SMOKE_IDS]
    if any(r.get("status") != "completed" or r.get("llm_equal") is None for r in results):
        raise ValueError("Smoke содержит технические ошибки; полный прогон запрещён")
    inputs = [json.loads((root / "caption_inputs" / f"{i:03d}.json").read_text()) for i in SMOKE_IDS]
    images = [row for item in inputs for row in item["images"]]
    if len(images) != 7 or any(not r["eos_finished"] or not r["caption"].strip() for r in images):
        raise ValueError("Smoke: неверное число изображений или незавершённые описания")
    for item in inputs:
        if item["image_payloads_after"] != 0 or item["other_media_sha256_before"] != item["other_media_sha256_after"]:
            raise ValueError("Smoke: нарушена замена изображений или сохранность других модальностей")
    events = [json.loads(line) for line in (root / "caption_transmissions.jsonl").read_text().splitlines()]
    sends = [r for r in events if r["event"] == "send"]
    if any(r["image_payload_count"] != 0 for r in sends):
        raise ValueError("Smoke: изображение попало в API-запрос")
    stats = summarize_caption_tokens(root, 7)
    if stats["image_occurrences"] != 7 or any(r["send_count"] < 1 for r in stats["images"]):
        raise ValueError("Smoke: не все изображения учтены в отправках")
    report = {"passed": True, "item_ids": SMOKE_IDS, "caption_metrics": stats,
              "correct": sum(r["llm_equal"] for r in results),
              "note_ru": "Переход разрешён по техническим проверкам; качество ответов не используется для отбора задач."}
    atomic_json(root / "smoke_audit.json", report)
    return manifest


def import_smoke(smoke, full, manifest):
    marker = full / "imported_results.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["source_run"] != str(smoke) or saved["item_ids"] != SMOKE_IDS:
            raise ValueError("Полный запуск уже содержит другой импорт")
        return
    if (full / "manifest.json").exists() or (full / "requests.jsonl").exists():
        raise ValueError("Каталог полного прогона занят; автоматическое смешивание запрещено")
    for folder in ("items", "caption_inputs"):
        (full / folder).mkdir(parents=True, exist_ok=True)
        for item_id in SMOKE_IDS:
            shutil.copy2(smoke / folder / f"{item_id:03d}.json", full / folder / f"{item_id:03d}.json")
    for name in ("requests.jsonl", "caption_transmissions.jsonl"):
        shutil.copy2(smoke / name, full / name)
    fields = ("model", "judge", "dataset_revision", "image_mode", "caption_protocol", "max_turns",
              "max_tokens", "sampling", "tools", "active_perception", "agent_protocol", "media_preprocessing", "jina_reader_mode")
    atomic_json(marker, {"source_run": str(smoke), "item_ids": SMOKE_IDS,
                        "protocol": {k: manifest[k] for k in fields},
                        "request_cost_usd": request_usage(smoke / "requests.jsonl")["cost_usd"],
                        "note_ru": "Импортированы все пять задач smoke, независимо от правильности. Передачи и расходы сохранены; повторный API-прогон этих задач не выполняется."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, default=ROOT / "outputs/openrouter_nemotron_caption_smoke5")
    parser.add_argument("--full-dir", type=Path, default=ROOT / "outputs/openrouter_nemotron_caption_fixed130")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    args.smoke_dir, args.full_dir = args.smoke_dir.resolve(), args.full_dir.resolve()
    args.smoke_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.smoke_dir / "experiment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fixed = ROOT / "data/omnigaia_fixed130/metadata.json"
    if hashlib.sha256(fixed.read_bytes()).hexdigest() != FIXED_SHA256:
        raise ValueError("Фиксированные 130 задач изменились")
    fixed_items = json.loads(fixed.read_text())
    smoke_items = json.loads((ROOT / "data/omnigaia_caption_smoke5/metadata.json").read_text())
    if len(fixed_items) != 130 or smoke_items != [r for r in fixed_items if r["id"] in SMOKE_IDS]:
        raise ValueError("Smoke не совпадает с исходными задачами фиксированного набора")

    def status(phase, **extra):
        atomic_json(args.smoke_dir / "experiment.json", {
            "phase": phase, "updated_at": datetime.now(timezone.utc).isoformat(),
            "smoke_dir": str(args.smoke_dir), "full_dir": str(args.full_dir), **extra})

    def run_job(directory, inputs):
        return subprocess.call([sys.executable, "-u", str(ROOT / "src/run_openrouter_job.py"),
                                "--run-dir", str(directory), "--model", MODEL,
                                "--image-mode", "image_captioning", "--input-file", str(inputs),
                                "--dataset-revision", REVISION, "--concurrency", str(args.concurrency)], cwd=ROOT,
                               env={**os.environ, "JINA_READER_MODE": "anonymous"})

    try:
        if not (args.smoke_dir / "smoke_audit.json").exists():
            status("smoke_running")
            code = run_job(args.smoke_dir, ROOT / "data/omnigaia_caption_smoke5/metadata.json")
            if code:
                raise RuntimeError(f"Smoke завершился с кодом {code}; нужны проверка и исправление")
        status("smoke_audit")
        manifest = audit_smoke(args.smoke_dir)
        import_smoke(args.smoke_dir, args.full_dir, manifest)
        status("full_running")
        code = run_job(args.full_dir, fixed)
        status("finished" if code == 0 else "full_has_errors", return_code=code)
        return code
    except Exception as exc:
        status("stopped", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
