"""Учёт изображений на каждой попытке API; визуальные токены — локальный контрфактуал."""
import json
import uuid
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path


def assert_no_images(value):
    if isinstance(value, dict):
        if value.get("type") in {"image_url", "input_image", "image"} or "image_url" in value:
            raise ValueError("В режиме captioning запрещены изображения в запросе основной модели")
        for child in value.values():
            assert_no_images(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_images(child)
    elif isinstance(value, str) and value.startswith("data:image/"):
        raise ValueError("В запрос основной модели попали байты изображения")


class CaptionLedger:
    def __init__(self, run_dir):
        self.root = Path(run_dir)
        self.path = self.root / "caption_transmissions.jsonl"
        self.images = {}

    def register(self, item_id, images):
        self.images[str(item_id)] = images

    def append(self, event):
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def before_send(self, item_id, messages, attempt):
        assert_no_images(messages)
        if str(item_id) not in self.images:
            raise ValueError("Для задачи не зарегистрирован результат замены изображений")
        # Считаем точные текстовые блоки исходного пользовательского сообщения;
        # цитирование описания ассистентом не считается повторной передачей изображения.
        parts = [p for m in messages if m.get("role") == "user"
                 for p in (m.get("content") if isinstance(m.get("content"), list) else [])]
        blocks = Counter(p.get("text") for p in parts if p.get("type") == "text")
        images = []
        for record in self.images.get(str(item_id), []):
            count = blocks[record["caption_block"]]
            if count != 1:
                raise ValueError("Описание отсутствует или продублировано в контексте")
            images.append({k: record[k] for k in (
                "occurrence_id", "image_sha256", "image_tokens_single", "caption_tokens_single")})
        request_id = str(uuid.uuid4())
        self.append({"event": "send", "request_id": request_id, "item_id": item_id,
                     "attempt": attempt, "image_payload_count": 0, "images": images})
        return request_id

    def received(self, request_id, body):
        self.append({"event": "response", "request_id": request_id,
                     "generation_id": body.get("id"), "usage": body.get("usage", {}),
                     "provider": body.get("provider"), "has_error": bool(body.get("error"))})


def summarize_caption_tokens(run_dir, expected_images=None):
    root = Path(run_dir)
    occurrences = {}
    for path in sorted((root / "caption_inputs").glob("*.json")):
        for row in json.loads(path.read_text())["images"]:
            occurrences[row["occurrence_id"]] = {
                **{k: row[k] for k in ("occurrence_id", "image_sha256", "image_tokens_single", "caption_tokens_single")},
                "send_count": 0, "image_tokens_total": 0, "caption_tokens_total": 0,
            }
    path = root / "caption_transmissions.jsonl"
    sends, responses = 0, set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["event"] == "response":
                responses.add(row["request_id"])
                continue
            if row["event"] != "send":
                continue
            sends += 1
            for img in row["images"]:
                target = occurrences.setdefault(img["occurrence_id"], {
                    **img, "send_count": 0, "image_tokens_total": 0, "caption_tokens_total": 0})
                target["send_count"] += 1
                target["image_tokens_total"] += img["image_tokens_single"]
                target["caption_tokens_total"] += img["caption_tokens_single"]
    exposed = [r for r in occurrences.values() if r["send_count"]]
    n = len(exposed)
    native = sum(r["image_tokens_total"] for r in exposed)
    captions = sum(r["caption_tokens_total"] for r in exposed)
    return {
        "expected_image_occurrences": expected_images, "prepared_image_occurrences": len(occurrences),
        "image_occurrences": n, "unique_images": len({r["image_sha256"] for r in exposed}),
        "image_transmissions": sum(r["send_count"] for r in exposed),
        "main_request_attempts": sends, "responses_with_json": len(responses),
        "image_tokens_total": native, "caption_tokens_total": captions,
        "image_tokens_mean_per_image": native / n if n else None,
        "caption_tokens_mean_per_image": captions / n if n else None,
        "saved_tokens_total": native - captions,
        "note_ru": "Повторные шаги и сетевые попытки включены. Попытка не доказывает обработку или оплату провайдером. Изображения оценены локальным процессором на фактической траектории captioning; API usage учитывается отдельно. Средние рассчитаны по вхождениям изображений с хотя бы одной отправкой.",
        "images": list(occurrences.values()),
    }
