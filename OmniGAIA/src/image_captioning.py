"""Замена отдельных изображений описаниями локального CapRL; видео и аудио сохраняются."""
import asyncio
import hashlib
import json
from pathlib import Path

from openrouter_baseline import ROOT, atomic_json
from caption_metrics import assert_no_images
from caption_protocol import protocol


class CaptioningAdapter:
    def __init__(self, run_dir, ledger):
        self.root = Path(run_dir)
        self.ledger = ledger
        self.process = None
        self.lock = asyncio.Lock()
        self.log = None

    async def request(self, data):
        async with self.lock:
            if self.process is None:
                self.root.mkdir(parents=True, exist_ok=True)
                self.log = (self.root / "caption_worker.log").open("a")
                self.process = await asyncio.create_subprocess_exec(
                    str(ROOT.parent / ".venv-caption/bin/python"), "-u", str(ROOT / "src/caption_worker.py"),
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self.log,
                    limit=4 * 1024**2)
            try:
                self.process.stdin.write((json.dumps(data) + "\n").encode())
                await self.process.stdin.drain()
                line = await self.process.stdout.readline()
            except BaseException:
                # После таймаута нельзя прочитать ответ старого изображения как новый.
                await self.close()
                raise
            if not line:
                await self.close()
                raise RuntimeError("Процесс captioner завершился; см. caption_worker.log")
            result = json.loads(line)
            if "error" in result:
                raise RuntimeError("Captioner: " + result["error"])
            return result

    async def transform(self, item, parts):
        result, images = [], []
        sources = [m["path"] for m in item["omni_modal_input"] if m["type"] == "image"]
        index = 0
        for part in parts:
            if part.get("type") != "image_url":
                result.append(part)
                continue
            index += 1
            record = await self.request({"image": part["image_url"]["url"], "index": index})
            record.update(occurrence_id=f"{item['id']}:{index}", source_path=sources[index - 1])
            if not record.get("eos_finished") or not record.get("caption", "").strip():
                raise RuntimeError("Caption не завершён или пуст; исходное изображение запрещено")
            images.append(record)
            result.append({"type": "text", "text": record["caption_block"]})
        if index != len(sources):
            raise RuntimeError("Число заменённых изображений не совпало с метаданными")
        assert_no_images(result)
        def other_media_hash(value):
            media = [p for p in value if p.get("type") in {"video_url", "input_audio"}]
            return hashlib.sha256(json.dumps(media, sort_keys=True).encode()).hexdigest()
        before, after = other_media_hash(parts), other_media_hash(result)
        if before != after:
            raise RuntimeError("Видео или аудио изменились при captioning")
        atomic_json(self.root / "caption_inputs" / f"{item['id']:03d}.json", {
            "item_id": item["id"], "images": images, "image_payloads_after": 0,
            "other_media_sha256_before": before, "other_media_sha256_after": after})
        self.ledger.register(item["id"], images)
        return result

    async def close(self):
        if self.process:
            if self.process.returncode is None:
                self.process.stdin.close()
                try:
                    await asyncio.wait_for(self.process.wait(), 20)
                except asyncio.TimeoutError:
                    self.process.terminate()
                    await self.process.wait()
            self.log.close()
            self.process = None
            self.log = None
