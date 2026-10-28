"""Проверки запрета изображений, сохранения модальностей и повторного учёта токенов."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from caption_metrics import CaptionLedger, assert_no_images, summarize_caption_tokens
from image_captioning import CaptioningAdapter
from caption_protocol import CAPTION_PROMPT
from openrouter_baseline import OpenRouterClient, atomic_json


def record(index=1):
    return {"occurrence_id": f"17:{index}", "image_sha256": "same-image",
            "caption": "A bottle.", "caption_block": f"Image {index}: A bottle.",
            "eos_finished": True, "image_tokens_single": 258, "caption_tokens_single": 7}


class CaptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_still_images_replaced_and_question_not_sent_to_captioner(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = CaptionLedger(directory)
            adapter = CaptioningAdapter(directory, ledger)
            adapter.request = AsyncMock(side_effect=[record(1), record(2)])
            video = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}}
            audio = {"type": "input_audio", "input_audio": {"data": "AA", "format": "mp3"}}
            image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
            item = {"id": 17, "question": "SECRET QUESTION", "answer": "SECRET ANSWER",
                    "omni_modal_input": [{"type": "image", "path": "a.png"},
                                         {"type": "video", "path": "b.mp4"},
                                         {"type": "image", "path": "a.png"}]}
            transformed = await adapter.transform(item, [image, video, audio, image])
            self.assertEqual([p["type"] for p in transformed], ["text", "video_url", "input_audio", "text"])
            self.assertIs(transformed[1], video)
            self.assertIs(transformed[2], audio)
            for call in adapter.request.call_args_list:
                self.assertEqual(set(call.args[0]), {"image", "index"})
            self.assertEqual(CAPTION_PROMPT, "Please describe this image in detail.")

    async def test_unfinished_caption_never_reaches_model_or_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = CaptionLedger(directory)
            adapter = CaptioningAdapter(directory, ledger)
            adapter.request = AsyncMock(return_value={**record(), "eos_finished": False})
            with self.assertRaisesRegex(RuntimeError, "не завершён"):
                await adapter.transform({"id": 17, "omni_modal_input": [{"type": "image", "path": "a"}]},
                                        [{"type": "image_url", "image_url": {"url": "x"}}])
            self.assertFalse(ledger.images)

    async def test_tool_round_and_retry_are_counted_but_judge_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            row = record()
            ledger = CaptionLedger(directory)
            ledger.register(17, [row])
            atomic_json(Path(directory) / "caption_inputs/017.json", {"images": [row]})
            client = OpenRouterClient("fake")
            await client.http.aclose()
            attempts = []

            def respond(request):
                attempts.append(json.loads(request.content))
                if len(attempts) == 1:
                    return httpx.Response(503)
                return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]})

            client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            client.caption_ledger = ledger
            messages = [{"role": "user", "content": [{"type": "text", "text": row["caption_block"]}]}]
            try:
                with patch("openrouter_baseline.asyncio.sleep", new_callable=AsyncMock):
                    await client.complete("model", messages, item_id=17)
                    await client.complete("model", messages + [{"role": "assistant", "content": "step"}], item_id=17)
                    await client.complete("judge", [{"role": "user", "content": "evaluate"}], item_id=17, judge=True)
            finally:
                await client.close()
            summary = summarize_caption_tokens(directory, 1)
            self.assertEqual(summary["main_request_attempts"], 3)
            self.assertEqual(summary["image_transmissions"], 3)
            self.assertEqual(summary["image_tokens_total"], 774)
            self.assertEqual(summary["caption_tokens_total"], 21)
            self.assertEqual(summary["caption_tokens_mean_per_image"], 21)
            self.assertEqual(len(attempts), 4)

    async def test_image_is_rejected_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            client = OpenRouterClient("fake")
            await client.http.aclose()
            calls = []
            client.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: calls.append(r)))
            client.caption_ledger = CaptionLedger(directory)
            try:
                with self.assertRaises(ValueError):
                    await client.complete("model", [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}]}])
                self.assertEqual(calls, [])
            finally:
                await client.close()

    def test_same_file_in_different_occurrences_counts_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = CaptionLedger(directory)
            rows = [record(1), record(2)]
            ledger.register(17, rows)
            messages = [{"role": "user", "content": [{"type": "text", "text": r["caption_block"]} for r in rows]}]
            ledger.before_send(17, messages, 1)
            ledger.before_send(17, messages, 1)
            stats = summarize_caption_tokens(directory, 2)
            self.assertEqual(stats["image_occurrences"], 2)
            self.assertEqual(stats["unique_images"], 1)
            self.assertEqual(stats["image_transmissions"], 4)
            self.assertEqual(stats["caption_tokens_mean_per_image"], 14)
            with self.assertRaisesRegex(ValueError, "отсутствует"):
                ledger.before_send(17, [], 1)

    def test_raw_image_in_nested_data_is_rejected(self):
        with self.assertRaises(ValueError):
            assert_no_images({"nested": ["data:image/jpeg;base64,AAAA"]})


if __name__ == "__main__":
    unittest.main()
