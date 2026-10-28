"""Проверки протокола, оценки и продолжения агентного диалога."""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openrouter_baseline import (
    BaselineAgent, evaluate, extract_answer, inline_to_openrouter,
    parse_judgement, redact_media, summarize, media_relative_path, OpenRouterClient,
)
from service_access import ServiceAccessError, check_service_access


class StubClient:
    def __init__(self, messages):
        self.messages = list(messages)
        self.requests = []

    async def complete(self, model, messages, **kwargs):
        self.requests.append(json.loads(json.dumps({"model": model, "messages": messages, **kwargs})))
        return self.messages.pop(0), {"usage": {}, "model": model}


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_exhausted_openrouter_credits_stop_without_retries(self):
        client = OpenRouterClient("test-key")
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(402, json={"error": {"message": "Insufficient credits"}})))
        try:
            with patch("openrouter_baseline.asyncio.sleep", new_callable=AsyncMock) as sleep:
                with self.assertRaises(ServiceAccessError):
                    await client.complete("model", [{"role": "user", "content": "Question"}])
                sleep.assert_not_awaited()
        finally:
            await client.close()

    async def test_multimodal_download_timeout_is_retried_but_invalid_input_is_not(self):
        for detail, retry in [("Download multimodal file timed out", True), ("Unsupported audio format", False)]:
            with self.subTest(detail=detail):
                payloads = []

                def respond(request):
                    payloads.append(json.loads(request.content))
                    if len(payloads) == 1:
                        return httpx.Response(400, json={"error": {"message": detail}})
                    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "done"}}]})

                client = OpenRouterClient("test-key")
                await client.http.aclose()
                client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
                try:
                    with patch("openrouter_baseline.asyncio.sleep", new_callable=AsyncMock):
                        if retry:
                            message, _ = await client.complete("model", [{"role": "user", "content": "Question"}])
                            self.assertEqual(message["content"], "done")
                            self.assertEqual(payloads[0], payloads[1])
                        else:
                            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                                await client.complete("model", [{"role": "user", "content": "Question"}])
                            self.assertEqual(len(payloads), 1)
                finally:
                    await client.close()

    async def test_provider_error_inside_http_200_retries_identical_context(self):
        payloads = []

        def respond(request):
            payloads.append(json.loads(request.content))
            failed = len(payloads) == 1
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "error" if failed else "stop",
                             "message": {"role": "assistant", "content": None if failed else "<answer>4</answer>"}}],
                "usage": {},
            })

        client = OpenRouterClient("test-key")
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            with patch("openrouter_baseline.asyncio.sleep", new_callable=AsyncMock):
                message, record = await client.complete("model", [{"role": "user", "content": "2 + 2?"}])
            self.assertEqual(message["content"], "<answer>4</answer>")
            self.assertEqual(record["attempt"], 2)
            self.assertEqual(len(payloads), 2)
            self.assertEqual(payloads[0], payloads[1])
        finally:
            await client.close()

    async def test_persistent_provider_error_is_not_an_answer(self):
        client = OpenRouterClient("test-key")
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            "choices": [{"finish_reason": "error", "message": {"content": None}}],
        })))
        try:
            with patch("openrouter_baseline.asyncio.sleep", new_callable=AsyncMock):
                with self.assertRaisesRegex(RuntimeError, "после 4 попыток"):
                    await client.complete("model", [{"role": "user", "content": "Question"}])
        finally:
            await client.close()


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_judge_access_failure_is_propagated(self):
        client = SimpleNamespace(complete=AsyncMock(side_effect=ServiceAccessError("OpenRouter", 402)))
        with self.assertRaises(ServiceAccessError):
            await evaluate(client, "Question", "<answer>prediction</answer>", "reference")
        self.assertEqual(client.complete.await_count, 1)

    def test_invalid_judge_responses_never_pass(self):
        for value in ["", None, "maybe", "Correct or Incorrect", "I cannot judge"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_judgement(value)
        self.assertFalse(parse_judgement("incorrect"))
        self.assertTrue(parse_judgement("Correct."))

    def test_last_answer_is_used(self):
        self.assertEqual(extract_answer("<answer>wrong</answer> then <answer>right</answer>"), "right")

    async def test_judge_failure_is_not_a_correct_answer(self):
        client = StubClient([{"content": ""}] * 3)
        result = await evaluate(client, "Question", "<answer>wrong</answer>", "right")
        self.assertIsNone(result["llm_equal"])
        self.assertTrue(result["judge_error"])

    async def test_exact_match_does_not_call_judge(self):
        client = StubClient([])
        result = await evaluate(client, "Question", "<answer>FOUR</answer>", "four")
        self.assertEqual(result["llm_equal"], 1)
        self.assertEqual(client.requests, [])

    def test_incomplete_run_does_not_report_full_score(self):
        results = [{"status": "completed", "llm_equal": 1, "Level": "Easy"}, {"status": "error"}]
        stats = summarize(results, 2)
        self.assertIsNone(stats["pass_at_1"])
        self.assertEqual(stats["errors"], 1)


class MediaTests(unittest.TestCase):
    def test_metadata_with_filename_only(self):
        self.assertEqual(str(media_relative_path({"path": "000000217710.jpg", "type": "image"})), "images/000000217710.jpg")
        self.assertEqual(str(media_relative_path({"path": "P4Tjj7ga5jE.wav", "type": "audio"})), "audios/P4Tjj7ga5jE.wav")

    def test_audio_uses_input_audio_not_audio_url(self):
        part = inline_to_openrouter({"inlineData": {"mimeType": "audio/mpeg", "data": "YWJj"}})
        self.assertEqual(part, {"type": "input_audio", "input_audio": {"data": "YWJj", "format": "mp3"}})

    def test_video_stays_video(self):
        part = inline_to_openrouter({"inlineData": {"mimeType": "video/mp4", "data": "YWJj"}})
        self.assertEqual(part["video_url"]["url"], "data:video/mp4;base64,YWJj")

    def test_logs_do_not_contain_encoded_media(self):
        value = [{"input_audio": {"data": "a" * 2000, "format": "mp3"}}, {"url": "data:video/mp4;base64,YWJj"}]
        redacted = json.dumps(redact_media(value))
        self.assertNotIn("a" * 2000, redacted)
        self.assertNotIn("base64", redacted)


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_quota_failure_is_not_sent_to_model_as_evidence(self):
        call = {"id": "call-1", "type": "function", "function": {"name": "web_search", "arguments": '{"query":"test"}'}}
        client = StubClient([{"tool_calls": [call]}])
        agent = BaselineAgent(client)
        agent.functions["web_search"] = AsyncMock(side_effect=ServiceAccessError("Serper", 403))
        with self.assertRaises(ServiceAccessError):
            await agent.run("Question", [])
        self.assertEqual(len(client.requests), 1)

    async def test_native_tools_round_trip_and_no_active_perception(self):
        call = {"id": "call-1", "type": "function", "function": {"name": "code_executor", "arguments": '{"code":"print(2 + 2)"}'}}
        client = StubClient([
            {"role": "assistant", "content": None, "tool_calls": [call], "reasoning_details": [{"type": "reasoning.text", "text": "Use a calculator."}]},
            {"role": "assistant", "content": "<answer>4</answer>"},
        ])
        agent = BaselineAgent(client)
        async def calculate(code):
            return {"stdout": "4", "error": None}
        agent.functions["code_executor"] = calculate
        result = await agent.run("What is 2 + 2?", [])
        self.assertEqual(result["output"], "<answer>4</answer>")
        names = {s["function"]["name"] for s in client.requests[0]["tools"]}
        self.assertEqual(names, {"web_search", "page_browser", "code_executor"})
        second = client.requests[1]["messages"]
        self.assertEqual(second[-1]["tool_call_id"], "call-1")
        self.assertTrue(second[-2]["reasoning_details"])
        self.assertEqual(json.loads(second[-1]["content"])["stdout"], "4")

    async def test_unknown_tool_is_not_executed(self):
        call = {"id": "call-1", "type": "function", "function": {"name": "read_video", "arguments": "{}"}}
        client = StubClient([{"tool_calls": [call]}, {"content": "<answer>unknown</answer>"}])
        result = await BaselineAgent(client).run("Question", [])
        self.assertIn("error", result["tool_calls"][0]["result"])


class ServiceAccessTests(unittest.IsolatedAsyncioTestCase):
    def test_target_site_forbidden_is_not_a_jina_billing_error(self):
        check_service_access("Jina", 403, '{"message":"Target site access denied"}')
        check_service_access("OpenRouter", 400, "Download multimodal file timed out")
        with self.assertRaises(ServiceAccessError):
            check_service_access("Jina", 403, '{"code":"AUTHZ_INSUFFICIENT_BALANCE"}')
        with self.assertRaises(ServiceAccessError):
            check_service_access("Serper", 403)

    async def test_page_browser_does_not_swallow_access_error_in_gather(self):
        from tools.web_tools import page_browser
        with patch("tools.web_tools._web_cache.get_page", new_callable=AsyncMock, return_value=None), \
             patch("tools.web_tools._extract_text_from_url_async", new_callable=AsyncMock,
                   side_effect=ServiceAccessError("Jina", 402)):
            with self.assertRaises(ServiceAccessError):
                await page_browser(["https://example.com"])

    async def test_runner_stops_queue_and_preserves_existing_incorrect_answer(self):
        from run_openrouter_baseline import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "items").mkdir()
            previous = {"id": 1, "status": "completed", "llm_equal": 0, "Level": "Easy", "output": "wrong"}
            (root / "items/001.json").write_text(json.dumps(previous))
            (root / "manifest.json").write_text(json.dumps({
                "model": "model", "judge": "judge", "item_ids": [1, 2, 3],
                "max_turns": 2, "max_tokens": 128, "dataset_revision": "test",
            }))
            args = SimpleNamespace(run_dir=root, model="model", judge="judge", max_turns=2,
                                   max_tokens=128, concurrency=1, item_timeout=30)
            items = [{"id": i, "question": "Question", "answer": "answer", "Level": "Easy"} for i in [1, 2, 3]]
            media = SimpleNamespace(prepare=AsyncMock(return_value=[]), close=AsyncMock())
            agent = SimpleNamespace(run=AsyncMock(side_effect=ServiceAccessError("Serper", 403)))
            with patch.dict(os.environ, SERPER_API_KEY="test", JINA_API_KEY="test"), \
                 patch("run_openrouter_baseline.validate_web_tools", new_callable=AsyncMock), \
                 patch("run_openrouter_baseline.MediaPreparation", return_value=media), \
                 patch("run_openrouter_baseline.BaselineAgent", return_value=agent), \
                 patch("tools.web_tools.save_web_cache"):
                with self.assertRaisesRegex(RuntimeError, "Прогон остановлен"):
                    await run(args, StubClient([]), items, {})
            self.assertEqual(agent.run.await_count, 1)
            self.assertEqual(json.loads((root / "items/001.json").read_text()), previous)
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["state"], "blocked")
            self.assertEqual(status["metrics"]["completed"], 1)
            self.assertEqual(status["metrics"]["correct"], 0)
            self.assertEqual(status["metrics"]["pending"], 1)
            self.assertIsNone(status["metrics"]["pass_at_1"])


if __name__ == "__main__":
    unittest.main()
