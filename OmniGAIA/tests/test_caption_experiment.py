"""Smoke проверяет корректность протокола, а не отбирает удачные ответы."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from caption_metrics import CaptionLedger
from caption_protocol import protocol
from openrouter_baseline import atomic_json
from run_caption_experiment import audit_smoke, MODEL, REVISION, SMOKE_IDS


class SmokeAuditTests(unittest.TestCase):
    def fixture(self, root):
        atomic_json(root / "manifest.json", {"item_ids": SMOKE_IDS, "caption_protocol": protocol(),
                                            "model": MODEL, "dataset_revision": REVISION})
        ledger = CaptionLedger(root)
        for item_id in SMOKE_IDS:
            atomic_json(root / "items" / f"{item_id:03d}.json", {"status": "completed", "llm_equal": 0})
            images = [{"occurrence_id": f"{item_id}:{index}", "image_sha256": f"sha-{item_id}-{index}",
                       "caption": "Full description", "caption_block": f"Description {index}", "eos_finished": True,
                       "image_tokens_single": 258, "caption_tokens_single": 5}
                      for index in range(2 if item_id in {8, 128} else 1)]
            atomic_json(root / "caption_inputs" / f"{item_id:03d}.json", {
                "images": images, "image_payloads_after": 0,
                "other_media_sha256_before": "same", "other_media_sha256_after": "same"})
            ledger.register(item_id, images)
            ledger.before_send(item_id, [{"role": "user", "content": [
                {"type": "text", "text": r["caption_block"]} for r in images]}], 1)

    def test_all_wrong_answers_still_pass_technical_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            audit_smoke(root)
            result = json.loads((root / "smoke_audit.json").read_text())
            self.assertTrue(result["passed"])
            self.assertEqual(result["correct"], 0)

    def test_failure_or_changed_audio_prevents_full_run(self):
        for corruption in ("failed_task", "changed_audio"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root)
                if corruption == "failed_task":
                    atomic_json(root / "items/017.json", {"status": "error"})
                else:
                    path = root / "caption_inputs/017.json"
                    value = json.loads(path.read_text())
                    value["other_media_sha256_after"] = "different"
                    atomic_json(path, value)
                with self.assertRaises(ValueError):
                    audit_smoke(root)
                self.assertFalse((root / "smoke_audit.json").exists())


if __name__ == "__main__":
    unittest.main()
