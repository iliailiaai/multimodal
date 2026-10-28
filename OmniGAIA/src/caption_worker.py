"""Локальный GPU-процесс: только изображение и фиксированный запрос, JSONL по stdin/stdout."""
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import traceback

from caption_protocol import CAPTION_PROMPT, GENERATION_BUDGETS, protocol

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT.parent / "models/CapRL-Qwen3VL-2B"
COUNTER_DIR = ROOT.parent / "models/Nemotron-3-Nano-Omni-tokenizer"
CACHE_DIR = ROOT / "cache/caprl_captions"


class Worker:
    def __init__(self):
        import torch
        from PIL import Image
        from tokenizers import Tokenizer
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        import transformers

        if not (MODEL_DIR / "model.safetensors").exists():
            raise RuntimeError("Вес CapRL ещё не загружен и не проверен")
        self.torch, self.Image = torch, Image
        torch.set_num_threads(4)
        self.tokenizer = Tokenizer.from_file(str(COUNTER_DIR / "tokenizer.json"))
        spec = importlib.util.spec_from_file_location("nemotron_image_counter", COUNTER_DIR / "image_processing.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = json.loads((COUNTER_DIR / "preprocessor_config.json").read_text())
        self.image_processor = module.NemotronH_Nano_Omni_Reasoning_V3ImageProcessor(**config)
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
                        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                        "dtype": "bfloat16", "attention": "sdpa"}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def count_text(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def generate(self, image):
        # Здесь нет вопроса бенчмарка, ответа, поиска или истории основной модели.
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        attempts = []
        for budget in GENERATION_BUDGETS:
            started = time.monotonic()
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=budget)
            generated = output[0, inputs.input_ids.shape[-1]:].tolist()
            finished = bool(generated) and generated[-1] in eos
            attempts.append({"max_new_tokens": budget, "generated_tokens": len(generated),
                             "eos_finished": finished, "seconds": round(time.monotonic() - started, 3)})
            if finished:
                caption = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
                if not caption:
                    raise RuntimeError("CapRL вернул пустое описание")
                return {"caption": caption, "eos_finished": True, "generation_attempts": attempts,
                        "captioner_input_tokens": inputs.input_ids.shape[-1],
                        "captioner_image_grid_thw": inputs.image_grid_thw.tolist(), "runtime": self.runtime}
        raise RuntimeError("CapRL не завершил описание EOS даже при 8192 токенах; обрезанный текст не используется")

    def handle(self, request):
        if set(request) != {"image", "index"}:
            raise ValueError("Captioner принимает только image и index")
        header, payload = request["image"].split(",", 1)
        if not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("Ожидалось исходное изображение в data URL")
        raw = base64.b64decode(payload, validate=True)
        sha = hashlib.sha256(raw).hexdigest()
        image = self.Image.open(io.BytesIO(raw)).convert("RGB")
        cache_key = hashlib.sha256((sha + json.dumps(protocol(), sort_keys=True)).encode()).hexdigest()
        cache_path = CACHE_DIR / (cache_key + ".json")
        cached = cache_path.exists()
        if cached:
            result = json.loads(cache_path.read_text())
            if not result.get("eos_finished") or not result.get("caption", "").strip():
                raise RuntimeError("Повреждённый кэш captioner")
        else:
            result = self.generate(image)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            temporary.replace(cache_path)
        # Вызываем непосредственно реализацию HF-препроцессинга: без весов
        # Nemotron и без устаревшего _get_num_multimodal_tokens.
        native = self.image_processor._preprocess([image], return_tensors="pt")
        visual_tokens = int(native["num_tokens"][0])
        block = f"[Image {int(request['index'])} description]\n{result['caption']}\n[/Image {int(request['index'])} description]"
        return {**result, "image_sha256": sha, "width": image.width, "height": image.height,
                "cache_hit": cached, "caption_block": block,
                "image_visual_tokens": visual_tokens,
                "native_image_size_hw": native["imgs_sizes"][0].tolist(),
                "image_tokens_single": self.count_text("<img>" + "<image>" * visual_tokens + "</img>"),
                "caption_text_tokens_single": self.count_text(result["caption"]),
                "caption_tokens_single": self.count_text(block),
                "image_counter_api_verified": False, "protocol": protocol()}


def main():
    # В stdout только протокол; диагностические сообщения библиотек — в stderr.
    with contextlib.redirect_stdout(sys.stderr):
        worker = Worker()
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = worker.handle(json.loads(line))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
