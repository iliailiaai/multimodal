"""Зафиксированные параметры captioning без зависимостей от API и GPU."""
CAPTION_PROMPT = "Please describe this image in detail."
CAPTION_REVISION = "b838cec8b6c4791ed6169d12db21db0b47bb70e5"
NEMOTRON_REVISION = "e5e9932441de940c9a62185c870ea5bcd4cd24e2"
GENERATION_BUDGETS = (2048, 8192)


def protocol():
    return {"mode": "image_captioning", "caption_model": "internlm/CapRL-Qwen3VL-2B",
            "caption_revision": CAPTION_REVISION, "prompt": CAPTION_PROMPT,
            "caption_sampling": "greedy; EOS required; budgets 2048 then 8192; no truncation",
            "counter_model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
            "counter_revision": NEMOTRON_REVISION,
            "counter_method": "native image_processor num_tokens + image delimiters; native tokenizer of complete caption block",
            "counter_scope": "all main-model HTTP attempts including retries; video/audio excluded",
            "image_counter_api_verified": False, "counter_version": 1}
