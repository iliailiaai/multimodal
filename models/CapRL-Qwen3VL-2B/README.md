---
language:
- en
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- multimodal
- image caption
- captioning
datasets:
- internlm/CapRL-2M
---

# CapRL
📖<a href="https://arxiv.org/abs/2509.22647">Paper</a> | 🏠<a href="https://github.com/InternLM/CapRL">Github</a> | 🤗<a href="https://huggingface.co/collections/long-xing1/caprl-68d64ac32ded31596c36e189">CapRL Collection</a> | 🤗<a href="https://huggingface.co/papers/2509.22647">Daily Paper</a> 

### CapRL Series Model & Dataset
| Series | Models & Resources |
| :--- | :--- |
| **CapRL 2.0 Series** | [🤗 CapRL-Qwen3VL-2B](https://huggingface.co/internlm/CapRL-Qwen3VL-2B) \| [🤗 CapRL-Qwen3VL-4B](https://huggingface.co/internlm/CapRL-Qwen3VL-4B) \| [📦 CapRL-Qwen3VL-2B-GGUF](https://huggingface.co/internlm/CapRL-Qwen3VL-2B-GGUF) \| [📦 CapRL-Qwen3VL-4B-GGUF](https://huggingface.co/internlm/CapRL-Qwen3VL-4B-GGUF) \| [🌈CapRL-Qwen3VL-4B Space](https://huggingface.co/spaces/yuhangzang/CapRL-Qwen3VL-4B)
| **CapRL 1.0 Series** | [🤗 CapRL-Qwen2.5VL-3B](https://huggingface.co/internlm/CapRL-3B) \| [🤗 CapRL-InternVL3.5-8B](https://huggingface.co/yuhangzang/CapRL-InternVL3.5-8B) \|[📊 CapRL-QA-75K Dataset](https://huggingface.co/datasets/internlm/CapRL-QA-75K) \| [📊 CapRL-2M Dataset](https://huggingface.co/datasets/internlm/CapRL-2M) \| [📦 CapRL-3B-GGUF](https://huggingface.co/mradermacher/CapRL-3B-GGUF) \| [📦 CapRL-3B-i1-GGUF](https://huggingface.co/mradermacher/CapRL-3B-i1-GGUF) \| [🌈CapRL-Qwen2.5VL-3B Space](https://huggingface.co/spaces/yuhangzang/caprl)

### CapRL-Qwen3VL-2B
We are excited to release the **CapRL 2.0 series**: **CapRL-Qwen3VL-2B** and **CapRL-Qwen3VL-4B**. These models feature fewer parameters while delivering even more powerful captioning performance. 
Notably, **CapRL-Qwen3VL-2B outperforms both CapRL-Qwen2.5VL-3B and Qwen2.5VL-72B in captioning tasks**.
This leap in efficiency is driven by our upgraded training recipe, which includes a more rigorous QA data filter and a significantly more diverse image dataset. We welcome everyone to try them out!


When selecting between the available CapRL models, it's essential to consider the trade-off between performance and computational cost.
This guide will help you choose the most suitable model for your specific needs:
|Model|Parameters|Strength|
|-|-|-|
|🤗[CapRL-Qwen3VL-2B](https://huggingface.co/internlm/CapRL-Qwen3VL-2B)|2B|Speed, Efficiency|
|🤗[CapRL-Qwen3VL-4B](https://huggingface.co/internlm/CapRL-Qwen3VL-4B)|4B|High Performance, Advanced Captioning Ability|

Now you can try out CapRL with your own images🎨!&nbsp;&nbsp;&nbsp;&nbsp;➡️&nbsp;&nbsp;&nbsp;&nbsp;[🌈CapRL-Qwen2.5VL-3B Space](https://huggingface.co/spaces/yuhangzang/caprl) and [🌈CapRL-Qwen3VL-4B Space](https://huggingface.co/spaces/yuhangzang/CapRL-Qwen3VL-4B).

## 📢 News
We are working on even stronger base models and upgrading our training recipe — stay tuned!
- 🔥 [04/16/2026] We have released the **[CapRL-QA-75K](https://huggingface.co/datasets/internlm/CapRL-QA-75K)** training dataset!
- 🔥 [12/24/2025] We are excited to release the CapRL 2.0 series: **[CapRL-Qwen3VL-2B](https://huggingface.co/internlm/CapRL-Qwen3VL-2B)** and **[CapRL-Qwen3VL-4B](https://huggingface.co/internlm/CapRL-Qwen3VL-4B)**!
- 🔥 [12/24/2025] The total downloads of the CapRL-related [models and dataset](https://huggingface.co/collections/long-xing1/caprl-68d64ac32ded31596c36e189) reached 17,000!
- 🔥 [10/15/2025] The total downloads of the CapRL-related [models and dataset](https://huggingface.co/collections/long-xing1/caprl-68d64ac32ded31596c36e189) reached 6,000 within just 20 days!
- 🚀 [10/15/2025] We are excited to announce the release of **[CapRL-InternVL3.5-8B](https://huggingface.co/internlm/CapRL-InternVL3.5-8B)**, whose image captioning capability outperforms Qwen2.5-VL-72B!
- 🚀 [10/15/2025] Thanks [mradermacher](https://huggingface.co/mradermacher) for the valuable contribution! [CapRL-3B-GGUF](https://huggingface.co/mradermacher/CapRL-3B-GGUF) is the static quants version, and [CapRL-3B-i1-GGUF](https://huggingface.co/mradermacher/CapRL-3B-i1-GGUF) is weighted/imatrix quants version.
- 🚀 [10/15/2025] We release [QA curation code](https://github.com/InternLM/CapRL).
- 🚀 [09/25/2025] We release **CapRL** repository, [CapRL-3B model](https://huggingface.co/internlm/CapRL-3B), [evaluation code](https://github.com/InternLM/CapRL) and [dataset](https://huggingface.co/datasets/internlm/CapRL-2M).

## Introduction of CapRL
We are excited to introduce [CapRL-3B](https://huggingface.co/internlm/CapRL-3B), a lightweight 3B image captioner that achieves perception capabilities comparable to Qwen2.5-VL-72B.

This is the first study of applying Reinforcement Learning with Verifiable Rewards for the
open-ended and subjective image captioning task. Unlike traditional Supervised Fine-Tuning, which
can lead to models memorizing a limited set of annotated captions, our method allows the model to
explore and generate a broader range of creative and general descriptions.
CapRL is a new training paradigm featuring a decoupled two-stage pipeline. The initial
stage uses LVLMs to generate rich and accurate captions. Subsequently, the second stage evaluates
caption quality by using a vision-only LLM to perform the QA task. We also created a specific QA
curation pipeline to ensure the quality of the questions and answers used for the second stage.

By employing the CapRL training framework, initializing with the Qwen2.5-VL-3B model, and using a carefully 
filtered 75K QA dataset as the training set, we obtained a highly capable captioner, [CapRL-3B](https://huggingface.co/internlm/CapRL-3B).

<p align="center">
  <img src="./assets/teaser.png"  width="750"/>
</p>
<p align="center">
  <img src="./assets/performance_caprl2_0.png" width="750"/>
</p>

## Key Features
* **Remarkable visual understanding for Chart, Infographics and Document**: [CapRL-3B](https://huggingface.co/internlm/CapRL-3B) achieves perception accuracy and visual information coverage comparable to Qwen2.5-VL-72B.
* **Well-organized output**: The outputs of CapRL-3B are relatively well-structured, making them clear and easy to understand.
* **Detailed description for natural images**: The outputs of [CapRL-3B](https://huggingface.co/internlm/CapRL-3B) can perfectly cover all valid visual information while containing fewer hallucinations.

## Usage
If you want to use **[CapRL-3B](https://huggingface.co/internlm/CapRL-3B)** for captioning, you can directly follow the exact same inference approach as in [Qwen2.5-VL-series](https://github.com/QwenLM/Qwen3-VL/tree/d2240f11656bfe404b9ba56db4e51cd09f522ff1).

We recommend using **vLLM** to speed up inference.



### Start an OpenAI API Service

Run the command below to start an OpenAI-compatible API service:

```bash
vllm serve "/PATH/CapRL-3B" \
    --trust-remote-code \
    --tensor-parallel-size=1 \
    --pipeline-parallel-size=1 \
    --gpu_memory_utilization=0.95 \
    --served-model-name=caprl \
    --port 8000 \
    --host 0.0.0.0
```

Then you can use the chat API as below: (see [OpenAI API protocol document](https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images) for more details):
```python
import base64
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
image_path = "/path/to/local/image.png"
with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read())
encoded_image_text = encoded_image.decode("utf-8")
base64_qwen = f"data:image;base64,{encoded_image_text}"
chat_response = client.chat.completions.create(
    model="caprl",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_qwen
                    },
                },
                {"type": "text", "text": "What is the text in the illustrate?"},
            ],
        },
    ],
    temperature=1.0,
    max_tokens=max_tokens,
    top_p=1.0,
    extra_body={
        "repetition_penalty": 1.0,
        },
)
print("Chat response:", chat_response)
```



## Cases
<p align="center">
  <img src="./assets/comparison.png"  width="750"/>
</p>

<p align="center">
  <img src="./assets/info_caprl.png"  width="750"/>
</p>

<p align="center">
  <img src="./assets/info_caprl2.png"  width="750"/>
</p>
<p align="center">
  <img src="./assets/natural_caprl.png"  width="750"/>
</p>