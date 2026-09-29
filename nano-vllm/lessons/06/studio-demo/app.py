import os
from collections.abc import Generator

import gradio as gr
from openai import OpenAI


API_BASE = os.getenv("MINISGL_API_BASE", "http://127.0.0.1:1919/v1")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-0.6B")

client = OpenAI(base_url=API_BASE, api_key=os.getenv("MINISGL_API_KEY", "dummy"))


def stream_chat(
    message: str,
    history: list[dict[str, str]],
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=temperature,
        max_tokens=int(max_tokens),
        stream=True,
    )

    text = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            text += delta
            yield text


with gr.Blocks(title="mini-SGLang 推理 Demo") as demo:
    gr.Markdown(
        "# mini-SGLang 推理 Demo\n"
        "模型由 mini-SGLang 提供 OpenAI-compatible API，页面通过容器内回环地址调用推理服务。"
    )
    system_prompt = gr.Textbox(
        label="System Prompt",
        value="你是一名简洁、准确的中文助手。",
        lines=2,
    )
    with gr.Row():
        temperature = gr.Slider(0.0, 1.5, value=0.6, step=0.1, label="Temperature")
        max_tokens = gr.Slider(32, 1024, value=256, step=32, label="Max Tokens")

    gr.ChatInterface(
        fn=stream_chat,
        type="messages",
        additional_inputs=[system_prompt, temperature, max_tokens],
        examples=[
            "用三句话解释 KV Cache。",
            "对比 prefill 与 decode 的计算特征。",
            "解释 Radix Cache 如何复用共享前缀。",
        ],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )
