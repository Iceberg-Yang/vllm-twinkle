# 推理 Infra 学习实录 02：nano-vLLM vs Transformers

> 归档状态（GitHub 发布副本）：历史手记数据，无原始运行输出或原始 JSON，待复核；本轮未做 GPU 复测。预热与采样口径不统一，不能用于排名。ModelScope 发布 URL 尚未知，不表示已发布。
> [配套 Notebook](main.ipynb) · [上一篇](../01/article.md) · [下一篇](../03/article.md)。Notebook 输出与执行计数已清空。
> 版本边界：安装与源码说明对齐已核对 [nano-vLLM commit bb823b3](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)，已有 checkout 不自动覆盖；版本不符时应另选空目录。固定源码不等于全部依赖已锁定。
> 路径说明：`/mnt/workspace` 是 ModelScope 平台缓存示例，模型目录以 `snapshot_download()` 返回值为准。

第一篇里，我已经在魔搭 Notebook 上跑通了 nano-vLLM，并用 `Qwen/Qwen3-0.6B` 做了一次基础 benchmark。

跑通之后，一个很自然的问题会冒出来：

> Transformers 也能加载 Qwen，也能 `generate()` 生成文本。那为什么还要学习 nano-vLLM 这样的推理引擎？

本篇不仅仅测评速度，因为如果只看一组 `tokens/s`，很容易把问题理解成“哪个框架更快”。但我真正想弄清楚的是：

```text
同一个 Qwen 模型，
为什么 Transformers.generate() 能生成，
nano-vLLM.generate() 也能生成，
但 nano-vLLM 背后要多做一整套推理引擎？
```

也就是说

```text
当一次生成从“单用户调用”变成“多请求服务化生成”时，
系统复杂度从哪里冒出来？
```

## 1. 两条生成路径的区别

Transformers 和 nano-vLLM 都能完成文本生成，但它们解决的问题不一样。

| 对比项 | Transformers | nano-vLLM |
|---|---|---|
| 核心定位 | 通用模型库 | 轻量推理引擎 |
| 常见入口 | `AutoModelForCausalLM.generate()` | `LLM.generate()` |
| 主要关注 | 模型加载、训练、实验、单次推理 | 请求调度、KV Cache、batch、吞吐 |
| 使用方式 | 用户自己组织输入和 batch | 引擎管理请求生命周期 |
| KV Cache | 模型生成过程中的缓存机制 | 显存资源的一部分，需要 block 管理 |
| 更适合 | 模型能力验证、原型实验 | 学习服务化推理系统机制 |

我现在对两者的区分是：

```text
Transformers 解决的是“模型如何被调用”，
nano-vLLM 解决的是“很多生成请求如何在 GPU 上被组织、调度和持续执行”。
```

这也是为什么我先拿 nano-vLLM 和 Transformers 对比，而不是直接和 vLLM 对比。

Transformers 是很多人最熟悉的大模型入口。先和它对比，可以更清楚地看到“模型库”和“推理引擎”的边界。vLLM 是生产级推理引擎，更适合放到后面，等我理解 nano-vLLM 的核心机制之后，再去看生产级框架多做了哪些工程优化。

## 2. 本篇实验设置

为了让对比尽量干净，本篇实验保持几个条件一致：

| 项目 | 设置 |
|---|---|
| 平台 | 魔搭 Notebook |
| 模型 | `Qwen/Qwen3-0.6B` |
| 模型来源 | ModelScope 模型库 |
| 推理方式 1 | Transformers |
| 推理方式 2 | nano-vLLM |
| 采样参数 | `temperature=0.6`, `max_new_tokens=256` / `max_tokens=256` |
| 对比重点 | 耗时、输出 token 数、tokens/s、显存、batch 行为 |

模型下载沿用第一篇的方式：

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-0.6B",
    cache_dir="/mnt/workspace/.cache/modelscope",
)

print("model_dir =", model_dir)
```

如果模型已经下载，也可以直接使用本地路径：

```text
/mnt/workspace/.cache/modelscope/Qwen/Qwen3-0___6B
```

或者根据自己的 Notebook 实际路径调整。

本篇使用同一组 prompt：

```python
from transformers import AutoTokenizer

raw_prompts = [
    "请用三句话解释什么是大模型推理引擎。",
    "请解释 KV Cache 在大模型推理中的作用。",
    "What is the difference between prefill and decode in LLM inference?",
    "如果有 100 个用户同时请求大模型服务，推理系统需要处理哪些问题？",
]

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

chat_prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in raw_prompts
]

print("prompt 数:", len(chat_prompts))
print(chat_prompts[0][:500])
```

这里故意包含中文、英文、基础概念和服务化场景。因为这篇关注的不是模型回答质量，而是两条推理路径在系统层面的差异。

Notebook 依赖还包括 `pandas`；下文 `device_map="auto"` 需要 `accelerate`：

```python
%pip install modelscope pandas accelerate
%pip install git+https://github.com/GeeeekExplorer/nano-vllm.git@bb823b3e06983d71485a8e1f23715ebd87d98ef8
```

实验中还会用到几个公共统计函数。显存除以 `1024**3`，单位是 GiB；`peak_memory_gb` 暂保留旧字段名，展示为 `caller_peak_allocated_gib`。它只覆盖调用进程的 PyTorch allocator，不一定覆盖 worker，也不是引擎总显存。nano-vLLM 初始化时预分配 KV Cache，预留容量不能解释为当前请求的增量显存。

```python
import gc
import time
import pandas as pd
import torch


def reset_cuda_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def current_peak_memory_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def cleanup_cuda(*objects):
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


results = []
```

## 3. 用 Transformers 跑同一个 Qwen 模型

先看最熟悉的 Transformers 路径。

```python
from transformers import AutoModelForCausalLM

tokenizer_tf = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
tokenizer_tf.padding_side = "left"
if tokenizer_tf.pad_token_id is None:
    tokenizer_tf.pad_token = tokenizer_tf.eos_token

model_tf = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
)
model_tf.eval()

inputs_tf = tokenizer_tf(
    chat_prompts,
    return_tensors="pt",
    padding=True,
).to(model_tf.device)

reset_cuda_stats()

t0 = time.perf_counter()
with torch.inference_mode():
    output_ids_tf = model_tf.generate(
        **inputs_tf,
        max_new_tokens=256,
        temperature=0.6,
        do_sample=True,
        pad_token_id=tokenizer_tf.pad_token_id,
        eos_token_id=tokenizer_tf.eos_token_id,
    )
torch.cuda.synchronize()
elapsed_tf = time.perf_counter() - t0

input_len_tf = inputs_tf["input_ids"].shape[1]
new_token_ids_tf = output_ids_tf[:, input_len_tf:]
output_tokens_tf = int((new_token_ids_tf != tokenizer_tf.pad_token_id).sum().item())
tokens_per_second_tf = output_tokens_tf / elapsed_tf
peak_memory_tf = current_peak_memory_gb()
texts_tf = tokenizer_tf.batch_decode(new_token_ids_tf, skip_special_tokens=True)

transformers_result = {
    "engine": "Transformers",
    "num_prompts": len(chat_prompts),
    "max_new_tokens": 256,
    "elapsed_s": elapsed_tf,
    "output_tokens": output_tokens_tf,
    "tokens_per_second": tokens_per_second_tf,
    "peak_memory_gb": peak_memory_tf,
}

results.append(transformers_result)
transformers_result
```

查看输出样例：

```python
for i, text in enumerate(texts_tf):
    print("=" * 80)
    print("Prompt", i)
    print(raw_prompts[i])
    print("-" * 80)
    print(text[:1200])
```

这段代码背后的流程很直接：

```text
加载 tokenizer
加载 model
把 prompt tokenize 成 input_ids
调用 model.generate()
decode 输出 token
```

这就是 Transformers 很适合作为模型使用入口的原因。它把模型加载、tokenizer、生成参数、权重格式这些事情都封装得很好。

但从推理 Infra 角度看，这里也有一个特点：

```text
用户自己组织 batch，用户自己调用 generate。
```

也就是说，Transformers 更像是给我一个强大的模型调用接口。至于请求什么时候到达、不同请求如何排队、显存不够怎么办、KV Cache 如何作为系统资源管理，这些不是它在这个使用方式下主要暴露给我的问题。

跑完 Transformers 后，如果后面还要在同一个 Notebook 里继续跑 nano-vLLM，建议先释放显存：

```python
del model_tf
del inputs_tf
del output_ids_tf
del new_token_ids_tf
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

print("CUDA memory allocated GiB:", torch.cuda.memory_allocated() / 1024**3)
print("CUDA memory reserved GiB:", torch.cuda.memory_reserved() / 1024**3)
```

## 4. 用 nano-vLLM 跑同一个 Qwen 模型

再看 nano-vLLM 路径。

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    model_dir,
    enforce_eager=True,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=256,
)

reset_cuda_stats()

t0 = time.perf_counter()
outputs_nv = llm.generate(chat_prompts, sampling_params)
torch.cuda.synchronize()
elapsed_nv = time.perf_counter() - t0

output_tokens_nv = sum(len(output["token_ids"]) for output in outputs_nv)
tokens_per_second_nv = output_tokens_nv / elapsed_nv
peak_memory_nv = current_peak_memory_gb()

nanovllm_result = {
    "engine": "nano-vLLM",
    "num_prompts": len(chat_prompts),
    "max_new_tokens": 256,
    "elapsed_s": elapsed_nv,
    "output_tokens": output_tokens_nv,
    "tokens_per_second": tokens_per_second_nv,
    "peak_memory_gb": peak_memory_nv,
}

results.append(nanovllm_result)
nanovllm_result
```

查看输出样例：

```python
for i, output in enumerate(outputs_nv):
    print("=" * 80)
    print("Prompt", i)
    print(raw_prompts[i])
    print("-" * 80)
    print(output["text"][:1200])
```

表面上看，这段代码也很简单：

```text
LLM(...)
SamplingParams(...)
llm.generate(...)
```

但从源码看，`LLM` 本身其实非常薄：

```python
from nanovllm.engine.llm_engine import LLMEngine

class LLM(LLMEngine):
    pass
```

真正的逻辑在 `LLMEngine` 里。

从代码可以看到，`LLMEngine.generate()` 会先把每个 prompt 变成一个 `Sequence`，再交给 scheduler：

```text
for prompt in prompts:
    add_request(prompt)

while not finished:
    seqs, is_prefill = scheduler.schedule()
    token_ids = model_runner.run(seqs, is_prefill)
    scheduler.postprocess(seqs, token_ids, is_prefill)
```

也就是说，nano-vLLM 的 `generate()` 不是单纯调用一次模型 forward，而是一个调度循环。

## 5. 结果记录表

下面保留历史手记中的数值，无原始运行输出或原始 JSON 可供复核，不能视为本轮运行结果。两条路径的预热与采样细节不统一，因此不能据此排名。

新运行时，Notebook 可用 `results` 生成结果表；不要自动覆盖历史表：

```python
df = pd.DataFrame(results)
df.rename(columns={"peak_memory_gb": "caller_peak_allocated_gib"})
```

| 指标 | Transformers | nano-vLLM |
|---|---:|---:|
| prompt 数 | 4 | 4 |
| max new tokens | 256 | 256 |
| 总输出 token | 1024 | 956 |
| 总耗时 | 7.7210 s | 11.1423 s |
| tokens/s | 132.6261 | 85.7988 |
| 调用进程 peak allocated（历史记录） | 1.2759 GiB | 19.2473 GiB |
| 是否显式暴露 scheduler | 否 | 是 |
| 是否显式管理 KV block | 否 | 是 |

这里要特别注意：这组实验不是严格生产 benchmark。

原因有几个：

- prompt 数量很少。
- 没有多轮重复测量。
- 没有区分 TTFT 和 TPOT。
- `enforce_eager=True` 会关闭 nano-vLLM 的 CUDA Graph 路径。
- Transformers 和 nano-vLLM 的生成细节不一定完全一致。

所以这组结果只能作为学习观察，而不是框架排名。

这篇更重要的是解释：为什么两者都能生成文本，但系统职责不同。

## 6. API 差异背后的系统差异

Transformers 的生成路径可以粗略理解成：

```text
tokenizer
-> model.generate()
-> output ids
-> tokenizer.decode()
```

nano-vLLM 的生成路径更像：

```text
prompt
-> Sequence
-> waiting queue
-> Scheduler.schedule()
-> ModelRunner.prepare_prefill() / prepare_decode()
-> Attention.forward()
-> Sampler
-> Scheduler.postprocess()
-> finished output
```

这个差异很关键。

Transformers 让我把注意力放在：

```text
模型怎么加载？
prompt 怎么 tokenize？
generate 参数怎么设置？
输出怎么 decode？
```

nano-vLLM 则迫使我把注意力放在：

```text
请求进入系统后是什么状态？
这一轮 GPU 跑哪些请求？
哪些请求处于 prefill？
哪些请求处于 decode？
KV Cache 放在哪里？
显存 block 不够时怎么办？
请求什么时候结束并释放资源？
```

这就是“模型调用”和“推理系统”的分界线。

## 7. Scheduler：多请求复杂度从哪里来

单用户调用时，问题很简单：

```text
输入一个 prompt
生成一个 answer
```

但服务化场景里，请求不会这么整齐：

```text
请求 A: prompt 很短，生成很长
请求 B: prompt 很长，生成很短
请求 C: 刚刚进入系统
请求 D: 已经 decode 到一半
请求 E: 生成到了 EOS，应该结束
```

这时系统要回答一组新问题：

```text
这一轮先跑谁？
最多允许多少个 sequence 一起跑？
一次 prefill 最多处理多少 token？
decode 阶段是否每个请求只追加一个 token？
KV Cache 显存不够时，是否要 preempt 某些请求？
```

nano-vLLM 的 `Scheduler` 里有两个队列：

```text
waiting
running
```

每个请求会被包装成 `Sequence`，状态在下面几种之间变化：

```text
WAITING
RUNNING
FINISHED
```

从 `Scheduler.schedule()` 可以看到，它先尝试调度 waiting 队列里的请求做 prefill。如果没有 prefill 任务，再从 running 队列里调度 decode。

简化之后是：

```text
如果 waiting 里有请求：
    尽量按 token budget 做 prefill
否则：
    从 running 里取请求做 decode
```

这里的重点是：推理引擎不是简单把所有 prompt 拼起来跑一次，而是在持续维护请求生命周期。

## 8. Prefill 和 Decode 为什么要分开

Prefill 和 decode 的计算形态不同。

| 维度 | Prefill | Decode |
|---|---|---|
| 处理内容 | prompt 或 prompt chunk | 每个请求通常 1 个新 token |
| 输入长度 | 可能很长 | 通常很短 |
| 主要压力 | 计算量 | KV Cache 读取、调度开销 |
| nano-vLLM 准备函数 | `prepare_prefill()` | `prepare_decode()` |
| Attention 调用 | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |

从 `ModelRunner` 代码可以看到，nano-vLLM 为两种阶段准备的元数据也不同。

Prefill 阶段会准备：

```text
input_ids
positions
cu_seqlens_q
cu_seqlens_k
slot_mapping
block_tables
```

Decode 阶段会准备：

```text
input_ids
positions
slot_mapping
context_lens
block_tables
```

这说明 prefill / decode 不只是两个概念名词，它们在推理引擎里对应不同的 tensor 组织方式和 attention kernel 调用方式。

## 9. KV Cache：从模型缓存变成显存资源

Transformers 里也有 KV Cache。否则 decode 会非常慢。

但 nano-vLLM 更值得学习的地方在于：它把 KV Cache 当成需要管理的系统资源。

在 nano-vLLM 中，KV Cache 被切成 block。`BlockManager` 维护了几个关键结构：

```text
blocks
free_block_ids
used_block_ids
hash_to_block_id
```

每个 `Sequence` 也会维护自己的：

```text
block_table
```

可以把它理解成：

```text
这个请求的 token 对应的 KV Cache 存在哪些 block 里。
```

这就很像操作系统里的分页内存管理。

为什么要这么做？

因为服务化推理里，KV Cache 的形态很不规整：

```text
不同请求 prompt 长度不同
不同请求生成长度不同
有的请求已经结束
有的请求还在继续 decode
有的请求共享相同前缀
```

如果把 KV Cache 当成一整块连续内存来处理，很容易产生浪费和碎片。

Block 管理带来几个好处：

- 请求结束时，可以释放它占用的 block。
- decode 追加 token 时，只在需要时追加新 block。
- prefix cache 可以复用相同前缀对应的 block。
- scheduler 可以根据 free block 数量判断是否还能继续接纳请求。

这就是 nano-vLLM 比普通 `model.generate()` 多出来的一层系统价值。

## 10. Prefix Cache：为什么相同前缀可以复用

很多真实业务请求有相同前缀。

例如 RAG 场景里，多个问题可能共享同一段系统提示词和检索文档：

```text
请求 A: 系统提示词 + 长文档 + 问题 1
请求 B: 系统提示词 + 长文档 + 问题 2
```

如果前面的大段 token 完全一样，它们对应的 KV Cache 理论上可以复用。

nano-vLLM 的 `BlockManager` 会对 block token 做 hash：

```text
compute_hash(token_ids, prefix_hash)
```

这里使用的是链式 hash。也就是说，一个 block 的 hash 不只取决于当前 block 的 token，也取决于前一个 block 的 hash。

这样做是为了表达：

```text
同一个 block 的 token 相同还不够，
它前面的上下文也必须相同。
```

这就是 prefix cache 的基本思想。

这部分也是后续源码学习里非常值得深入的地方。

## 11. 为什么单次小 batch 不一定能看出 nano-vLLM 优势

如果只跑 1 到 4 条 prompt，nano-vLLM 未必会显得比 Transformers 更快。

这不奇怪。

因为推理引擎多做了很多系统工作：

```text
创建 Sequence
进入 waiting queue
Scheduler 调度
BlockManager 分配 block
ModelRunner 准备元数据
Attention 读写 KV Cache
postprocess 更新状态
```

这些机制的价值通常在更接近服务化的场景里体现出来：

- 请求数更多。
- prompt 长短不一。
- 输出长度不一。
- decode 过程持续很多轮。
- GPU 需要保持更高利用率。
- KV Cache 显存需要被复用和释放。

所以这篇的重点不是证明 nano-vLLM 在任何小实验里都更快，而是理解：

```text
推理引擎为了服务化生成，额外管理了哪些系统问题。
```

## 12. Transformers 和 nano-vLLM 该怎么选

经过这次对比，我对两者的使用场景有了更清楚的区分。

如果目标是：

- 快速验证模型能不能用。
- 看模型回答质量。
- 做 prompt 原型。
- 做训练、微调或模型实验。
- 需要最大模型兼容性。

那 Transformers 是更自然的入口。

如果目标是：

- 理解推理引擎内部怎么组织请求。
- 学习 prefill / decode。
- 学习 KV Cache block 管理。
- 学习 batch scheduler。
- 学习 prefix cache、FlashAttention、CUDA Graph 这些推理优化机制。

那 nano-vLLM 是更好的学习对象。

可以这样总结：

```text
Transformers 适合回答：这个模型怎么用？
nano-vLLM 适合回答：这个模型如何被组织成一个推理系统？
```

## 13. 这篇学到了什么

这次对比让我把“大模型生成”拆成了两层。

第一层是模型调用：

```text
tokenizer
model
generate
decode
```

这是 Transformers 最擅长的部分。

第二层是推理系统：

```text
request
sequence
scheduler
prefill
decode
KV Cache block
sampling
postprocess
```

这是 nano-vLLM 想让我看见的部分。

同一个 Qwen 模型，在 Transformers 里更像一个可以调用的模型；在 nano-vLLM 里则变成了一个被调度、被分配显存、被持续执行的请求流。

这就是我对第二篇标题的理解：

```text
同一个 Qwen 模型，推理引擎到底多做了什么？
```

答案不是“多写了一个 generate API”，而是：

```text
它把生成过程从一次模型调用，变成了一个持续运行的 GPU 资源调度问题。
```

## 14. 下一篇读源码看什么

有了这篇对比，下一步就可以进入源码导读。

我打算从这条链路开始：

```text
example.py
-> LLM(...)
-> LLMEngine.generate(...)
-> LLMEngine.step(...)
-> Scheduler.schedule(...)
-> ModelRunner.run(...)
-> Attention.forward(...)
-> Sampler(...)
-> Scheduler.postprocess(...)
```

第三篇的目标不再是比较两个框架，而是回答：

```text
一次 nano-vLLM generate 背后到底发生了什么？
```

尤其是这几个问题：

- `Sequence` 如何表示一个请求？
- `Scheduler` 如何区分 prefill 和 decode？
- `BlockManager` 如何分配和释放 KV block？
- `ModelRunner` 如何准备 GPU tensor？
- `Attention.forward()` 如何写入和读取 KV Cache？

如果第一篇是“跑起来”，第二篇是“为什么需要推理引擎”，那第三篇就应该开始真正进入：

```text
nano-vLLM 源码主链路。
```
