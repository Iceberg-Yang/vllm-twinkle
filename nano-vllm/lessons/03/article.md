# 推理 Infra 学习实录 03：一次 generate 背后发生了什么

> 归档状态（GitHub 发布副本）：源码学习稿，本轮未做 GPU 验证。前篇历史记录的证据边界见各篇状态栏。ModelScope 发布 URL 尚未知，不表示已发布。
> [配套 Notebook](main.ipynb) · [上一篇](../02/article.md) · [下一篇](../04/article.md)。Notebook 输出与执行计数已清空。
> 版本边界：安装与源码说明对齐已核对 [nano-vLLM commit bb823b3](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)，已有 checkout 不自动覆盖；版本不符时应另选空目录。固定源码不等于全部依赖已锁定。
> 路径说明：`/mnt/workspace` 是 ModelScope 平台缓存示例，模型目录以 `snapshot_download()` 返回值为准。

前两篇完成了两件事：

1. 在魔搭 Notebook 上跑通 nano-vLLM，并记录了一次基础 benchmark。
2. 用同一个 Qwen 模型对比 Transformers 和 nano-vLLM，理解“模型调用”和“推理引擎”的区别。

跑到这里之后，问题变具体了：

```text
llm.generate(prompts, sampling_params)
```

这行代码背后发生了什么？

从使用者角度看，`generate()` 接收 prompt，然后输出文本。从推理 Infra 角度看，它把一次生成拆成了一串系统动作：

```text
prompt
-> tokenizer
-> Sequence
-> waiting queue
-> Scheduler.schedule()
-> ModelRunner.run()
-> Attention.forward()
-> Sampler
-> Scheduler.postprocess()
-> output text
```

这一篇顺着 nano-vLLM 的源码，把这条主链路读清楚。

这个最应该被关心的问题是：

```text
一个请求进入 nano-vLLM 后，如何被调度、执行、写入 KV Cache，并最终变成输出文本？
```

## 1. 第一步：跑一个最小 generate

源码阅读需要一个具体入口。这里从最小调用开始。

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-0.6B",
    cache_dir="/mnt/workspace/.cache/modelscope",
)

print(model_dir)
```

然后初始化 nano-vLLM：

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    model_dir,
    enforce_eager=True,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=128,
)

outputs = llm.generate(
    ["请用三句话解释什么是大模型推理引擎。"],
    sampling_params,
)

print(outputs[0]["text"])
```

运行正常时，会看到类似这样的进度条：

```text
Generating: 100%|██████████| 1/1 [..., Prefill=..., Decode=...]
```

第一篇里，这一步代表“跑通了”。这一篇继续追问：

```text
为什么进度条里会分别出现 Prefill 和 Decode？
为什么 generate 没有在一次 forward 后结束？
为什么输出里既有 text，也有 token_ids？
```

答案都在 `LLMEngine.generate()` 的主循环里。

## 2. 第二步：确认 LLM 外壳

nano-vLLM 对外暴露的类叫 `LLM`。

打开 [`nanovllm/llm.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/llm.py) 可以看到，`LLM` 本身没有额外逻辑：

```python
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    pass
```

主要逻辑在 `LLMEngine`。

这也是读源码时的第一个提醒：

```text
API 名字叫 LLM，
推理系统的复杂度在 engine、scheduler、model runner、block manager 和 attention。
```

## 3. 第三步：看初始化阶段

初始化 `LLM(model_dir, ...)` 时，`LLMEngine.__init__()` 主要做几件事：

```text
Config(model, **kwargs)
Sequence.block_size = config.kvcache_block_size
ModelRunner(config, rank=0, events)
AutoTokenizer.from_pretrained(config.model)
Scheduler(config)
```

对应到源码，可以看到这些关键对象：

| 对象 | 作用 |
|---|---|
| `Config` | 读取模型配置和推理参数 |
| `Sequence` | 保存单个请求的 token、状态和 KV block 信息 |
| `ModelRunner` | 加载模型、分配 KV Cache、执行 forward 和 sampling |
| `AutoTokenizer` | 文本和 token id 之间转换 |
| `Scheduler` | 维护 waiting/running 队列，决定每一步跑哪些请求 |

这里需要注意这一行：

```python
Sequence.block_size = config.kvcache_block_size
```

它说明 `Sequence` 后面会按固定 block size 来切分 token。KV Cache 的 block 管理从请求对象创建时就参与进来了。

## 4. 第四步：prompt 进入系统后变成 Sequence

`generate()` 开始时，把 prompt 加入系统：

```python
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)
```

`add_request()` 做三件事：

```python
if isinstance(prompt, str):
    prompt = self.tokenizer.encode(prompt)
seq = Sequence(prompt, sampling_params)
self.scheduler.add(seq)
```

这个过程对应为：

```text
自然语言 prompt
-> tokenizer.encode
-> token ids
-> Sequence
-> Scheduler waiting queue
```

`Sequence` 是 nano-vLLM 里理解请求生命周期的核心对象。

它保存 prompt token，也保存这些状态：

| 字段 | 含义 |
|---|---|
| `seq_id` | 请求编号 |
| `status` | `WAITING` / `RUNNING` / `FINISHED` |
| `token_ids` | prompt token 和生成 token 放在同一个列表里 |
| `last_token` | decode 阶段本轮要输入的上一个 token |
| `num_prompt_tokens` | prompt 长度 |
| `num_cached_tokens` | 已经写入 KV Cache 的 token 数 |
| `num_scheduled_tokens` | 当前 step 被调度执行的 token 数 |
| `is_prefill` | 当前是否处于 prefill 阶段 |
| `block_table` | 这个请求对应的 KV Cache block id 列表 |
| `temperature` / `max_tokens` / `ignore_eos` | 采样参数 |

这一步改变了我对“请求”的理解。

在普通模型调用里，请求可以是一段文本。在推理引擎里，请求必须变成一个能被持续调度的状态对象。

## 5. 第五步：理解 generate 的 while loop

`LLMEngine.generate()` 的主体是一个循环：

```python
while not self.is_finished():
    t = perf_counter()
    output, num_tokens = self.step()
    ...
```

循环的原因是：

一次生成不会在一个 GPU step 里完成：

- prompt 需要进入 prefill。
- 生成阶段通常每轮 decode 一个 token。
- 多个请求可能处于不同状态。
- 有的请求刚进入 waiting queue。
- 有的请求已经在 running queue 里 decode 到一半。
- 有的请求生成到 EOS，需要释放 KV block。

所以 `generate()` 会持续推进所有请求，直到 scheduler 里没有 waiting/running 请求。

这是 nano-vLLM 和 `transformers.generate()` 在学习视角上的一个重要差异：

```text
Transformers.generate 更像模型库接口。
nano-vLLM.generate 更像一个小型推理系统的事件循环。
```

## 6. 第六步：看 step 的一轮执行

每一轮 `step()` 做三件事：

```python
seqs, is_prefill = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

流程可以写成：

```text
Scheduler.schedule()
    选出本轮要跑的请求

ModelRunner.run()
    准备 input tensor
    执行模型 forward
    sampling 得到新 token

Scheduler.postprocess()
    更新 Sequence 状态
    追加 token
    释放已结束请求的 KV block
```

`step()` 返回的 `num_tokens` 用来区分 prefill 和 decode：

```python
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
```

nano-vLLM 用正负号区分 prefill 和 decode：

- `num_tokens > 0`：本轮是 prefill，统计本轮处理了多少 prompt token。
- `num_tokens < 0`：本轮是 decode，统计本轮给多少个 sequence 各生成一个 token。

因此进度条会分开显示：

```text
Prefill=...tok/s
Decode=...tok/s
```

原因是 prefill 和 decode 的计算形态不同。

## 7. 第七步：看 Scheduler 如何选择本轮任务

`Scheduler` 维护两个队列：

```text
waiting
running
```

请求进入系统时，会被放进 `waiting`。

在 `Scheduler.schedule()` 调度最后一段 prefill 时，就会将请求设为 `RUNNING` 并移入 `running`；此时模型尚未执行这段 prefill。未调度完整 prompt 的 chunked prefill 请求仍留在 `waiting`。因此 `RUNNING` 是调度状态，不是“prefill 已执行完成”的证明。

当生成结束后，请求变成 `FINISHED`，并从 running 中移除。

`Scheduler.schedule()` 的策略可以概括为：

```text
优先调度 waiting 队列做 prefill
如果本轮调度到了 prefill，直接返回
如果没有 prefill，再调度 running 队列做 decode
```

prefill 阶段受两个参数限制：

```text
max_num_seqs
max_num_batched_tokens
```

本轮最多跑多少条请求、本轮最多处理多少 token，都由 scheduler 控制。

decode 阶段会检查是否还有 KV block 可以追加：

```python
while not self.block_manager.can_append(seq):
    ...
```

如果 KV block 不够，scheduler 会触发 preempt，把部分请求重新放回 waiting。

推理引擎必须管理请求状态，因为 GPU 和 KV Cache 都是有限资源。

## 8. 第八步：区分 prefill 和 decode 的输入

`ModelRunner.run()` 根据 `is_prefill` 走不同准备路径：

```python
input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
temperatures = self.prepare_sample(seqs)
logits = self.run_model(input_ids, positions, is_prefill)
token_ids = self.sampler(logits, temperatures).tolist()
reset_context()
```

prefill 准备的是一段 prompt token：

```text
input_ids
positions
cu_seqlens_q
cu_seqlens_k
slot_mapping
block_tables
```

decode 准备的是每个 sequence 的 `last_token`：

```text
input_ids
positions
slot_mapping
context_lens
block_tables
```

这里可以这样理解：

```text
prefill 是把一段 prompt 写入模型上下文和 KV Cache；
decode 是拿每个请求的 last token 继续生成下一个 token，同时读取历史 KV Cache。
```

因此两者会使用不同的 attention 调用。

## 9. 第九步：确认 KV Cache 写入位置

KV Cache 的空间在 `ModelRunner.allocate_kv_cache()` 里分配。

代码里对单个 block 的大小估算是：

```text
block_bytes =
    2
  * num_hidden_layers
  * block_size
  * num_kv_heads_per_rank
  * head_dim
  * dtype.itemsize
```

这里的 `2` 表示 K 和 V。

分配出来的 cache 形状大致是：

```text
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

每一层 attention 模块会拿到自己对应的 `k_cache` 和 `v_cache`。

KV Cache 的写入位置由 `slot_mapping` 决定。

`Attention.forward()` 中可以看到：

```python
store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

对应关系是：

```text
Scheduler / BlockManager 决定这个请求有哪些 block
ModelRunner 根据 block_table 算出 slot_mapping
Attention 根据 slot_mapping 把 K/V 写进对应 cache slot
```

KV Cache 的写入位置由上层调度和 block 管理共同决定。

## 10. 第十步：看 Attention 如何区分 prefill 和 decode

`Attention.forward()` 里有一个分支：

```python
if context.is_prefill:
    o = flash_attn_varlen_func(...)
else:
    o = flash_attn_with_kvcache(...)
```

prefill 使用 `flash_attn_varlen_func`。

这是因为一批请求的 prompt 长度可能不同，需要 variable length attention。

decode 使用 `flash_attn_with_kvcache`。

这是因为 decode 阶段每个请求通常只输入一个新 token，同时需要通过 `block_tables` 和 `context_lens` 读取完整历史 KV Cache。

因此 prefill / decode 在源码中对应不同的数据组织和 kernel 调用。

## 11. 第十一步：看 postprocess 如何推进请求状态

模型执行完成后，`Scheduler.postprocess()` 回写状态：

```python
self.block_manager.hash_blocks(seq)
seq.num_cached_tokens += seq.num_scheduled_tokens
seq.num_scheduled_tokens = 0
...
seq.append_token(token_id)
```

这里包含几个关键动作：

1. 对刚写满的 KV block 做 hash，用于后续 prefix cache。
2. 更新 `num_cached_tokens`，表示这些 token 已经进入 KV Cache。
3. 如果 prefill 还没处理完，不 append 新 token。
4. 如果可以生成新 token，把 sampled token append 到 `Sequence.token_ids`。
5. 如果遇到 EOS 或达到 `max_tokens`，把请求标记为 `FINISHED` 并释放 KV block。

这一步让请求生命周期闭合起来：

```text
WAITING
-> 调度 prefill（未调度完整时仍在 WAITING）
-> 调度最后一段 prefill 时设为 RUNNING
-> 执行最后一段 prefill，采样并 append 首 token
-> decode 继续采样并 append token
-> FINISHED
-> free KV blocks
```

## 12. 第十二步：定位 Prefix Cache 的入口

这条主链路里已经能看到 prefix cache 的入口。

`BlockManager` 维护了：

```text
hash_to_block_id
free_block_ids
used_block_ids
```

它会对完整 block 做 hash：

```python
compute_hash(token_ids, prefix_hash)
```

这里会 hash 当前 block 的 token，也会把前一个 block 的 hash 作为 prefix 放进去。

这样做用来保证：

```text
当前 block token 相同还不够，
它前面的上下文也必须相同，
才能安全复用 KV Cache。
```

这个设计适合放到下一篇继续讲。

沿着“generate 主链路”继续往下，下一篇进入：

```text
KV Cache block 管理和 prefix cache。
```

## 13. 第十三步：串起整条链路

再回头看 `llm.generate()`，背后的流程可以整理成：

```text
1. 用户传入 prompt 和 SamplingParams
2. tokenizer.encode(prompt)
3. 创建 Sequence
4. Scheduler.add(seq)，进入 waiting queue
5. generate() 进入 while loop
6. step() 调用 Scheduler.schedule()
7. waiting 中有请求时，优先做 prefill
8. 没有 prefill 时，从 running 中调度 decode
9. ModelRunner 准备 input_ids / positions / attention metadata
10. Attention 写入或读取 KV Cache
11. Sampler 根据 logits 和 temperature 选出 token
12. Scheduler.postprocess() 更新 Sequence
13. 请求完成后释放 KV block
14. tokenizer.decode(completion_token_ids)
15. 返回 text 和 token_ids
```

一句话概括：

```text
nano-vLLM 的 generate 是一个围绕 Sequence、Scheduler、KV Cache 和 ModelRunner 展开的请求推进循环。
```

## 14. 这一篇学到了什么

读完这条主链路后，我对 nano-vLLM 有了几个具体认识。

第一，`LLM` 只保留入口外壳，推理系统的核心在 `LLMEngine`。

第二，`generate()` 使用 while loop，因为推理服务要持续推进多个请求。

第三，`Sequence` 是请求状态中心。它把 prompt token、生成 token、调度状态、KV block table 和采样参数放在一起管理。

第四，`Scheduler.schedule()` 决定本轮是 prefill 还是 decode，并且要考虑 token budget、sequence 数量和 KV block 是否足够。

第五，`ModelRunner` 负责模型调用，也负责准备 prefill/decode 所需的 tensor、attention metadata、sampling 参数，并处理 CUDA Graph 路径。

第六，KV Cache 是贯穿 scheduler、model runner 和 attention 的系统资源。

这也回到第二篇里的问题：

```text
为什么推理引擎不同于 transformers.generate？
```

因为推理引擎要把“生成文本”变成“在有限 GPU 显存和调度预算下持续推进多请求”的系统问题。

## 15. 下一篇继续看什么

这一篇打通了主链路。

下一篇继续看两个问题：

```text
KV Cache block 是怎么分配、追加、释放的？
Prefix Cache 为什么要求完整前缀一致？
```

对应源码包括：

- [`nanovllm/engine/block_manager.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py)
- [`nanovllm/engine/scheduler.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py)
- [`nanovllm/engine/model_runner.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py)
- [`nanovllm/layers/attention.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/attention.py)

这一篇回答：

```text
一次 generate 背后发生了什么？
```

下一篇回答：

```text
KV Cache 为什么要做 block 管理？
```
