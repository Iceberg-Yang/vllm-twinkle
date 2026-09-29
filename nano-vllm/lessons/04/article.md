# 推理 Infra 学习实录 04：KV Cache Block 的分配、复用与回收

> 归档状态（GitHub 发布副本）：源码学习稿，本轮未做 GPU 验证；block 状态模拟不等于模型运行验证。ModelScope 发布 URL 尚未知，不表示已发布。
> [配套 Notebook](main.ipynb) · [上一篇](../03/article.md) · [下一篇](../05/article.md)。Notebook 输出与执行计数已清空。
> 版本边界：安装与源码说明对齐已核对 [nano-vLLM commit bb823b3](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)，已有 checkout 不自动覆盖；版本不符时应另选空目录。固定源码不等于全部依赖已锁定。
> 路径说明：`/mnt/workspace` 是 ModelScope 平台缓存示例，模型目录以 `snapshot_download()` 返回值为准。

前一篇里，我顺着 `LLMEngine.generate()` 读完了一次生成的主链路：prompt 先变成 `Sequence`，再由 `Scheduler` 决定本轮做 prefill 还是 decode，`ModelRunner` 准备 GPU 输入，`Attention` 负责读写 KV Cache。

`block_table` 在主链路中反复出现，仍有以下问题需要说明：

```text
KV Cache 为什么不能只给每个请求分一整块连续显存？
一个 Sequence 的 block_table 里到底存了什么？
两个请求共享相同前缀时，哪些 block 可以复用？
一个请求结束后，KV block 是立即清空，还是仍有机会被命中？
```

本篇聚焦 nano-vLLM 的 `BlockManager`，通过一组小型 token 序列观察 block 的分配、共享和释放。

本篇会完成四件事：

1. 用源码中的真实 `BlockManager` 做一个小型 block 实验。
2. 观察 `free_block_ids`、`used_block_ids`、`block_table` 和 `ref_count`。
3. 验证 Prefix Cache 为什么要求从开头连续一致。
4. 把 nano-vLLM 的实现和 vLLM 的 PagedAttention 思想联系起来。

## 1. KV Cache 为什么会成为显存问题

自回归生成分成 prefill 和 decode。

prefill 会处理整段 prompt，并为每一层 attention 计算 K 和 V。进入 decode 后，模型每次通常只生成一个新 token，但这个 token 仍然要关注前面的全部上下文。

如果每轮都重新计算历史 token 的 K/V，计算会大量重复。因此推理引擎会保存已经计算过的 K/V，这就是 KV Cache。

对单个 token 来说，KV Cache 的字节数可以近似写成：

```text
每 token KV 字节数
= 2 × 层数 × KV head 数 × head_dim × dtype 字节数
```

其中 `2` 表示 K 和 V。

如果再乘上 token 数和并发请求数：

```text
KV Cache 总量
≈ 2 × 层数 × token 数 × KV head 数 × head_dim × dtype 字节数 × 请求数
```

问题在于，请求的长度并不整齐：

```text
请求 A：prompt 短，输出很长
请求 B：prompt 很长，输出很短
请求 C：刚完成 prefill
请求 D：decode 到一半
请求 E：已经结束，可以释放资源
```

如果每条请求都提前占用一块按最大长度计算的连续 KV Cache，会造成内部浪费；如果只按当前长度分配连续区域，生成过程中又要不断扩容或搬迁。

固定大小 block 提供了另一种方式：

```text
逻辑 token block
-> Sequence.block_table
-> 物理 KV block
```

请求只需要记录自己使用了哪些物理 block，不要求这些 block 在显存里连续。

## 2. 实验环境与源码对象

实验运行环境为魔搭 Notebook，并沿用前几篇使用的 `Qwen/Qwen3-0.6B` 配置估算 KV block 大小。block 生命周期实验只操作 token id 和 `BlockManager`，不执行模型 forward，也不重复初始化 GPU runtime。

| 项目 | 设置 |
|---|---|
| 平台 | 魔搭 Notebook |
| 推理框架 | nano-vLLM |
| 配置参考模型 | `Qwen/Qwen3-0.6B` |
| 核心实验 | 真实 `BlockManager` + 人工 token 序列 |
| 实验 block size | 4，便于观察边界 |
| nano-vLLM 默认 block size | 256 |

Notebook 中仍然通过 ModelScope 下载模型配置：

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-0.6B",
    cache_dir="/mnt/workspace/.cache/modelscope",
)
```

核心源码文件如下：

- [`nanovllm/engine/sequence.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/sequence.py)
- [`nanovllm/engine/block_manager.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py)
- [`nanovllm/engine/scheduler.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py)

职责内容如下表：

| 对象 | 作用 |
|---|---|
| `Sequence` | 保存请求 token、状态、缓存进度和 `block_table` |
| `Block` | 保存物理 block id、引用计数、hash 和对应 token |
| `BlockManager` | 分配、复用、追加和释放 block |
| `Scheduler` | 根据 token budget 和空闲 block 决定哪些请求可以执行 |

`BlockManager` 内部最关键的四个容器是：

```text
blocks
free_block_ids
used_block_ids
hash_to_block_id
```

其中：

- `free_block_ids` 表示当前可以分配的物理 block。
- `used_block_ids` 表示至少被一个请求引用的物理 block。
- `hash_to_block_id` 是 Prefix Cache 的检索入口。
- `Sequence.block_table` 保存这条请求使用的物理 block id 顺序。

## 3. 用小 block 观察分配过程

正式运行 nano-vLLM 时，默认 `kvcache_block_size` 是 256。实验将 block size 缩小到 4，使 block 边界更加直观，并构造一条 10-token 请求：

```python
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm import SamplingParams

Sequence.block_size = 4
manager = BlockManager(num_blocks=8, block_size=4)

seq_a = Sequence(
    list(range(10)),
    SamplingParams(max_tokens=4),
)

cached_a = manager.can_allocate(seq_a)
manager.allocate(seq_a, cached_a)

print("num_blocks:", seq_a.num_blocks)
print("block_table:", seq_a.block_table)
print("free:", list(manager.free_block_ids))
print("used:", sorted(manager.used_block_ids))
```

10 个 token 按每块 4 个 token 切分：

```text
逻辑 block 0: [0, 1, 2, 3]
逻辑 block 1: [4, 5, 6, 7]
逻辑 block 2: [8, 9]       # partial block
```

`seq_a.num_blocks == 3`，`block_table` 将得到三个物理 block id，例如：

```text
[0, 1, 2]
```

相关概念分为两类：

```text
逻辑 block 序号：这是请求里的第几个 token block
物理 block id：这段 KV 实际放在哪个 cache block
```

`block_table[1] == 6` 的含义不是“第 6 个 token block”，而是：

```text
这条请求的第 1 个逻辑 block，存放在物理 KV block 6。
```

这层间接映射正是分页式 KV 管理的核心。

## 4. 什么时候 block 才能进入 Prefix Cache

分配 block 不代表其中已经有有效 KV。

每轮模型执行后、K/V 写入对应 slot，`Scheduler.postprocess()` 会调用下列方法。这不仅发生在 prefill 后：decode 后也会登记本轮新写满的完整 block。

```python
self.block_manager.hash_blocks(seq)
```

以下代码只模拟一次完整 prefill 的状态变化，不计算真实 K/V：

```python
seq_a.num_scheduled_tokens = len(seq_a)
manager.hash_blocks(seq_a)
seq_a.num_cached_tokens += seq_a.num_scheduled_tokens
seq_a.num_scheduled_tokens = 0
```

`hash_blocks()` 根据 `num_cached_tokens` 和 `num_scheduled_tokens` 计算新增的完整 block 范围：

```text
start = num_cached_tokens // block_size
end   = (num_cached_tokens + num_scheduled_tokens) // block_size
```

对于 10 个 token、block size 为 4 的请求：

```text
end = 10 // 4 = 2
```

因此只有前两个完整 block 会被 hash：

```text
[0, 1, 2, 3] -> 有 hash
[4, 5, 6, 7] -> 有 hash
[8, 9]       -> partial block，不进入 Prefix Cache
```

该结果说明 Prefix Cache 不是“任意 token 片段缓存”。在 nano-vLLM 里，它以固定大小的完整 block 为复用单位。

`can_allocate()` 使用 `range(seq.num_blocks - 1)` 探测可复用前缀，即保守地排除请求的最后一个逻辑 block。通常最后一个 block 是不完整的；即使 prompt 长度恰好是 block size 的整数倍，这个教学实现仍然会排除最后一个逻辑 block。因此，该实现不能被解释为“所有已经写满的 block 都一定会被复用”。

## 5. 相同前缀如何共享物理 block

第二条请求定义如下：

```python
seq_b = Sequence(
    [0, 1, 2, 3, 4, 5, 6, 7, 100, 101],
    SamplingParams(max_tokens=4),
)

cached_b = manager.can_allocate(seq_b)
print("cached blocks:", cached_b)

manager.allocate(seq_b, cached_b)
print("seq_a:", seq_a.block_table)
print("seq_b:", seq_b.block_table)
```

它和 `seq_a` 的前 8 个 token 一样，最后两个 token 不同：

```text
seq_a: [0 1 2 3] [4 5 6 7] [8 9]
seq_b: [0 1 2 3] [4 5 6 7] [100 101]
```

前两个完整 block 可以命中，第三个 partial block 需要新分配。两条 `block_table` 的关系如下：

```text
seq_a.block_table = [0, 1, 2]
seq_b.block_table = [0, 1, 3]
```

两条请求共享物理 block 0 和 1，并分别拥有独立的尾部 block。

共享之后，block 的 `ref_count` 会从 1 增加到 2：

```python
for block_id in seq_b.block_table[:cached_b]:
    print(block_id, manager.blocks[block_id].ref_count)
```

`ref_count` 是必要的。

如果 `seq_a` 先结束，只能把共享 block 的引用数从 2 减到 1，不能直接把 block 放回 free list，否则 `seq_b.block_table` 会指向已经可能被其他请求覆盖的 KV Cache。

## 6. 为什么“局部 token 一样”仍不能命中

Prefix Cache 要求的是完整前缀相同，不是某个局部 block 恰好相同。

例如：

```text
seq_a: [0 1 2 3]   [4 5 6 7]   [8 9]
seq_c: [50 51 52 53] [4 5 6 7] [200 201]
```

两条请求的第二个 block 都是 `[4, 5, 6, 7]`，但它们前面的上下文不同，所以第二个 block 对应的 K/V 也不同，不能直接共享。

nano-vLLM 使用链式 hash：

```text
h0 = hash(block0_tokens)
h1 = hash(h0, block1_tokens)
h2 = hash(h1, block2_tokens)
```

因此，第二个 block 的 hash 不只取决于 `[4, 5, 6, 7]`，还取决于前一个 block 的 hash。

源码还会在 hash 命中后比较 `token_ids`：

```python
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

链式 hash 保证“从开头连续一致”，token 比较则为 hash 冲突再加一道正确性检查。

## 7. Block 从 free 到 shared 再回到 free

一个 block 的完整生命周期如下：

```text
初始化
-> ref_count = 0
-> 位于 free_block_ids

_allocate_block()
-> 从 free_block_ids 取出
-> ref_count = 1
-> 加入 used_block_ids

Prefix Cache 被另一请求复用
-> ref_count = 2
-> 两条 Sequence 的 block_table 指向同一个 block

第一条请求结束
-> ref_count = 1
-> block 仍在 used_block_ids

最后一条引用结束
-> ref_count = 0
-> 回到 free_block_ids
```

`deallocate()` 把 block 放回 free list 时，并不会立即清空它的 `hash` 和 `token_ids`。因此，一个已经没有活跃请求引用的 free block，仍可能被后续相同前缀重新命中。

只有当 `_allocate_block()` 真正要把这个物理 block 分给新内容时，旧的 hash 索引才会失效，block 随后被 `reset()`。

这是一种“内容还没被覆盖，就继续保留复用机会”的设计。

## 8. Decode 为什么在 `len(seq) % block_size == 1` 时追加 block

`BlockManager.can_append()` 的边界判断如下：

```python
return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

常见的 block 边界判断是 `% block_size == 0`，而该实现判断的是 `== 1`。

原因在于 nano-vLLM 的时序：

```text
本轮 forward 当前 last_token
-> 把当前 last_token 的 K/V 写入 cache
-> sample 下一个 token
-> postprocess 把新 token append 到 Sequence
```

prefill 完成时，`postprocess()` 已经把第一个 sampled token 加进 `token_ids`，但它的 K/V 还没有写入 cache。它会成为下一轮 decode 的 `last_token`。

假设 block size 是 4，prompt 恰好有 4 个 token：

```text
prefill 后已有缓存: token 0, 1, 2, 3
append 第一个生成 token: token 4
此时 len(seq) = 5
5 % 4 == 1
```

当前 `last_token` 位于一个新 block 的第一个位置，因此 scheduler 必须先追加物理 block，再让 attention 写入它的 K/V。

这不是数学上的边界偏移，而是由“sample 后先 append、下一轮再计算这个 token 的 KV”这一执行时序决定的。

## 9. 为什么不直接用 CUDA 总显存观察 block 使用量

原实验计划通过改变 prompt 长度、`max_tokens` 和 batch size 观察显存变化。

`ModelRunner.allocate_kv_cache()` 的实现表明，该实验需要更换观察指标。

nano-vLLM 初始化时根据以下数据计算 KV Cache 容量：

```text
GPU 总显存
gpu_memory_utilization
模型当前占用与峰值
单个 KV block 字节数
```

计算 `num_kvcache_blocks`，随后一次性创建完整的 KV Cache tensor：

```python
self.kv_cache = torch.empty(
    2,
    num_hidden_layers,
    num_kvcache_blocks,
    block_size,
    num_kv_heads,
    head_dim,
)
```

因此，在同一个已经初始化的引擎里，仅仅改变请求长度，`torch.cuda.memory_allocated()` 不一定会随着活跃 block 数线性变化。大块 KV Cache 空间已经提前保留，变化的是其中哪些 block 被请求占用。

这篇更可靠的观测指标是：

- `len(free_block_ids)`
- `len(used_block_ids)`
- 每条请求的 `block_table`
- block 的 `ref_count`
- `num_cached_tokens`
- `can_allocate()` 返回的 prefix 命中 block 数

后续完整性能实验应记录 Prefix Cache 命中前后的 prefill token 数、TTFT 或 prefill 耗时，而不是只看 CUDA 总显存。

## 10. nano-vLLM 和 PagedAttention 是什么关系

PagedAttention 论文把操作系统分页思想带到 KV Cache 管理：一条请求的 KV 不必放在连续物理空间，而是拆成固定大小 block，通过类似页表的映射定位物理 KV block。这样可以减少显存碎片，并支持更灵活的 KV 共享。

nano-vLLM 中的对应关系如下表：

| 分页式概念 | nano-vLLM 对应对象 |
|---|---|
| 固定大小物理页 | `Block` |
| 页表式映射 | `Sequence.block_table` |
| 空闲页列表 | `free_block_ids` |
| 引用共享 | `ref_count` |
| 当前 token 写入位置 | `slot_mapping` |
| 前缀缓存索引 | `hash_to_block_id` |

但需要保持边界：nano-vLLM 是便于阅读的轻量实现，不等同于当前生产版 vLLM 的完整 block manager、调度策略或 Prefix Cache 实现。

vLLM 官方文档对 Automatic Prefix Caching 的描述也强调：一个 KV block 的身份不仅取决于 block 内 token，还取决于它之前的完整前缀。这个原则和 nano-vLLM 的链式 hash 是一致的，但两者的工程细节和能力范围不能直接画等号。

## 11. 实验结论

`BlockManager` 的实现呈现了以下关键机制。

第一，KV Cache 的 block 管理不是把 tensor 随便切小，而是用一层逻辑到物理的间接映射，让变长请求不必占用连续 KV 空间。

第二，`block_table` 存的是物理 KV block id。Scheduler 负责分配它，ModelRunner 再根据它构造 `slot_mapping` 和 GPU 侧 `block_tables`。

第三，Prefix Cache 只复用满足实现约束的前缀 block。局部 token 一样没有意义，前面的上下文也必须一致。

第四，`ref_count` 保护共享 block。一个请求结束，不能破坏其他请求仍在使用的 KV Cache。

第五，nano-vLLM 的 KV tensor 在初始化时预分配，所以“已分配 CUDA 显存”和“活跃 KV block 使用量”不是同一个指标。

完整链路如下：

```text
Sequence 的 token
-> 按 block_size 切成逻辑 block
-> BlockManager 分配或复用物理 block
-> block_table 保存映射
-> ModelRunner 计算 slot_mapping
-> Attention 把 K/V 写入对应 cache slot
-> 请求结束后按 ref_count 释放
```

## 12. 下一篇继续看什么

到这一篇为止，nano-vLLM 的几条核心线索已经连起来了：

```text
generate 主循环
-> scheduler
-> prefill / decode
-> KV block
-> Prefix Cache
```

下一篇将视角从 nano-vLLM 扩展到 mini-SGLang，重点分析：

```text
nano-vLLM 的 block Prefix Cache
和 mini-SGLang 的 Radix Cache 有什么区别？
两个小型推理框架分别适合学习哪些 serving 机制？
```

如果前四篇是在把一个轻量 vLLM 实现读透，下一篇就开始观察：同样是缓存和调度，不同推理框架会怎样组织系统结构。

## 参考资料

1. [nano-vLLM `block_manager.py`（固定 commit）](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py)
2. [nano-vLLM `scheduler.py`（固定 commit）](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py)
3. [nano-vLLM `model_runner.py`（固定 commit）](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py)
4. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
5. [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
6. [vLLM Automatic Prefix Caching Design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
