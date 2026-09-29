# 推理 Infra 学习实录 05：从 nano-vLLM 到 mini-SGLang

> 归档状态（GitHub 发布副本）：流程草稿，尚未实跑，不提供性能结论。ModelScope 发布 URL 尚未知，不表示已发布。
> [配套 Notebook](main.ipynb) · [上一篇](../04/article.md) · [下一篇](../06/article.md)。Notebook 输出与执行计数已清空。
> 版本边界：nano-vLLM 对照说明基于已核对 [commit bb823b3](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)；mini-SGLang 固定版本见下文。已有 checkout 不自动覆盖，固定源码不等于全部依赖已锁定。
> 路径说明：`/mnt/workspace` 是 ModelScope 平台缓存示例，模型目录以 `snapshot_download()` 返回值为准。

前四篇围绕 nano-vLLM 建立了一条完整的推理主线：

```text
prompt
-> Sequence
-> Scheduler
-> prefill / decode
-> KV Cache Block
-> Prefix Cache
-> output
```

nano-vLLM 用较小的代码规模呈现了推理引擎的核心结构。请求状态、调度循环、KV Cache block、prefix hash、FlashAttention 和 CUDA Graph 都能在源码中找到明确入口。

完成这条主线后，学习对象扩展到 mini-SGLang。

mini-SGLang 同样以轻量和可读为目标，但系统边界明显更宽。它不仅提供离线 `LLM.generate()`，还包含 API Server、Tokenizer/Detokenizer Worker、Scheduler Worker、Radix Cache、Chunked Prefill、Overlap Scheduling、Tensor Parallel 和 OpenAI-compatible API。

本篇通过基础运行和源码结构对照，分析两个框架分别适合学习哪些推理系统机制。

主要内容包括：

1. 在独立环境中安装 mini-SGLang。
2. 通过 ModelScope 获取 `Qwen/Qwen3-0.6B`。
3. 使用 mini-SGLang 完成离线生成。
4. 对比 nano-vLLM Block Prefix Cache 与 mini-SGLang Radix Cache。
5. 对比两套框架的 Chunked Prefill 和调度结构。
6. 定位 mini-SGLang 的服务化边界。

## 1. 两个项目的定位

nano-vLLM 和 mini-SGLang 都不是对生产框架的逐行缩写。它们保留的系统边界不同。

| 维度 | nano-vLLM | mini-SGLang |
|---|---|---|
| 学习对象 | vLLM 核心推理链路 | 现代 LLM serving 系统 |
| Python 代码规模 | README 标注约 1200 行 | README 标注约 5000 行 |
| 对外入口 | 离线 `LLM.generate()` | 离线 `LLM.generate()`、Shell、API Server |
| 请求组织 | `Sequence` + waiting/running | `Req`/`Batch` + Prefill/Decode Manager |
| 前缀复用 | 完整 block 链式 hash | Radix Tree |
| 长 prompt | 简化的 chunked prefill | 独立 `ChunkedReq` 与 token budget |
| CPU/GPU 重叠 | 未实现完整 overlap scheduler | Overlap Scheduling |
| 服务接口 | 无内置 API Server | OpenAI-compatible API |
| 并行能力 | Tensor Parallel | Tensor Parallel + 多进程 serving 结构 |

nano-vLLM 适合作为第一份推理引擎源码教材。它把复杂度集中在：

```text
generate
scheduler
block manager
model runner
attention
sampler
```

mini-SGLang 更适合作为下一阶段教材。它把研究范围扩展到：

```text
API 接入
tokenize / detokenize
prefill / decode manager
Radix Cache
page table
overlap scheduling
多进程通信
OpenAI-compatible serving
```

二者不是替代关系，而是不同学习阶段的入口。

## 2. 实验环境与版本边界

实验延续前几篇的魔搭 Notebook 环境和 Qwen3-0.6B 模型。

| 项目 | 设置 |
|---|---|
| 平台 | 魔搭 Notebook |
| GPU | NVIDIA CUDA GPU |
| 模型 | `Qwen/Qwen3-0.6B` |
| 模型来源 | ModelScope |
| mini-SGLang 来源 | `sgl-project/mini-sglang` 官方仓库 |
| 运行方式 | 离线 `LLM.generate()` |

本文核对的 mini-SGLang 官方源码版本为：

```text
commit: 9a91cfafe754aa85daee49998176275667eb58f2
date: 2026-05-17
```

mini-SGLang 当前存在几项明确的环境约束：

- 支持 Linux x86_64 / aarch64。
- 依赖 NVIDIA CUDA 和 Linux-specific kernels。
- Python 要求为 3.10 及以上。
- 当前 `pyproject.toml` 约束 `torch<2.10.0`。
- `transformers` 约束为 `>=4.56.0,<=4.57.3`。
- 依赖 FlashInfer、`sgl-kernel`、TVM FFI 等组件。

前几篇魔搭环境曾使用 PyTorch 2.10。直接在当前 kernel 内安装 mini-SGLang，可能触发 PyTorch 降级并影响 nano-vLLM 环境。

因此 Notebook 使用独立虚拟环境：

```text
当前 Notebook kernel
├── 负责环境检查和结果展示
└── .venv-minisgl
    ├── 安装 mini-SGLang
    ├── 下载或读取 ModelScope 模型
    └── 执行离线生成脚本
```

依赖隔离比在一个 kernel 中反复切换 torch、FlashAttention 和 FlashInfer 更稳定。

## 3. 安装 mini-SGLang

官方 README 推荐通过 `uv` 创建独立环境并从源码安装。

在 Notebook 中执行：

```python
%pip install -q uv modelscope pandas
```

随后使用与 Notebook 一致的目录和固定 commit。首次 clone 在当前工作目录生成 `mini-sglang/`；归档修订本身不执行这些命令，也不影响本次发布副本。

```python
from pathlib import Path
import subprocess

repo_dir = Path("mini-sglang")
commit = "9a91cfafe754aa85daee49998176275667eb58f2"
if not repo_dir.exists():
    subprocess.run(
        ["git", "clone", "https://github.com/sgl-project/mini-sglang.git", str(repo_dir)],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo_dir), "checkout", commit], check=True)
else:
    print("保留已有 checkout，不自动 fetch、pull 或覆盖：", repo_dir)
actual_commit = subprocess.check_output(
    ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True,
).strip()
if actual_commit != commit:
    raise RuntimeError(f"源码版本不符：{actual_commit}；请另选空目录克隆固定版本。")

venv_dir = repo_dir / ".venv-minisgl"
venv_python = venv_dir / "bin" / "python"
if not venv_python.exists():
    subprocess.run(["uv", "venv", "--python", "3.12", str(venv_dir)], check=True)
subprocess.run(
    ["uv", "pip", "install", "--python", str(venv_python), "-e", str(repo_dir)],
    check=True,
)
```

Notebook kernel 另需 `modelscope` 和 `pandas`，用于下载与结果展示，不在其中安装 mini-SGLang。

安装完成后检查关键版本：

```bash
mini-sglang/.venv-minisgl/bin/python -c \
  "import torch, transformers, minisgl; print(torch.__version__); print(transformers.__version__)"
```

这一步检查的是运行时依赖是否落在 `.venv-minisgl` 中，而不是当前 Notebook kernel。

## 4. 通过 ModelScope 准备模型

mini-SGLang 的命令行入口支持：

```text
--model-source modelscope
```

源码中的 `parse_args()` 会在模型参数不是本地目录时调用 ModelScope `snapshot_download()`，再把下载后的本地路径写回 `model_path`。

服务模式在同一独立环境中使用（回环监听不是鉴权，对外提供前仍需访问控制）：

```bash
mini-sglang/.venv-minisgl/bin/python -m minisgl \
  --model Qwen/Qwen3-0.6B \
  --model-source modelscope \
  --host 127.0.0.1
```

离线 `LLM` 接口接收的是 `model_path`。Notebook 先显式下载模型，再把本地目录交给 mini-SGLang：

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-0.6B",
    cache_dir="/mnt/workspace/.cache/modelscope",
)
```

显式保存 `model_dir` 有两个优点：

- nano-vLLM 与 mini-SGLang 可以引用同一份模型权重。
- 实验记录能够明确模型的实际本地路径。

## 5. mini-SGLang 的最小离线生成

mini-SGLang 的离线 API 与 nano-vLLM 接近。下面是独立脚本的核心逻辑，须在 `mini-sglang/.venv-minisgl/bin/python` 中运行，不要直接放进编排 kernel；Notebook 已用该解释器的 `subprocess.run(..., check=True)` 传入 `model_dir` 并等待退出：

```python
from minisgl.core import SamplingParams
from minisgl.llm import LLM

llm = LLM(
    model_dir,
    max_seq_len_override=4096,
    max_extend_tokens=2048,
    cuda_graph_max_bs=8,
    page_size=16,
)

outputs = llm.generate(
    ["请用三句话解释什么是大模型推理引擎。"],
    SamplingParams(
        temperature=0.6,
        max_tokens=128,
    ),
)

print(outputs[0]["text"])
print("output tokens:", len(outputs[0]["token_ids"]))
llm.shutdown()
```

这段代码验证以下链路：

```text
prompt
-> tokenizer
-> pending request
-> PrefillManager
-> Scheduler
-> Engine
-> KV Cache / attention
-> decode
-> output tokens
-> tokenizer.decode
```

`LLM` 继承 `Scheduler`。它把输入加入 `pending_requests`，再复用 scheduler 的运行循环完成离线推理。输出格式仍包含：

```text
text
token_ids
```

因此，从 nano-vLLM 迁移到 mini-SGLang 时，最外层使用体验并不陌生，变化集中在接口背后的系统结构。

## 6. 外层 API 相似，内部结构不同

nano-vLLM 的离线主链路为：

```text
LLMEngine.generate()
-> add_request()
-> Scheduler.schedule()
-> ModelRunner.run()
-> Scheduler.postprocess()
```

mini-SGLang 的离线主链路为：

```text
LLM.generate()
-> pending_requests
-> Scheduler.run_forever()
-> PrefillManager / DecodeManager
-> Engine.forward_batch()
-> cache_req() / request finish
```

两套系统的对象映射如下表：

| nano-vLLM | mini-SGLang | 作用 |
|---|---|---|
| `Sequence` | `Req` / `PendingReq` | 保存请求 token 和执行状态 |
| `Scheduler.waiting` | `PrefillManager.pending_list` | 等待进入 prefill 的请求 |
| `Scheduler.running` | `DecodeManager` | 正在 decode 的请求 |
| `BlockManager` | `CacheManager` + `BasePrefixCache` | 管理 KV Cache 物理空间与前缀复用 |
| `Sequence.block_table` | `TableManager.page_table` | 请求到物理 KV page 的映射 |
| `ModelRunner` | `Engine` | 准备元数据并执行模型 |
| `Attention` | Attention Backend | 执行 prefill/decode attention |

nano-vLLM 使用少量对象串起完整链路，适合建立整体直觉。mini-SGLang 将职责拆分得更细，适合继续研究 serving 模块之间的边界。

## 7. Block Prefix Cache 与 Radix Cache

[第 04 篇](../04/article.md) 分析了 nano-vLLM 的 Prefix Cache：

```text
完整 token block
-> 链式 hash
-> hash_to_block_id
-> 复用物理 KV block
```

一个 block 的 hash 同时依赖当前 block token 和前一个 block 的 hash。只有从请求开头连续一致的完整 block 才能命中。

mini-SGLang 使用 Radix Tree 组织前缀：

```text
root
└── shared prefix A
    ├── suffix B
    └── suffix C
```

`RadixPrefixCache.match_prefix()` 从 root 开始沿树匹配 token。`insert_prefix()` 将新前缀插入树中；当新请求只匹配某个节点的一部分时，`split_at()` 会拆分节点并形成新的公共父节点。

两种实现的核心目标一致：

```text
只有完整前缀相同，历史 KV 才能安全复用。
```

组织方式存在明显差异：

| 维度 | nano-vLLM Block Prefix Cache | mini-SGLang Radix Cache |
|---|---|---|
| 索引结构 | `hash_to_block_id` | Radix Tree |
| 匹配单位 | 完整 block | page 对齐的 token 前缀 |
| 分支表达 | hash 链隐式表达 | 树节点显式表达 |
| 部分节点匹配 | 不拆分 block | `split_at()` 拆分 Radix 节点 |
| 共享保护 | block `ref_count` | 节点 `ref_count` 与 lock/unlock |
| 回收策略 | free/used block 生命周期 | 可驱逐叶节点 + 时间戳 |
| 空间统计 | free/used block 数 | `evictable_size` / `protected_size` |

Radix Cache 不代表“任何 token 片段都可以复用”。mini-SGLang 的 `insert_prefix()` 会通过 `align_down(..., page_size)` 将插入长度向 page size 对齐，匹配结果也会向 page 边界对齐。

因此，Radix Tree 解决的是前缀分支的组织和查找问题，page table 仍负责底层 KV 空间映射。

## 8. Radix Cache 的保护与驱逐

mini-SGLang 将已缓存 token 分成两类：

```text
protected_size：仍被活跃请求引用，不能驱逐
evictable_size：没有活跃引用，可以在空间不足时驱逐
```

`lock_handle()` 沿匹配节点向 root 更新 `ref_count`。

请求使用缓存前执行 lock：

```text
ref_count: 0 -> 1
evictable_size 减少
protected_size 增加
```

请求不再使用缓存时执行 unlock：

```text
ref_count: 1 -> 0
protected_size 减少
evictable_size 增加
```

空间不足时，`evict()` 收集 `ref_count == 0` 的叶节点，并根据时间戳形成优先队列。被保护的节点和 root 不会进入驱逐集合。

这套机制比 nano-vLLM 的 free/used block 集合多出一层显式缓存淘汰策略，也更接近在线 serving 系统的资源管理问题。

## 9. 两种 Chunked Prefill

长 prompt 一次占满 prefill token budget，会增加峰值资源压力，也可能阻塞其他请求。

nano-vLLM 的实现规则较简单：

```text
本轮剩余 token budget 不足
且当前请求是本轮第一条请求
-> 允许只处理一部分 prompt
```

prefill 未完成的 `Sequence` 保留在 waiting 队列，下一轮继续执行。

mini-SGLang 使用独立的 `ChunkedReq` 表示未完成 prefill 的请求。`PrefillAdder` 根据 `token_budget` 计算：

```text
remain_len = input_len - cached_len
chunk_size = min(token_budget, remain_len)
is_chunked = chunk_size < remain_len
```

如果 prompt 尚未处理完，请求以 `ChunkedReq` 的形式留在 `pending_list`，并在后续 prefill batch 中继续推进。

| 维度 | nano-vLLM | mini-SGLang |
|---|---|---|
| 未完成 prefill 状态 | 原 `Sequence` 留在 waiting | `ChunkedReq` |
| token budget | `max_num_batched_tokens` | `max_extend_tokens` |
| 前缀命中参与预算 | `num_cached_tokens` | `cached_len` |
| 输出空间预留 | 简化处理 | `estimated_len = extend_len + output_len` |
| decode 并发影响 | 独立简化调度 | `inflight_tokens` 计入 `reserved_size` |

mini-SGLang 的代码展示了一个更完整的问题：prefill 调度不仅要考虑本轮算多少 token，还要为已有 decode 和请求未来输出预留 KV 空间。

## 10. Overlap Scheduling 扩展了调度问题

nano-vLLM 的 `step()` 顺序执行：

```text
schedule
-> GPU run
-> postprocess
```

mini-SGLang 的 overlap loop 将当前 batch 的 GPU 执行与上一批结果处理交叠：

```text
接收请求
-> 调度下一批
-> GPU 执行当前批
-> CPU 处理上一批结果
```

源码中的 `overlap_loop()` 使用独立 CUDA stream，并维护 `last_data` 与 `ongoing_data`。

该设计对应一个新的性能问题：

```text
当 decode 每轮只处理少量 token 时，
CPU 调度、元数据准备和 kernel launch 开销如何被隐藏？
```

环境变量可用于关闭 overlap scheduling，形成消融实验：

```bash
MINISGL_DISABLE_OVERLAP_SCHEDULING=1
```

关闭前后的吞吐差异必须在相同模型、相同 workload、相同 GPU 和相同预热条件下测量。本篇不使用官方 H200 数据推断魔搭 A10 的表现，也不在缺少实测结果时给出性能排名。

## 11. mini-SGLang 的服务化边界

nano-vLLM 的核心使用方式是离线生成。mini-SGLang 已经包含完整的在线请求入口：

```text
API Server
-> Tokenizer Worker
-> Scheduler Worker
-> Engine
-> Detokenizer Worker
-> streaming response
```

官方结构文档列出的关键组件包括：

- FastAPI API Server。
- Tokenizer Worker。
- Detokenizer Worker。
- 每个 TP rank 对应的 Scheduler Worker。
- ZeroMQ 控制消息。
- Tensor Parallel 场景下的 NCCL 通信。

启动命令如下：

```bash
mini-sglang/.venv-minisgl/bin/python -m minisgl \
  --model Qwen/Qwen3-0.6B \
  --model-source modelscope \
  --host 127.0.0.1 \
  --port 1919
```

服务启动后，可以通过 OpenAI Python SDK 调用：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1919/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "请解释 Radix Cache。"}
    ],
    max_tokens=128,
)
```

这部分为下一篇创空间 Demo 提供了基础：应用层可以调用标准 API，而不必把模型推理过程直接嵌入 Web UI 进程。

## 12. 安装和使用时需要注意的细节

### 12.1 README 与当前 CLI 参数存在命名差异

当前源码的 CLI 参数是：

```text
--cache-type radix
--cache-type naive
```

README benchmark 示例仍使用：

```text
--cache naive
```

运行时应以当前 `python -m minisgl --help` 和 `server/args.py` 为准。

### 12.2 ModelScope 参数属于命令行入口

`--model-source modelscope` 在 `parse_args()` 中处理。离线 `LLM` 构造函数没有 `model_source` 参数，因此离线模式应先调用 `snapshot_download()`，再传入本地 `model_dir`。

### 12.3 不在同一 kernel 中重复初始化两个引擎

nano-vLLM 和 mini-SGLang 都会加载模型并预留 KV Cache。连续在同一 Python 进程初始化两套引擎容易造成显存竞争，也会使峰值显存统计失真。

合理的执行方式包括：

- 分别使用独立虚拟环境和独立进程。
- 完成一个框架实验后结束进程，再运行另一个框架。
- 性能横评使用统一脚本、独立冷启动和多轮预热。

## 13. 两个框架分别适合学习什么

### nano-vLLM

适合建立以下基础能力：

- 从 `generate()` 追踪到 scheduler、runner、attention 和 sampler。
- 理解 prefill 与 decode 的执行差异。
- 理解 KV Cache block table、slot mapping 和 prefix hash。
- 手动模拟 waiting/running 队列。
- 在较少代码中建立推理引擎整体图景。

### mini-SGLang

适合继续研究以下系统能力：

- Radix Tree 如何组织共享前缀。
- KV page、page table 与缓存淘汰如何协作。
- Chunked Prefill 如何参与资源预算。
- CPU scheduling 与 GPU computation 如何 overlap。
- Tokenizer、Scheduler、Detokenizer 和 API Server 如何组成在线服务。
- Tensor Parallel 和多进程消息如何进入 serving 主链路。

学习顺序可以表示为：

```text
nano-vLLM
-> 读懂一台 GPU 内部如何完成生成

mini-SGLang
-> 读懂一个 serving 系统如何组织请求、缓存和进程

SGLang / vLLM
-> 继续研究生产级调度、模型覆盖、分布式与可观测性
```

## 14. 本篇结论

nano-vLLM 与 mini-SGLang 的外层 API 都保持简洁，但内部系统边界不同。

nano-vLLM 将核心复杂度集中在一次生成如何完成。`Sequence`、`Scheduler`、`BlockManager`、`ModelRunner` 和 `Attention` 构成紧凑的学习路径。

mini-SGLang 将核心复杂度扩展到请求如何进入并持续运行。Radix Cache、page table、Chunked Prefill、Overlap Scheduling、Tokenizer/Detokenizer 和 API Server 共同构成更完整的 serving 结构。

两套框架的关系如下：

```text
nano-vLLM：理解推理引擎核心机制
mini-SGLang：理解现代 LLM serving 系统结构
```

从 nano-vLLM 进入 mini-SGLang，不是更换一个生成 API，而是将学习范围从“模型如何执行”扩展到“请求、缓存、计算与服务如何协作”。

## 15. 下一篇内容

下一篇将使用 mini-SGLang 的 OpenAI-compatible API，构建魔搭创空间 Demo，重点处理：

```text
推理服务与 Web UI 的进程边界
模型下载与缓存
服务启动和健康检查
GPU 显存限制
Gradio 调用 OpenAI-compatible API
体验 Demo 与生产服务的边界
```

## 参考资料

1. [mini-SGLang 官方仓库](https://github.com/sgl-project/mini-sglang)
2. [mini-SGLang Features](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/docs/features.md)
3. [mini-SGLang System Architecture](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/docs/structures.md)
4. [mini-SGLang `RadixPrefixCache`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/kvcache/radix_cache.py)
5. [mini-SGLang Scheduler](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py)
6. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
7. [Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)
8. [nano-vLLM `block_manager.py`（固定 commit）](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py)
9. [nano-vLLM `scheduler.py`（固定 commit）](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py)

