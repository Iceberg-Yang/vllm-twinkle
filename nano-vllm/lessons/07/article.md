# 推理 Infra 学习实录 07：四种推理路径的统一横评

> 归档状态（GitHub 发布副本）：横评方案，尚无 48 条同机实测记录，本轮未做 GPU 验证，不提供性能排名。ModelScope 发布 URL 尚未知，不表示已发布。
> [配套 Notebook](main.ipynb) · [上一篇](../06/article.md)。Notebook 输出与执行计数已清空。
> 版本边界：nano-vLLM 安装与源码说明对齐已核对 [commit bb823b3](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)，mini-SGLang 版本见下文；固定源码不等于全部依赖已锁定。
> 路径说明：`/mnt/workspace` 是 ModelScope 平台缓存示例，模型目录以 `snapshot_download()` 返回值为准。本文路径以合集根目录为基准。

前六篇依次展开了 nano-vLLM 的运行、调度与 KV Cache，mini-SGLang 的 serving 结构，以及魔搭创空间的应用编排。

这些内容对应四种常见推理路径：

```text
Transformers
nano-vLLM
mini-SGLang
vLLM
```

四者都能完成文本生成，但系统边界并不相同。Transformers 首先是模型库；nano-vLLM 首先是推理引擎教材；mini-SGLang 将轻量源码扩展到 serving 系统；vLLM 面向更完整的生产推理能力。

因此，横评不能只保留一列 tokens/s。本篇建立一套统一的实验口径，把能力边界、离线吞吐、服务延迟、显存和学习成本放在同一张图中观察。

当前仓库尚未保存四个框架在同一 GPU 上的完整实测记录。文中的结果表保持为空，Notebook 在目标 GPU 上执行后生成数据。官方仓库或其他硬件上的数字不进入本地排名。

## 1. 横评对象

四种路径的定位如下表：

| 路径 | 核心定位 | 适合观察的内容 |
|---|---|---|
| Transformers | 通用模型库与生成接口 | 模型结构、tokenizer、基础生成 |
| nano-vLLM | 轻量 vLLM 核心实现 | scheduler、KV block、prefix cache、CUDA Graph |
| mini-SGLang | 轻量 serving 系统 | Radix Cache、Chunked Prefill、Overlap Scheduling、API Server |
| vLLM | 生产级推理与服务框架 | 连续批处理、广泛模型支持、服务化、分布式与工程能力 |

Transformers 放入横评，不代表它与三个推理引擎属于完全相同的产品类别。它构成基础参照：当生成任务从一个静态 batch 扩展到动态请求流时，推理引擎额外处理了哪些系统问题。

## 2. 两组实验分别回答不同问题

横评拆成离线和在线两组。

### 离线生成

四个框架都通过本地 Python API 接收相同 token 序列：

```text
fixed token prompts
-> generate()
-> fixed output tokens
-> elapsed time / peak memory
```

记录指标包括：

- 输出 tokens/s。
- 输入与输出合计 tokens/s。
- 完成时间。
- PyTorch 峰值 allocated memory。
- OOM 和失败阶段。

离线测试用于比较单进程生成路径，不提供请求级 TTFT 和 TPOT。

### 在线服务

在线测试只覆盖内置 OpenAI-compatible API 的框架：

```text
mini-SGLang
vLLM
```

记录指标包括：

- TTFT：请求发出到第一个流式内容到达的时间。
- TPOT：首个内容之后，每个输出单位的平均间隔。
- 端到端延迟。
- 成功率和错误率。
- 总输出吞吐。

nano-vLLM 没有内置 API Server。为它临时包装 FastAPI 后再参与在线横评，会把自定义服务层开销混入框架差异，因此不放入当前在线表。

## 3. 公平比较的固定条件

实验固定以下变量：

| 项目 | 统一设置 |
|---|---|
| 模型 | `Qwen/Qwen3-0.6B` |
| 模型文件 | 同一份 ModelScope snapshot |
| GPU | 同一张卡，逐个框架运行 |
| 输入 | 相同 seed 生成的 token IDs |
| 输出 | `ignore_eos=True` 或强制 `min_new_tokens` |
| 最大上下文 | 4096 |
| 精度 | bf16，框架与 GPU 均支持时 |
| 预热 | 正式计时前完成一次短生成 |
| 重复次数 | 每个 case 独立运行 3 次，取中位数 |
| 并发进程 | 1，不同时加载多个框架 |

模型文件相同仍不足以保证环境相同。每条结果还需要保存：

```text
GPU name
driver
CUDA runtime
Python
PyTorch
Transformers
framework version or commit
command line
```

mini-SGLang 当前固定到上一篇核对的 commit：

```text
9a91cfafe754aa85daee49998176275667eb58f2
```

nano-vLLM 安装固定 commit，与源码说明对齐。在 `NANOVLLM_PYTHON` 指定的独立环境中安装：

```bash
"${NANOVLLM_PYTHON:?请先设置独立环境的绝对 Python 路径}" -m pip install \
  git+https://github.com/GeeeekExplorer/nano-vllm.git@bb823b3e06983d71485a8e1f23715ebd87d98ef8
```

已有 checkout 不自动覆盖，版本不符时应另选空目录。vLLM 和 Transformers 需要在运行记录中保存实际安装版本，不能只写“latest”。

## 4. 四组 workload

单一 workload 容易把框架差异压缩成偶然结果。实验使用四组输入输出组合：

| Case | 请求数 | 输入长度 | 输出长度 | 观察重点 |
|---|---:|---:|---:|---|
| A | 4 | 128 | 128 | smoke test |
| B | 16 | 512 | 256 | 混合负载 |
| C | 64 | 128 | 512 | decode-heavy |
| D | 16 | 2048 | 128 | prefill-heavy |

Case C 的输出明显长于输入，执行时间更容易受到 decode loop、CUDA Graph 和批调度影响。

Case D 的输入明显长于输出，执行时间更容易受到 prefill attention、chunked prefill 和输入 token budget 影响。

这两组数据放在一起，才能避免用一个总吞吐数字概括所有场景。

## 5. 统一 token 输入

benchmark 不使用自然语言 prompt，而是根据 tokenizer 词表大小生成固定长度 token IDs：

```python
def make_random_token_prompts(num_requests, input_len, vocab_size, seed):
    rng = random.Random(seed)
    return [
        [rng.randint(0, vocab_size - 1) for _ in range(input_len)]
        for _ in range(num_requests)
    ]
```

该设计直接固定 token 数量，避免不同 chat template、空格和文本内容造成长度偏差。

随机 token 不能用于评估生成质量。它只服务于系统性能实验，观察固定 token workload 下的执行行为。

## 6. 固定输出长度

nano-vLLM、mini-SGLang 和 vLLM 都将 `ignore_eos` 设为 `True`：

```python
SamplingParams(
    temperature=0.6,
    ignore_eos=True,
    max_tokens=output_len,
)
```

Transformers 使用：

```python
model.generate(
    ...,
    max_new_tokens=output_len,
    min_new_tokens=output_len,
)
```

只有 `max_new_tokens` 时，模型可能提前生成 EOS。某个框架少生成了一部分 token，完成时间自然更短，但这不是相同工作量下的性能差异。

## 7. 四个离线入口

### Transformers

```python
output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=output_len,
    min_new_tokens=output_len,
    do_sample=True,
    temperature=0.6,
)
```

所有请求组成一个静态 batch。请求不会在生成过程中动态进入或离开这次调用。

### nano-vLLM

```python
llm = LLM(
    model_dir,
    max_model_len=4096,
)

outputs = llm.generate(
    prompts,
    sampling_params,
    use_tqdm=False,
)
```

`LLMEngine.generate()` 内部持续调用 scheduler 和 model runner，维护 waiting/running 请求与 KV block。

### mini-SGLang

```python
llm = LLM(
    model_dir,
    max_seq_len_override=4096,
    page_size=16,
    cuda_graph_max_bs=8,
    cache_type="radix",
)

outputs = llm.generate(prompts, sampling_params)
```

离线 `LLM` 复用 Scheduler、PrefillManager、DecodeManager 和 Radix Cache，不经过 HTTP 层。

### vLLM

```python
llm = LLM(
    model=model_dir,
    max_model_len=4096,
)

outputs = llm.generate(
    [{"prompt_token_ids": prompt} for prompt in prompts],
    sampling_params,
)
```

四段代码的外层形式相似，差异集中在请求组织、KV Cache、attention backend、调度和服务能力。

## 8. 独立环境

四个框架不安装在同一个 Python 环境中。

```text
.venv-transformers
.venv-nanovllm
.venv-minisgl
.venv-vllm
```

mini-SGLang 固定版本对 PyTorch 和 Transformers 有自己的约束，nano-vLLM 还需要 FlashAttention，vLLM 也会绑定特定 PyTorch/CUDA 组合。强行合并环境可能发生以下变化：

```text
安装后一个框架
-> PyTorch 被升级或降级
-> Transformers 版本变化
-> CUDA extension 重新安装
-> 前一个框架的结果不再可复现
```

独立环境不是额外的形式工作，而是实验变量控制的一部分。

Notebook 必须显式配置 `TRANSFORMERS_PYTHON`、`NANOVLLM_PYTHON`、`MINISGL_PYTHON`、`VLLM_PYTHON`。四个值都必须非空、是绝对路径、通过 `is_file()` 和可执行检查，且路径及环境目录不同；不会回退到 `sys.executable`。未配置时默认 dry-run 并显示缺项，只有全部检查通过且设置 `RUN_FRAMEWORK_BENCHMARKS=1` 才允许执行框架 benchmark。虚拟环境里的 Python 可共享基础解释器符号链接，但不能把同一环境的 `python` / `python3` 当成两个独立环境。

该开关不控制依赖安装和模型下载：执行这些单元仍可能联网、写入本地缓存。如果只检查编排逻辑，请使用已装好 pandas 的 kernel，跳过安装和下载，并手动将 `model_dir` 设为已有模型目录。

## 9. benchmark runner

仓库中的 runner 位于：

```text
nano-vllm/experiments/benchmarks/
├── common.py
├── run_transformers_offline.py
├── run_nanovllm_offline.py
├── run_minisgl_offline.py
├── run_vllm_offline.py
└── summarize_results.py
```

Notebook 从当前目录向上查找 `nano-vllm/experiments/benchmarks`，将其所在的合集根目录设为 `repo_root`。每次生成实验矩阵都会用 UUID 分配独立 `run_id`；同一轮的 runner 写入下列 JSONL，不与旧实验混存：

```text
nano-vllm/experiments/benchmarks/results/<run_id>/offline_results.jsonl
```

每条 `config.notes` 都记录 `session=<run_id> case=A run=1` 等唯一组合。重跑应重新生成 `run_id`，不要重复执行旧 session。下面的展示记录尚未运行，示意 notes 位于 config 中：

```json
{
  "engine": "mini-SGLang",
  "config": {"notes": "session=<run_id> case=B run=1"},
  "num_requests": 16,
  "input_tokens": 8192,
  "output_tokens": 4096,
  "elapsed_s": null,
  "output_tokens_per_s": null,
  "peak_memory_gb": null,
  "oom": false,
  "error": null
}
```

上例中的性能字段为 `null`，表示尚未运行，不是零。

## 10. 执行顺序

建议先单独做 Case A smoke test，通过后再新建完整实验 session。完整 48 组矩阵由 [Notebook](main.ipynb) 生成和校验；下面仅展示其命令构造方式，应在 Notebook 的四环境校验通过后运行。下载路径必须使用返回值，不猜测磁盘缓存目录：

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-0.6B",
    cache_dir="/mnt/workspace/.cache/modelscope",
)
```

以 nano-vLLM 的 Case A 三次独立进程为例，沿用 Notebook 新生成的 `repo_root`、`engine_pythons`、`run_id` 和 `result_path`：

```python
import subprocess

for run in range(1, 4):
    command = [
        str(engine_pythons["nano-vLLM"]),
        str(repo_root / "nano-vllm/experiments/benchmarks/run_nanovllm_offline.py"),
        "--model", str(model_dir),
        "--num-requests", "4",
        "--input-len", "128",
        "--output-len", "128",
        "--max-model-len", "4096",
        "--dtype", "bfloat16",
        "--results-path", str(result_path),
        "--notes", f"session={run_id} case=A run={run}",
    ]
    subprocess.run(command, cwd=repo_root, check=True)
```

每一次 `(engine, case, run)` 都是独立阻塞子进程；退出后才进入下一次。不要在完整矩阵已运行的同一 session 再执行上述示例，否则会产生重复组合。Notebook 会拒绝重复执行已有 session，并另记子进程启动/退出失败；runner 写入的 error/OOM 记录也必须检查，退出码为零不等于实验成功。

这样每轮都保留完整的冷进程边界，同时仍把模型加载排除在 `generate()` 计时区间之外。

## 11. 结果表

本归档未提供四框架同机的 48 条完整 JSONL 记录，因此不填入推测数字。

Notebook 只从本轮 `session=<run_id>` 的成功记录计算中位数和峰值；失败单独计数，不混入性能统计，其他 session 不进入本轮汇总。完整性必须集合校验 4 engine × 4 case × 3 run 的唯一组合：无缺项、无重复、全部属于本轮且无 error/OOM/无效指标，仅有 `len == 48` 不够。

执行 Notebook 后生成以下表格：

| Case | Engine | Median out tok/s | Median elapsed s | caller_peak_allocated_gib | Errors |
|---|---|---:|---:|---:|---:|
| A | Transformers | 待实测 | 待实测 | 待实测 | 待实测 |
| A | nano-vLLM | 待实测 | 待实测 | 待实测 | 待实测 |
| A | mini-SGLang | 待实测 | 待实测 | 待实测 | 待实测 |
| A | vLLM | 待实测 | 待实测 | 待实测 | 待实测 |

官方 README 中的 benchmark 只能说明对应作者环境中的结果。硬件、版本、输入分布和参数不同，不能复制为本地 A10 或创空间 L20 的结论。

## 12. 显存指标的边界

runner 使用：

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
```

该指标表示调用进程的 PyTorch allocator 观测到的 peak allocated memory，除以 `1024**3` 后单位为 GiB；不一定覆盖引擎 worker，更不等于引擎总显存或 `nvidia-smi` 的进程占用。JSONL 暂保留旧兼容字段 `peak_memory_gb`，展示名称使用 `caller_peak_allocated_gib`。

差异可能来自：

- CUDA context。
- 非 PyTorch allocator 管理的内存。
- NCCL、FlashAttention 或其他 kernel workspace。
- 其他 worker 进程的模型和 KV Cache。
- 框架 reserved 但尚未 allocated 的内存；另需注意 nano-vLLM 初始化时就预分配 KV tensor，不能把其容量当作请求增量。

如果要比较引擎总显存，需要另行采样全部相关进程或设备的显存，并记录采样方法，不能直接用这一列排名。

## 13. 在线服务口径

mini-SGLang 和 vLLM 分别启动 OpenAI-compatible API，然后使用同一客户端：

```bash
"${MINISGL_PYTHON:?请先配置独立环境}" nano-vllm/experiments/benchmarks/run_serving_client.py \
  --url http://127.0.0.1:1919/v1/chat/completions \
  --endpoint chat \
  --model Qwen/Qwen3-0.6B \
  --num-requests 16 \
  --concurrency 4 \
  --prompt-words 128 \
  --max-tokens 128
```

服务横评增加以下控制项：

| 项目 | 处理方式 |
|---|---|
| 服务启动 | 模型加载和 warmup 完成后再压测 |
| 连接方式 | 本机回环地址，避免外网噪声 |
| 流式传输 | 两个框架都启用 |
| 并发 | 1、4、8 分别测试 |
| 请求速率 | 明确记录，不能同时使用无限速和固定 QPS 得出同一结论 |
| 输出 token | 优先使用服务端 usage；缺失时标记近似来源 |

mini-SGLang 固定版本的流式接口不返回 `usage.completion_tokens`。客户端会回退到 stream chunk 数量，此时 TPOT 的单位来源必须保留在结果中，不能与精确 token 计数混写。

## 14. 能力对照

不依赖性能数字的能力表如下：

| 维度 | Transformers | nano-vLLM | mini-SGLang | vLLM |
|---|---|---|---|---|
| 基础文本生成 | 支持 | 支持 | 支持 | 支持 |
| 离线 Python API | 支持 | 支持 | 支持 | 支持 |
| 内置 OpenAI API | 不属于核心生成 API | 未内置 | 支持 | 支持 |
| 请求调度 | 静态 generate 调用 | waiting/running scheduler | Prefill/Decode manager | 生产级 scheduler |
| KV 管理 | 模型 generation cache | block table | page table + Radix Cache | paged KV cache |
| Chunked Prefill | 非当前对照重点 | 简化实现 | 支持 | 支持，具体行为依版本配置 |
| Prefix Cache | 非 serving cache | block chain hash | Radix Tree | Automatic Prefix Caching |
| CUDA Graph | 非当前对照重点 | decode 路径 | 支持 | 支持，具体模式依版本配置 |
| Tensor Parallel | 模型级能力需另行配置 | 支持 | 支持 | 支持 |
| 源码学习成本 | 模型代码范围大 | 最低 | 中等 | 最高 |
| 生产服务定位 | 需要外部服务层 | 否 | 学习与实验优先 | 是 |

表中的“支持”不表示四个实现的成熟度、模型覆盖和工程细节相同。

## 15. 选型结论

选型结果取决于目标。

### 模型结构与基础生成

选择 Transformers。它提供 tokenizer、模型加载、GenerationConfig 和广泛的模型实现，适合作为正确性参照和模型结构入口。

### 推理引擎源码入门

选择 nano-vLLM。其代码规模较小，`Sequence -> Scheduler -> BlockManager -> ModelRunner` 调用链清晰，适合建立第一张推理系统地图。

### serving 系统结构学习

选择 mini-SGLang。它在相对紧凑的代码中呈现 API Server、Tokenizer/Detokenizer、Radix Cache、Chunked Prefill 和 Overlap Scheduling。

### 生产部署与完整生态

选择 vLLM，并基于目标版本重新验证模型兼容、量化、并行、服务参数和监控方案。生产选型还需要容量测试、故障恢复和升级策略，不能只依据单卡离线吞吐。

## 16. 横评结论

四种路径不存在脱离场景的统一排名。

```text
Transformers
-> 模型库与正确性参照

nano-vLLM
-> 推理引擎核心机制教材

mini-SGLang
-> serving 系统结构教材与实验框架

vLLM
-> 生产级推理与服务平台
```

性能表需要同机、同模型、同 token workload、同输出长度和多轮重复。能力表则用于解释数字背后的系统边界。

第 07 篇完成后，系列已经从“运行一个模型”推进到“建立可复现的框架选型方法”。下一阶段可以沿两条路线继续：

```text
性能路线：A10 实测、TTFT/TPOT、profiling、CUDA kernel
适配路线：Qwen3.5 新架构、config、权重映射、attention 与 KV Cache
```

## 参考资料

1. [Transformers Generation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
2. [nano-vLLM 官方仓库](https://github.com/GeeeekExplorer/nano-vllm)
3. [mini-SGLang 官方仓库](https://github.com/sgl-project/mini-sglang)
4. [vLLM 官方文档](https://docs.vllm.ai/)
5. [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
6. [本仓库 benchmark runner](../../experiments/benchmarks/)（`nano-vllm/experiments/benchmarks/`）
