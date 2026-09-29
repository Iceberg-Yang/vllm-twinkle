# 推理基准脚本（待 GPU 实测）

这里提供 Transformers、nano-vLLM、mini-SGLang、vLLM 四个离线 runner，以及 OpenAI 兼容 HTTP 客户端、汇总与 profiler。**当前没有可发布的实测结果，`results/` 仅保留说明模板；示例参数不代表性能结论。** 本次修正只用标准库 mock 测试验证，未验证真实框架版本、模型或 GPU 执行。

## 运行位置与环境

以下命令均从文章合集根目录 `modelscope-tech-notes/` 运行，脚本路径以 `nano-vllm/experiments/benchmarks/` 开头。

- 四套框架使用**四个隔离 Python 环境**，分别按对应上游版本准备 PyTorch、CUDA、Transformers 等依赖；不要将四套依赖混装。记录所用源码版本/提交及环境，尤其是 nano-vLLM 和 mini-SGLang。
- 合集不包含可执行的 nano-vLLM 源码包。需要在其独立环境中预先安装 `nanovllm`（可为上游的 editable 安装）。兼容函数 `add_nanovllm_source_to_path()` 只检查当前环境能否找到包，不再注入旧源码路径；profiler 同样使用此检查。
- Transformers 的显式 `device_map={"": "cuda:0"}` **需要 accelerate**，固定使用第一张可见 GPU，不允许自动分片或 CPU offload。建议所有离线环境均以 `CUDA_VISIBLE_DEVICES=0` 限制为单卡。
- 模型由使用者预先准备在本地。`--model` 默认为 `~/huggingface/Qwen3-0.6B/`，可改为下面的 `models/Qwen3-0.6B`；不要将下载模型当作测试的一部分。

## 四个离线 runner

每条命令分别在对应环境运行，而不是在同一环境依次安装依赖：

```bash
# nano-vLLM 环境
CUDA_VISIBLE_DEVICES=0 python nano-vllm/experiments/benchmarks/run_nanovllm_offline.py \
  --model models/Qwen3-0.6B --num-requests 4 --input-len 128 --output-len 128

# Transformers + accelerate 环境
CUDA_VISIBLE_DEVICES=0 python nano-vllm/experiments/benchmarks/run_transformers_offline.py \
  --model models/Qwen3-0.6B --num-requests 4 --input-len 128 --output-len 128

# mini-SGLang 环境（接口依赖所用上游版本）
CUDA_VISIBLE_DEVICES=0 python nano-vllm/experiments/benchmarks/run_minisgl_offline.py \
  --model models/Qwen3-0.6B --num-requests 4 --input-len 128 --output-len 128

# vLLM 环境
CUDA_VISIBLE_DEVICES=0 python nano-vllm/experiments/benchmarks/run_vllm_offline.py \
  --model models/Qwen3-0.6B --num-requests 4 --input-len 128 --output-len 128
```

口径与限制：

- 请求数、长度和可选调度容量须为正数；`gpu_memory_utilization` 在 `(0, 1]` 内；`input_len + output_len <= max_model_len`，另须留出至少两个预热输出 token 的空间。
- 四个 runner 均以 `seed + 1` 生成随机预热输入，与正式 `seed` 不同，保持相同 batch 和 input_len，生成两个新 token 以经过 decode，随后同步。正式计时不包含模型加载与预热。输入为随机 token，不是自然语言质量评测；seed 控制输入，不保证跨框架采样输出一致。
- 温度固定为 `0.6`，忽略 EOS 保持指定输出长度。Transformers 显式关闭 EOS 提前停止与 `top_k`/`top_p` 过滤。mini-SGLang 只使用已确认的 `temperature`、`ignore_eos`、`max_tokens` 字段，不假定其余采样默认值相同。
- nano-vLLM 的 `ModelRunner` 使用 `torch.set_default_dtype(hf_config.dtype)`。runner 通过 `AutoConfig.from_pretrained(...).dtype` 核验 `--dtype`；不一致时明确失败，不假装支持覆盖，实际类型记录为 `effective_dtype`。所选 Transformers 版本需要提供该配置属性。
- Transformers 明确拒绝 `--max-num-batched-tokens`、`--max-num-seqs`、`--gpu-memory-utilization`，这些参数并未被实现。`--max-model-len` 在该 runner 中是输入/输出长度校验，不是修改模型位置编码上限。
- vLLM 显式传入 `enable_prefix_caching=False`（默认）或 `True`（指定 `--enable-prefix-caching`），不依赖版本默认值；使用独立预热输入也不是彻底清空前缀缓存的保证。
- 初始化、参数语义校验、预热或运行失败仍先追加 JSONL，再以退出码 `1` 结束。mini 的 `shutdown()` 错误另记 `cleanup_error`，不覆盖原始错误；仅清理失败也视为失败。命令行语法错误由 argparse 报错；结果路径必须可写。

离线结果默认追加至 `nano-vllm/experiments/benchmarks/results/offline_results.jsonl`。保留 `config.notes`、`engine`、`error`、`peak_memory_gb`、`elapsed_s`、`output_tokens_per_s` 等 Notebook 依赖字段。

`peak_memory_gb` 为兼容旧数据保留的名称，**实际单位是 GiB（除以 `1024**3`）**，记录同时注明 `memory_unit="GiB"`、`memory_scope="caller_process_torch_allocator"`。它只统计调用进程、当前 CUDA 设备的 torch allocator 峰值分配，不含其他进程、保留未分配缓存或非 torch 分配，**不能解释成整卡显存，也不能代表 vLLM 等 worker 进程的峰值**。离线 `generate()` 耗时不提供请求级精确 TTFT/TPOT。

环境记录包括 Python、torch/CUDA 和可获得的已安装框架 distribution 版本（只读本机元数据，不联网）。未安装/缺失 distribution 元数据的版本不填，不据此推断源码提交。

## HTTP serving 客户端

服务需由使用者另行启动。客户端默认流式访问本机服务：

```bash
python nano-vllm/experiments/benchmarks/run_serving_client.py \
  --url http://127.0.0.1:8000/v1/completions --endpoint completions \
  --model Qwen3-0.6B --num-requests 16 --concurrency 4 \
  --prompt-words 128 --max-tokens 128

python nano-vllm/experiments/benchmarks/run_serving_client.py \
  --url http://127.0.0.1:8000/v1/chat/completions --endpoint chat \
  --model Qwen3-0.6B --no-stream
```

- `--request-rate 0` 立即提交所有请求；线程池仅限制同时执行数量，多余请求在客户端排队。正值限制提交速率。`--prompt-words` 是固定前缀之外的填充词数，不等于 tokenizer 的输入 token 数；HTTP `max_tokens` 是上限，服务仍可能因 EOS 提前结束。
- `queue_s`：提交到 worker 开始；`latency_s`：worker 开始到响应结束（不含客户端排队）；`end_to_end_s`：提交到响应结束。逐请求字段保存在 `request_results`。
- TTFT 是 worker 开始到**首个非空文本 chunk**，跳过仅 role、空文本和 usage 事件；不是服务内部第一 token 时间。非流式 chat 读取 `choices[0].message.content`，无 TTFT/TPOT。
- TPOT 是 `(latency_s - ttft_s) / (output_units - 1)` 的客户端估计。优先使用 `usage.completion_tokens`；缺失时流式只能按非空文本 chunk 数近似，一个 chunk 可能包含多个 token。非流式缺 usage 时仅数空白分词。即使有 usage，也无法从网络 chunk 获得精确逐 token 时间。
- `latency_p50/p95`、TTFT/TPOT 分位数仅针对成功请求（表头为 `success-only`）；`all_latency_p95`、`queue_p95`、`end_to_end_p95` 包含成功与失败的已完成请求。`success_rate` 为成功数 / 提交计划请求数，结合 `completed_count`、`error_count` 和轮次 `error` 查看。
- 成功输出按 `usage.completion_tokens`、`stream_chunks`、`whitespace_words` 分组。混合单位时整体 `output_units`、`output_units_per_s` 为 `null`，只展示 `output_units_per_s_by_source`；不同 TPOT 单位也不混算分位数。吞吐分母为整轮 elapsed，包含排队、提交间隔和失败等待。
- HTTP 200 中的 `error` 仍是失败；流式 EOF 前没有 `[DONE]` **也没有**非空值的 `finish_reason` 则判截断失败。错误明细全部保存，不只保存前五条；任何失败请求/轮次落盘后退出码为 `1`。
- `--timeout` 是 socket **单次读写/阻塞操作超时**，不是严格的整请求或整轮 deadline。持续有数据的流可能运行很久；本版本不实现总期限或主动取消所有 worker。客户端针对常见 OpenAI 单行 `data:` SSE 文本协议，不承诺支持所有兼容服务扩展。

默认结果为 `nano-vllm/experiments/benchmarks/results/serving_results.jsonl`。

## 汇总、隐私与发布

```bash
python nano-vllm/experiments/benchmarks/summarize_results.py
python nano-vllm/experiments/benchmarks/summarize_serving_results.py

# 确实需要时才主动过滤失败；默认保留所有失败轮次
python nano-vllm/experiments/benchmarks/summarize_serving_results.py --only-success
```

两个汇总脚本支持 `--results-path` 和 `--markdown-path`；未指定后者时只输出到终端。`--include-errors` 保留兼容，离线的旧 `--include-failures` 也可继续用。缺文件、空文件或无效 JSON 会明确失败，不产生伪空成功表。Markdown 单元格中的管道符会转义。

默认结果目录由合集根 `.gitignore` 排除，只有 `results/README.md` 模板保留。**gitignore 不是脱敏**：自定义输出位置可能不被忽略，已跟踪文件也不受它保护。虽然不再默认记录 hostname，模型路径、runner 路径、URL、错误、notes 及 profiler trace 仍可能含个人目录、服务地址或凭据。发布前必须人工审阅与脱敏，再决定是否纳入文章；不要直接提交原始日志。

## Profiler 与本地测试

在已准备好的 nano-vLLM 环境中，可另行执行（本次未执行）：

```bash
CUDA_VISIBLE_DEVICES=0 python nano-vllm/experiments/benchmarks/profile_nanovllm.py \
  --model models/Qwen3-0.6B --num-requests 16 --input-len 512 \
  --output-len 256 --profile-memory
```

输出位于 `nano-vllm/experiments/benchmarks/results/profiles/`。Profiler 带来额外开销，不能与无 profiler 的吞吐直接混比。

标准库测试不依赖 torch 或框架，HTTP 和模型接口均在内存 mock，不访问外部服务、不下载模型、不安装软件、不运行 GPU：

```bash
python3 -B -m unittest discover -s nano-vllm/experiments/benchmarks -p test_benchmarks.py -v
```

mock 可验证协议、统计、参数拒绝、失败落盘与退出、汇总行为，但不能证明上游接口/内核可运行或任何性能优劣。真实单卡复现、框架版本兼容性和测量稳定性仍需后续验证。
