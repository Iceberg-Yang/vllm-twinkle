# 推理 Infra 学习实录 06：从推理服务到魔搭创空间

> 归档状态（GitHub 发布副本）：部署草案，尚未完成目标创空间验证，不代表已上线。ModelScope 发布 URL 尚未知，不表示已发布。
> [配套 Notebook](main.ipynb) · [上一篇](../05/article.md) · [下一篇](../07/article.md)。Notebook 输出与执行计数已清空。
> 路径说明：`/mnt/workspace` 是目标平台缓存路径示例，实际挂载、持久性和权限仍需验收；[studio-demo 草案](studio-demo/) 由其他任务维护，本轮未修改。

前五篇完成了两条学习路径。

第一条路径围绕 nano-vLLM 展开：

```text
generate
-> scheduler
-> prefill / decode
-> KV Cache Block
-> Prefix Cache
```

第二条路径进入 mini-SGLang：

```text
API Server
-> Tokenizer / Detokenizer
-> PrefillManager / DecodeManager
-> Scheduler
-> Engine
```

推理引擎具备在线 API 后，还需要应用入口、启动编排、模型缓存、健康检查和资源限制。魔搭创空间承载这些应用层能力，mini-SGLang 继续负责模型推理。

本篇将 `Qwen/Qwen3-0.6B`、mini-SGLang 和 Gradio 组合为一个 Docker 创空间草案。目标不是把推理代码塞进页面进程，而是建立清晰的服务边界：

```text
浏览器
-> Gradio Web UI
-> OpenAI-compatible API
-> mini-SGLang
-> Qwen3-0.6B
```

文中的显存比例、并发和上下文长度是部署初值。实际 GPU、CUDA 兼容性、峰值显存和启动时间需要在目标创空间中测量。

## 1. 创空间中的系统边界

mini-SGLang 已经包含 FastAPI 和 OpenAI-compatible API。Gradio 只处理交互页面和 API 调用，不负责初始化模型。

职责内容如下表：

| 组件 | 职责 | 监听地址 |
|---|---|---|
| Gradio | 页面、对话历史、生成参数、流式展示 | `0.0.0.0:7860` |
| mini-SGLang API Server | OpenAI-compatible 请求、tokenize、调度、推理、detokenize | `127.0.0.1:1919` |
| ModelScope Cache | 保存模型文件 | `/mnt/workspace/.cache/modelscope` |
| Kernel Cache | 保存 FlashInfer、TVM FFI 等缓存 | `/mnt/workspace/.cache/...` |

进程结构如下：

```text
Docker container
├── entrypoint.sh
│   ├── start mini-SGLang :1919
│   ├── poll /v1/models
│   └── start Gradio :7860
├── mini-SGLang process tree
│   ├── API Server
│   ├── Tokenizer / Detokenizer Worker
│   └── Scheduler Worker / Engine
└── Gradio process
    └── OpenAI Python client
```

`1919` 只在容器内部使用，这只是网络可达范围限制，不等于鉴权。创空间对外暴露的 `7860` 仍包含 Gradio 页面及事件/API 入口，应配置访问控制、私有可见性和限流，不能据此声称用户无法绕过页面交互调用后端。`show_api=False` 只隐藏 API 文档，不会禁用 API；示例中的 `api_key="dummy"` 也不是有效鉴权。

## 2. SDK 选择

该应用采用 Docker SDK。

mini-SGLang 的依赖包括 CUDA、FlashInfer、`sgl-kernel`、TVM FFI 和 Linux-specific kernels。普通 Gradio SDK 更适合纯 Python 应用，无法完整表达 CUDA 基础镜像、系统包、多进程启动和 kernel 依赖。

Docker 镜像负责固定以下运行边界：

```text
Ubuntu 24.04
CUDA 12.8.1 devel
Python virtual environment
mini-SGLang fixed commit
Gradio / OpenAI client
```

本文沿用上一篇核对的 mini-SGLang commit：

```text
9a91cfafe754aa85daee49998176275667eb58f2
```

固定源码 commit 可以避免 CLI 参数、依赖上限和服务行为随 main 分支变化。

Docker 创空间还要求完成阿里云账号绑定和实名认证。这属于平台开通条件，不属于应用代码。

## 3. 模型与工作负载边界

模型使用 `Qwen/Qwen3-0.6B`。0.6B 模型便于将注意力放在 serving 和部署结构上，同时保留完整的对话生成链路。

Demo 的推荐工作负载定义如下：

| 项目 | 初始值 |
|---|---:|
| 模型 | `Qwen/Qwen3-0.6B` |
| 上下文上限 | 4096 tokens |
| 单次输出 | 256 tokens |
| Gradio 并发 | 1 |
| 等待队列 | 8 |
| mini-SGLang running requests | 8 |
| CUDA Graph 最大 batch | 8 |
| 显存预算比例 | 0.70 |

这些参数构成体验 Demo 的资源护栏，不代表模型能力上限，也不代表目标 GPU 的最优配置。

Gradio 并发限制为 1，mini-SGLang 仍保留 8 个 running request 配额。二者控制的层级不同：

```text
Gradio concurrency
-> 页面同时进入生成函数的任务数

mini-SGLang max running requests
-> 推理调度器内部允许同时运行的请求数
```

当前页面只开放单用户串行生成。后续接入独立 API、批处理或多页面入口时，后端配额仍有调整空间。

## 4. 持久化缓存

创空间实例重启后，容器文件系统中的临时数据可能丢失。模型和编译缓存统一写入 `/mnt/workspace`：

```text
/mnt/workspace/.cache/
├── modelscope/
├── huggingface/
├── flashinfer/
└── tvm-ffi/
```

启动脚本设置以下环境变量：

```bash
export MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope
export FLASHINFER_WORKSPACE_BASE=/mnt/workspace/.cache/flashinfer
export TVM_FFI_CACHE_DIR=/mnt/workspace/.cache/tvm-ffi
```

缓存设计需要分别验证两个阶段：

1. 首次启动完成模型下载和必要编译。
2. 重启后命中已有文件，不重复下载模型。

只验证首次启动不足以证明缓存有效。部署记录应同时包含冷启动时间和缓存重启时间。

## 5. mini-SGLang 启动参数

后端启动命令如下：

```bash
python -m minisgl \
  --model Qwen/Qwen3-0.6B \
  --model-source modelscope \
  --host 127.0.0.1 \
  --port 1919 \
  --cache-type radix \
  --memory-ratio 0.70 \
  --max-running-requests 8 \
  --max-seq-len-override 4096 \
  --cuda-graph-max-bs 8 \
  --max-prefill-length 2048 \
  --page-size 16
```

参数分成四组：

| 类别 | 参数 | 作用 |
|---|---|---|
| 模型来源 | `--model`、`--model-source` | 从 ModelScope 获取模型 |
| 服务边界 | `--host`、`--port` | 仅在容器内部提供 API |
| 请求规模 | `--max-running-requests`、`--max-seq-len-override` | 限制并发与上下文 |
| 显存结构 | `--memory-ratio`、`--cuda-graph-max-bs`、`--page-size` | 控制 KV Cache 和 CUDA Graph 相关预算 |

`memory-ratio=0.70` 为 Gradio、CUDA context、kernel workspace 和运行时峰值留下余量。实际余量必须通过 `nvidia-smi`、PyTorch 显存统计和完整生成请求确认。

## 6. 就绪检查

Gradio 不能与 mini-SGLang 同时无条件启动。后端加载模型期间，页面已经可访问会产生大量连接失败。

`entrypoint.sh` 按以下顺序执行：

```text
创建持久化缓存目录
-> 后台启动 mini-SGLang
-> 轮询 http://127.0.0.1:1919/v1/models
-> 后端就绪
-> 启动 Gradio
```

脚本同时检查 mini-SGLang 进程是否提前退出。模型加载或 CUDA kernel 初始化失败时，容器直接失败并保留后端日志，不会留下一个无法生成内容的空页面。

就绪检查与存活检查是两个概念：

| 检查 | 含义 |
|---|---|
| 进程存活 | mini-SGLang 进程仍在运行 |
| 服务就绪 | API 已经完成模型加载并能够响应请求 |

创空间开放页面需要以后者为准。

## 7. Gradio 调用 OpenAI-compatible API

页面使用 OpenAI Python SDK 连接容器内服务：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1919/v1",
    api_key="dummy",
)
```

生成函数将 Gradio 对话历史转换为 `messages`：

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens,
    stream=True,
)
```

流式返回逐步拼接文本：

```python
text = ""
for chunk in response:
    delta = chunk.choices[0].delta.content
    if delta:
        text += delta
        yield text
```

Gradio 不导入 mini-SGLang，也不持有模型对象。推理引擎依赖变化时，页面只需要保持 OpenAI-compatible 协议一致。

## 8. Dockerfile 的构建边界

Dockerfile 完成四个阶段：

```text
CUDA 基础镜像
-> 安装 Python 与 Git
-> 拉取并固定 mini-SGLang commit
-> 安装 mini-SGLang、Gradio 和 OpenAI client
```

核心结构如下：

```dockerfile
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG MINI_SGLANG_COMMIT=9a91cfafe754aa85daee49998176275667eb58f2

RUN git clone https://github.com/sgl-project/mini-sglang.git /opt/mini-sglang \
    && git -C /opt/mini-sglang checkout "${MINI_SGLANG_COMMIT}"

RUN pip install -e /opt/mini-sglang \
    && pip install -r /app/requirements.txt

EXPOSE 7860
ENTRYPOINT ["/app/entrypoint.sh"]
```

基础镜像中的 CUDA toolkit 不能替代宿主机 NVIDIA driver。目标创空间仍需验证：

```text
实际 GPU compute capability
宿主机 driver
容器 CUDA runtime
PyTorch CUDA build
FlashInfer / sgl-kernel
```

这条兼容链中任意一项不匹配，都可能在镜像构建成功后出现运行时错误。

## 9. 创空间部署草案

本篇目录包含可检查的部署草案：

```text
nano-vllm/lessons/06/studio-demo/
├── Dockerfile
├── app.py
├── entrypoint.sh
├── requirements.txt
├── README.md
└── studio-assessment.yaml
```

创空间配置为：

| 项目 | 配置 |
|---|---|
| SDK | Docker |
| 公开端口 | 7860 |
| 资源 | xGPU，按 L20 48 GB 基线评估 |
| 可见性 | 首轮使用私有 |
| 持久化目录 | `/mnt/workspace` |

L20 48 GB 是评估基线，不是运行时事实。部署后应记录实际 GPU 名称和可见显存，再判断资源是否与申请标签一致。

## 10. 部署前评估结论

`studio-assessment.yaml` 给出的结论为 `needs_information`。

已经明确的部分包括：

- 模型来源和开源许可。
- mini-SGLang 固定源码版本。
- Docker SDK 和双进程结构。
- 公开端口与内部端口。
- 持久化缓存目录。
- 单用户 Demo 的初始 workload。

仍需目标环境验证的部分包括：

- 创空间实际 GPU 和可见显存。
- CUDA 12.8、driver、FlashInfer 与 `sgl-kernel` 的兼容性。
- 推荐 workload 的峰值显存。
- 首次下载和 kernel 编译耗时。
- 缓存重启是否避免重复下载。

这些未知项不会阻止整理部署草案，但会阻止正式部署和“已经可稳定上线”的结论。下一步应在具备授权的私有创空间中完成运行时探测，再更新评估状态。

## 11. 验收顺序

私有创空间按以下顺序验收：

```text
1. 容器构建成功
2. nvidia-smi 与 PyTorch 识别 GPU
3. mini-SGLang 完成模型加载
4. /v1/models 返回成功
5. Gradio 页面可访问
6. 单轮流式生成成功
7. 多轮对话成功
8. 4096 token 上下文边界测试
9. 记录峰值显存
10. 重启并验证模型缓存
```

验收记录至少包含：

| 指标 | 记录方式 |
|---|---|
| GPU 型号 | `nvidia-smi`、`torch.cuda.get_device_name(0)` |
| 可见显存 | `nvidia-smi --query-gpu=memory.total` |
| 冷启动时间 | 容器启动到 `/v1/models` 首次成功 |
| 缓存重启时间 | 已有 `/mnt/workspace` 缓存后的同一路径 |
| 峰值显存 | 完整生成请求期间的峰值 |
| 运行时网络请求 | 模型加载完成后的外部访问记录 |

参数调整建立在这些记录之上。

## 12. 故障定位路径

双进程结构将故障分成三层：

### 后端未就绪

现象：容器启动失败，`/v1/models` 超时。

检查内容：

```text
模型下载
CUDA driver/runtime
FlashInfer / sgl-kernel
显存预留
CLI 参数
```

### 页面能够打开但生成失败

现象：Gradio 正常，OpenAI client 抛出连接或 API 错误。

检查内容：

```text
MINISGL_API_BASE
mini-SGLang 进程状态
请求模型 ID
messages 格式
后端错误日志
```

### 生成过程中 OOM

调整顺序如下：

```text
降低 MAX_SEQ_LEN
-> 降低 CUDA_GRAPH_MAX_BS
-> 降低 MAX_RUNNING_REQUESTS
-> 降低 MEMORY_RATIO 并重新观察整体峰值
```

`MEMORY_RATIO` 主要影响 mini-SGLang 内部显存预算。将其调高不等于系统能够承载更多请求，过高还会挤压运行时临时空间。

## 13. Demo 与生产服务的边界

该创空间面向学习和交互展示，不等同于生产推理服务。

| 能力 | 当前 Demo | 生产服务常见要求 |
|---|---|---|
| 访问入口 | Gradio 页面 | API Gateway、SDK、多租户入口 |
| 鉴权 | 内部端口不等于鉴权，Gradio 仍需访问控制 | API Key、用户权限、配额 |
| 调度 | 单实例、低并发 | 多副本、弹性、负载均衡 |
| 可观测性 | 容器日志和人工记录 | 指标、链路、告警、SLO |
| 发布 | 手工验收 | 灰度、回滚、版本治理 |
| 数据治理 | 不保存业务数据 | 审计、隐私、保留策略 |

把 Gradio 页面运行起来，只完成了体验入口。稳定服务还需要鉴权、限流、监控、容量规划和故障恢复。

## 14. 本篇结论

mini-SGLang 与魔搭创空间之间需要一层明确的应用编排。

完整结构如下：

```text
ModelScope Studio Docker
├── public: Gradio 0.0.0.0:7860
├── internal: mini-SGLang 127.0.0.1:1919
├── model: Qwen/Qwen3-0.6B
└── persistent cache: /mnt/workspace
```

Gradio 负责交互，mini-SGLang 负责推理，启动脚本负责就绪顺序，`/mnt/workspace` 负责跨重启缓存。四部分各自保持单一职责。

当前产物是部署草案，不是已上线记录。私有创空间中的运行时探测、完整生成、峰值显存和缓存重启验证完成后，才能确定公开发布参数。

## 参考资料

1. [mini-SGLang 官方仓库](https://github.com/sgl-project/mini-sglang)
2. [mini-SGLang 固定源码版本](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2)
3. [Qwen3-0.6B ModelScope 模型页](https://modelscope.cn/models/Qwen/Qwen3-0.6B)
4. [ModelScope 创空间 Docker 文档](https://modelscope.cn/docs/studios/docker)
5. [OpenAI Python SDK](https://github.com/openai/openai-python)
6. [Gradio ChatInterface 文档](https://www.gradio.app/docs/gradio/chatinterface)
