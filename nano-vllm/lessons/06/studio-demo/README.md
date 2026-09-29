# mini-SGLang ModelScope Studio 草案

> 仅作为文章配套的未验收原型归档，不建议直接公开部署。本次只检查 Python/Shell 语法，没有构建镜像或运行 GPU。历史评估文件已标记为 `stale` / `incomplete`，其中旧来源观察、模型许可和平台资源仍需重新核验。

已知待办：就绪探测的 curl 缺少单次超时；双进程监督尚不完善；聊天历史未做 token 预算；流式响应为空 `choices` 时缺少保护；尚未完善后端异常提示和访问控制。`show_api=False` 只隐藏 Gradio API 说明，不提供鉴权。后端只监听回环地址也不等于 UI 已有身份验证。

目录内容对应文章中的双进程方案：

```text
Gradio :7860
  -> OpenAI-compatible API 127.0.0.1:1919
     -> mini-SGLang
        -> Qwen/Qwen3-0.6B
```

## 创空间设置

- SDK：Docker
- GPU：xGPU，目标配置按 L20 48 GB 评估
- 公开端口：`7860`
- 持久化目录：`/mnt/workspace`
- 模型：`Qwen/Qwen3-0.6B`

Docker 创空间需要完成阿里云账号绑定和实名认证。平台实际 GPU、可见显存、CUDA 驱动和启动耗时必须在部署后记录。

## 配置参数

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `MODEL_ID` | `Qwen/Qwen3-0.6B` | ModelScope 模型 ID |
| `MEMORY_RATIO` | `0.70` | mini-SGLang 显存预算比例 |
| `MAX_RUNNING_REQUESTS` | `8` | 同时运行的请求上限 |
| `MAX_SEQ_LEN` | `4096` | Demo 上下文上限 |
| `CUDA_GRAPH_MAX_BS` | `8` | CUDA Graph 最大 batch size |
| `MAX_PREFILL_LENGTH` | `2048` | 单轮 prefill 长度上限 |
| `PAGE_SIZE` | `16` | KV Cache page size |

这些值是面向单人交互 Demo 的保守初始值，不是实测最优值。目标创空间完成冷启动、连续对话、并发和峰值显存验证后再调整。

## 发布前检查

```bash
python -m py_compile app.py
bash -n entrypoint.sh
docker build -t minisgl-studio-demo .
```

本地 Docker 运行需要 NVIDIA Container Toolkit：

```bash
docker run --rm --gpus all -p 7860:7860 \
  -v minisgl-workspace:/mnt/workspace \
  minisgl-studio-demo
```

部署验收至少记录：实际 GPU 型号、可见显存、首次启动耗时、缓存命中后的重启耗时、峰值显存和剩余运行时网络请求。
