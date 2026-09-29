# nano-vLLM 推理 Infra 学习系列

## 阅读顺序

建议按 **跑通 → 对比 → 调用链 → KV Cache → serving → 部署 → 横评方法** 阅读。前四篇聚焦 nano-vLLM，后三篇延伸至 mini-SGLang、ModelScope 创空间及 vLLM。

| 期数 | 文章 | 可执行配套 |
| --- | --- | --- |
| 01 | [从 nano-vLLM 开始](lessons/01/article.md) | [Notebook](lessons/01/main.ipynb) |
| 02 | [nano-vLLM vs Transformers](lessons/02/article.md) | [Notebook](lessons/02/main.ipynb) |
| 03 | [一次 generate 背后发生了什么](lessons/03/article.md) | [Notebook](lessons/03/main.ipynb) |
| 04 | [KV Cache Block 的分配、复用与回收](lessons/04/article.md) | [Notebook](lessons/04/main.ipynb) |
| 05 | [从 nano-vLLM 到 mini-SGLang](lessons/05/article.md) | [Notebook](lessons/05/main.ipynb) |
| 06 | [从推理服务到魔搭创空间](lessons/06/article.md) | [Notebook](lessons/06/main.ipynb) · [部署草案](lessons/06/studio-demo/README.md) |
| 07 | [四种推理路径的统一横评方案](lessons/07/article.md) | [Notebook](lessons/07/main.ipynb) · [benchmark 脚本](experiments/benchmarks/README.md) |

## 环境和版本

nano-vLLM 的 GPU 路径面向 Linux + NVIDIA CUDA，不是 macOS CPU/MPS 教程。推荐从已有 NVIDIA GPU 的 ModelScope Notebook 开始；硬件、驱动和账号资源是否可用需要自行确认。

本地核对的 nano-vLLM 提交为 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，与第 01 篇原 Notebook 的安装日志一致。其包约束为 Python 3.10–3.12。安装示例：

```bash
python -m pip install 'git+https://github.com/GeeeekExplorer/nano-vllm.git@bb823b3e06983d71485a8e1f23715ebd87d98ef8'
python -m pip install modelscope pandas accelerate
```

这不是完整环境锁文件。FlashAttention、Torch、Triton 与 GPU/驱动必须配套；本次未在新 GPU 环境验证安装。不要只凭 `nvidia-smi` 顶部 CUDA 字样判断所有 wheel 的兼容性。

模型通过 ModelScope `snapshot_download("Qwen/Qwen3-0.6B")` 获取，以返回值作为实际模型路径，不推测缓存目录名字。Notebook 中 `/mnt/workspace` 是魔搭环境示例；在其他平台应换成自己的可写目录。

mini-SGLang 的参考提交来自原稿：`9a91cfafe754aa85daee49998176275667eb58f2`，本次未联网复核该提交及安装可用性。Transformers、nano-vLLM、mini-SGLang、vLLM 应各自建立环境，记录每套 `pip freeze`、模型 revision、GPU 与运行参数；不能把不同环境的吞吐表直接当作公平排名。

## 实验边界

第 01 篇保留两张历史截图：[GPU](lessons/01/assets/nvidia-smi.png)、[吞吐结果](lessons/01/assets/nanovllm-benchmark.png)。133,966 个输出 token / 59.633176 秒约为 2,246.50 output tok/s，只对应该次随机 token workload，不是通用性能承诺。

第 02 篇数字为历史手记，缺少可独立复核的原始结果。第 05–07 篇是待验证流程与方案，`results/` 没有完整实测数据。本次已清空 Notebook 运行输出，以免修订后的代码展示旧运行结果。

第 01 篇生成与 benchmark 分别使用独立进程，避免同一个 Notebook 进程重复初始化 nano-vLLM 默认进程组、保留旧 KV Cache 显存。第 07 篇默认不执行完整横评；须配置四套 Python 路径并显式启用实验。

显存字段的历史名字 `peak_memory_gb` 实际按 1024³ 换算为 GiB，只是调用进程的 PyTorch allocator 峰值，不能代表 vLLM/mini-SGLang 全部 worker 或整卡显存。服务 chunk 近似指标也不能当作精确 token 指标。

封面图是原目录附带素材，并非实验结果；其来源与发布权仍待作者确认。上游项目来源及许可见 [来源说明](../docs/SOURCES.md)。
