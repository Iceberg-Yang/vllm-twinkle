# 本地基准结果（仅模板）

当前没有实测数据或可发布的性能结果。此目录用于使用者后续保存 JSONL、汇总表及 profiler trace，不应以示例参数推断吞吐或显存。

默认产物：

```text
offline_results.jsonl
offline_results.md
serving_results.jsonl
serving_results.md
profiles/
```

这些结果默认被合集根 `.gitignore` 排除，只有本 README 模板保留。自定义位置或已跟踪文件不受该规则保护；发布前人工检查模型路径、URL、错误、notes 和 trace 中的隐私，不直接上传原始日志。

从合集根目录运行以下汇总命令（需要先有真实结果，缺文件会失败）：

```bash
python nano-vllm/experiments/benchmarks/summarize_results.py \
  --results-path nano-vllm/experiments/benchmarks/results/offline_results.jsonl \
  --markdown-path nano-vllm/experiments/benchmarks/results/offline_results.md

python nano-vllm/experiments/benchmarks/summarize_serving_results.py \
  --results-path nano-vllm/experiments/benchmarks/results/serving_results.jsonl \
  --markdown-path nano-vllm/experiments/benchmarks/results/serving_results.md
```

默认保留失败，只有显式 `--only-success` 才过滤；旧 `--include-errors`（离线也兼容 `--include-failures`）仍可使用。完整说明见 [基准脚本 README](../README.md)。

## 后续实测记录模板

以下字段尚未填写，不代表已验证：

```text
日期：
GPU / 驱动 / CUDA：
Python / torch / Transformers / accelerate：
引擎版本与源码提交：
模型版本与有效 dtype：
请求数 / 输入输出长度 / seed：
运行命令（脱敏）：
预热和缓存设置：
成功数 / 失败数 / 退出码：
结果路径（脱敏）：
指标单位与范围：
重复次数与稳定性：
限制与备注：
```

离线 `peak_memory_gb` 实为调用进程 torch allocator 的 **GiB**，不是整卡显存。HTTP 成功请求分位数需与所有请求延迟、排队和成功率一起阅读；chunk/词数近似不可当作精确 token 数。
