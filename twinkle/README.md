# Twinkle TaaS：中文法律数据 LoRA 实践

[阅读主文章](article.md) · [历史实验观察记录](notes/experience-log.md) · [返回合集](../README.md)

示例展示客户端数据处理与训练循环、远端模型计算、checkpoint 保存和同题推理。它不是经验证的法律助手产品；任何输出都不构成法律建议。

## 内容与状态

| 文件 | 用途 |
| --- | --- |
| `article.md` | 技术文章主版本；2026 年 7 月实验记录的整理稿 |
| `examples/main.py` | 40 条输入切片，过滤超长样本后执行 LoRA 训练 |
| `examples/legal_eval.py` | 两道固定问题的 base / LoRA 同题采样，支持 JSON 留档 |
| `examples/inference.py` | 指定自己 checkpoint 的单问题采样 |
| `examples/common.py` | 环境读取与配置校验，不在 import 时启动服务 |
| `examples/test_examples.py` | 无远端请求的隔离单元测试 |

原记录记载：40 条输入经过长度过滤留下 32 条，batch size 4，共 8 个 step。过滤结果取决于数据版本和模板，并非未来每次都恰好 32 条。本次没有重新执行训练或推理。

## 环境与权限

使用 Python 3.11 独立环境。以下命令先进入本仓库 `twinkle/` 目录，再执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` 固定的是归档时检查过的本地上游源码提交，以及原记录中的部分依赖版本；它不是全部传递依赖锁文件，也不是本次重新安装验证过的环境。原实验只记录 `twinkle-kit==0.4.0.dev0`，未记录具体安装提交。

不要单独安装名为 `twinkle-client` 的未核实包；这里的 `twinkle_client` 模块来自 `twinkle-kit` 源码，`[client]` 是其可选依赖组。

在本地 `.env` 填入自己的 `MODELSCOPE_TOKEN`。脚本不会用 `.env` 覆盖已导出的环境变量，也不会把 token 写入 JSON。`.env` 的位置始终按脚本所在目录定位，不依赖启动时的工作目录。

运行前需确认 ModelScope TaaS 准入、当前支持的基础模型、资源配额及服务规则。默认的 `Qwen/Qwen3.6-27B` 是原实验配置，不代表今天必然开放。若修改 `TWINKLE_BASE_MODEL`，也要人工确认它仍适配代码里的 `Qwen3_5Template`；不能任意换模型而继续沿用模板。

客户端的 tokenization 在本地进行，但训练 datum 中的 token IDs、标签等会发送给远端；“本地预处理”不等于训练内容不出本机。不要使用未获授权上传的数据。这里只提供数据集标识，不打包原始训练数据或模型权重。

## 执行流程

下面命令会访问远端服务并可能消耗配额。安装后可以先通过 `--help` 查看参数，它不会加载模型或创建训练任务。

```bash
python examples/main.py --help
python examples/legal_eval.py --help
```

先做 base 采样。即使 `.env` 已有 LoRA 路径，`--mode base` 也不会加载它：

```bash
python examples/legal_eval.py --mode base --output outputs/base.json
```

然后执行训练：

```bash
python -u examples/main.py
```

训练结束后，将服务返回的真实 `twinkle://.../weights/...` 路径填入 `.env` 的 `TWINKLE_MODEL_PATH`。文章中的 `YOUR_RUN_ID` 仅是脱敏占位符，不能直接使用。

```bash
python examples/legal_eval.py --mode lora --output outputs/lora.json
python examples/inference.py --prompt '请介绍一下你自己。'
```

JSON 保存两道题的全部返回文本和采样配置，适合后续重复实验；其中的 checkpoint、prompt 和输出可能敏感，所以 `outputs/` 默认不提交。脚本会创建输出父目录，并在 `--output` 已存在时拒绝继续，请为每次实验使用不同文件名。

## 结果如何解释

默认每题采样 2 次，温度 0.2、top-p 0.9、最多 512 token。两道题的非确定性输出与可能发生的长度截断，不足以支持“微调提高法律能力”或“风格稳定改变”的结论；这里只观察输出并验证链路。

若要得出效果结论，需补全独立测试集、数据去重、统一解码、多次重复实验及可信评分。保留特殊结束 token 的原始输出，展示时如清理需说明处理方式，不能靠清洗制造提升。

## 本地检查

在合集根目录运行：

```bash
python3 -m unittest discover -s twinkle/examples -p 'test_*.py'
```

这里的模拟测试只验证程序行为，不代表服务权限、模型兼容、checkpoint 生命周期和训练效果已经通过验证。上游来源见 [SOURCES](../docs/SOURCES.md)。
