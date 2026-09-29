# Twinkle TaaS 法律助手 LoRA 微调实验记录

> 历史观察记录（2026 年 7 月），不是完整原始日志；本次没有远端复跑。个人路径和任务 ID 已脱敏，下文输出摘要及“能力较强”“格式变化”等措辞是当时的主观观察，不是有评分和重复实验支持的结论。实际训练样本为长度过滤后的 32 条，不能把 40 条输入切片当成有效样本数。法律文本仅为模型输出观察，不构成法律建议。执行方法以 [当前 README](../README.md) 为准，下面命令仅记录当时的调用方式。

## 实验目标

使用 ModelScope Twinkle TaaS 跑通一次云端 LoRA 微调流程，并以中文法律助手作为可观察的领域化示例，记录：

- 原模型推理输出
- 远程 LoRA 训练过程
- checkpoint 保存路径
- 加载 LoRA 后的同题推理输出
- 对 Twinkle TaaS 体验的观察

## 实验配置

| 项目 | 配置 |
| --- | --- |
| Base model | `Qwen/Qwen3.6-27B` |
| TaaS endpoint | `https://www.modelscope.cn/twinkle` |
| 数据集 | `ms://AI-ModelScope/DISC-Law-SFT` |
| 数据集用途 | 中文法律 SFT，覆盖法律问答、文本摘要、判决预测、信息抽取等场景 |
| 数据集许可 | 原笔记记为 Apache-2.0，本次未核对数据卡及数据来源条款；不据此再分发数据 |
| 训练方式 | LoRA SFT |
| LoRA rank | 16 |
| Loss | `cross_entropy` |
| 数据切片 | `range(40)` |
| Batch size | 4 |
| Epoch | 1 |
| Learning rate | `1e-4` |
| Max length | 2048 |
| Truncation strategy | `delete` |

## 对比问题

### case_summary

请阅读以下案情，并用“案情摘要、争议焦点、分析思路、结论”四个小标题回答：
甲向乙借款10万元，双方通过微信约定月利率2%，乙当天转账。到期后甲只归还2万元，并称双方没有签订纸质借条，所以剩余款项不应继续偿还。乙准备起诉。

### legal_qa

房屋租赁合同到期后，承租人继续居住并按月支付租金，出租人也一直收取。半年后出租人突然要求承租人三天内搬离。请分析双方之间的法律关系以及承租人可以如何应对。

## 运行记录

### 1. 原模型推理

运行命令：

```bash
python legal_eval.py
```

环境现象：

- 首次运行会下载/读取 `Qwen/Qwen3.6-27B` 的 tokenizer/config 等文件。
- 本地会出现提示：`return_assistant_tokens_mask==True but chat template does not contain {% generation %}`。
- 本地会出现提示：未安装 flash-linear-attention/causal-conv1d，fallback 到 torch implementation。

#### case_summary 原模型输出摘要

原模型能够按照用户要求输出“案情摘要、争议焦点、分析思路、结论”四段结构，内容较完整，已经能识别：

- 微信聊天记录和转账记录可以形成证据链。
- 无纸质借条不必然影响借贷关系成立。
- 月利率 2% 折合年利率 24%，需要结合民间借贷利率司法保护上限判断。

观察：

- 原模型法律问答能力已经较强。
- 输出较长，解释比较充分。
- 结论部分在 512 tokens 限制下可能被截断。

#### legal_qa 原模型输出摘要

原模型能够识别合同到期后继续居住、持续交租且出租人收租的情形构成不定期租赁，并指出：

- 出租人可以解除不定期租赁。
- 解除需要提前合理期限通知。
- 三天通常不构成合理期限。
- 承租人应保存租金支付记录、沟通记录、房屋现状证据。

观察：

- 原模型已经能给出较专业的租赁合同分析。
- 这说明本实验不能简单写成“微调后才会法律”，更适合写成“用 Twinkle 跑通领域 LoRA 流程，并观察输出格式和领域表达的变化”。

### 2. LoRA 训练

首次训练使用 `range(500)`、`max_length=2048`，在本地数据编码阶段失败：

```text
ValueError: Input length 2376 exceeds max_length 2048
```

原因：

- DISC-Law-SFT 里的法律样本比 self-cognition 更长。
- 默认 `truncation_strategy='raise'`，遇到超长样本会直接报错。

处理方式：

```python
dataset.set_template(
    'Qwen3_5Template',
    model_id=base_model,
    max_length=2048,
    truncation_strategy='delete',
)
```

为了快速完成端到端体验链路，最终使用 40 条样本进行小样本 LoRA 训练：

```python
dataset = Dataset(dataset_meta=DatasetMeta(
    'ms://AI-ModelScope/DISC-Law-SFT',
    subset_name='default',
    split='train',
    data_slice=range(40),
))
```

运行命令：

```bash
python -u main.py
```

训练过程记录：

```text
Map: 40/40
Filter: 32/32
8it [00:28,  3.51s/it]
```

40 条样本经过 `max_length=2048` 和 `delete` 策略后，实际保留 32 条，batch size 为 4，因此训练 8 个 step。

训练 metrics：

```text
step 0 loss=1.3493 grad_norm=0.284810 lr=1e-4
step 1 loss=1.3264 grad_norm=0.647434 lr=1e-4
step 2 loss=1.1378 grad_norm=0.939008 lr=1e-4
step 3 loss=0.8595 grad_norm=1.202189 lr=1e-4
step 4 loss=1.0579 grad_norm=1.211018 lr=1e-4
step 5 loss=0.7383 grad_norm=1.150941 lr=1e-4
step 6 loss=0.7682 grad_norm=1.105996 lr=1e-4
step 7 loss=0.7455 grad_norm=1.143099 lr=1e-4
```

checkpoint：

```text
twinkle://YOUR_RUN_ID/weights/twinkle-lora-0
```

### 3. 微调后推理

运行命令：

```bash
TWINKLE_MODEL_PATH=twinkle://YOUR_RUN_ID/weights/twinkle-lora-0 \
python legal_eval.py
```

#### case_summary 微调后输出摘要

微调后仍按四段结构输出，但相比原模型更短、更像法律 SFT 数据中的标准问答格式：

- 案情摘要更贴近题干，不展开太多背景。
- 争议焦点更集中。
- 分析思路按法条依据逐点展开。
- 结论更直接。

代表性输出片段：

```text
### 争议焦点
1. 甲乙之间是否存在合法有效的民间借贷关系？
2. 甲以“未签订纸质借条”为由拒绝偿还剩余借款本息，该抗辩理由是否成立？
3. 双方约定的月利率2%是否受法律保护？
```

观察：

- 小样本 LoRA 没有显著改变“是否懂法律”这件事。
- 更明显的变化是回答压缩、争议焦点归纳和法律分析格式。
- 输出末尾出现 `<|im_end|>`，说明推理 decode/stop token 还需要进一步处理。

#### legal_qa 微调后输出摘要

微调后仍然识别出不定期租赁关系，并给出承租人应对策略：

- 固定出租人要求三天搬离的沟通记录。
- 保存租金支付记录和原合同。
- 主张出租人应给予合理期限。

观察：

- 与原模型相比，这道题变化不明显。
- 原因是 base model 本身已经能很好回答该问题。
- 对体验文章而言，这反而是一个诚实的结论：少量 LoRA 在强 base model 上，更容易改变输出风格和格式，不一定能明显提升已经很强的能力项。

## 初步观察

1. Twinkle TaaS 的核心体验不是“一键训练任务”，而是“本地写训练循环，远端执行大模型训练计算”。
2. 数据准备、template、processor、batch 构造都发生在本地；`forward_backward`、`optim_step`、`save_state` 发生在远端。
3. 法律数据比官方 self-cognition 样本长，必须处理 `max_length` 和超长样本策略。
4. 40 条样本可以快速完成端到端链路，但效果更像 sanity check；若要获得更稳定的领域风格变化，建议扩大到 500 到 2000 条并跑更多 epoch。
5. Qwen3.6-27B 原模型法律问答能力已经很强，所以体验文章应避免夸大“能力提升”，重点写 Twinkle 的可编程训练流程、checkpoint 管理和 LoRA 加载验证。
6. 当前 after 输出中出现 `<|im_end|>`，后续可以通过 stop 参数或 decode 清理来优化体验文档展示。
