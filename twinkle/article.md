# ModelScope Twinkle TaaS 实践指南：用法律数据完成一次云端 LoRA 微调

本文记录一次使用 Twinkle 框架和 ModelScope 社区资源，在云端对 `Qwen/Qwen3.6-27B` 进行 LoRA 微调的实践。实验选用中文法律 SFT 数据集 `AI-ModelScope/DISC-Law-SFT`，并摘录训练前后的同题推理结果。

> 归档说明：实验记录来自 2026 年 7 月；本次整理未重新执行远端训练或推理。以下日志和输出是原记录的摘录，不是完整、独立复核的原始实验数据。服务模型、准入权限和配额可能已变化，请先核对官方说明。[环境与运行入口](README.md) · [原实验观察记录](notes/experience-log.md)。两道题的随机采样仅用于流程验证，不能证明法律能力提升或输出变化由 LoRA 导致。

实践链路如下：

```text
本地读取法律 SFT 数据
本地构造 chat trajectory
本地 template encode
本地写训练循环
远端执行 forward_backward
远端执行 optim_step
远端保存 LoRA checkpoint
远端加载 LoRA 做 sampling
本地记录训练前后输出差异
```

checkpoint 路径格式如下（所有摘录中的个人任务 ID 已脱敏，`YOUR_RUN_ID` 是占位符，并非可访问模型）：

```text
twinkle://YOUR_RUN_ID/weights/twinkle-lora-0
```

文中保留原实验的训练配置、关键代码与长度报错处理方式。复用时以配套脚本和官方当前接口为准；请使用自己的训练返回路径，不要直接运行占位符。

## 1. Twinkle 是什么

Twinkle 官方将自己定义为：

```text
Twinkle 是一个 client-server training framework。
```

这次实践中的职责划分如下。

客户端负责：

```text
数据读取
数据清洗
template
tokenize / encode
dataloader
训练循环控制
loss 调用
optimizer step 调用
checkpoint 保存调用
```

远端服务负责：

```text
加载大模型
维护 LoRA 训练状态
执行 forward / backward
执行 optimizer step
保存 checkpoint
加载 checkpoint 推理
```

本文走的是 Twinkle 的 TaaS（Training-as-a-Service）路线。本地脚本掌握训练循环，远端服务完成模型计算和训练状态管理。

训练脚本通过以下 API 组织：

```python
training_client.forward_backward(input_datum, "cross_entropy")
training_client.optim_step(types.AdamParams(learning_rate=1e-4))
training_client.save_state("twinkle-lora-0")
```

本地代码负责数据流和训练节奏，27B 模型的计算与状态留在远端。数据处理、loss 和训练步骤的改动，都可落在 Python 训练循环中。

## 2. 数据集与任务选择

官方示例里常用的是 `swift/self-cognition`。这个数据集适合展示“模型知道自己是谁”的变化，比如训练前问“你是谁”，训练后回答成用户设定的模型名字。

CodeAlpaca 的 `instruction/input/output` 字段很规整，可以直接接 `AlpacaProcessor`。Qwen3.6 本身已有较好的代码能力，少量样本训练后的差异不容易从一两道题里判断。

最后选了中文法律 SFT 数据集：

```text
ms://AI-ModelScope/DISC-Law-SFT
```

这个数据集覆盖法律问答、法律文本摘要、判决预测、法律信息抽取等任务。它的字段也很简单：

```text
input
output
```

中文法律数据提供了明确的领域语境，`input/output` 也能直接映射成对话数据，适合用于本次 LoRA SFT 实践。

本文实验用于验证 Twinkle TaaS 训练流程，模型输出不构成法律建议。

## 3. 环境准备

我的本地环境是 Mac 上的 conda 环境：

```text
conda env: llm
Python: 3.11.15
```

依赖版本：

```text
twinkle-kit: 0.4.0.dev0
tinker: 0.16.1
python-dotenv: 1.2.2
modelscope: 1.38.0
datasets: 4.8.4
```

上表是原实验环境记录，不是完整锁文件。归档时核对的本地 Twinkle 源码版本为 `e1c28b32cc10ee75901121fe708ae03fb5fc421e`，但原实验未记录安装提交，不能据此认定二者完全相同。安装方式见 [README](README.md)，不要把 `0.4.0.dev0` 当成一定存在于 PyPI 的稳定版本。

下面命令均在合集的 `twinkle/` 目录、已激活的 Python 3.11 环境执行。仅客户端在本地运行；首次数据加载仍可能下载完整数据文件。

token 放在 `twinkle/.env` 中（从 `.env.example` 复制，禁止提交）：

```text
MODELSCOPE_TOKEN=xxxx
```

代码里会读取这个 token：

```python
import os
from dotenv import load_dotenv

# 在 twinkle/ 目录执行；配套脚本用相对脚本文件位置定位 .env。
load_dotenv(".env", override=False)
token = os.environ.get("MODELSCOPE_TOKEN", "").strip()
if not token:
    raise RuntimeError("请先设置 MODELSCOPE_TOKEN")
os.environ["MODELSCOPE_API_TOKEN"] = token
```

## 4. 原始 Twinkle TaaS 示例代码做了什么

原始的 `main.py` 是一个典型的 Twinkle TaaS LoRA 训练脚本。

它的结构大概是：

```text
加载环境变量
设置 base model 和 TaaS endpoint
加载数据集
设置 template
数据预处理
数据 encode
构造 dataloader
初始化 Tinker-compatible client
创建 LoRA training client
循环执行 forward_backward
循环执行 optim_step
save_state 保存 checkpoint
```

训练循环只有这些：

```python
for epoch in range(1):
    for step, batch in tqdm(enumerate(dataloader)):
        input_datum = [input_feature_to_datum(input_feature) for input_feature in batch]

        fwdbwd_future = training_client.forward_backward(input_datum, "cross_entropy")
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()
        print(f"Training Metrics: {optim_result}")

    result = training_client.save_state(f"twinkle-lora-{epoch}").result()
    print(f"Saved checkpoint for epoch {epoch} to {result.path}")
```

循环留在本地，调用远端 client 完成一次前向、反向和参数更新。写起来接近常见的 PyTorch 训练脚本，只是计算设备换成了 TaaS。

## 5. 把示例改成中文法律 LoRA 微调

原始示例使用的是 self-cognition 数据集。这里改成 DISC-Law-SFT。

数据集加载部分改成：

```python
dataset = Dataset(dataset_meta=DatasetMeta(
    "ms://AI-ModelScope/DISC-Law-SFT",
    subset_name="default",
    split="train",
    data_slice=range(40),
))
```

法律数据集的字段是 `input/output`，我写了一个很薄的 processor，把它转成 Twinkle 的 chat trajectory。

```python
from twinkle.data_format import Message, Trajectory
from twinkle.preprocessor import Preprocessor


class LegalSFTProcessor(Preprocessor):
    """Convert DISC-Law-SFT rows into Twinkle chat trajectories."""

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        return self.map_row_to_col(rows)

    def preprocess(self, row):
        return Trajectory(messages=[
            Message(role="system", content="你是一名严谨的中文法律助手。请基于用户给出的事实和问题进行分析。"),
            Message(role="user", content=row["input"]),
            Message(role="assistant", content=row["output"]),
        ])
```

template 和 encode 部分：

```python
dataset.set_template(
    "Qwen3_5Template",
    model_id=base_model,
    max_length=2048,
    truncation_strategy="delete",
)
dataset.map(LegalSFTProcessor(), load_from_cache_file=False)
dataset.encode(batched=True, load_from_cache_file=False)
dataloader = DataLoader(dataset=dataset, batch_size=4)
```

数据预处理在本地完成。Twinkle 客户端先把原始法律样本整理成模型训练需要的 input feature，再通过 `input_feature_to_datum` 转成远端 API 接收的 datum 格式。datum 中的 token IDs、标签等训练内容仍会发送给远端，所以“本地处理”不等于数据不出本机；使用非公开数据前必须确认上传授权。

## 6. 训练前基线推理

训练前后使用相同问题推理，用于记录输出格式、措辞和回答密度的变化。`legal_eval.py` 提供了这部分验证。

它支持两种模式：

```text
--mode base：明确使用 base model，即使 .env 内配置了 checkpoint 也不加载
--mode lora：从 TWINKLE_MODEL_PATH 加载 LoRA checkpoint；未设置则报错
```

评测问题固定两道。

第一道是民间借贷：

```text
请阅读以下案情，并用“案情摘要、争议焦点、分析思路、结论”四个小标题回答：
甲向乙借款10万元，双方通过微信约定月利率2%，乙当天转账。到期后甲只归还2万元，并称双方没有签订纸质借条，所以剩余款项不应继续偿还。乙准备起诉。
```

第二道是房屋租赁：

```text
房屋租赁合同到期后，承租人继续居住并按月支付租金，出租人也一直收取。半年后出租人突然要求承租人三天内搬离。请分析双方之间的法律关系以及承租人可以如何应对。
```

base model 推理命令（需要具备服务访问权限，会发送远端请求）：

```bash
python examples/legal_eval.py --mode base --output outputs/base.json
```

原实验的采样关键步骤如下；整理版脚本通过 `--mode` 显式选择模型，并在两道题之间复用同一个 sampling client：

```python
def sample_one(service_client, template, case):
    sampling_client = service_client.create_sampling_client(
        model_path=model_path,
        base_model=base_model,
    )
    trajectory = Trajectory(messages=[
        Message(role="system", content="你是一名严谨的中文法律助手。请基于用户给出的事实和问题进行分析。"),
        Message(role="user", content=case["prompt"]),
    ])
    input_feature = template.batch_encode([trajectory], add_generation_prompt=True)[0]
    prompt = types.ModelInput.from_ints(input_feature["input_ids"].tolist())
    params = types.SamplingParams(max_tokens=512, temperature=0.2, top_p=0.9)
    result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=2).result()
    return [template.decode(seq.tokens) for seq in result.sequences]
```

原模型在第一道问题上的输出节选：

```text
### 案情摘要
甲向乙借款人民币10万元，双方通过微信聊天记录约定月利率为2%，乙随即通过银行转账方式将款项交付给甲。借款到期后，甲仅偿还本金2万元，剩余8万元本金及相应利息未予支付。甲以双方未签订纸质借条为由，主张剩余款项无需继续偿还。乙拟通过诉讼途径维护自身权益。

### 争议焦点
1. 在缺乏纸质借条的情况下，仅凭微信聊天记录和转账凭证，能否认定甲乙之间存在合法有效的民间借贷法律关系？
2. 甲以“未签订纸质借条”为由拒绝偿还剩余债务的主张是否成立？
3. 双方约定的月利率2%是否符合法律规定，乙主张的利息是否受法律保护？
```

原模型在第二道问题上也能识别不定期租赁关系：

```text
根据《中华人民共和国民法典》第七百三十四条规定：“租赁期限届满，承租人继续使用租赁物，出租人没有提出异议的，原租赁合同继续有效，但是租赁期限为不定期。”

在本案中：
* 原合同到期后，承租人继续居住并支付租金；
* 出租人继续收取租金且未提出异议；
* 双方行为表明达成了事实上的租赁关系延续。
```

从 baseline 可以看到，Qwen3.6 原模型已经能处理中文法律问答。因此，后续对比聚焦于 LoRA 对表达结构、措辞和回答密度的影响。

## 7. 启动 Twinkle TaaS 云端 LoRA 训练

连接 TaaS 的代码：

```python
init_tinker_client()
from tinker import ServiceClient

service_client = ServiceClient(
    base_url="https://www.modelscope.cn/twinkle",
    api_key=api_key,
)

training_client = service_client.create_lora_training_client(
    base_model=base_model[len("ms://"):],
    rank=16,
)
```

训练配置：

```text
Base model: Qwen/Qwen3.6-27B
Dataset: ms://AI-ModelScope/DISC-Law-SFT
LoRA rank: 16
Loss: cross_entropy
Batch size: 4
Epoch: 1
Learning rate: 1e-4
Max length: 2048
Data slice: range(40)
```

运行命令：

```bash
python -u examples/main.py
```

第一次启动时，本地会下载并缓存 DISC-Law-SFT。数据文件为 `DISC-Law-SFT-Pair.csv`，约 341MB。

## 8. 一个真实踩坑：法律数据样本更长

使用 `range(500)` 与 `max_length=2048` 时，encode 阶段遇到了超长样本。

报错信息：

```text
ValueError: Input length 2376 exceeds max_length 2048
```

报错来得很直接：DISC-Law-SFT 里的部分样本超过了 2048 token。法律数据里常见长案情、长分析和长答案，沿用官方短样本示例的长度配置，很快就会碰到这个边界。

处理方式是在 template 里加入：

```python
truncation_strategy="delete"
```

完整配置：

```python
dataset.set_template(
    "Qwen3_5Template",
    model_id=base_model,
    max_length=2048,
    truncation_strategy="delete",
)
```

这个策略会跳过超过 `max_length` 的样本。本次实践采用跳过策略，避免截断法律答案后留下不完整的监督信号。

本次实践将数据切片设为 40 条，用于验证完整训练链路。

最终编码结果：

```text
Map: 40/40
Filter: 32/32
```

40 条样本经过 `max_length=2048` 和 `delete` 后，保留 32 条。batch size 为 4，对应 8 个训练 step。

## 9. 训练日志与 checkpoint

训练过程日志：

```text
0it [00:00, ?it/s]
1it [00:03,  3.29s/it]
2it [00:06,  3.33s/it]
3it [00:10,  3.48s/it]
4it [00:13,  3.44s/it]
5it [00:16,  3.25s/it]
6it [00:20,  3.41s/it]
7it [00:24,  3.52s/it]
8it [00:28,  3.51s/it]
```

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

最后保存 checkpoint：

```text
Saved checkpoint for epoch 0 to twinkle://YOUR_RUN_ID/weights/twinkle-lora-0
```

这个 `twinkle://.../weights/...` 路径会在下一步直接传给 sampling client。

## 10. 加载 LoRA 做同题推理

微调后推理命令：

```bash
# 先将 TWINKLE_MODEL_PATH 设置为自己训练返回的真实路径。
python examples/legal_eval.py --mode lora --output outputs/lora.json
```

推理脚本会读取环境变量：

```python
model_path = os.environ.get("TWINKLE_MODEL_PATH")
```

然后加载 LoRA：

```python
sampling_client = service_client.create_sampling_client(
    model_path=model_path,
    base_model=base_model,
)
```

同一道民间借贷问题，微调后输出节选：

```text
### 案情摘要
甲向乙借款10万元，双方通过微信约定月利率2%，乙当天转账。到期后甲只归还2万元，并称双方没有签订纸质借条，所以剩余款项不应继续偿还。乙准备起诉。

### 争议焦点
1. 甲乙之间是否存在合法有效的民间借贷关系？
2. 甲以“未签订纸质借条”为由拒绝偿还剩余借款本息，该抗辩理由是否成立？
3. 双方约定的月利率2%是否受法律保护？

### 分析思路
1. 借贷关系的认定：根据《最高人民法院关于审理民间借贷案件适用法律若干问题的规定》，自然人之间的借款合同，自贷款人提供借款时成立。本案中，乙已通过微信转账实际交付了10万元借款，且双方通过微信明确约定了借款金额及利率，形成了完整的借贷合意与交付事实。因此，甲乙之间的民间借贷关系合法有效。
2. 证据效力分析：甲主张“未签订纸质借条”故无需还款，该观点缺乏法律依据。根据《中华人民共和国民事诉讼法》及《最高人民法院关于民事诉讼证据的若干规定》，电子数据（包括微信聊天记录、转账凭证等）属于法定证据种类。
```

两次输出的差异主要落在表达方式上。

训练前的回答更展开：

```text
在缺乏纸质借条的情况下，仅凭微信聊天记录和转账凭证，能否认定甲乙之间存在合法有效的民间借贷法律关系？
```

训练后的回答更收束：

```text
甲乙之间是否存在合法有效的民间借贷关系？
```

训练前的案情摘要会补充更多背景表达：

```text
乙随即通过银行转账方式将款项交付给甲。借款到期后，甲仅偿还本金2万元，剩余8万元本金及相应利息未予支付。
```

训练后的案情摘要更贴近原始案情：

```text
甲向乙借款10万元，双方通过微信约定月利率2%，乙当天转账。到期后甲只归还2万元...
```

在已摘录的这组输出中，微调后回答较短，争议焦点更集中，案情摘要也更贴近题干。但输入切片是 40 条、实际训练为过滤后的 32 条；比较仅有两道题，温度为 0.2，未做多随机种子重复实验，512 token 上限还可能截断回答。因此这里只记录观察，不把差异归因于 LoRA，也不据此判断法律能力提升。

第二道房屋租赁问题的变化没有第一道明显。原模型已经能识别不定期租赁，也能给出承租人保存证据、主张合理通知期等建议。微调后仍然保持这些要点，并继续使用较规范的法律分析结构。

部分 LoRA 输出末尾出现了 `<|im_end|>`。展示场景可通过 stop 参数或 decode 清理处理这个结束标记。

## 11. 本次实践中的 Twinkle 行为特征

### 训练循环留在本地

数据怎么处理、batch 怎么构造、何时调用 forward/backward、何时更新参数、何时保存 checkpoint，都由 Python 脚本决定。实验时要插日志、调整数据流或替换一段训练逻辑，位置比较明确。

### 27B 计算留在远端

本地 Mac 没有加载 27B 权重，也没有准备 GPU。forward、backward 和 optimizer step 都由 TaaS 执行，本地看到的是请求、日志和结果。

### checkpoint 能直接进入下一步

训练结束后拿到的是一个明确的路径：

```text
twinkle://YOUR_RUN_ID/weights/twinkle-lora-0
```

这个路径可直接放进后续推理流程，训练和验证之间少了一次手动导出、下载或转换。

### 数据准备会直接影响调试体验

官方 self-cognition 示例很短，换成法律数据后第一处报错就是 `max_length`。字段格式、template、长度分布和截断策略，都会很快从“数据准备”变成实际训练的一部分。

## 12. 本次实践中 Twinkle 与 ms-swift 的分工

ms-swift 提供了成熟的训练工具链，适合通过配置和 CLI 快速跑 SFT、DPO、GRPO、部署和评测。

Twinkle 提供可编程训练框架和 TaaS 基础设施，把训练循环拆成 API，客户端能保留更多控制权。

在本次实践中：

```text
数据集来自 ModelScope 社区
数据处理发生在本地 Twinkle 代码中
训练循环由本地 Python 控制
大模型训练计算由 Twinkle TaaS 执行
LoRA checkpoint 由 Twinkle TaaS 保存
推理验证通过 sampling client 完成
```

标准 SFT 任务追求尽快跑出结果时，ms-swift 的配置和现成配方会更省事。研究训练循环、接入自定义数据流，或想管理远端训练与 checkpoint 生命周期时，Twinkle 提供的接口空间更大。

## 13. 实践结果

本次实践完成了 ModelScope Twinkle TaaS 上的一次云端 LoRA 微调。

最后得到的流程是：

```text
base model 推理基线
DISC-Law-SFT 数据准备
LegalSFTProcessor 转换样本格式
Qwen3_5Template 编码
create_lora_training_client
forward_backward
optim_step
save_state
create_sampling_client
LoRA 推理验证
```

这次记录支持的结论是：在原实验环境下，完成了数据处理、32 条有效样本的云端 LoRA 训练、checkpoint 保存与同题采样链路。输出摘录存在措辞和结构差异，但尚不足以证明风格发生稳定改变，更不足以证明法律能力提升。若要研究效果，应补充独立测试集、训练集去重检查、统一采样设定、多次重复实验和专家评分。

本次验证的路径为：

```text
用一个法律领域 LoRA 示例完成 Twinkle TaaS 云端训练、checkpoint 保存和推理验证。
```

这次使用中，Twinkle 将“训练循环控制”和“大模型训练计算”分开：本地代码表达训练逻辑，远端服务承担模型状态和计算负载。这套链路可继续用于数据实验、算法实验和训练平台接入。
