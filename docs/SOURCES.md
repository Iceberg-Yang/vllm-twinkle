# 来源、版本与权利边界

本次整理依据本地材料及本地上游 checkout，未联网确认所有外链、当前服务状态或软件发布情况。下列 URL 是来源入口，不表示对应页面已在本次逐一访问成功。

## 上游项目

| 项目 | 官方来源 | 归档依据 | 许可与处理 |
| --- | --- | --- | --- |
| nano-vLLM | [GitHub](https://github.com/GeeeekExplorer/nano-vllm) | 本地 HEAD `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，提交日期 2026-04-26；01 原安装输出同一提交 | MIT，Xingkai Yu；保留 [原许可](../licenses/nano-vllm-MIT.txt)，不复制完整仓库 |
| Twinkle | [GitHub](https://github.com/modelscope/twinkle) · [中文文档](https://modelscope.github.io/twinkle-web/zh/docs/) | 本地 HEAD `e1c28b32cc10ee75901121fe708ae03fb5fc421e`，提交日期 2026-07-04；原实验安装提交未记录 | Apache-2.0；保留 [原许可](../licenses/twinkle-Apache-2.0.txt)，示例有整理改动 |
| mini-SGLang | [GitHub](https://github.com/sgl-project/mini-sglang) | 原 Notebook 和 Dockerfile 指定 `9a91cfafe754aa85daee49998176275667eb58f2`，本次未拉取核验 | 不打包其源码；引用与安装前仍须核对该版本许可及兼容性 |
| vLLM | [GitHub](https://github.com/vllm-project/vllm) | 仅调用其公开接口；未提供新的实测或已验证版本锁 | 不打包其源码，不用本地模拟测试替代其 GPU 验证 |
| Transformers | [GitHub](https://github.com/huggingface/transformers) | 用于加载模型、模板与对照实验 | 不打包其源码；实际使用遵循对应版本许可 |

两份本地上游 checkout 没有已跟踪源码修改；nano-vLLM 存在的系统缓存文件未收入合集。`docs/source-manifest.json` 记录选中文件原路径（相对于原材料根目录）及整理前 SHA-256，便于作者追溯，不要求读者拥有那些原目录。

## 模型与数据

- [Qwen/Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)：nano-vLLM 系列示例模型；仓库不包含模型权重。
- [Qwen/Qwen3.6-27B](https://www.modelscope.cn/models/Qwen/Qwen3.6-27B)：Twinkle 历史实验配置；不是对当前 TaaS 模型支持情况的承诺。
- [AI-ModelScope/DISC-Law-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/DISC-Law-SFT)：Twinkle 示例读取的数据集；只保留标识和处理代码，不打包训练数据。

模型、数据、代码可能适用不同条款，不能从 Twinkle 的 Apache-2.0 或 nano-vLLM 的 MIT 推导模型/数据许可。本次未重新核验模型和数据卡的 revision、许可及服务准入；使用或再分发前需按实际版本确认。

## 哪些内容属于原稿，哪些是整理修改

nano-vLLM 第 01–07 篇、Notebook、benchmark 脚本、封面和截图来自原 `nano_vllm/learn/` 目录。第 01 篇选择“推理 Infra 学习实录”正文和 `main.ipynb` 为主版本；早期另一组 draft/Notebook 未混入，以免不同 workload 数据混用。

Twinkle 主文、实验观察与三个示例来自原 `twinkle/` 顶层。它们调用或改编了官方 cookbook 展示的训练/采样流程，并非本合集原创的训练框架。整理修改包括入口保护、环境读取、移除个人 checkpoint、明确 base/LoRA 模式、输出留档与轻量测试。

来源代码片段及实质性改编仍受上游许可约束，原作者署名不得删除。新增的合集索引、审阅说明和修订，不把第三方框架冒称为作者自行实现。

## 图片、原创材料与公开发布

第 01 篇的两张截图已人工查看，内容为 GPU 信息和 benchmark 数字；未发现可见 token 或个人账号。系列封面与第 03 篇封面来自原目录，生成方式、素材许可和对外发布权未确认，不将其默认标为 CC 或 MIT。

本次未统一给原创文章、封面及新增代码套用许可证。作者可在确认权属后，分别决定代码许可与文档/图片许可；工作相关稿件还应确认所在团队的发布规则。在完成这一步前，私有仓库备份是更稳妥的默认选择。

## 未收入上传候选目录的内容

| 原材料 | 原因 | 处理 |
| --- | --- | --- |
| 两个上游完整源码目录及 `.git` | 避免嵌套仓库、重复镜像与作者身份混淆 | 保留在原处，只记录上游地址/版本 |
| Twinkle `.env`、IDE 配置、数据缓存锁 | 凭据和本地状态不应发布 | 原处保留，合集只含空白 `.env.example` |
| nano-vLLM 第 01 篇早期草稿、另一版 Notebook | 主版本重复，benchmark workload 不同 | 原处保留，不混用数值 |
| 求职路线、面试问答、简历模板、个人设备规划、助手配置 | 不属于公开技术文章配套 | 原处保留 |
| GEMM 扩展规划和空结果模板 | 当前两组技术文章之外的延伸方向，未提供实测 | 原处保留，未来可独立成篇 |
| Twinkle 行动大纲 | 已由主文与实验记录覆盖，包含未核实旧安装指令 | 原处保留，主入口不推荐旧命令 |

没有删除、移动或覆盖原始目录中的任何材料，也没有创建远端仓库、提交或推送。
