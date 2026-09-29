# ModelScope 技术实践笔记

围绕 ModelScope 生态整理的技术文章、Notebook 和实验脚本。目前包含 **nano-vLLM 推理 Infra 学习系列**与 **Twinkle TaaS 云端 LoRA 实践**。

这是个人学习与实践材料合集，不是 nano-vLLM / Twinkle 的官方仓库或源码镜像。文章中的历史观察、待验证方案和已完成的本地检查分别标记，不把草案作为实测结论。

## 从哪里开始

| 主题 | 内容 | 当前材料状态 | 入口 |
| --- | --- | --- | --- |
| nano-vLLM 01 | 从跑通模型到 benchmark | 有历史截图与输出，修订版待 GPU 复跑 | [文章](nano-vllm/lessons/01/article.md) · [Notebook](nano-vllm/lessons/01/main.ipynb) |
| nano-vLLM 02 | nano-vLLM 与 Transformers | 历史手记数据，缺原始运行证据 | [文章](nano-vllm/lessons/02/article.md) · [Notebook](nano-vllm/lessons/02/main.ipynb) |
| nano-vLLM 03 | 一次 generate 的调用链 | 源码学习稿；代码示例待 GPU 验证 | [文章](nano-vllm/lessons/03/article.md) · [Notebook](nano-vllm/lessons/03/main.ipynb) |
| nano-vLLM 04 | KV Cache Block 分配、复用与回收 | 源码学习稿；实验待验证 | [文章](nano-vllm/lessons/04/article.md) · [Notebook](nano-vllm/lessons/04/main.ipynb) |
| nano-vLLM 05 | 从 nano-vLLM 到 mini-SGLang | 流程草稿，未保存执行输出 | [文章](nano-vllm/lessons/05/article.md) · [Notebook](nano-vllm/lessons/05/main.ipynb) |
| nano-vLLM 06 | 推理服务与 ModelScope 创空间 | 部署草案，未完成容器及云端验证 | [文章](nano-vllm/lessons/06/article.md) · [Notebook](nano-vllm/lessons/06/main.ipynb) |
| nano-vLLM 07 | 四种推理路径的横评方案 | 方案与脚本，无完整实测数据或排名 | [文章](nano-vllm/lessons/07/article.md) · [Notebook](nano-vllm/lessons/07/main.ipynb) |
| Twinkle TaaS | 中文法律数据 LoRA 实践 | 有历史训练日志摘录与输出观察，非能力评测 | [文章](twinkle/article.md) · [运行说明](twinkle/README.md) |

文章在 ModelScope 的公开发布状态与 URL 尚未确认；本表只描述仓库内材料状态，不表示“已发表”或“全部跑通”。

## 目录

```text
nano-vllm/
  README.md                   系列导航、环境与阅读顺序
  lessons/01..07/              每篇 article.md + main.ipynb
  lessons/01/assets/           历史实验截图
  lessons/pictures/            原系列封面素材
  lessons/06/studio-demo/      明确标记的部署草案
  experiments/benchmarks/     离线、服务压测与汇总脚本
twinkle/
  README.md                   环境、配置与执行方法
  article.md                  主文章
  examples/                   训练、推理、同题对比及轻量测试
  notes/experience-log.md      脱敏后的历史观察记录
  .env.example                仅空值和公开配置
  requirements.txt            源码版本参考与客户端依赖
licenses/                     已携带的第三方许可证
scripts/check_repository.py   无需 GPU 的发布前静态检查
docs/REVIEW.md                 审阅结论、修订与待验证事项
docs/SOURCES.md                上游来源、版本和权利边界
docs/source-manifest.json      原材料相对路径与整理前 SHA-256
```

## 使用方式

只阅读文章不需要安装任何依赖。需要执行时，分别遵循 [nano-vLLM 说明](nano-vllm/README.md) 和 [Twinkle 说明](twinkle/README.md)，不要把四个推理框架混装到同一环境。

在合集根目录运行不触发 GPU 或远端服务的检查：

```bash
python3 scripts/check_repository.py
python3 -m unittest discover -s twinkle/examples -p 'test_*.py'
python3 -m unittest discover -s nano-vllm/experiments/benchmarks -p 'test_*.py'
```

本地通过的语法、链接与模拟测试不等于 GPU benchmark、Docker 构建或 TaaS 端到端通过。详细边界见 [审阅报告](docs/REVIEW.md)。Notebook 默认不保存运行输出；公开实验结果前请另行整理环境、完整日志、输出与统计方法。

## 上传与权利说明

这个目录是上传候选内容。原始目录中的 `.env`、上游 `.git`、个人求职规划、旧稿和训练产物未收入。请只上传本目录，不要将它与原工作区一起提交。

当前未为原创文章、图片与新增代码统一指定开源许可；这不妨碍权利人备份自己的材料，但不等于授予他人任意再利用的权利。公开前请确认工作相关内容和封面素材的发布权，再自行选择适合的代码/文档许可。已引用或改编的第三方材料仍遵循原许可，详见 [来源与许可边界](docs/SOURCES.md)。

已初始化本地 `main` 分支；没有远端仓库配置，也没有执行 Git 提交或推送。实际上传前建议先使用私有仓库备份，确认发布权限与材料状态后再决定是否公开。
