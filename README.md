# playATryModel

从零学习中文语言模型：从 NumPy 字符级 RNN，逐步实现 PyTorch RNN/GRU 和小型因果 Transformer，学习数据处理、训练、评估与优化。下一阶段目标是小于1GB的中文聊天模型。

当前最新已训练模型为168万参数Transformer，权重约6.72MB，完整训练检查点约20.5MB；它仍是续写模型，尚不具备可用聊天能力。见[Transformer课程](docs/lesson45_tiny_transformer.md)。

聊天训练从[MiniMind参考路线与预算](docs/chat_training_roadmap.md)及[对话格式练习](scripts/learn/26_chat_sft_format.py)开始，30M/64M候选尚未实现或训练。本轮[问答复盘](docs/lesson45_qa_review.md)保留真实错答与纠正过程。

以下快速体验使用早期已训练的NumPy教学示例：**39,119 个参数，权重约 144 KB，CPU 即可运行**。它主要复现和组合示例语料中的句子，不具备通用问答能力。

## 快速体验

需要 Python 3.10 或更新版本，本地已在 Python 3.13 上运行验证。

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt # 安装 NumPy 依赖。
python scripts/example/generate.py --prompt "小猫" --length 100 # 加载已有权重，续写 100 个字符。
```

重新训练示例模型：

```powershell
python scripts/example/train.py # 使用内置语料训练，覆盖默认示例权重。
```

## 文件结构

```text
model/                  模型权重，仓库保留微型示例权重
scripts/example/tiny_model.py   RNN 结构、反向传播、保存、加载和生成
scripts/example/train.py        训练入口
scripts/example/generate.py     生成入口
scripts/learn/          分块学习练习与下一课的SFT格式演示
scripts/experiments/    已实际训练的PyTorch循环模型、Transformer及预算工具
data/corpus/            四份原始小说文本
data/corpus_processed/  第二课生成的清洗文本
data/chapters/          第四课保存的 JSONL 章节记录
data/splits/            第五课固定的训练、验证、测试集及内容指纹清单
data/vocab/             第六课仅由训练集建立的字符词表
data/transformer45/     新故事、独立分组划分与Transformer词表
data/chat_demo/         两条聊天格式教学样例，不作训练或测评
docs/                   使用说明、语料和学习笔记
.github/workflows/      GitHub 自动检查
```

## 学习与文档

- [模型使用说明](docs/README.md)：训练参数、生成参数和能力边界。
- [学习笔记](docs/learning_notes.md)：学习路线、每节任务、实操结果和纠错。
- [语料与清洗结果](data/README.md)：来源线索、清洗规则和复现方法。
- [Git 仓库管理约定](docs/git_workflow.md)：提交、同步、分支和权重管理。

每次先讲解一个小目标，由学习者输入代码并运行，检查通过后记录学习笔记。现有示例不代表学习任务已经完成。

## 当前验证范围

已有本地验证包括：反向传播的数值梯度检查、权重加载、确定性采样、生成长度，以及短训练。当前示例未划分独立验证集和测试集，训练损失不能代表通用语言能力。

GitHub 自动检查覆盖已完成课程的数据复现、梯度、训练保存、固定验证、RNN/GRU与Transformer产物，以及新增SFT格式和候选参数预算。长训练结果主要通过只读检查复核，不在CI重跑。示例短训练写入临时位置。
