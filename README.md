# playATryModel

从零学习一个能生成中文短文的微型语言模型：用 NumPy 实现字符级 RNN，并逐步学习数据采集、预处理、模型搭建、训练、测评与优化。

当前提供一个已经训练好的教学示例：**39,119 个参数，权重约 144 KB，CPU 即可运行**。它主要复现和组合示例语料中的句子，不具备通用问答能力。

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
scripts/learn/          学习者亲手编写的数据检查和清洗练习
data/corpus/            四份原始小说文本
data/corpus_processed/  第二课生成的清洗文本
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

GitHub 自动检查会检查 Python 语法，并运行示例生成、两步训练、第一课数据检查和第二课清洗复现检查。训练检查写入临时位置，不覆盖仓库中的示例权重。
