# 微型中文文本生成模型

这是从零实现并训练的字符级 RNN 神经网络，仅依赖 NumPy，CPU 即可运行。`model/` 存放权重，`scripts/` 存放代码，`docs/` 存放说明和示例语料。

默认隐藏层 64 维，使用内置的小型中文语料，权重远小于 1 GB。实际参数量、文件大小以训练输出为准。1 GB 限制指保存的模型权重，不包括 Python 环境、依赖和训练内存。

## 快速运行

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
python scripts/generate.py --prompt "春天" --length 100
```

如果没有权重，先训练：

```powershell
python scripts/train.py
```

生成更多不同的文字：

```powershell
python scripts/generate.py --prompt "小猫" --length 150 --temperature 0.8 --seed 7
python scripts/generate.py --prompt "夜晚" --length 100 --temperature 0.5
```

`length` 是新增字符数。温度越低通常越保守，越高越随机；`top-k` 默认只从概率最高的 8 个字符中采样，设为 0 可取消限制。相同权重、提示词和随机种子通常会得到相同结果。

## 换成自己的语料

准备 UTF-8 纯文本，长度大于 32 字，推荐使用数千字以上、风格一致的文本，然后执行：

```powershell
python scripts/train.py --data docs/my_corpus.txt --steps 3000 --output model/my_rnn.npz
python scripts/generate.py --model model/my_rnn.npz --prompt "你的开头" --length 100
```

提示词中的每个字符都必须出现在相应的训练语料里；否则脚本会提示缺少哪些字符。重新训练会建立新词表，并从随机参数开始训练，不是继续微调。默认训练会覆盖 `model/tiny_rnn.npz`；想保留原模型请指定其他 `--output`。

## 原理和能力边界

模型把上一个字和隐藏状态作为输入，学习预测下一个字，用反向传播和 Adam 更新参数。生成时逐字采样。权重文件同时包含词表，加载时不使用 pickle。

这是教学用的微型语言模型，不是问答助手。示例语料非常少，训练后的输出主要是学过句子的复现和组合，可能重复、不连贯或出现错字搭配。训练损失只反映对训练语料的拟合，没有独立评测集，不能代表通用语言能力。它没有预训练知识，也不能可靠回答问题。想增加表达范围，需要更丰富的语料并重新训练；增大模型本身不能替代数据。

可通过 `python scripts/train.py --help` 和 `python scripts/generate.py --help` 查看参数。
