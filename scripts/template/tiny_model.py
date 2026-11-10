"""A small character RNN implemented with NumPy only."""  # 说明本文件用 NumPy 实现一个按字预测的小模型。

import json  # 导入 JSON 工具，用来把词表转换成可保存的文字。
from pathlib import Path  # 导入路径工具，用来处理权重文件的位置。

import numpy as np  # 导入 NumPy，后面用 np 这个简称进行数组计算。


class TinyRNN:  # 定义模型类，把训练、保存和生成方法放在一起。
    def __init__(self, chars, hidden_size=64, seed=42):  # 创建模型；默认用 64 个数字记录上下文，随机种子默认是 42。
        self.chars = list(chars)  # 把词表转成列表，方便通过编号找到对应的字。
        self.index = {c: i for i, c in enumerate(self.chars)}  # 建立“字到编号”的对应关系，方便把文字交给模型。
        rng = np.random.default_rng(seed)  # 创建随机数生成器；相同种子可以得到相同的初始参数。
        v, h = len(chars), hidden_size  # v 是词表中的字数，h 是记录上下文的数字个数。
        self.params = {  # 把模型需要学习的所有参数放进一个字典。
            "embedding": rng.normal(0, 0.1, (v, h)).astype(np.float32),  # 给每个字准备一组随机数字，用来表示这个字。
            "recurrent": (rng.normal(0, 1 / np.sqrt(h), (h, h))).astype(np.float32),  # 准备连接前后隐藏状态的权重，并按维度缩小初始数值。
            "bias": np.zeros(h, dtype=np.float32),  # 把隐藏层的偏移量设为零，之后由训练调整。
            "output": rng.normal(0, 0.1, (h, v)).astype(np.float32),  # 准备把隐藏状态转换成每个候选字分数的权重。
            "output_bias": np.zeros(v, dtype=np.float32),  # 把每个候选字的额外偏移量设为零。
        }  # 结束模型参数字典。

    def step(self, tokens, hidden):  # 定义一次计算：读入当前字的编号和之前的隐藏状态。
        p = self.params  # 给参数字典起一个短名字，方便后面的计算。
        hidden = np.tanh(p["embedding"][tokens] + hidden @ p["recurrent"] + p["bias"])  # 结合当前字和之前的状态，用 tanh 把新状态限制在 -1 到 1 之间。
        return hidden, hidden @ p["output"] + p["output_bias"]  # 返回新状态，以及下一个字在词表中各个位置的分数。

    def loss_and_grads(self, inputs, targets):  # 定义训练计算：计算预测误差，以及各参数该如何调整。
        p = self.params  # 取出模型当前的参数。
        batch, length = inputs.shape  # 读取一批有多少段文字，以及每段文字有多少个字。
        states = [np.zeros((batch, p["bias"].size), dtype=np.float32)]  # 给这一批文字各准备一个全零的初始隐藏状态。
        probs = []  # 准备列表，保存每个位置预测出的字的概率。
        loss = 0.0  # 把这一批文字的总预测误差从零开始累计。
        rows = np.arange(batch)  # 生成每段文字的行号，方便一次取出各行的目标字。
        for t in range(length):  # 按照文字顺序，逐个位置进行预测。
            hidden, logits = self.step(inputs[:, t], states[-1])  # 读取所有文字在当前位置的字，并用上一个状态预测下一个字。
            logits -= logits.max(axis=1, keepdims=True)  # 每行都减去最高分，避免后续求指数时数值太大。
            prob = np.exp(logits)  # 把分数转成正数，作为计算概率的中间结果。
            prob /= prob.sum(axis=1, keepdims=True)  # 让每一行的数值加起来等于 1，得到各个字的概率。
            loss -= np.log(prob[rows, targets[:, t]] + 1e-12).sum()  # 累计正确答案对应的负对数概率；预测越准，误差越小。
            states.append(hidden)  # 保存当前状态，反向计算时还会用到。
            probs.append(prob)  # 保存当前位置预测的概率。
        grads = {k: np.zeros_like(v) for k, v in p.items()}  # 为每个参数准备同样形状的全零数组，用来累计梯度。
        dh = np.zeros_like(states[0])  # 准备从后一个位置传回来的状态梯度，初始为零。
        for t in reversed(range(length)):  # 从最后一个字往前算，把误差传回之前的每个位置。
            dy = probs[t].copy()  # 复制预测概率，避免修改之前保存的结果。
            dy[rows, targets[:, t]] -= 1  # 在正确答案的位置减去 1，得到分数对应的梯度。
            dy /= batch * length  # 除以这一批的总字数，让梯度对应平均误差。
            grads["output"] += states[t + 1].T @ dy  # 累计输出权重的梯度；T 表示转置，@ 表示矩阵乘法。
            grads["output_bias"] += dy.sum(axis=0)  # 把各行的梯度相加，累计输出偏移量的梯度。
            dt = (dy @ p["output"].T + dh) * (1 - states[t + 1] ** 2)  # 合并当前和后续位置的影响，再算出经过 tanh 之前的梯度。
            np.add.at(grads["embedding"], inputs[:, t], dt)  # 把同一个字在不同文字中产生的梯度累加到它自己的表示上。
            grads["recurrent"] += states[t].T @ dt  # 结合上一个隐藏状态，累计循环连接权重的梯度。
            grads["bias"] += dt.sum(axis=0)  # 累计隐藏层偏移量的梯度。
            dh = dt @ p["recurrent"].T  # 算出需要继续传给上一个位置的状态梯度。
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))  # 计算所有梯度合在一起的长度，用来判断更新幅度是否过大。
        for g in grads.values():  # 逐个处理每组参数的梯度。
            g *= min(1.0, 5.0 / (norm + 1e-8))  # 如果总梯度长度超过 5，就按比例缩小，避免更新过猛。
        return float(loss / (batch * length)), grads  # 返回每个字的平均误差，以及全部参数的梯度。

    def save(self, path):  # 定义保存模型的方法。
        path = Path(path)  # 把传入的文件名转换为路径对象。
        path.parent.mkdir(parents=True, exist_ok=True)  # 创建权重文件所在的文件夹；已经存在也不会报错。
        # 词表以 JSON 文字形式放进权重文件，读取时不需要 pickle。
        with path.open("wb") as stream:  # 以二进制写入方式打开文件；同名旧文件会被覆盖。
            np.savez_compressed(stream, chars=np.array(json.dumps(self.chars)), **self.params)  # 把词表和全部权重压缩保存到同一个 npz 文件中。

    @classmethod  # 声明下面的方法属于类，可以不先创建模型就调用。
    def load(cls, path):  # 定义读取模型的方法，cls 代表这个模型类。
        with np.load(path, allow_pickle=False) as data:  # 打开权重文件，并禁止加载 pickle 对象。
            model = cls(json.loads(str(data["chars"])), data["bias"].size)  # 从文件读取词表和隐藏层大小，创建相同结构的模型。
            model.params = {key: data[key].copy() for key in model.params}  # 用文件里的权重替换随机初始权重，并复制到内存中。
        return model  # 返回已经装好权重的模型。

    def generate(self, prompt, length=100, temperature=0.7, top_k=8, seed=42):  # 定义文字生成方法，接收开头、长度、随机程度、候选数和种子。
        unknown = sorted(set(prompt) - self.index.keys())  # 找出提示词中没有收录进模型词表的字。
        if unknown:  # 如果存在模型不认识的字，就进入报错分支。
            raise ValueError("提示词包含词表之外的字符：" + "".join(unknown) + "。请换用训练文本中的字，或加入新语料重新训练。")  # 指出哪些字不在词表里，并说明如何处理。
        rng = np.random.default_rng(seed)  # 创建生成文字时使用的随机数生成器。
        hidden = np.zeros((1, self.params["bias"].size), dtype=np.float32)  # 给这一次生成准备一个全零的初始隐藏状态。
        text = prompt or self.chars[0]  # 使用用户提供的开头；开头为空时，用词表中的第一个字起步。
        for char in text:  # 按顺序读取开头中的每一个字。
            hidden, logits = self.step(np.array([self.index[char]]), hidden)  # 把开头逐字交给模型，让隐藏状态记录这些字的信息。
        for _ in range(length):  # 重复指定次数，每次生成一个新字。
            scores = logits[0].astype(np.float64) / temperature  # 用温度调整分数；温度低时更倾向于高分字，高时更随机。
            if top_k > 0 and top_k < len(scores):  # 如果限制了候选数，并且候选数小于词表大小，就筛选候选字。
                selected = np.argpartition(scores, -top_k)[-top_k:]  # 找出分数最高的 top_k 个字的编号，不要求这些编号有序。
                mask = np.full_like(scores, -np.inf)  # 先把所有位置设为负无穷，表示这些字暂时不能被选中。
                mask[selected] = scores[selected]  # 只保留入选字原来的分数。
                scores = mask  # 使用筛选之后的分数继续计算。
            probability = np.exp(scores - scores.max())  # 先减去最高分再求指数，避免数值过大。
            probability /= probability.sum()  # 把这些数值换成总和为 1 的抽样概率。
            token = rng.choice(len(self.chars), p=probability)  # 按各字的概率随机选出一个字的编号。
            text += self.chars[token]  # 把选中的字追加到生成结果后面。
            hidden, logits = self.step(np.array([token]), hidden)  # 把新字交回模型，更新状态并预测再下一个字。
        return text  # 返回开头和新生成的全部文字。
