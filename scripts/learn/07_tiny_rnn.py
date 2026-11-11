import json  # 导入 JSON 工具，读取固定词表。
from pathlib import Path  # 导入路径工具，找到词表文件。
import numpy as np  # 导入 NumPy，创建参数矩阵并执行查表。


class TinyRNN:  # 模型包含字符表示、循环隐藏状态和候选字符分数输出层。
    def __init__(self, vocab_size, hidden_size=64, seed=42):  # 接收词表大小、向量维度和初始化种子。
        assert vocab_size > 0 and hidden_size > 0, "模型维度必须为正"  # 防止创建无效参数矩阵。
        self.vocab_size = vocab_size  # 保存词表项数，检查合法编号时会用到。
        self.hidden_size = hidden_size  # 保存向量维度，后续也作为隐藏状态维度。
        rng = np.random.default_rng(seed)  # 创建局部随机数生成器，固定初始参数。
        self.params = {  # 集中保存字符表示、循环权重和偏移参数。
            "embedding": rng.normal(0, 0.1, (vocab_size, hidden_size)).astype(np.float32),  # 保留原来的字符向量初始化。
            "recurrent": rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, hidden_size)).astype(np.float32),  # 创建共享循环权重，并按维度缩小初始标准差。
            "bias": np.zeros(hidden_size, dtype=np.float32),  # 隐藏层偏移从零开始，后续训练会调整。
            "output": rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, vocab_size)).astype(np.float32),  # 为每个候选字符准备一列输出权重。
            "output_bias": np.zeros(vocab_size, dtype=np.float32),  # 为每个候选字符准备一个偏移，初始为零。
        }  # 每次创建模型只初始化一次，所有时间位置共用这些参数。

    def embed(self, token_ids):  # 将二维字符编号转换成三维字符向量。
        token_ids = np.asarray(token_ids)  # 接受列表或数组，但不偷偷把浮点编号转换成整数。
        assert token_ids.ndim == 2, "输入应为二维编号数组"  # 本课规定输入形状为 (B,L)。
        assert np.issubdtype(token_ids.dtype, np.integer), "字符编号必须是整数"  # 防止用浮点数作为索引。
        assert ((token_ids >= 0) & (token_ids < self.vocab_size)).all(), "编号超出词表范围"  # 拒绝负数和越界编号。
        return self.params["embedding"][token_ids]  # 按每个位置的编号取行，得到 (B,L,H)。

    def forward_hidden(self, token_ids):  # 按文字顺序计算并保存每个位置的隐藏状态。
        vectors = self.embed(token_ids)  # 将 (B,L) 编号转换成 (B,L,H) 字符向量。
        batch_size, seq_length, _ = vectors.shape  # 从实际输入读取批大小和长度。
        assert batch_size > 0 and seq_length > 0, "输入批次和序列不能为空"  # 本课不处理空批次或空序列。
        hidden = np.zeros((batch_size, self.hidden_size), dtype=np.float32)  # 为每条窗口准备独立的全零初始状态。
        states = np.zeros((batch_size, seq_length, self.hidden_size), dtype=np.float32)  # 保存全部位置的隐藏状态。
        for t in range(seq_length):  # 按位置顺序递推，先算前面的字，再算后面的字。
            current = vectors[:, t, :]  # 同时取出各条窗口在当前位置的字符向量。
            hidden = np.tanh(current + hidden @ self.params["recurrent"] + self.params["bias"])  # 结合当前字符和之前状态。
            states[:, t, :] = hidden  # 将当前状态复制到对应位置保存。
        return states  # 返回 (B,L,H)，此时还没有下一字符分数。

    def forward(self, token_ids):  # 从字符编号计算每个位置的候选字符原始分数。
        states = self.forward_hidden(token_ids)  # 先得到 (B,L,H) 的上下文状态。
        logits = states @ self.params["output"] + self.params["output_bias"]  # 通过共享输出层得到 (B,L,V) 分数。
        return logits  # 返回原始分数，暂不计算概率和损失。

def main():  # 定义演示入口，被其他文件导入时不自动执行。
    root = Path(__file__).resolve().parents[2]  # 找到项目根目录。
    with (root / "data" / "vocab" / "vocab.json").open(encoding="utf-8") as file:  # 打开固定词表。
        vocab = json.load(file)  # 读取词表配置。
    tokens = vocab["tokens"]  # 保留字符编号顺序。
    char_to_id = {char: index for index, char in enumerate(tokens)}  # 根据列表位置重建编号映射。
    model = TinyRNN(len(tokens), hidden_size=64, seed=42)  # 创建带有随机表示表的模型雏形。
    sample_texts = ["春天", "春风"]  # 两条文本首字相同，方便验证查表结果。
    x = np.asarray([[char_to_id[char] for char in text] for text in sample_texts], dtype=np.int64)  # 得到形状为 (2,2) 的编号数组。
    vectors = model.embed(x)  # 查表，得到形状为 (2,2,64) 的浮点数组。
    assert vectors.shape == (x.shape[0], x.shape[1], model.hidden_size), "向量形状不正确"  # 根据实际输入形状检查输出。
    assert vectors.dtype == np.float32, "向量应为 float32"  # 编号是整数，查出的参数是浮点数。
    assert np.array_equal(vectors[0, 0], vectors[1, 0]), "同一字符查表结果不一致"  # 两个位置都是春，应取到同一行。
    assert model.embed(x[:1]).shape == (1, 2, model.hidden_size), "较小批次不能处理"  # 检查只有一条文本时也能运行。
    print("输入编号：", x)  # 显示两条文本对应的整数数组。
    print("参数矩阵形状：", model.params["embedding"].shape)  # 应为 (5547,64)。
    print("字符向量形状：", vectors.shape)  # 应为 (2,2,64)。
    print("字符向量类型：", vectors.dtype)  # 应为 float32。
    print("表示表参数数：", model.params["embedding"].size)  # 应为 355008。
    print("表示表字节数：", model.params["embedding"].nbytes)  # 应为 1420032，只统计矩阵数据区。
    print("查表、重复字符和动态批大小检查：通过")  # 所有断言通过后显示。

    states = model.forward_hidden(x)  # 对原样例“春天”和“春风”计算状态。
    assert states.shape == (x.shape[0], x.shape[1], model.hidden_size), "隐藏状态形状错误"  # 每条输入的每个位置都应有状态。
    assert states.dtype == np.float32 and np.isfinite(states).all(), "隐藏状态数值异常"  # 检查类型以及是否出现无穷或无效数。
    assert np.allclose(states[0, 0], states[1, 0]), "后面的字影响了前面的状态"  # 首字都是春，从零状态开始应得到相同结果。
    assert np.allclose(model.forward_hidden(x[:1]), states[:1]), "不同批次行发生干扰"  # 单独计算第一行，应与批量计算一致。

    context_texts = ["春天", "秋天"]  # 前字不同，后一个字都是天。
    context_x = np.asarray([[char_to_id[char] for char in text] for text in context_texts], dtype=np.int64)  # 使用固定词表编码。
    context_vectors = model.embed(context_x)  # 先获取不包含前文的字符向量。
    assert np.array_equal(context_vectors[0, 1], context_vectors[1, 1]), "相同字符查表不一致"  # 两个天应取到同一行向量。
    context_states = model.forward_hidden(context_x)  # 使用循环层加入前文状态。
    assert not np.allclose(context_states[0, 1], context_states[1, 1]), "本例前文变化未影响状态"  # 当前固定随机参数下，两个天的状态应不同。

    print("循环权重形状：", model.params["recurrent"].shape)  # 应为 (64,64)。
    print("隐藏偏移形状：", model.params["bias"].shape)  # 应为 (64,)。
    print("全部隐藏状态形状：", states.shape)  # 应为 (2,2,64)。
    print("同字不同前文的状态不同：", not np.allclose(context_states[0, 1], context_states[1, 1]))  # 当前示例应为 True。
    print("当前总参数数：", sum(param.size for param in model.params.values()))  # 加入输出层后应为 719723。
    print("递推、前文依赖和批次独立性检查：通过")  # 所有断言通过后显示。

    logits = model.forward(x)  # 对原样例计算所有位置的下一字符分数。
    assert logits.shape == (x.shape[0], x.shape[1], len(tokens)), "输出分数形状错误"  # 每条窗口每个位置都要对整个词表评分。
    assert logits.dtype == np.float32 and np.isfinite(logits).all(), "输出分数数值异常"  # 检查类型和数值是否有限。
    expected_scores = states[0, 1] @ model.params["output"] + model.params["output_bias"]  # 单独计算第一条窗口第二个位置的分数。
    assert np.allclose(logits[0, 1], expected_scores, rtol=1e-5, atol=1e-6), "输出层计算不符合公式"  # 允许 float32 累加顺序造成的微小误差。
    assert np.allclose(logits[0, 0], logits[1, 0], rtol=1e-5, atol=1e-6), "相同前文的分数不一致"  # 两条原样例首字都是春，应给出接近的分数。
    assert np.allclose(model.forward(x[:1]), logits[:1], rtol=1e-5, atol=1e-6), "单条与批量前向不一致"  # 检查不同实际批大小。
    print("输出权重形状：", model.params["output"].shape)  # 应为 (64,5547)。
    print("输出偏移形状：", model.params["output_bias"].shape)  # 应为 (5547,)。
    print("全部分数形状：", logits.shape)  # 应为 (2,2,5547)。
    print("首条末位置的前五项分数：", logits[0, -1, :5])  # 展示词表前五项的分数，不是前五名或概率。
    print("含输出层的总参数数：", sum(param.size for param in model.params.values()))  # 应为 719723。
    print("输出层形状、公式和批次独立性检查：通过")  # 所有断言通过后显示。

if __name__ == "__main__":  # 只有直接运行本文件时才启动演示。
    main()  # 调用入口，其他文件导入模型类时不会运行演示。
