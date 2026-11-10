import json  # 导入 JSON 工具，读取固定词表。
from pathlib import Path  # 导入路径工具，找到词表文件。
import numpy as np  # 导入 NumPy，创建参数矩阵并执行查表。


class TinyRNN:  # 先搭建模型类，本课只实现字符向量部分。
    def __init__(self, vocab_size, hidden_size=64, seed=42):  # 接收词表大小、向量维度和初始化种子。
        assert vocab_size > 0 and hidden_size > 0, "模型维度必须为正"  # 防止创建无效参数矩阵。
        self.vocab_size = vocab_size  # 保存词表项数，检查合法编号时会用到。
        self.hidden_size = hidden_size  # 保存向量维度，后续也作为隐藏状态维度。
        rng = np.random.default_rng(seed)  # 创建局部随机数生成器，固定初始参数。
        self.params = {  # 用字典集中保存之后需要训练的参数。
            "embedding": rng.normal(0, 0.1, (vocab_size, hidden_size)).astype(np.float32),  # 随机生成字符向量表，均值为 0，标准差为 0.1。
        }  # 当前只有这一组参数，后续再加入循环层和输出层。

    def embed(self, token_ids):  # 将二维字符编号转换成三维字符向量。
        token_ids = np.asarray(token_ids)  # 接受列表或数组，但不偷偷把浮点编号转换成整数。
        assert token_ids.ndim == 2, "输入应为二维编号数组"  # 本课规定输入形状为 (B,L)。
        assert np.issubdtype(token_ids.dtype, np.integer), "字符编号必须是整数"  # 防止用浮点数作为索引。
        assert ((token_ids >= 0) & (token_ids < self.vocab_size)).all(), "编号超出词表范围"  # 拒绝负数和越界编号。
        return self.params["embedding"][token_ids]  # 按每个位置的编号取行，得到 (B,L,H)。

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


if __name__ == "__main__":  # 只有直接运行本文件时才启动演示。
    main()  # 调用入口，其他文件导入模型类时不会运行演示。
