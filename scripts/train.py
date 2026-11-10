"""Train from a UTF-8 plain text file and save a small checkpoint."""  # 说明本文件从 UTF-8 文本训练模型，并保存权重。

import os  # 导入系统环境工具，用来设置计算库的线程数。

# 这些矩阵很小，默认使用单线程，避免线程调度反而拖慢计算。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 如果用户没有设置，就让 OpenBLAS 使用一个线程，减少小矩阵计算的额外开销。
os.environ.setdefault("OMP_NUM_THREADS", "1")  # 如果用户没有设置，就把 OpenMP 默认线程数设为 1。
os.environ.setdefault("MKL_NUM_THREADS", "1")  # 如果用户没有设置，就把 MKL 默认线程数设为 1。

import argparse  # 导入命令行参数工具，让运行脚本时可以指定训练设置。
from pathlib import Path  # 导入路径工具，用来定位语料和权重文件。
import numpy as np  # 导入 NumPy，用来处理数组和训练计算。
from tiny_model import TinyRNN  # 导入我们自己实现的小模型。

ROOT = Path(__file__).resolve().parents[1]  # 根据脚本位置找到项目根目录，避免依赖终端当前所在目录。


def main():  # 定义训练脚本的主流程。
    parser = argparse.ArgumentParser(description=__doc__)  # 创建命令行参数解析器，使用文件开头的说明作为帮助文字。
    parser.add_argument("--data", type=Path, default=ROOT / "docs" / "sample_corpus.txt")  # 设置语料文件参数，默认读取 docs 中的示例文本。
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "tiny_rnn.npz")  # 设置权重保存位置，默认保存到 model 文件夹。
    parser.add_argument("--steps", type=int, default=1500)  # 设置训练更新次数，默认更新 1500 次。
    parser.add_argument("--hidden-size", type=int, default=64)  # 设置隐藏层大小，默认用 64 个数字记录上下文。
    parser.add_argument("--seq-length", type=int, default=32)  # 设置每个训练片段的长度，默认一次学习 32 个字的预测。
    parser.add_argument("--batch-size", type=int, default=8)  # 设置每批训练片段数量，默认同时处理 8 段。
    parser.add_argument("--learning-rate", type=float, default=0.003)  # 设置学习率，控制每次调整参数的幅度。
    parser.add_argument("--seed", type=int, default=42)  # 设置随机种子，让相同配置下的训练更容易复现。
    args = parser.parse_args()  # 读取用户在命令行中提供的参数。
    if min(args.steps, args.hidden_size, args.seq_length, args.batch_size) < 1 or args.learning_rate <= 0:  # 检查训练次数、隐藏层大小、片段长度、批大小和学习率是否为正。
        parser.error("训练步数、维度、批大小和学习率必须大于零")  # 发现无效数值时，显示原因并结束程序。
    text = args.data.read_text(encoding="utf-8-sig")  # 以 UTF-8 读取语料，同时兼容带 BOM 标记的文件。
    if len(text) <= args.seq_length:  # 检查语料是否足够长，能否组成输入和它后面的目标字。
        parser.error("训练文本长度必须大于 seq-length")  # 语料太短时，提示用户增加文本或减小片段长度。
    chars = sorted(set(text))  # 去除重复字符并排序，得到固定顺序的词表。
    # 下面按每个参数 4 字节、每个词表字符 4 字节，粗略估算大小。
    v, h = len(chars), args.hidden_size  # 记录词表大小 v 和隐藏层大小 h。
    estimated_bytes = 4 * (2 * v * h + h * h + h + v + v)  # 粗略估算参数和词表占用的字节数，不包含文件格式开销。
    if estimated_bytes >= 1_000_000_000:  # 检查估算大小是否已经达到 10 亿字节。
        parser.error("当前配置的权重预计超过 1 GB，请减小 hidden-size 或词表")  # 模型太大时停止训练，并提示缩小配置。
    model = TinyRNN(chars, h, args.seed)  # 用词表、隐藏层大小和种子创建一个新模型。
    data = np.array([model.index[c] for c in text], dtype=np.int64)  # 把整篇语料中的每个字替换成它在词表里的编号。
    rng = np.random.default_rng(args.seed)  # 创建随机数生成器，用来随机选择训练片段。
    moments = {k: np.zeros_like(v) for k, v in model.params.items()}  # 给每个参数准备一个数组，记录梯度的滑动平均。
    variances = {k: np.zeros_like(v) for k, v in model.params.items()}  # 给每个参数准备一个数组，记录梯度平方的滑动平均。
    print(f"字符数={len(text)}，词表={v}，参数={sum(p.size for p in model.params.values()):,}", flush=True)  # 打印语料字数、词表大小和模型参数总数，并立即显示。
    for step in range(1, args.steps + 1):  # 从第 1 次到指定次数，循环更新模型。
        starts = rng.integers(0, len(data) - args.seq_length, size=args.batch_size)  # 为这一批片段随机选择起点，并确保后面还有足够的字。
        positions = starts[:, None] + np.arange(args.seq_length + 1)  # 从每个起点向后取位置编号，多取一个字作为最后的预测目标。
        batch = data[positions]  # 按位置从整篇语料中取出这一批文字编号。
        loss, grads = model.loss_and_grads(batch[:, :-1], batch[:, 1:])  # 用前面的字预测后面的字，得到误差和参数梯度。
        if not np.isfinite(loss):  # 检查误差是否出现无穷大或无效数值。
            raise RuntimeError("训练出现非有限损失，请降低学习率")  # 数值异常时停止训练，并提示降低学习率。
        for key, param in model.params.items():  # 逐个更新模型中的每一组参数。
            moments[key] *= 0.9  # 把之前累计的梯度平均值保留九成。
            moments[key] += 0.1 * grads[key]  # 加入当前梯度的一成，得到新的滑动平均。
            variances[key] *= 0.999  # 把之前累计的梯度平方平均值保留 99.9%。
            variances[key] += 0.001 * grads[key] ** 2  # 加入当前梯度平方的 0.1%，记录最近的梯度幅度。
            param -= args.learning_rate * (moments[key] / (1 - 0.9**step)) / (  # 按 Adam 方法调整参数；先修正梯度平均值在训练初期偏小的问题。
                np.sqrt(variances[key] / (1 - 0.999**step)) + 1e-8  # 再除以修正后的梯度幅度；加上很小的数以避免除以零。
            )  # 结束本次参数更新公式。
        if step == 1 or step % 100 == 0 or step == args.steps:  # 在第 1 次、每 100 次和最后一次更新时输出进度。
            print(f"step {step:5d}/{args.steps}  loss={loss:.4f}", flush=True)  # 显示当前步数和平均预测误差，误差越低表示越贴近训练文本。
    model.save(args.output)  # 训练完成后，把词表和权重写入指定文件。
    print(f"已保存：{args.output}（{args.output.stat().st_size / 1024:.1f} KB）")  # 打印保存路径，以及实际权重文件的大小，单位为 KB。


if __name__ == "__main__":  # 只有直接运行这个脚本时才启动训练，被其他文件导入时不执行。
    main()  # 调用训练主流程。
