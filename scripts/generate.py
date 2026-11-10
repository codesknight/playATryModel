"""Generate text using a trained TinyRNN checkpoint."""  # 说明本文件读取训练好的小模型并生成文字。

import argparse  # 导入命令行参数工具，用来读取生成设置。
from pathlib import Path  # 导入路径工具，用来定位模型文件。
from tiny_model import TinyRNN  # 导入我们自己实现的小模型。

ROOT = Path(__file__).resolve().parents[1]  # 根据脚本位置找到项目根目录。


def main():  # 定义生成文字的主流程。
    parser = argparse.ArgumentParser(description=__doc__)  # 创建参数解析器，并用文件开头的说明作为帮助文字。
    parser.add_argument("--model", type=Path, default=ROOT / "model" / "tiny_rnn.npz")  # 设置权重文件位置，默认读取 model 中已经训练好的模型。
    parser.add_argument("--prompt", default="春天")  # 设置生成开头，默认从“春天”两个字开始。
    parser.add_argument("--length", type=int, default=100, help="新增字符数")  # 设置新增字符数，默认在开头后再生成 100 个字符。
    parser.add_argument("--temperature", type=float, default=0.7)  # 设置采样温度，较低时更保守，较高时更随机。
    parser.add_argument("--top-k", type=int, default=8, help="0 表示不限制候选字符")  # 设置每一步最多保留多少个候选字，0 表示不限制。
    parser.add_argument("--seed", type=int, default=42)  # 设置随机种子，方便复现同一次生成结果。
    args = parser.parse_args()  # 读取用户在命令行中传入的设置。
    if args.length < 0 or args.temperature <= 0 or args.top_k < 0:  # 检查字符数和候选数不能为负，温度必须大于零。
        parser.error("length、top-k 不能为负，temperature 必须大于零")  # 参数不合法时显示原因，并结束程序。
    if not args.model.is_file():  # 检查权重文件是否存在，而且确实是一个文件。
        parser.error("找不到模型，请先运行 python scripts/train.py")  # 找不到权重时，提示用户先运行训练脚本。
    try:  # 尝试读取模型并生成文字，同时准备处理词表不匹配等错误。
        print(TinyRNN.load(args.model).generate(args.prompt, args.length, args.temperature, args.top_k, args.seed))  # 加载权重，按用户设置续写，再把完整结果打印出来。
    except ValueError as error:  # 如果加载或生成时抛出 ValueError，就接收具体错误信息。
        parser.error(str(error))  # 用命令行错误提示显示原因，并结束程序。


if __name__ == "__main__":  # 只有直接运行这个文件时才开始生成，被导入时不执行。
    main()  # 调用生成主流程。
