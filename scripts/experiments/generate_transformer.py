import argparse  # 接收模型文件和续写开头。
from pathlib import Path  # 读取命令行模型路径。
import tiny_transformer as lm  # 复用同一网络与生成规则。
from tiny_transformer import ROOT, torch  # 使用一致的项目根目录和PyTorch。
parser = argparse.ArgumentParser()  # 独立生成入口不会启动训练。
parser.add_argument("--model", type=Path, default=ROOT / "model/lesson45/lesson45_best.pt")  # 默认加载本课最佳文件。
parser.add_argument("--prompt", default="从前，有一个小男孩")  # 也可以替换为自己的开头。
parser.add_argument("--length", type=int, default=60)  # 最多生成的新编号数。
parser.add_argument("--seed", type=int, default=42)  # 固定采样随机种子。
parser.add_argument("--greedy", action="store_true")  # 可单独观察贪心重复。
args = parser.parse_args()  # 读取用户参数。
assert args.length > 0, "生成长度必须为正"  # 不接受无意义的生成预算。
torch.set_num_threads(4)  # 限制CPU线程。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用当前可用设备。
model, _, state = lm.restore(args.model, device)  # 模型参数和词表来自同一文件。
print(lm.generate(model, state["vocab"], args.prompt, args.length, args.seed, args.greedy)["text"])  # 只生成，不更新参数。
