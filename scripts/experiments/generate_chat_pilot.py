import argparse  # 独立续写入口，不启动训练。
from pathlib import Path  # 从命令行读取权重文件位置。
from chat_model import ChatLM, torch  # 使用同一模型结构。
from tokenizers import Tokenizer  # 必须使用对应的BPE。
from prepare_chat_corpus import ROOT, DIRECTORY, sha  # 固定分词器来源与指纹。
from train_chat_pilot import generate  # 复用固定解码规则。
parser = argparse.ArgumentParser()  # 这是预训练模型的续写演示，不是SFT聊天界面。
parser.add_argument("--model", type=Path, default=ROOT / "model/lesson48/pretrain_fp16.pt")  # 默认加载本课最佳推理文件。
parser.add_argument("--prompt", default="春风")  # 用户可以指定续写开头。
parser.add_argument("--greedy", action="store_true")  # 显式选择贪心，否则固定top-k40和温度0.7。
args = parser.parse_args()  # 读取参数。
torch.set_num_threads(4)  # 控制CPU线程。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 与当前硬件匹配。
state = torch.load(args.model, map_location="cpu", weights_only=True)  # 只读取张量和普通配置。
assert state["format"] == "chat_lm48_inference_v1", "不是本课推理文件"  # 不混用旧字符模型。
path = DIRECTORY / "tokenizer/tokenizer.json"  # 使用新分词器而不是旧vocab.json。
assert sha(path.read_bytes()) == state["tokenizer_sha256"], "分词器与模型不对应"  # 词表大小相同也不能替换规则。
tokenizer = Tokenizer.from_file(str(path))  # 加载实际切分规则。
model = ChatLM(**state["model_config"]).to(device)  # 重建30M模型。
model.load_state_dict(state["model_state"])  # FP16保存的参数载入FP32主存储。
print(generate(model, tokenizer, args.prompt, args.greedy)["text"])  # 只续写，不更新参数，不宣称已能chat。
