import argparse  # 默认只检查续训来源，不启动训练。
import hashlib  # 核对完整状态与来源报告。
import json  # 读取语料和分词器版本。
from pathlib import Path  # 定位项目与可选检查点。
import numpy as np  # 复现游标所指向的下一批排列。
import torch  # 在CPU读取完整模型和优化器张量。
ROOT = Path(__file__).resolve().parents[2]  # 项目根目录。
parser = argparse.ArgumentParser()  # 可选择检查下一课产生的新状态。
parser.add_argument("--checkpoint", type=Path, default=ROOT / "model/lesson48/last_train.pt")  # 本课从第1020步开始。
args = parser.parse_args()  # 只读检查，不重建网络或占用显卡。
assert args.checkpoint.is_file(), "缺少完整状态，请从对应Release下载last_train.pt，不能使用推理FP16文件"  # 克隆仓库不会自动包含360MB附件。
state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)  # 只载入张量和普通配置。
assert state["format"] == "chat_lm48_train_v1", "需要完整训练状态"  # 不把无优化器的推理文件当断点。
directory = ROOT / "data/chat48"  # 所有数据游标都依赖这一冻结版本。
manifest = json.loads((directory / "tokenizer/manifest.json").read_text(encoding="utf-8"))  # 保存了编码产物的指纹。
assert hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest() == state["corpus_manifest_sha256"], "数据版本改变"  # 游标不能用于另一份语料。
assert hashlib.sha256((directory / "tokenizer/tokenizer.json").read_bytes()).hexdigest() == state["tokenizer_sha256"], "分词器版本改变"  # 同词表大小也不能互换。
stream_path = directory / "tokenizer/pretrain_train.npy"  # 训练窗口来自固定编号流。
assert hashlib.sha256(stream_path.read_bytes()).hexdigest() == manifest["files"]["pretrain_train.npy"], "编号流版本改变"  # 划分版本相同也要检查编码产物。
stream = np.load(stream_path, mmap_mode="r", allow_pickle=False)  # 不修改任何编号。
windows = int(np.ceil((len(stream) - 1) / state["model_config"]["context"]))  # 当前完整训练流的窗口数。
epoch, offset = state["cursor"]["epoch"], state["cursor"]["offset"]  # 这里轮号和偏移都从0计数。
assert 0 <= offset < windows, "下一批游标越界"  # 轮末应已推进到下一轮起点。
order = np.random.default_rng(state["data_seed"] + epoch).permutation(windows)  # 重建该轮的相同排列。
next_ids, peek_epoch, peek_offset = [], epoch, offset  # 只预览，不修改检查点中的游标。
for _ in range(state["gradient_accumulation"]):  # 即使下一组跨轮也应展示完整8个窗口。
    preview_order = np.random.default_rng(state["data_seed"] + peek_epoch).permutation(windows)  # 按各自轮号重建排列。
    next_ids.append(int(preview_order[peek_offset]))  # 这是窗口编号，不是词表中的token编号。
    peek_offset += 1  # 推进临时预览位置。
    if peek_offset == windows:  # 预览用完一轮时切换下一轮。
        peek_epoch, peek_offset = peek_epoch + 1, 0  # 不重复尾窗。
adam_steps = sorted({int(item["step"].item()) for item in state["optimizer_state"]["state"].values()})  # 检查所有参数的优化器累计步数。
assert adam_steps == [state["step"]], "模型与AdamW不在同一步"  # 新建空优化器不能通过这项检查。
print("完整状态累计更新：", state["step"])  # 不是当前轮的批次序号。
print("AdamW各参数累计步数：", adam_steps)  # 预期来源为[1020]。
print("已处理有效目标：", state["seen_positions"])  # 不能只按更新数乘固定长度估计。
print("下一微批所属轮号（从0计）：", epoch, "下一微批偏移（从0计）：", offset)  # 完成过的批次不再重读。
print("下一组窗口编号：", next_ids)  # 这是打乱后的编号，不是连续字符编号。
print("梯度累积微批数：", state["gradient_accumulation"])  # 八个微批才增加一次step。
print("GradScaler缩放比例：", state["scaler_state"]["scale"])  # 缩放历史也必须恢复。
print("学习率：", [group["lr"] for group in state["optimizer_state"]["param_groups"]])  # 恢复优化器后使用原参数组。
print("完整状态与数据游标检查：通过；本脚本没有执行下一次参数更新。")  # 检查不会把1020变成1021。
