import hashlib  # 导入哈希工具，检查训练文件与建立词表时是否一致。
import json  # 导入 JSON 工具，加载词表和训练章节。
from pathlib import Path  # 导入路径工具，定位数据文件。

ROOT = Path(__file__).resolve().parents[2]  # 找到项目根目录。
train_path = ROOT / "data" / "splits" / "train.jsonl"  # 本课只读取训练集。
vocab_path = ROOT / "data" / "vocab" / "vocab.json"  # 读取上一课保存的固定词表。

with vocab_path.open(encoding="utf-8") as file:  # 打开词表文件，结束后自动关闭。
    vocab = json.load(file)  # 读取整个词表字典。

tokens = vocab["tokens"]  # 保留已确定的编号顺序，不重新排序。
char_to_id = {char: index for index, char in enumerate(tokens)}  # 由固定列表重建字符到编号的映射。
unk_id = vocab["unk_id"]  # 读取未知字符编号，不在这里重新指定。

assert vocab["train_sha256"] == hashlib.sha256(train_path.read_bytes()).hexdigest(), "训练文件与词表来源不一致"  # 检查数据版本是否匹配。

with train_path.open(encoding="utf-8") as file:  # 打开训练集文件。
    train_records = [json.loads(line) for line in file]  # 逐行还原章节记录。

def iter_windows(records, seq_length, stride):  # 定义生成器，按需提供一个个输入与目标窗口。
    assert seq_length > 0 and stride > 0, "窗口长度和步长必须大于零"  # 防止零长度窗口或无效步长。
    for record in records:  # 每次处理一章，章节之间不拼接。
        token_ids = [char_to_id.get(char, unk_id) for char in record["text"]]  # 将当前章转换成编号列表。
        for start in range(0, len(token_ids) - seq_length, stride):  # 只选择后面还有 L+1 个编号的位置。
            block = token_ids[start:start + seq_length + 1]  # 多取一个编号，制作最后一个位置的答案。
            x = block[:-1]  # 去掉最后一个编号，得到 L 个输入。
            y = block[1:]  # 去掉第一个编号，得到向后错开一位的 L 个答案。
            yield {"chapter_id": record["id"], "start": start, "x": x, "y": y}  # 提供当前窗口并暂停，下次请求时继续。

preview = next(iter_windows(train_records[:1], seq_length=8, stride=8))  # 先从第一章取一个短窗口，方便逐字检查。

print("样例章节：", preview["chapter_id"], "起点：", preview["start"])  # 显示窗口来源，起点从 0 开始。
print("输入编号：", preview["x"])  # 显示长度为 8 的输入编号。
print("目标编号：", preview["y"])  # 显示向后错开一位的答案编号。
print("输入文字：", repr("".join(tokens[index] for index in preview["x"])))  # 查表显示输入文字，不删除空格。
print("目标文字：", repr("".join(tokens[index] for index in preview["y"])))  # 查表显示答案文字。

SEQ_LENGTH = 32  # 每个窗口预测 32 个位置。
STRIDE = 32  # 相邻窗口起点向后移动 32 个编号。
total_windows = 0  # 从零开始累计窗口数，不把所有窗口保存到列表。

for window in iter_windows(train_records, SEQ_LENGTH, STRIDE):  # 逐个生成训练窗口。
    assert len(window["x"]) == len(window["y"]) == SEQ_LENGTH, "窗口长度不一致"  # 检查输入和答案都足够长。
    assert window["x"][1:] == window["y"][:-1], "输入与答案没有正确错开"  # 检查重叠部分是否符合下一个字符预测。
    total_windows += 1  # 当前窗口检查通过后计数。

expected_windows = sum(max(0, (len(record["text"]) - SEQ_LENGTH - 1) // STRIDE + 1) for record in train_records)  # 用窗口数量公式独立核对。
assert total_windows == expected_windows, "窗口数量与公式不一致"  # 检查循环起点和终点是否正确。

print("训练窗口数：", total_windows)  # 当前应为 76584。
print("预测位置数：", total_windows * SEQ_LENGTH)  # 当前应为 2450688。
print("窗口长度、错位和数量检查：通过")  # 所有断言通过后显示。
