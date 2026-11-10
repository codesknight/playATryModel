import numpy as np  # 导入 NumPy，把多个窗口转换成规则的整数数组。
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


def pack_batch(windows):  # 把已经收集好的非空窗口列表转换成一个批次。
    x = np.asarray([window["x"] for window in windows], dtype=np.int64)  # 每条输入作为一行，组成二维整数数组。
    y = np.asarray([window["y"] for window in windows], dtype=np.int64)  # 按同样行序组成目标数组。
    chapter_ids = [window["chapter_id"] for window in windows]  # 保留每行对应的章节标识。
    starts = [window["start"] for window in windows]  # 保留每行在章节中的起点。
    return {"x": x, "y": y, "chapter_ids": chapter_ids, "starts": starts}  # 返回数组及其来源信息。

def iter_batches(windows, batch_size):  # 将窗口生成器按指定数量分组，逐批提供数据。
    assert batch_size > 0, "批大小必须大于零"  # 防止出现无效批大小。
    buffer = []  # 暂存尚未凑满一批的窗口。
    for window in windows:  # 逐个请求上一课的窗口。
        buffer.append(window)  # 把当前窗口加入临时列表。
        if len(buffer) == batch_size:  # 窗口数达到一批时就打包。
            yield pack_batch(buffer)  # 提供一批数据并暂停，等待下一次请求。
            buffer = []  # 恢复执行后清空缓冲，准备下一批。
    if buffer:  # 遍历结束后，检查是否还有不足一批的窗口。
        yield pack_batch(buffer)  # 保留最后的小批次，不丢弃剩余样本。

preview_batch = next(iter_batches(iter_windows(train_records, 8, 8), 4))  # 先取 4 条长度为 8 的窗口，便于观察。

print("预览 X 形状：", preview_batch["x"].shape)  # 应为 (4, 8)，分别代表窗口数和每条长度。
print("预览 Y 形状：", preview_batch["y"].shape)  # 目标形状同样为 (4, 8)。
print("预览数据类型：", preview_batch["x"].dtype)  # 应为 int64。
print("预览 X 第一行：", preview_batch["x"][0])  # 取第一条输入窗口。
print("预览 Y 第一行：", preview_batch["y"][0])  # 取与它对应的目标窗口。

BATCH_SIZE = 16  # 正式统计时每批最多取 16 条窗口。
batch_count = 0  # 累计批次数。
sample_count = 0  # 累计真正处理的窗口数。
last_batch_size = 0  # 记录最后一批的实际窗口数。

for batch in iter_batches(iter_windows(train_records, SEQ_LENGTH, STRIDE), BATCH_SIZE):  # 顺序遍历所有训练批次。
    x, y = batch["x"], batch["y"]  # 取出输入与目标数组。
    assert x.ndim == 2 and x.shape == y.shape, "数组形状不正确"  # 两者必须是形状一致的二维数组。
    assert 0 < x.shape[0] <= BATCH_SIZE and x.shape[1] == SEQ_LENGTH, "批次尺寸不正确"  # 最后一批可以不足 16 条。
    assert x.dtype == y.dtype == np.int64, "编号必须使用 int64"  # 检查编号没有误用浮点数。
    assert np.array_equal(x[:, 1:], y[:, :-1]), "批次中的目标错位不正确"  # 逐行检查错位关系。
    assert len(batch["chapter_ids"]) == len(batch["starts"]) == x.shape[0], "来源信息与行数不一致"  # 每行都要有对应来源。
    if batch_count == 0:  # 用第一个完整批次展示数组占用。
        print("首批 X 和 Y 字节数：", x.nbytes + y.nbytes)  # 应为 8192，只计算两个数组的数据区。
    batch_count += 1  # 当前批检查通过后计数。
    sample_count += x.shape[0]  # 按实际行数计数，不假设每批都满。
    last_batch_size = x.shape[0]  # 更新最后一批大小。

assert sample_count == total_windows, "批处理遗漏或重复样本"  # 核对处理的总窗口数。
assert batch_count == (total_windows + BATCH_SIZE - 1) // BATCH_SIZE, "批次数不正确"  # 用向上取整公式核对批数。

print("批次数：", batch_count)  # 应为 4787。
print("处理窗口数：", sample_count)  # 应为 76584。
print("最后一批窗口数：", last_batch_size)  # 应为 8。
print("批次形状、错位和覆盖数量检查：通过")  # 所有断言通过后显示。
