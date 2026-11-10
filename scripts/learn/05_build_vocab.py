# 第一块：只读取训练集，建立词表。
import hashlib  # 导入哈希工具，记录建立词表所用训练文件的指纹。
import json  # 导入 JSON 工具，读取章节并保存词表。
from collections import Counter  # 导入计数器，统计验证集未见字符的出现次数。
from pathlib import Path  # 导入路径工具，定位数据文件。

ROOT = Path(__file__).resolve().parents[2]  # 找到项目根目录。
train_path = ROOT / "data" / "splits" / "train.jsonl"  # 指定唯一用于建词表的数据。
validation_path = ROOT / "data" / "splits" / "validation.jsonl"  # 验证集只用于统计未知字符。
UNK_TOKEN = "<UNK>"  # 定义表示未知字符的特殊词表项。
UNK_ID = 0  # 将编号 0 固定保留给未知字符。

with train_path.open(encoding="utf-8") as file:  # 读取训练章节，结束后关闭文件。
    train_records = [json.loads(line) for line in file]  # 将每行 JSON 还原为章节字典。

characters = set()  # 创建集合，收集训练集出现的不同字符。

for record in train_records:  # 逐章查看训练正文。
    characters.update(record["text"])  # 把每个字符加入集合，重复字符只保留一次。

tokens = [UNK_TOKEN] + sorted(characters)  # 未知符号放在首位，其余字符排序后固定编号。
char_to_id = {char: index for index, char in enumerate(tokens)}  # 建立字符或特殊符号到编号的映射。

print("训练集不同字符数：", len(characters))  # 当前应为 5546。
print("含未知符号的词表大小：", len(tokens))  # 当前应为 5547。

def encode(text):  # 定义编码函数，把文字转换成编号列表。
    return [char_to_id.get(char, UNK_ID) for char in text]  # 已知字符使用固定编号，未见字符使用 0。

def decode(token_ids):  # 定义解码函数，把编号列表转换成文字表示。
    return "".join(tokens[token_id] for token_id in token_ids)  # 逐个查表连接，0 会显示为 <UNK>。

sample = train_records[0]["text"][:20]  # 从第一条训练记录取前 20 个字符。
encoded = encode(sample)  # 把样例转换成整数列表。

assert decode(encoded) == sample, "训练样例不能还原"  # 已知字符往返转换后应保持不变。
assert all(decode(encode(record["text"])) == record["text"] for record in train_records), "训练文本不能完整还原"  # 逐章检查所有训练文字。

print("样例原文：", repr(sample))  # 显示原始字符及其中的换行。
print("样例编号：", encoded)  # 显示每个字符对应的整数。
print("解码结果：", repr(decode(encoded)))  # 显示查表还原后的文字。
print("未知字符示例：", encode("🧪"), decode(encode("🧪")))  # 当前训练集未收录这个字符，应显示 [0] 和 <UNK>。

# 第三块：统计验证集的未知字符。
unknown_counts = Counter()  # 记录每种未见字符出现了多少次。
validation_characters = 0  # 累计验证集的全部字符数。

with validation_path.open(encoding="utf-8") as file:  # 只读取验证集，不修改词表。
    for line in file:  # 一次读取一条章节记录。
        text = json.loads(line)["text"]  # 从记录中取出完整章节文字。
        validation_characters += len(text)  # 将当前章节长度加入验证集总长度。
        unknown_counts.update(char for char in text if char not in char_to_id)  # 累计所有未收录字符的出现次数。

assert validation_characters > 0, "验证集为空"  # 防止计算未知率时除以零。

unknown_total = sum(unknown_counts.values())  # 计算未见字符出现的总次数，重复出现也计数。
unknown_rate = unknown_total / validation_characters  # 未知率按出现次数计算，不是按种类数计算。

print("验证集字符总数：", validation_characters)  # 当前应为 317717。
print("未见字符种类数：", len(unknown_counts))  # 当前应为 68。
print("未见字符出现次数：", unknown_total)  # 当前应为 102。
print("未知率：", f"{unknown_rate:.4%}")  # 当前约为 0.0321%，注意百分比与小数的区别。
print("最常见的未见字符：", unknown_counts.most_common(10))  # 展示高频未见字符。

# 第四块：保存词表并重新读取。
output_dir = ROOT / "data" / "vocab"  # 指定词表文件目录。
output_dir.mkdir(parents=True, exist_ok=True)  # 创建目录，已存在时继续使用。
output_path = output_dir / "vocab.json"  # 指定词表文件名。

vocab = {"type": "character", "unk_token": UNK_TOKEN, "unk_id": UNK_ID, "tokens": tokens, "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()}  # 保存有序词表、特殊编号和训练来源指纹。

with output_path.open("w", encoding="utf-8", newline="\n") as file:  # 以 UTF-8 和 LF 覆盖写入当前词表。
    file.write(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n")  # 保存为可阅读的 JSON，并保留 tokens 列表顺序。

with output_path.open(encoding="utf-8") as file:  # 重新打开词表文件。
    loaded_vocab = json.load(file)  # 将整个 JSON 文件还原成字典。

assert loaded_vocab == vocab, "词表保存后发生变化"  # 检查词表顺序、编号配置和来源都未改变。

print("词表路径：", output_path)  # 显示生成位置。
print("训练文本往返及词表重载检查：通过")  # 所有断言均通过后显示。
