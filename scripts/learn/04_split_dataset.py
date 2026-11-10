import hashlib  # 导入哈希工具，为输入和输出文件计算内容指纹。
import json  # 导入 JSON 工具，读取和保存章节记录。
import random  # 导入随机工具，按固定种子打乱章节顺序。
from pathlib import Path  # 导入路径工具，定位数据文件。

ROOT = Path(__file__).resolve().parents[2]  # 找到项目根目录。
source_path = ROOT / "data" / "chapters" / "chapters.jsonl"  # 指定上一课生成的章节文件。
SEED = 42  # 固定随机种子，42 本身没有特殊含义。

with source_path.open(encoding="utf-8") as file:  # 打开章节文件，读取后自动关闭。
    records = [json.loads(line) for line in file]  # 将每行还原成字典，组成记录列表。

assert len({record["id"] for record in records}) == len(records), "输入 ID 重复"  # 先检查每条记录的身份唯一。

groups = {}  # 创建字典，按书名分别保存章节。
for record in records:  # 依次处理每条章节记录。
    groups.setdefault(record["book"], []).append(record)  # 书名尚不存在时创建列表，再加入当前章节。

rng = random.Random(SEED)  # 创建独立的随机数生成器，不依赖全局随机状态。
splits = {"train": [], "validation": [], "test": []}  # 为三个集合分别准备列表。

for book_name in sorted(groups):  # 固定书籍处理顺序，避免依赖输入记录的排列。
    chapters = sorted(groups[book_name], key=lambda record: (record["chapter_index"], record["id"]))  # 按顺序号和 ID 生成新的有序列表。
    rng.shuffle(chapters)  # 打乱当前列表；章节字典和正文内容不变。
    count = len(chapters)  # 取得当前书籍的章节数。
    train_end = int(count * 0.8)  # 计算训练集右端点，正数取整相当于向下取整。
    validation_end = train_end + int(count * 0.1)  # 在训练集之后再取约一成章节作为验证集。
    assert 0 < train_end < validation_end < count, "章节太少，无法分成三个非空集合"  # 防止小分组产生空集合。
    splits["train"].extend(chapters[:train_end])  # 将前八成章节逐条加入训练集。
    splits["validation"].extend(chapters[train_end:validation_end])  # 将中间一成章节加入验证集。
    splits["test"].extend(chapters[validation_end:])  # 将剩余章节全部加入测试集，避免取整后遗漏。
    print(book_name, "总章数：", count, "训练：", train_end, "验证：", validation_end - train_end, "测试：", count - validation_end)  # 显示每本书的划分结果。

ids = {name: {record["id"] for record in rows} for name, rows in splits.items()}  # 为每个集合提取章节 ID 集合。
texts = {name: {record["text"] for record in rows} for name, rows in splits.items()}  # 为每个集合提取完整章节文本集合。

for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:  # 检查三种集合配对。
    assert ids[left].isdisjoint(ids[right]), "同一章节进入了多个集合"  # 两个集合不能共享章节 ID。
    assert texts[left].isdisjoint(texts[right]), "不同集合含有完全相同的整章正文"  # 检查不同 ID 下是否存在相同整章文本。

assert ids["train"] | ids["validation"] | ids["test"] == {record["id"] for record in records}, "划分遗漏或多出章节"  # 三个集合合起来必须覆盖全部输入 ID。
assert sum(len(rows) for rows in splits.values()) == len(records), "划分后的总记录数不一致"  # 检查总数没有增加或减少。
print("集合交集、完整章节重复和覆盖检查：通过")  # 所有断言通过后显示。

output_dir = ROOT / "data" / "splits"  # 指定三个数据集和清单的保存目录。
output_dir.mkdir(parents=True, exist_ok=True)  # 创建目录，已经存在时继续使用。
file_info = {}  # 保存每个输出文件的记录数、字符数和内容指纹。

for name, rows in splits.items():  # 依次保存训练集、验证集和测试集。
    output_path = output_dir / f"{name}.jsonl"  # 用集合名称生成文件名。
    with output_path.open("w", encoding="utf-8", newline="\n") as file:  # 以 UTF-8 和 LF 覆盖写入当前划分。
        for record in rows:  # 按划分后的固定顺序取出章节。
            file.write(json.dumps(record, ensure_ascii=False) + "\n")  # 每章写成一行 JSON，正文保持不变。
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()  # 根据文件的全部字节计算 SHA-256。
    characters = sum(len(record["text"]) for record in rows)  # 统计正文字符总数，不是 JSON 文件字节数。
    file_info[name] = {"count": len(rows), "characters": characters, "sha256": digest}  # 把统计和指纹加入清单。
    print(name, "章节数：", len(rows), "字符数：", characters, "SHA-256：", digest)  # 显示结果，供重复运行时对比。

manifest = {"seed": SEED, "algorithm": "per_book_sorted_shuffle_v1", "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1}, "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "files": file_info}  # 记录划分配置、输入指纹和输出统计。
manifest_path = output_dir / "manifest.json"  # 指定划分清单位置。

with manifest_path.open("w", encoding="utf-8", newline="\n") as file:  # 打开清单文件并明确换行方式。
    file.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")  # 普通 JSON 清单可以使用缩进。

print("保存位置：", output_dir)  # 显示本次划分的输出目录。
