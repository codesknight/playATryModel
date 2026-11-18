import csv  # 解析原始故事的CSV字段。
import hashlib  # 保存来源与产物指纹。
import io  # 将下载字节交给CSV解析器。
import json  # 保存划分和训练词表。
from pathlib import Path  # 定位独立的新语料目录。
import random  # 用固定种子打乱故事分组。
import unicodedata  # 统一全角形式以检测重复正文。
import urllib.request  # 下载固定提交的公开语料。
ROOT = Path(__file__).resolve().parents[2]  # 当前脚本位于实验目录。
DIRECTORY = ROOT / "data/transformer45"  # 不改动旧四书的数据产物。
REVISION = "ad789c9d622bfbdeecdc90df2041b3ba458e0368"  # 固定作者仓库版本。
BASE = "https://huggingface.co/datasets/joe-chiu/TinyChineseStories"  # 保留作者署名和来源。


def digest(payload):  # 所有文件使用同一哈希算法。
    return hashlib.sha256(payload).hexdigest()  # 返回可写入JSON的指纹。


def read_jsonl(path):  # 不把换行当成故事切分条件。
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]  # 每条JSON才是一个完整文档。


def combined(split):  # 训练与验证分别组合，不混淆旧划分。
    novels = [{"id": row["id"], "source": "classics", "text": row["text"]} for row in read_jsonl(ROOT / f"data/splits/{split}.jsonl")]  # 原章节归属保持不变。
    return novels + read_jsonl(DIRECTORY / f"{split}.jsonl")  # 新故事独立划分后才加入。


def main():  # 下载、清洗、成组划分、建立只来自训练集的词表。
    DIRECTORY.mkdir(parents=True, exist_ok=True)  # 单独保存新数据产物。
    raw_path = DIRECTORY / "stories.csv"  # 保存固定版本原始内容以便离线复现。
    url = f"{BASE}/resolve/{REVISION}/stories.csv"  # 不使用可变化的main下载地址。
    if not raw_path.exists():  # 已下载的文件直接复用，不反复访问网络。
        with urllib.request.urlopen(url, timeout=30) as response:  # 网络异常明确退出。
            payload = response.read(10 * 1024 * 1024 + 1)  # 最多下载10MiB并检测是否超限。
        assert len(payload) <= 10 * 1024 * 1024, "下载超出10MiB预算"  # 不截断后冒充完整CSV。
        raw_path.write_bytes(payload)  # 留存原始文件及其哈希。
    payload = raw_path.read_bytes()  # 原始CSV不会被清洗结果覆盖。
    assert digest(payload) == "7ee24009ff4e0a12256d0458b345b725ff611bb2481c7f6a8f8f29adec562374", "原始文件与固定提交不一致"  # 防止已有本地CSV被改动后仍冒用原来源版本。
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))  # 正确处理引用字段与内嵌换行。
    rows, seen, duplicate_count = [], set(), 0  # 精确去重后的文档列表。
    for index, source in enumerate(reader):  # 保留源行号便于追溯。
        source = {key.strip(): value for key, value in source.items() if key is not None}  # 原表头含前导空格。
        text = source["story"].replace("\r\n", "\n").replace("\r", "\n").strip()  # 只统一换行和首尾空白，不删正文标点。
        assert text, "发现空故事"  # 空正文不能进入训练。
        fingerprint = digest("".join(unicodedata.normalize("NFKC", text).split()).encode("utf-8"))  # 检测仅空白或全角形式不同的重复。
        if fingerprint in seen:  # 重复文本只保留第一条。
            duplicate_count += 1  # 如实记录删除数量。
            continue  # 不把同正文放到两个集合。
        seen.add(fingerprint)  # 先去重，再划分。
        conditions = {key: source.get(key, "").strip() for key in ("names", "genre", "theme")}  # 保存原生成条件，同主题的名字和类型变体仍归为一组。
        group = digest(json.dumps({"theme": conditions["theme"]}, ensure_ascii=False, sort_keys=True).encode("utf-8"))  # 确定性分组编号。
        rows.append({"id": f"tiny_stories:{index:05d}", "source": "tiny_stories", "source_index": index, "group": group, "text_sha256": fingerprint, "conditions": conditions, "text": text})  # 完整保留来源信息。
    groups = sorted({row["group"] for row in rows})  # 分组排序后再打乱，可重复得到同一划分。
    random.Random(45).shuffle(groups)  # 独立固定的数据划分种子。
    count = len(groups)  # 比例按组数计算，故事条数可能略有差异。
    assignment = {group: "train" if i < int(count * 0.8) else "validation" if i < int(count * 0.9) else "test" for i, group in enumerate(groups)}  # 整组按80/10/10划分。
    counts, files = {}, {}  # 为每个产物记录完整证据。
    for split in ("train", "validation", "test"):  # 测试故事保存但训练不读取。
        selected = [row for row in rows if assignment[row["group"]] == split]  # 同组只能属于一个集合。
        output = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode("utf-8")  # 一条完整故事一行JSON。
        (DIRECTORY / f"{split}.jsonl").write_bytes(output)  # 可离线检查划分。
        counts[split] = {"documents": len(selected), "characters": sum(len(row["text"]) for row in selected), "groups": len({row["group"] for row in selected})}  # 不用组数冒充故事数。
        files[f"{split}.jsonl"] = digest(output)  # 后续训练记录实际读入的版本。
    train = combined("train")  # 建词表只读取训练集，包括旧四书训练章节。
    tokens = ["<UNK>", "<BOS>", "<EOS>"] + sorted({char for row in train for char in row["text"]})  # 字符级词表保留入门可解释性。
    vocab = {"tokens": tokens, "unk_id": 0, "bos_id": 1, "eos_id": 2}  # 文档边界由专用编号表示。
    vocabulary = (json.dumps(vocab, ensure_ascii=False, indent=2) + "\n").encode("utf-8")  # 固定序列化便于核验。
    (DIRECTORY / "vocab.json").write_bytes(vocabulary)  # 新词表独立于旧RNN词表。
    files["vocab.json"] = digest(vocabulary)  # 训练加载时核对编号含义。
    manifest = {"dataset": "joe-chiu/TinyChineseStories", "revision": REVISION, "source_url": url, "card_url": f"{BASE}/blob/{REVISION}/README.md", "credit": "joe-chiu, TinyChineseStories", "license_declared_in_readme": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "synthetic_source": "GPT-3.5, as declared by author", "raw_sha256": digest(payload), "raw_bytes": len(payload), "duplicates_removed": duplicate_count, "retained_stories": len(rows), "split_seed": 45, "split_unit": "theme group; names and genre variants kept together", "near_duplicate_check": "not performed; normalized exact duplicates only", "new_corpus_splits": counts, "files": files, "vocab_size": len(tokens), "combined_train_characters": sum(len(row["text"]) for row in train), "old_split_sha256": {split: digest((ROOT / f"data/splits/{split}.jsonl").read_bytes()) for split in ("train", "validation")}}  # 测试正文不参与建词表与训练。
    (DIRECTORY / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 记录下载与变更方式。
    (DIRECTORY / "README.md").write_text(f"# 新故事语料\n\njoe-chiu, TinyChineseStories，固定提交{REVISION}。\n来源：{BASE}\n作者README声明GPT-3.5合成，许可CC BY 4.0：https://creativecommons.org/licenses/by/4.0/ 。\n本仓库保留原CSV，衍生JSONL仅统一换行和首尾空白、去掉规范化完全重复正文，再按生成条件成组划分；详细来源与统计见manifest.json。近似重复检测尚未进行。\n", encoding="utf-8")  # 署名、许可与改动说明随数据分发。
    print(json.dumps(manifest, ensure_ascii=False, indent=2))  # 显示实际收集规模。


if __name__ == "__main__":  # 导入工具时不下载数据。
    main()  # 执行一次可复现的数据准备。
