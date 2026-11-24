import hashlib  # 检查已经采集的数据版本。
import json  # 读取冻结清单而不重新下载数据。
from pathlib import Path  # 定位本课目录。
from tokenizers import Tokenizer  # 实际观察ByteLevel BPE编码。
ROOT = Path(__file__).resolve().parents[2]  # 项目根目录。
directory = ROOT / "data/chat48"  # 正式训练管线首次有界采样。
manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))  # 来源、过滤与划分已保存。
groups = {name: set() for name in ("train", "validation", "test")}  # 联合检查两个阶段的组归属。
for name, expected in manifest["files"].items():  # 不只检查输出文件是否存在。
    assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, "语料产物与版本不符"  # 输入必须对应保存的指纹。
for kind in ("pretrain", "sft"):  # 预训练正文与后续对话分别检查。
    for split in groups:  # 这里只作完整性检查，不计算测试模型指标。
        path = directory / f"{kind}_{split}.jsonl"  # 不把两个阶段混为同一训练文件。
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]  # 一行一条完整来源记录。
        groups[split].update(row["group_id"] for row in rows)  # 相似组跨阶段也不能泄漏到另一集合。
        print(kind, split, "记录数：", len(rows), "字符数：", manifest["stats"][f"{kind}_{split}"]["characters"])  # 字符数与token数分开理解。
assert not groups["train"] & groups["validation"] and not groups["train"] & groups["test"] and not groups["validation"] & groups["test"], "联合分组有交集"  # 三集合两两互斥。
tokenizer = Tokenizer.from_file(str(directory / "tokenizer/tokenizer.json"))  # 切分规则与词表都从文件加载。
text = "春风又起，Hello🙂"  # 一个字符不一定对应一个token。
encoded = tokenizer.encode(text, add_special_tokens=False)  # 不额外插入文档标记。
decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)  # 完整还原字节文本。
assert decoded == text and 7 not in encoded.ids, "编码往返或未知符检查失败"  # ByteLevel覆盖所有UTF-8字节。
print("词表大小：", tokenizer.get_vocab_size())  # 当前6400对应30M结构预算。
print("原文：", text, "字符数：", len(text))  # 先观察字符长度。
print("编号：", encoded.ids, "token数：", len(encoded.ids))  # 再观察BPE长度。
print("重载解码：", decoded)  # 编号只有配套分词器才能解读。
print("正式采样数据版本、跨阶段分组与BPE往返检查：通过；本脚本不训练神经网络。")  # 数据准备与模型训练明确区分。
