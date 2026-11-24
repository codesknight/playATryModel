import hashlib  # 保存来源块和训练数据指纹。
import json  # 原始和清洗后的数据都使用JSONL。
from pathlib import Path  # 定位独立的数据版本目录。
import random  # 固定按组划分的随机种子。
import re  # 检查格式、身份文本和控制字符。
import unicodedata  # 规范化去重用的文本。
import urllib.request  # 只下载明确指定的字节范围。
ROOT = Path(__file__).resolve().parents[2]  # 项目根目录。
DIRECTORY = ROOT / "data/chat48"  # 不覆盖旧课程数据。
REVISION = "312afb4f76391145c6902f765bb51691c09a12f5"  # 固定数据集提交。
BASE = "https://huggingface.co/datasets/jingyaogong/minimind_dataset"  # 保留来源署名。
SOURCES = {"pretrain": ("pretrain_t2t_mini.jsonl", 1241043656, 2 * 1024 * 1024), "sft": ("sft_t2t_mini.jsonl", 1739201170, 1024 * 1024)}  # 八块各2MiB或1MiB，总24MiB。


def sha(data):  # 只校验实际收到的字节，不冒充整文件下载校验。
    return hashlib.sha256(data).hexdigest()  # 返回可保存的指纹。


def dump(value):  # 固定跨平台JSON格式。
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")  # 写入字节时不会变成CRLF。


def normalize(text):  # 仅用于去重，不改写保存的正文。
    return "".join(c for c in unicodedata.normalize("NFKC", text) if c.isalnum())  # 忽略空白和标点。


def simhash(text):  # 字符近似重复启发式，不是语义相似性的完整证明。
    grams = sorted({text[i:i + 5] for i in range(0, max(1, len(text) - 4), max(1, len(text) // 128))})  # 抽取约128个五字符片段。
    numbers = [int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little") for s in grams]  # 使用固定64位哈希。
    return sum(1 << b for b in range(64) if sum(1 if n & (1 << b) else -1 for n in numbers) >= 0)  # 按每一位多数表决。


def clean(source, kind):  # 格式筛选不等于已经人工验证回答正确。
    if kind == "pretrain":  # 预训练来源已整理为text字段。
        text = source["text"].replace("\r\n", "\n").strip()  # 统一换行和首尾空白。
        assert isinstance(text, str) and 40 <= len(text) <= 20000, "length"  # 排除异常长度。
        document = {"text": text}  # 只保存正文。
        question = re.split(r"[？?]", text, maxsplit=1)[0] if re.search(r"[？?]", text[:120]) else text[:60]  # 用首问或前缀辅助跨阶段分组。
    else:  # SFT只接入普通可见对话，不采用思考过程和工具任务。
        messages, expected = [], "user"  # 可选系统提示后从用户开始。
        for i, item in enumerate(source["conversations"]):  # 一条完整会话作为一个样本。
            role, content = item["role"], item.get("content", "")  # reasoning_content字段不收录。
            assert not item.get("tools") and not item.get("tool_calls") and role in ("system", "user", "assistant"), "tools_or_role"  # 工具任务留待以后。
            assert isinstance(content, str) and content.strip(), "empty_message"  # 不学习空回答。
            if role == "system":  # 系统提示只允许位于开头。
                assert i == 0, "role_order"  # 中间系统消息不符合本轮模板。
            else:  # 用户与助手严格交替。
                assert role == expected, "role_order"  # 拒绝不完整顺序。
                expected = "assistant" if role == "user" else "user"  # 更新下一条角色。
            messages.append({"role": role, "content": content.strip()})  # 保留可见内容，不读隐藏推理字段。
        assert messages and messages[-1]["role"] == "assistant" and expected == "user", "unfinished_dialogue"  # 必须有完整最后回答。
        text = "".join(m["content"] for m in messages if m["role"] != "system")  # 跨阶段去重忽略附加系统提示。
        assert 20 <= len(text) <= 12000, "length"  # 限制初次训练长度。
        question = next(m["content"] for m in messages if m["role"] == "user")  # 同一首问的会话保守归组。
        document = {"messages": messages}  # 显式转换字段名称。
    assert not re.search(r"jingyaogong|minimind|<think>|<tool_call>|<\|(?:pad|bos|eos|system|user|assistant|end|unk)\|>", text, re.I), "identity_or_reserved_format"  # 不学习来源身份或本轮保留格式。
    assert "\ufffd" not in text and not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text), "control_character"  # 排除损坏文本。
    return document, normalize(text), re.sub(r"\d+", "#", normalize(question))  # 保存正文与两种分组线索。


def main():  # 限量下载、清洗、联合分组并冻结版本。
    raw = DIRECTORY / "raw"  # 原始字节块单独保存。
    raw.mkdir(parents=True, exist_ok=True)  # 不读取原轻小说压缩包。
    card = raw / "upstream_README.md"  # 保留许可与上游来源说明。
    if not card.exists():  # 已下载则离线复用。
        with urllib.request.urlopen(f"{BASE}/resolve/{REVISION}/README.md", timeout=30) as response:  # 固定版本数据卡。
            card.write_bytes(response.read(1024 * 1024))  # 数据卡最多1MiB。
    records, downloads, rejected, seen, duplicates = [], [], {}, set(), 0  # 统计采集与过滤过程。
    for kind, (filename, size, block_size) in SOURCES.items():  # 预训练和SFT分别准备。
        for block in range(8):  # 在八个区间的中点各取一块。
            start = min(size - block_size, size * (2 * block + 1) // 16)  # 固定偏移，不宣称全库均匀随机抽样。
            end, path = start + block_size - 1, raw / f"{kind}_{block:02d}.bin"  # 明确范围和本地位置。
            if not path.exists():  # 中断后仅补缺失块。
                req = urllib.request.Request(f"{BASE}/resolve/{REVISION}/{filename}", headers={"Range": f"bytes={start}-{end}"})  # HTTP Range限制下载量。
                with urllib.request.urlopen(req, timeout=60) as response:  # 失败时保留已完成采集。
                    assert response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"), "服务器忽略Range"  # 不接受意外GB级整文件响应。
                    payload = response.read(block_size + 1)  # 检测超限或截断。
                assert len(payload) == block_size, "采集块长度异常"  # 每块固定字节数。
                path.write_bytes(payload)  # 原始块不被清洗覆盖。
            payload = path.read_bytes()  # 已有块复用相同字节。
            assert len(payload) == block_size, "已有块长度错误"  # 不使用中断残留。
            downloads.append({"kind": kind, "file": filename, "block": block, "start": start, "end": end, "bytes": len(payload), "sha256": sha(payload)})  # 实际范围与指纹。
            accepted = 0  # 每块记录保留数量。
            for offset, line in enumerate(payload.split(b"\n")[1:-1]):  # 丢弃边缘可能不完整的JSON行。
                try:  # 有问题的行计数后排除。
                    document, normalized, question = clean(json.loads(line), kind)  # 不执行语料中的任何指令。
                    fingerprint = sha(normalized.encode("utf-8"))  # 跨阶段规范化完全去重。
                    if fingerprint in seen:  # 重复正文只保留第一条。
                        duplicates += 1  # 记录去重数量。
                        continue  # 不重复计入训练量。
                    seen.add(fingerprint)  # 划分前去重。
                    records.append({"id": f"{kind}:{block:02d}:{offset:05d}", "kind": kind, "block": block, "source_line_within_block": offset, "text_sha256": fingerprint, "question_key": question, "similarity": simhash(normalized), **document})  # 来源和临时分组线索。
                    accepted += 1  # 统计该块。
                except (AssertionError, KeyError, TypeError, ValueError, StopIteration, AttributeError) as error:  # 不静默吞掉格式问题。
                    reason = str(error) if isinstance(error, AssertionError) else type(error).__name__  # 只记录原因，不打印整篇内容。
                    rejected[reason] = rejected.get(reason, 0) + 1  # 报告筛选计数。
            print(kind, block, "保留：", accepted, flush=True)  # 显示实际进度。
    parent = list(range(len(records)))  # 用并查集跨阶段合并近似组。

    def root(i):  # 查询连通组代表。
        while parent[i] != i:  # 路径压缩加快后续查询。
            parent[i] = parent[parent[i]]  # 缩短路径。
            i = parent[i]  # 继续查找根。
        return i  # 返回代表索引。

    keys, buckets = {}, {}  # 首问与近似指纹分别建立索引。
    for i, row in enumerate(records):  # 联合分组后才分配集合。
        key = row["question_key"]  # 同首问和数字模板使用同组。
        if key in keys:  # 已见过这个辅助模板。
            parent[root(i)] = root(keys[key])  # 两个阶段也不拆组。
        keys[key] = i  # 为后续记录建立索引。
        fingerprint = row["similarity"]  # 64位SimHash。
        for band in range(4):  # 分四个16位段寻找候选。
            bucket = band, (fingerprint >> (band * 16)) & 65535  # 三位以内差异必有一段相同。
            for j in buckets.get(bucket, []):  # 避免全库两两比较。
                if (fingerprint ^ records[j]["similarity"]).bit_count() <= 3:  # 字符级启发式仍可能漏掉语义改写。
                    parent[root(i)] = root(j)  # 近似记录整组划分。
            buckets.setdefault(bucket, []).append(i)  # 添加当前候选。
    members = {}  # 用稳定id为连通组命名。
    for i, row in enumerate(records):  # 收集每组成员。
        members.setdefault(root(i), []).append(row["id"])  # 保存来源编号。
    names = {key: min(ids) for key, ids in members.items()}  # 最小稳定id作组名。
    groups = sorted(names.values())  # 先排序再打乱。
    random.Random(48).shuffle(groups)  # 固定本版划分种子。
    mapping = {g: "train" if i < len(groups) * 8 // 10 else "validation" if i < len(groups) * 9 // 10 else "test" for i, g in enumerate(groups)}  # 按组数80/10/10。
    outputs = {(kind, split): [] for kind in SOURCES for split in ("train", "validation", "test")}  # 两阶段共用映射。
    for i, row in enumerate(records):  # 划分后移除临时相似度字段。
        group = names[root(i)]  # 当前记录的稳定组名。
        cleaned = {k: v for k, v in row.items() if k not in ("question_key", "similarity")}  # 保留来源和正文。
        cleaned["group_id"] = group  # 下游可以检查组交集。
        outputs[row["kind"], mapping[group]].append(cleaned)  # 同组跨阶段也只属一个集合。
    payloads = {f"{kind}_{split}.jsonl": ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)).encode("utf-8") for (kind, split), rows in outputs.items()}  # 固定UTF-8与LF。
    payloads["group_to_split.json"] = dump(mapping)  # 保存分组映射。
    stats = {f"{kind}_{split}": {"records": len(rows), "groups": len({r["group_id"] for r in rows}), "characters": sum(len(r["text"]) if kind == "pretrain" else sum(len(m["content"]) for m in r["messages"]) for r in rows)} for (kind, split), rows in outputs.items()}  # 实际条数与字符数。
    manifest = {"format": "chat_corpus48_v1", "dataset": "jingyaogong/minimind_dataset", "revision": REVISION, "source_card": f"{BASE}/blob/{REVISION}/README.md", "credit": "jingyaogong and upstream contributors named in dataset card", "licenses_declared": ["Apache-2.0", "CC-BY-NC-2.0"], "scope": "educational noncommercial pilot; mixed upstream terms retained", "sampling": "eight fixed byte-block midpoints per file, not representative random sampling", "downloaded_range_bytes": sum(x["bytes"] for x in downloads), "blocks": downloads, "filtered": rejected, "duplicates_removed": duplicates, "group_method": "joint first-question/numeric-template key and sampled character SimHash Hamming<=3; incomplete semantic dedup heuristic", "seed": 48, "stats": stats, "files": {name: sha(value) for name, value in payloads.items()}}  # 不宣称机械筛选等于人工内容审核。
    payloads["manifest.json"] = dump(manifest)  # 清单也冻结保存。
    if (DIRECTORY / "manifest.json").exists():  # 完成的版本只检查，不覆盖。
        assert all((DIRECTORY / name).read_bytes() == value for name, value in payloads.items()), "版本产物变化，请另建版本"  # 不悄悄改变验证尺子。
    else:  # 首次生成整套产物。
        for name, value in payloads.items():  # 写入明确版本目录。
            (DIRECTORY / name).write_bytes(value)  # 固定跨平台字节。
    print(json.dumps({"stats": stats, "filtered": rejected, "duplicates_removed": duplicates, "groups": len(groups)}, ensure_ascii=False, indent=2))  # 报告真实规模。


if __name__ == "__main__":  # 导入工具不会下载数据。
    main()  # 运行有界采集与分组流程。
