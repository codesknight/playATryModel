import hashlib  # 固定分词器与编码产物的版本。
import json  # 读取已冻结的训练划分。
import os  # 关闭分词器并行，减少教学环境中的不确定性。
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 在导入分词器前设置。
from pathlib import Path  # 项目路径不依赖当前终端位置。
import numpy as np  # 保存紧凑的整数编号数组。
import tokenizers  # 使用现有环境中固定版本的分词器库。
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders  # 显式实现ByteLevel BPE。
from prepare_chat_corpus import DIRECTORY, dump, sha  # 复用固定数据版本与LF字节格式。
SPECIAL = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|unk|>"]  # 后续模型与聊天必须使用同一组编号。


def records(kind, split):  # 测试集合不在分词器训练或预训练入口读取。
    path = DIRECTORY / f"{kind}_{split}.jsonl"  # 根据阶段和集合显式定位。
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]  # 每条仍是完整文档或对话。


def training_texts():  # 分词器只能学习训练集的统计。
    for row in records("pretrain", "train"):  # 预训练训练正文。
        yield row["text"]  # 不混入验证文本。
    for row in records("sft", "train"):  # SFT训练正文也属于允许的训练部分。
        for message in row["messages"]:  # 全部角色正文用于词表统计，不等于全部位置都作SFT损失。
            yield message["content"]  # 不加入reasoning_content或测试数据。


def main():  # 训练新分词器并准备首段预训练编号。
    folder = DIRECTORY / "tokenizer"  # 新词表独立于旧字符模型。
    folder.mkdir(parents=True, exist_ok=True)  # 只创建本课的产物目录。
    path = folder / "tokenizer.json"  # 分词器的完整切分规则和词表。
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL[7]))  # BPE逐步合并频繁的相邻片段。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # 先映射UTF-8字节，不强制在文本前加空格。
    tokenizer.decoder = decoders.ByteLevel()  # 编号序列可以还原原始字节文本。
    trainer = trainers.BpeTrainer(vocab_size=6400, min_frequency=2, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)  # 完整字节表使新字符可分解，不必都落到UNK。
    tokenizer.train_from_iterator(training_texts(), trainer=trainer)  # 只有训练正文参与合并规则学习。
    assert tokenizer.get_vocab_size() == 6400, "实际词表未达到预算6400"  # 结构参数量取决于实际词表。
    assert [tokenizer.token_to_id(token) for token in SPECIAL] == list(range(8)), "特殊符号编号改变"  # 后续模型不能猜测角色编号。
    probes = ["春风又起。", "你好！Hello 2026", "繁體中文與简体中文", "生僻字𠮷和表情🙂\n第二行"]  # 覆盖多语、稀有字、换行和表情。
    for text in probes:  # 检查编解码与未知编号。
        ids = tokenizer.encode(text, add_special_tokens=False).ids  # 不自动插入BOS/EOS。
        assert 7 not in ids and tokenizer.decode(ids, skip_special_tokens=False) == text, "字节BPE往返失败"  # 新字可能拆成多个token，但不能丢字。
    serialized = tokenizer.to_str(pretty=True).encode("utf-8")  # 固定保存实际训练出的分词器。
    if path.exists():  # 已冻结的分词器不悄悄改写。
        assert path.read_bytes() == serialized, "分词器重训结果不同，请另建版本"  # 变化必须导致新的模型版本。
    else:  # 首次训练时写入完整规则。
        path.write_bytes(serialized)  # 不只保存token字符串列表。
    restored = Tokenizer.from_file(str(path))  # 重载真实文件再做一次验证。
    assert restored.encode(probes[0]).ids == tokenizer.encode(probes[0]).ids, "分词器重载结果不同"  # 推理阶段必须得到相同编号。
    train_tokens, character_count = [], 0  # 首次试验数据不大，明确累积实际编号数。
    for row in records("pretrain", "train"):  # 预训练只使用预训练训练集，不拿SFT或验证补足步数。
        ids = tokenizer.encode(row["text"], add_special_tokens=False).ids  # 保留正文原有顺序。
        train_tokens.extend([1] + ids + [2])  # 文档间用BOS/EOS明确标识，随后可以做标准流式打包。
        character_count += len(row["text"])  # 压缩率分母为正文字符数。
    np.save(folder / "pretrain_train.npy", np.asarray(train_tokens, dtype=np.uint16), allow_pickle=False)  # 6400编号可安全存入16位整数。
    xs, ys, unknowns = [], [], 0  # 验证按文档保留全部目标，不混合不同文档的前文。
    for row in records("pretrain", "validation"):  # 冻结验证集，不读取测试集。
        body = tokenizer.encode(row["text"], add_special_tokens=False).ids  # 仅用已训练的分词器编码。
        unknowns += body.count(7)  # ByteLevel全字节覆盖应使未知率为0。
        ids = [1] + body + [2]  # 每篇独立从BOS开始。
        for start in range(0, len(ids) - 1, 256):  # 窗口256，保留尾段。
            chunk = ids[start:start + 257]  # 需要多一个编号才能错位。
            x, y = chunk[:-1], chunk[1:]  # 每个目标恰好一次。
            xs.append(x + [0] * (256 - len(x)))  # 右侧补PAD以便组批。
            ys.append(y + [-100] * (256 - len(y)))  # 补齐目标不计损失。
    np.save(folder / "pretrain_validation_x.npy", np.asarray(xs, dtype=np.uint16), allow_pickle=False)  # 输入保存小整数。
    np.save(folder / "pretrain_validation_y.npy", np.asarray(ys, dtype=np.int32), allow_pickle=False)  # 目标还需要存-100。
    report = {"format": "chat_tokenizer48_v1", "tokenizers_version": tokenizers.__version__, "corpus_manifest_sha256": sha((DIRECTORY / "manifest.json").read_bytes()), "tokenizer_sha256": sha(serialized), "vocab_size": 6400, "special_tokens": dict(zip(SPECIAL, range(8))), "train_source": "pretrain_train and sft_train only", "pretrain_train_characters": character_count, "pretrain_train_tokens_with_boundaries": len(train_tokens), "body_tokens_per_character": (len(train_tokens) - 2 * len(records("pretrain", "train"))) / character_count, "validation_windows": len(xs), "validation_positions": int((np.asarray(ys) != -100).sum()), "validation_unknown_tokens": unknowns, "roundtrip_probes": probes, "files": {name: sha((folder / name).read_bytes()) for name in ("tokenizer.json", "pretrain_train.npy", "pretrain_validation_x.npy", "pretrain_validation_y.npy")}}  # 完整记录词表与模型输入的对应关系。
    (folder / "manifest.json").write_bytes(dump(report))  # 新模型检查点引用此版本。
    print(json.dumps(report, ensure_ascii=False, indent=2))  # 真实token数用于决定后续训练预算。


if __name__ == "__main__":  # 导入模块不训练分词器。
    main()  # 首次运行训练BPE，重跑核对规则一致。
