import json  # 读取人工审核清单和冻结的训练对话。
from pathlib import Path  # 用项目根目录定位数据。
from tokenizers import Tokenizer  # 复用6400词BPE，不重新训练词表。
from prepare_chat_corpus import ROOT, sha  # 复用项目路径与文件摘要。
SPECIAL = {"pad": 0, "bos": 1, "system": 3, "user": 4, "assistant": 5, "end": 6}  # 与预训练保存的编号完全一致。


def reviewed_rows():  # 只返回审核通过且来自训练划分的完整会话。
    audit = json.loads((ROOT / "data/chat51/audit.json").read_text(encoding="utf-8"))  # 本课审核决定可追溯。
    source = ROOT / audit["source_path"]  # 不读验证集或测试集。
    assert audit["source_path"] == "data/chat48/sft_train.jsonl" and sha(source.read_bytes()) == audit["source_sha256"], "训练来源版本改变"  # 不能把留出对话临时转入训练。
    rows = {row["id"]: row for row in map(json.loads, source.read_text(encoding="utf-8").splitlines())}  # 依据稳定标识定位原对话。
    selected = []  # 本课只接纳显式批准的两条。
    for item in audit["reviews"]:  # 拒绝样例也保留理由，而非静默删除证据。
        row = rows[item["id"]]  # 所有审核项均应属于同一冻结训练文件。
        assert row["text_sha256"] == item["text_sha256"], "审核正文版本改变"  # 防止审核后替换答案。
        if item["decision"] == "accept_probe":  # 只批准训练接线检查，不批准整库正式SFT。
            selected.append(row)  # 保留完整多轮顺序及来源字段。
    assert len(selected) == 2, "本课固定使用两条已审会话"  # 不自动扩大训练量。
    return selected, audit  # 返回审核版本供报告记录。


def encode_dialogue(messages, tokenizer, context=256):  # 正文用BPE，角色边界用专用token。
    ids, flags = [SPECIAL["bos"]], [False]  # BOS保留在上下文，不监督生成BOS。
    expected = "user"  # 第一轮必须由用户发起。
    for index, message in enumerate(messages):  # 保留整个多轮会话，不拆散用户与答案。
        role, content = message["role"], message["content"]  # 分开读取角色与正文。
        assert isinstance(content, str) and content.strip() and "<|" not in content, "空正文或保留标记冲突"  # 普通文本不能冒充控制标记。
        if role == "system":  # 可选系统提示只能在最开头。
            assert index == 0, "系统消息位置错误"  # 中间不能插入新的系统角色。
        else:  # 用户和助手必须交替。
            assert role == expected, "对话角色顺序错误"  # 拒绝缺少答案或连续用户消息。
            expected = "assistant" if role == "user" else "user"  # 切换下一角色。
        body = tokenizer.encode(content, add_special_tokens=False).ids  # 中文字符数不等于BPE token数。
        assert body and all(token >= 8 for token in body), "正文出现特殊编号"  # 控制token只由模板显式写入。
        ids.extend([SPECIAL[role], *body, SPECIAL["end"]])  # 每条消息完整保留角色、正文、结束符。
        flags.extend([False, *([role == "assistant"] * len(body)), role == "assistant"])  # 只监督助手正文及其结束符。
    assert messages and messages[-1]["role"] == "assistant" and expected == "user", "会话缺少完整回答"  # 不能将待生成前缀作为训练答案。
    assert len(ids) - 1 <= context, "完整会话超长：本课拒绝，不截断答案"  # 后续分段策略另作实验，不能静默损坏对话。
    x = ids[:-1]  # 输入当前token。
    y = [token if keep else -100 for token, keep in zip(ids[1:], flags[1:])]  # 标签与监督标记一起向左错一位。
    assert any(target != -100 for target in y), "没有有效助手目标"  # 不能对全忽略的批次求平均。
    return x, y, ids, flags  # 完整信息供学习者检查对齐。


def make_batch(rows, tokenizer, context=256):  # 右侧填充不会让较短样本提前结束的部分参与损失。
    encoded = [encode_dialogue(row["messages"], tokenizer, context) for row in rows]  # 先逐条验证完整性。
    length = max(len(item[0]) for item in encoded)  # 本批实际最长输入，而非盲目填满256。
    xs = [x + [SPECIAL["pad"]] * (length - len(x)) for x, _, _, _ in encoded]  # 输入PAD编号是0。
    ys = [y + [-100] * (length - len(y)) for _, y, _, _ in encoded]  # 填充标签必须忽略，不能用0代替。
    return xs, ys, encoded  # 在调用处选择CPU或GPU张量。


def load_tokenizer():  # 加载并核对旧分词器版本。
    path = ROOT / "data/chat48/tokenizer/tokenizer.json"  # 新阶段保持同一词表。
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))  # 原始编号对应关系的证据。
    assert sha(path.read_bytes()) == manifest["tokenizer_sha256"], "分词器内容改变"  # 词表大小相同也不能替换编号。
    tokenizer = Tokenizer.from_file(str(path))  # 不调用任何词表训练方法。
    assert tokenizer.get_vocab_size() == 6400 and all(tokenizer.token_to_id("<|" + name + "|>") == value for name, value in SPECIAL.items()), "特殊编号不一致"  # 模板必须与预训练模型兼容。
    return tokenizer  # 编码和解码均复用此对象。
