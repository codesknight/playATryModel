import json  # 读取一行一条的完整对话。
from pathlib import Path  # 相对脚本位置寻找项目数据。
ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于scripts/learn。
SPECIAL = ["<UNK>", "<BOS>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]  # 本课演示专用编号，不复用旧模型编号。


def encode_dialogue(messages, tokens):  # 把完整多轮对话转换为输入和只监督回答的目标。
    mapping = {token: index for index, token in enumerate(tokens)}  # 建立字符和角色标记到编号的映射。
    ids, supervise = [mapping["<BOS>"]], [False]  # 序列起点不作为回答目标。
    expected = "user"  # 除可选系统消息外，第一条必须由用户发起。
    for index, message in enumerate(messages):  # 保留整段会话的原始顺序。
        role, content = message["role"], message["content"]  # 读取角色和消息正文。
        assert isinstance(content, str) and content, "消息正文必须是非空字符串"  # 空回答不能用于当前演示。
        if role == "system":  # 系统提示只允许出现在会话开头。
            assert index == 0, "系统消息只能在开头"  # 不把中间消息误作系统配置。
        else:  # 用户与助手必须交替出现。
            assert role == expected, "用户和助手顺序不正确"  # 缺少回答或连续用户消息需要先整理。
            expected = "assistant" if role == "user" else "user"  # 记录下一条应有的角色。
        ids.append(mapping[f"<|{role}|>"])  # 使用一个专用编号标识消息角色。
        supervise.append(False)  # 当前示例不训练模型生成角色头，只训练回答正文和结束符。
        ids.extend(mapping.get(char, 0) for char in content)  # 正文逐字符编码，未知字符映射为UNK。
        supervise.extend([role == "assistant"] * len(content))  # 只有助手正文作为损失目标。
        ids.append(mapping["<|end|>"])  # 每条消息均保留结束标记。
        supervise.append(role == "assistant")  # 助手结束符也要学习，才能知道何时停止回答。
    assert messages and messages[-1]["role"] == "assistant" and expected == "user", "会话必须以完整助手回答结束"  # 不把待生成前缀冒充完整SFT样本。
    x = ids[:-1]  # 每个位置输入当前已知编号。
    y = [token if train else -100 for token, train in zip(ids[1:], supervise[1:])]  # 监督标记也必须随目标向左错位一格。
    assert len(x) == len(y), "输入目标长度不同"  # 交叉熵每行对应一个目标。
    assert sum(target != -100 for target in y) == sum(len(item["content"]) + 1 for item in messages if item["role"] == "assistant"), "回答目标数量错误"  # 每个回答字符加一个结束符。
    assert all(target == -100 or target == original for target, original in zip(y, ids[1:])), "监督目标错位"  # 有效目标必须是实际下一符号。
    return x, y, ids, supervise  # 返回完整序列方便检查角色与目标的关系。


def main():  # 只演示格式与损失掩码，不执行任何模型训练。
    path = ROOT / "data/chat_demo/messages.jsonl"  # 本课只有两条人工编写的格式样例。
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]  # 每条JSON保留完整多轮会话。
    characters = sorted({char for row in records for message in row["messages"] for char in message["content"]})  # 此词表只为演示，不能替换已有模型词表。
    tokens = SPECIAL + characters  # 把专用角色标记放在词表前面。
    for row in records:  # 分别检查单轮和多轮对话。
        x, y, ids, supervise = encode_dialogue(row["messages"], tokens)  # 输入含用户问题，损失只监督助手回答。
        learned = "".join(tokens[target] for target in y if target != -100)  # 展示真正参与损失的目标。
        print("对话：", row["id"], "输入长度：", len(x), "有效回答目标：", sum(target != -100 for target in y))  # 与填充后的长度区分。
        print("参与损失的内容：", learned)  # 用户和系统正文不会出现在这里。
        assert all(target == -100 for target, flag in zip(y, supervise[1:]) if not flag), "非回答位置参与了损失"  # 不意外监督用户提示。
    print("聊天角色、完整多轮、目标错位和助手回答损失掩码检查：通过；本脚本没有训练模型。")  # 格式正确不代表模型已经会聊天。


if __name__ == "__main__":  # 导入检查工具时不打印或执行演示。
    main()  # 学习者可在base环境直接运行，不依赖PyTorch。
