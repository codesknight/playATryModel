import sys  # 导入正式SFT批次函数。
from pathlib import Path  # 定位项目目录。
ROOT = Path(__file__).resolve().parents[2]  # 不依赖终端当前目录。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 只导入编码函数，不导入训练入口。
from sft_batches import reviewed_rows, load_tokenizer, make_batch, encode_dialogue  # 使用真实BPE与审核记录。


def main():  # 学习者只读实操，不载入30M模型或更新参数。
    rows, audit = reviewed_rows()  # 使用两条经人工检查的训练会话。
    tokenizer = load_tokenizer()  # 6400词BPE与预训练完全一致。
    xs, ys, encoded = make_batch(rows, tokenizer)  # 完整对话右侧填充。
    print("审核条数：", len(audit["reviews"]), "接纳接线检查：", len(rows))  # 不把局部抽查推广成全库质量认证。
    print("批次X/Y形状：", (len(xs), len(xs[0])), (len(ys), len(ys[0])))  # token轴按本批最长输入对齐。
    counts = []  # 有效目标数用于真正的损失分母。
    for row, padded_x, padded_y, item in zip(rows, xs, ys, encoded):  # 同时检查原序列与填充结果。
        x, y, ids, flags = item  # flags在错位前与完整ids逐项对应。
        assert all(target == (token if flag else -100) for target, token, flag in zip(y, ids[1:], flags[1:])), "目标或掩码错位"  # 不能将flags[:-1]误配到下一token。
        assert all(value == 0 for value in padded_x[len(x):]) and all(value == -100 for value in padded_y[len(y):]), "填充位置错误"  # PAD输入与忽略标签是两个不同概念。
        count = sum(value != -100 for value in y)  # 只计助手正文和结束符。
        expected = sum(len(tokenizer.encode(message["content"], add_special_tokens=False).ids) + 1 for message in row["messages"] if message["role"] == "assistant")  # 每轮结束符也算一个目标。
        assert count == expected, "助手目标数量不一致"  # 不以字符数代替BPE token数。
        counts.append(count)  # 不同会话目标数可以不同。
        print("会话：", row["id"], "输入token：", len(x), "助手目标：", count)  # 标识可回溯到训练来源。
        print("参与损失的内容：", tokenizer.decode([value for value in y if value != -100], skip_special_tokens=False))  # 展示助手结束符，不显示用户目标。
    assert counts[0] != counts[1], "本课应覆盖有效目标不等长的情况"  # 为加权损失提供真实样例。
    try:  # 超长会话不能静默丢掉末尾答案。
        encode_dialogue(rows[0]["messages"], tokenizer, context=8)  # 故意给出不足以容纳完整对话的上下文。
    except AssertionError as error:  # 只接受预期的长度拒绝。
        assert "超长" in str(error), "长度测试被其他错误提前中断"  # 保证测到了指定约束。
    else:  # 未拒绝就说明实现有问题。
        raise AssertionError("超长会话未拒绝")  # 不能把被截断答案当正常目标。
    print("本批损失分母：", sum(counts))  # 平均交叉熵除以总有效助手目标数。
    print("正式BPE、完整会话、错位掩码和填充检查：通过；本脚本没有训练模型。")  # 运行这份脚本不增加预训练或SFT步数。


if __name__ == "__main__":  # 导入时不自动执行学习者检查。
    main()  # 学习者显式运行才打印数据。
