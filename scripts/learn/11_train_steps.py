from importlib import import_module  # 按字符串导入已有数据与模型模块。
from itertools import islice  # 从批次生成器中只取指定数量的批次。
import numpy as np  # 检查批次数组与训练损失。


def main():  # 定义连续训练少量批次的演示入口。
    data = import_module("06_make_windows")  # 复用真实训练章节和批次函数，首次导入会执行旧数据检查。
    network = import_module("07_tiny_rnn")  # 复用模型、完整反向和梯度裁剪。
    batch_size = 16  # 每个批次最多包含 16 个窗口。
    seq_length = 32  # 每个窗口预测 32 个位置，窗口步长也取 32。
    max_steps = 5  # 本课只更新五个批次，不训练完整一轮。
    learning_rate = 0.1  # 每一步使用相同的普通梯度下降学习率。
    max_norm = 1.0  # 每一步都对完整梯度使用相同的全局范数上限。
    model = network.TinyRNN(len(data.tokens), hidden_size=64, seed=42)  # 只在循环外创建一次模型，之后持续更新同一份参数。
    windows = data.iter_windows(data.train_records, seq_length, seq_length)  # 只创建一次窗口生成器，使窗口位置持续向后移动。
    batches = data.iter_batches(windows, batch_size)  # 把这一个窗口流打包为连续批次。
    completed_steps = 0  # 累计真正完成的参数更新次数。
    seen_windows = 0  # 累计真正用于更新的窗口数。
    seen_positions = 0  # 累计真正参与损失计算的预测位置数。
    loss_sum = 0.0  # 累计各批平均损失乘其位置数，准备计算加权平均。

    for batch in islice(batches, max_steps):  # 顺着同一个批次生成器取前五批，每批更新一次。
        x, y = batch["x"], batch["y"]  # 取本批输入和正确的下一字编号。
        assert x.shape == y.shape and 0 < x.shape[0] <= batch_size and x.shape[1] == seq_length, "批次尺寸错误"  # 按实际行数检查，也允许未来遇到不足一批的尾批。
        assert x.dtype == y.dtype == np.int64 and np.array_equal(x[:, 1:], y[:, :-1]), "批次类型或错位错误"  # 保证训练格式仍然正确。
        loss, raw_grads = model.loss_and_grad(x, y)  # 用当前参数计算本批更新前的平均损失与完整梯度。
        assert np.isfinite(loss), "训练损失不是有限数"  # 发现异常数值就停止，不继续污染参数。
        grads, raw_norm, scale = network.clip_gradients(raw_grads, max_norm)  # 每步完整反向结束后再做一次全局裁剪。
        for name in model.params:  # 使用同一批次算出的五组梯度统一更新参数。
            model.params[name] -= learning_rate * grads[name]  # 本步新参数会被下一批直接使用。
        assert all(np.isfinite(param).all() for param in model.params.values()), "更新后参数异常"  # 确认连续更新没有产生无效参数。
        completed_steps += 1  # 五组参数更新完成，记作一个 step。
        seen_windows += x.shape[0]  # 统计实际行数，不假定永远是完整批次。
        seen_positions += y.size  # 统计本批实际预测位置数。
        loss_sum += loss * y.size  # 将平均损失还原为本批损失总和，再跨批累计。
        print(f"step={completed_steps} loss={loss:.6f} grad_norm={raw_norm:.6f} scale={scale:.6f} 来源={batch['chapter_ids'][0]} 起点={batch['starts'][0]}")  # 显示每步更新前的损失和首行窗口来源。

    assert completed_steps == max_steps and seen_positions > 0, "当前训练数据不足以完成本课五步"  # 本例训练集应能提供足够批次。
    windows_per_epoch = sum(max(0, (len(record["text"]) - seq_length - 1) // seq_length + 1) for record in data.train_records)  # 根据当前窗口长度与步长，独立计算完整一轮的窗口数。
    steps_per_epoch = (windows_per_epoch + batch_size - 1) // batch_size  # 尾批也更新一次，用向上取整计算一轮步数。
    print("已完成更新步数：", completed_steps)  # 应为 5，而不是五组参数乘五步。
    print("已训练窗口数：", seen_windows)  # 当前五个完整批次共 80 个窗口。
    print("已训练预测位置数：", seen_positions)  # 当前共 2560 个预测位置。
    print("训练过程按位置加权的平均损失：", loss_sum / seen_positions)  # 汇总各步更新前的损失，模型参数在这些步骤间持续变化。
    print("完整一轮需要的更新步数：", steps_per_epoch)  # 当前配置为 4787 步，包含最后八行的尾批。
    print("本次是否遍历完整训练集：", seen_windows == windows_per_epoch)  # 本课只取五批，因此应为 False。
    print("同一模型连续多批更新检查：通过")  # 所有本课断言通过后显示。


if __name__ == "__main__":  # 只有直接运行文件时才开始训练演示。
    main()  # 创建一次模型并连续执行五步更新。