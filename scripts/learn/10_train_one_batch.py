from importlib import import_module  # 按模块文件名导入已有代码，支持数字开头的文件名。
import numpy as np  # 导入 NumPy，检查编号数组、参数和损失。


def main():  # 定义一次真实训练批次的演示入口。
    data = import_module("06_make_windows")  # 复用已有训练记录、固定词表、窗口和批次函数，并执行该模块顶层的旧检查。
    network = import_module("07_tiny_rnn")  # 复用 TinyRNN、交叉熵与梯度裁剪，该模块的 main 不会被自动执行。
    batch_size = 16  # 最多收集 16 个窗口作为本次训练批次。
    seq_length = 32  # 每个窗口预测 32 个位置。
    learning_rate = 0.1  # 本次演示使用的普通梯度下降学习率。
    max_norm = 1.0  # 将所有参数梯度的总长度限制在 1 以内。
    windows = data.iter_windows(data.train_records, seq_length, seq_length)  # 从真实训练章节按顺序产生窗口，窗口不跨章节。
    batch = next(data.iter_batches(windows, batch_size))  # 只取第一个批次，不遍历训练更新整个数据集。
    x, y = batch["x"], batch["y"]  # 取出真实输入编号和向后错开一位的目标编号。
    assert x.shape == y.shape == (batch_size, seq_length), "本例首批尺寸不符合预期"  # 当前数据足够组成 16 行的完整首批。
    assert x.dtype == y.dtype == np.int64 and np.array_equal(x[:, 1:], y[:, :-1]), "真实批次错位或类型不正确"  # 再次核对接入模型的数据。
    model = network.TinyRNN(vocab_size=len(data.tokens), hidden_size=64, seed=42)  # 用真实固定词表创建全新的随机模型。

    params_before = {name: param.copy() for name, param in model.params.items()}  # 保存本次前向使用的参数，用来检查反向不会更新它们。
    loss_before, raw_grads = model.loss_and_grad(x, y)  # 对 512 个真实预测位置完成前向、平均损失和全部反向。
    assert all(np.array_equal(param, params_before[name]) for name, param in model.params.items()), "求梯度过程中修改了参数"  # 完整反向结束前参数应保持不变。
    clipped_grads, raw_norm, scale = network.clip_gradients(raw_grads, max_norm)  # 汇总完成后，对五组梯度做一次全局裁剪。
    clipped_norm = network.gradient_norm(clipped_grads)  # 得到本次实际用于更新的梯度总长度。
    assert clipped_norm <= max_norm + 1e-6, "裁剪后的梯度超过阈值"  # 为 float32 数组的舍入误差保留容差。
    for name in model.params:  # 使用同一批计算出的完整裁剪梯度更新五组参数。
        model.params[name] -= learning_rate * clipped_grads[name]  # 每组参数都减去学习率乘对应梯度。
    assert all(np.isfinite(param).all() for param in model.params.values()), "更新后参数出现异常"  # 确认更新结果仍是有限数值。
    assert any(not np.array_equal(param, params_before[name]) for name, param in model.params.items()), "本次更新没有改变任何参数"  # 确认真实执行了一次参数更新。
    loss_after, _ = network.cross_entropy(model.forward(x), y)  # 使用更新后的参数，对同一个批次重新前向计算损失。
    assert np.isfinite(loss_after) and loss_after < loss_before, "本例单批更新没有降低损失"  # 验证当前数据、种子和学习率的具体结果。

    print("真实批次来源首行：", batch["chapter_ids"][0], "起点：", batch["starts"][0])  # 显示第一行窗口的章节和零起点。
    print("真实批次 X/Y 形状：", x.shape, y.shape)  # 应为两个 (16,32)。
    print("本批预测位置数：", y.size)  # 应为 16*32=512。
    print("首行输入前八字：", repr("".join(data.tokens[index] for index in x[0, :8])))  # 将前八个输入编号还原成文字。
    print("首行目标前八字：", repr("".join(data.tokens[index] for index in y[0, :8])))  # 展示向后错开一字的目标。
    print("真实批次原始梯度范数：", raw_norm)  # 显示本批完整梯度的总长度。
    print("真实批次裁剪比例：", scale)  # 没超过阈值时比例为 1，超过时小于 1。
    print("真实批次裁剪后范数：", clipped_norm)  # 应不超过阈值 1，允许浮点舍入误差。
    print("真实批次更新前后损失：", loss_before, loss_after)  # 两个数都在同一个训练批次上计算。
    print("真实语料单批训练更新检查：通过")  # 所有检查通过后显示。


if __name__ == "__main__":  # 只有直接运行新脚本时才开始本次演示。
    main()  # 运行一次真实批次更新。