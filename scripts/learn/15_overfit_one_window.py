from importlib import import_module  # 复用数据、模型、梯度裁剪、保存和生成函数。
import json  # 保存单窗口拟合的配置、指标与生成结果。
import numpy as np  # 检查标签、计算逐位置预测正确数。

network = import_module("07_tiny_rnn")  # 导入完整模型与裁剪工具，不执行旧演示。
training = import_module("11_train_steps")  # 复用权重保存与加载，不执行旧训练入口。
decoding = import_module("12_generate_text")  # 复用逐字生成，不执行旧生成演示。


def measure_fit(model, x, y, step):  # 在同一个训练窗口上测量当前拟合程度。
    logits = model.forward(x)  # 用当前参数计算该窗口各位置的下一字分数。
    loss, _ = network.cross_entropy(logits, y)  # 计算这个训练窗口的平均交叉熵，不是验证损失。
    correct = int((np.argmax(logits, axis=-1) == y).sum())  # 每个位置取最高分编号，与真实下一字比较。
    print(f"step={step} 固定训练窗口损失={loss:.6f} 正确位置={correct}/{y.size}")  # 日志比较的是同一组输入和目标。
    return {"step": step, "loss": loss, "correct": correct, "positions": int(y.size)}  # 记录拟合过程，供之后复查。


def main():  # 用一个真实窗口进行小样本拟合诊断。
    data = import_module("06_make_windows")  # 复用训练集首窗口和固定词表，首次导入仍会显示旧数据检查。
    seq_length = 32  # 只学习一个包含 32 个预测位置的窗口。
    max_steps = 300  # 对同一窗口反复更新三百次。
    learning_rate = 0.1  # 使用当前普通梯度下降学习率。
    max_norm = 1.0  # 保留全局梯度裁剪。
    batch = next(data.iter_batches(data.iter_windows(data.train_records, seq_length, seq_length), 1))  # 只取一个窗口，此后不再获取新批次。
    x, y = batch["x"], batch["y"]  # 固定输入与向后错位一字的目标。
    assert x.shape == y.shape == (1, seq_length) and np.array_equal(x[:, 1:], y[:, :-1]), "诊断窗口格式错误"  # 确认任务仍是下一字预测。
    saved_x, saved_y = x.copy(), y.copy()  # 保存数据副本，检查三百步使用的是同一份样本。
    model = network.TinyRNN(len(data.tokens), hidden_size=64, seed=42)  # 新建独立诊断模型，不修改上一课的两百步模型。
    history = [measure_fit(model, x, y, 0)]  # 训练前记录同一窗口的基线。
    for step in range(1, max_steps + 1):  # 每步仍然使用相同的 x 和 y。
        loss, raw_grads = model.loss_and_grad(x, y)  # 在固定训练窗口上求完整梯度。
        assert np.isfinite(loss), "训练损失异常"  # 无效数值不能继续用于更新。
        grads, _, _ = network.clip_gradients(raw_grads, max_norm)  # 对全部共享参数梯度做一次全局裁剪。
        for name in model.params:  # 完整反向结束后统一更新五组参数。
            model.params[name] -= learning_rate * grads[name]  # 新参数继续用于下一次相同样本的学习。
        assert all(np.isfinite(param).all() for param in model.params.values()), "参数更新异常"  # 确认拟合过程数值有限。
        if step in (20, 100, 300):  # 在预先指定的步数测量同窗口更新后的结果。
            history.append(measure_fit(model, x, y, step))  # 既记录损失，也记录正确位置数。
    assert np.array_equal(x, saved_x) and np.array_equal(y, saved_y), "训练样本被意外改变"  # 防止误以为固定样本，实际却改了数据。
    assert history[-1]["loss"] < history[0]["loss"] * 0.1 and history[-1]["correct"] == y.size, "本例单窗口拟合未达到预期"  # 对当前配置检查损失显著下降且三十二个位置全对。
    prompt = data.tokens[int(x[0, 0])]  # 只给这个训练窗口的第一个字符作为生成开头。
    expected_text = prompt + "".join(data.tokens[int(index)] for index in y[0])  # 拼出该窗口对应的完整三十三字训练片段。
    generated_text, _ = decoding.generate(model, data.vocab, prompt, new_tokens=seq_length, context_size=seq_length)  # 不提供后续正确字符，让模型贪心生成三十二步。
    assert generated_text == expected_text, "本例尚不能从首字复现训练片段"  # 这只检查已训练片段的记忆，不证明泛化能力。
    checkpoint_path = data.ROOT / "model" / "lesson31_single_window.npz"  # 用独立名称保存诊断权重，不覆盖全语料实验检查点。
    training.save_checkpoint(checkpoint_path, model, data.vocab, max_steps)  # 保存只反复学过一个窗口的模型。
    print("训练片段：", repr(expected_text))  # 保留换行与空格，便于逐字核对。
    print("只给首字后的生成：", repr(generated_text))  # 当前例子应与训练片段相同。
    result = {  # 保存诊断条件与观察，不把训练指标写成验证指标。
        "config": {"steps": max_steps, "seq_length": seq_length, "batch_size": 1, "learning_rate": learning_rate, "max_norm": max_norm, "model_seed": 42},  # 记录固定训练设置。
        "train_sha256": data.vocab["train_sha256"], "chapter_id": batch["chapter_ids"][0], "start": batch["starts"][0],  # 记录唯一训练窗口的来源。
        "input_ids": x.tolist(), "target_ids": y.tolist(), "history": history,  # 保留输入、真实目标和同窗口拟合指标。
        "prompt": prompt, "expected_training_text": expected_text, "generated_training_text": generated_text, "checkpoint": checkpoint_path.name,  # 标明生成结果对应的是已训练片段。
    }  # 本次没有使用验证集或测试集进行拟合。
    report_path = data.ROOT / "docs" / "training_runs" / "lesson31.json"  # 将本课诊断记录与之前实验分开。
    report_path.parent.mkdir(parents=True, exist_ok=True)  # 创建记录目录，已有目录继续使用。
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存中文可读的实验记录。
    print("诊断权重：", checkpoint_path)  # 文件只用于单窗口记忆检查，不能作为泛化已达标的证据。
    print("诊断记录：", report_path)  # 显示可复查的配置与指标文件。
    print("单窗口拟合、数据固定与训练片段复现检查：通过")  # 说明当前小样本诊断通过。


if __name__ == "__main__":  # 直接运行脚本时才执行单窗口实验。
    main()  # 从头训练独立的小样本诊断模型。