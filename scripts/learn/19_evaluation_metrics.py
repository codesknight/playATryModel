from importlib import import_module  # 复用已有数据、模型和字频工具。
from pathlib import Path  # 根据当前脚本位置找到项目目录。
import json  # 读取上一课记录，并保存新的评估报告。
import numpy as np  # 计算预测编号、自然指数和统计指标。

ROOT = Path(__file__).resolve().parents[2]  # 本文件位于 scripts/learn，向上两级得到根目录。
network = import_module("07_tiny_rnn")  # 使用已经完成的模型结构与交叉熵函数。
training = import_module("11_train_steps")  # 复用检查点加载函数。
baseline = import_module("14_frequency_baseline")  # 复用训练集计数的平滑概率函数。


def summarize(loss_sum, correct, positions):  # 将全体位置的累计值转换为评估指标。
    assert positions > 0 and 0 <= correct <= positions, "评估数量不合法"  # 避免空数据及错误命中数。
    loss = loss_sum / positions  # 先求所有位置的平均交叉熵。
    perplexity = float(np.exp(loss))  # 对平均交叉熵取自然指数，得到困惑度。
    assert np.isfinite(loss) and loss >= 0 and np.isfinite(perplexity), "指标不是有效有限数"  # 发现无效计算时及时停止。
    return {"loss": loss, "perplexity": perplexity, "correct": correct, "accuracy": correct / positions}  # 准确率保存为零到一之间的小数。

def main():  # 在相同完整验证位置比较 RNN 和字频基线。
    data = import_module("06_make_windows")  # 复用既有词表与窗口，首次导入会打印旧数据检查。
    source = json.loads((ROOT / "docs/training_runs/lesson33.json").read_text(encoding="utf-8"))  # 读取验证选出的最佳检查点信息。
    assert source["status"] == "completed" and source["train_sha256"] == data.vocab["train_sha256"], "起点或数据版本不正确"  # 防止不同实验数据混用。
    model, vocab, step = training.load_checkpoint(ROOT / "model" / source["best_checkpoint"], network)  # 加载已完成训练的最佳模型。
    assert vocab == data.vocab, "模型词表与窗口编码不一致"  # 相同编号必须对应相同字符。
    saved = {name: value.copy() for name, value in model.params.items()}  # 评估结束后检查参数未变。
    counts = np.zeros(model.vocab_size, dtype=np.int64)  # 给每个目标编号准备训练计数。
    for batch in data.iter_batches(data.iter_windows(data.train_records, 32, 32), 16):  # 只从训练窗口统计目标字符。
        counts += np.bincount(batch["y"].reshape(-1), minlength=model.vocab_size)  # 累加每个编号的实际出现次数。
    assert int(counts.sum()) == 2450688, "训练目标计数范围不一致"  # 与之前字频基线使用相同目标集合。
    probabilities = baseline.smoothed_probabilities(counts, alpha=1.0)  # 沿用预先固定的加一平滑。
    log_probabilities = np.log(probabilities)  # 字频分布不随输入变化，只需计算一次对数。
    frequency_prediction = int(np.argmax(probabilities))  # 字频基线在每个位置都选同一个最高频编号。
    records = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 验证文本只用于测量。
    rnn_loss_sum, frequency_loss_sum = 0.0, 0.0  # 分别累计两个模型的逐位置损失。
    rnn_correct, frequency_correct, positions = 0, 0, 0  # 命中数与位置数都按实际数量统计。

    for batch in data.iter_batches(data.iter_windows(records, 32, 32), 16):  # 两种预测在完全相同的批次上评估。
        x, y = batch["x"], batch["y"]  # 读取输入及每个位置正确的下一字编号。
        logits = model.forward(x)  # 只前向计算，不求梯度或更新参数。
        _, position_losses = network.cross_entropy(logits, y)  # 保留每个预测位置的交叉熵。
        predictions = np.argmax(logits, axis=-1)  # 对词表维度取最高分，得到与 y 相同的二维形状。
        rnn_loss_sum += float(position_losses.sum(dtype=np.float64))  # 双精度累计逐位损失，避免把尾批等权处理。
        frequency_loss_sum += float((-log_probabilities[y]).sum(dtype=np.float64))  # 同样统计字频基线正确答案的负对数概率。
        rnn_correct += int(np.count_nonzero(predictions == y))  # 统计模型最高分编号等于正确答案的数量。
        frequency_correct += int(np.count_nonzero(y == frequency_prediction))  # 统计字频基线的命中位置数量。
        positions += y.size  # 尾批只有实际存在的位置参与评估。
    assert positions == 316960, "完整验证位置数不正确"  # 本课仍使用相同完整验证窗口集。
    rnn_metrics = summarize(rnn_loss_sum, rnn_correct, positions)  # 先汇总全部位置，再计算困惑度。
    frequency_metrics = summarize(frequency_loss_sum, frequency_correct, positions)  # 对基线使用完全相同的公式。
    assert np.isclose(rnn_metrics["loss"], source["best_validation_loss"], rtol=1e-6, atol=1e-6), "RNN 损失与旧记录不一致"  # 累计精度不同可能产生微小浮点差。
    assert np.isclose(frequency_metrics["loss"], source["full_frequency_loss"], rtol=1e-6, atol=1e-6), "字频损失与旧记录不一致"  # 确认概率拟合和评估范围没有改变。
    assert all(np.array_equal(value, saved[name]) for name, value in model.params.items()), "评估改变了参数"  # 本课全程固定模型。
    report = {  # 保存可复查的评估条件、命中数及指标。
        "checkpoint": source["best_checkpoint"], "completed_steps": step, "train_sha256": vocab["train_sha256"],  # 记录来源。
        "config": {"batch_size": 16, "seq_length": 32, "frequency_alpha": 1.0, "log_base": "e", "accuracy": "top1_all_encoded_positions"},  # 明确自然对数及全部编码位置的准确率。
        "positions": positions, "frequency_prediction_token": vocab["tokens"][frequency_prediction],  # 保留分母和基线恒定预测字符。
        "rnn": rnn_metrics, "frequency": frequency_metrics,  # 两组结果使用相同词表和验证范围。
    }  # 未知编号、标点和换行也计入本次全部位置评估。
    for name in ("rnn", "frequency"):  # 逐项显示两个模型的结果。
        metrics = report[name]  # 当前模型的一组指标。
        print(f"{name} loss={metrics['loss']:.6f} perplexity={metrics['perplexity']:.3f} accuracy={metrics['accuracy']:.2%} correct={metrics['correct']}/{positions}", flush=True)  # 准确率仅在显示时转成百分比。
    print("同范围损失、困惑度、准确率与参数保留检查：通过", flush=True)  # 指标正确不代表生成文字已经流畅。
    return report  # 返回结果，便于检查工具只读复算。

if __name__ == "__main__":  # 直接运行脚本时才保存本课报告。
    result = main()  # 只评估已有模型，不重新训练。
    path = ROOT / "docs/training_runs/lesson35.json"  # 本课使用独立的评估记录。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存 JSON，保留中文字符。
    print("评估报告：", path)  # 输出结果文件位置。
