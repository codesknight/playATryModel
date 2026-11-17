from importlib import import_module  # 复用已有数据、训练、生成和字频基线代码。
from itertools import islice  # 只缓存中途评估用的固定验证前三批。
from time import perf_counter  # 测量训练与周期检查点所需的实际时间。
import json  # 保存本轮配置、覆盖统计和评估记录。
import random  # 用固定种子打乱训练章节顺序。
import numpy as np  # 累加目标计数，并检查训练数值。

network = import_module("07_tiny_rnn")  # 复用完整 RNN 和全局梯度裁剪。
training = import_module("11_train_steps")  # 复用固定参数评估与权重保存。
decoding = import_module("12_generate_text")  # 复用相同开头下的贪心和采样生成。
baseline = import_module("14_frequency_baseline")  # 复用加一平滑和字频损失，不执行旧诊断入口。

def record_checkpoint(model, vocab, batches, step, output_dir):  # 测量固定子集、生成样例并保存本步权重。
    loss, positions = training.evaluate(model, batches, network)  # 参数固定，只评估此前同一份验证子集。
    greedy, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32)  # 固定开头与贪心设置。
    sampled, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32, temperature=1.0, seed=42)  # 固定采样温度和种子。
    path = output_dir / f"lesson32_step{step:04d}.npz"  # 用本课独立名称保存，避免覆盖旧实验。
    training.save_checkpoint(path, model, vocab, step)  # 保存完整五组参数和配套词表。
    print(f"检查点 step={step} 固定验证子集损失={loss:.6f}", flush=True)  # 长训练时立即显示检查点进度。
    print("贪心：", repr(greedy), flush=True)  # 原样记录生成质量，包括重复。
    print("采样：", repr(sampled), flush=True)  # 不只挑选看起来好的样例。
    return {"step": step, "validation_loss": loss, "validation_positions": positions, "greedy": greedy, "sampled": sampled, "checkpoint": path.name}  # 返回本次检查点记录。


def main():  # 从初始模型开始，完整遍历一次四书训练窗口集。
    data = import_module("06_make_windows")  # 复用训练章节、词表与按章窗口，首次导入仍显示旧检查。
    batch_size, seq_length = 16, 32  # 保持每批最多十六个窗口、每窗口三十二个预测位置。
    learning_rate, max_norm = 0.1, 1.0  # 与两百步实验保持相同更新设置。
    records = list(data.train_records)  # 复制章节列表，不改动磁盘数据。
    random.Random(42).shuffle(records)  # 沿用相同的固定章节打乱顺序。
    expected_windows = sum(max(0, (len(record["text"]) - seq_length - 1) // seq_length + 1) for record in records)  # 按相同窗口规则计算整轮样本数。
    expected_steps = (expected_windows + batch_size - 1) // batch_size  # 向上取整，最后的小批次也更新一次。
    with (data.ROOT / "data" / "splits" / "validation.jsonl").open(encoding="utf-8") as file:  # 读取独立验证章节，不用于参数更新。
        validation_records = [json.loads(line) for line in file]  # 保留验证章节顺序。
    subset = list(islice(data.iter_batches(data.iter_windows(validation_records, seq_length, seq_length), batch_size), 3))  # 中途仍只测同样的前三批。
    assert len(subset) == 3, "固定验证子集不足三批"  # 不允许悄悄缩小原有比较范围。
    model = network.TinyRNN(len(data.tokens), hidden_size=64, seed=42)  # 新建一次模型，从头训练完整一轮，不加载单窗口诊断权重。
    output_dir = data.ROOT / "model"  # 本课检查点使用独立文件名保存。
    history = [record_checkpoint(model, data.vocab, subset, 0, output_dir)]  # 记录尚未更新的零步基线。
    checkpoints = {200, 1000, 3000, expected_steps}  # 按预先固定的步数记录参数快照。
    chapter_to_book = {record["id"]: record["book"] for record in records}  # 用章节标识追踪四书训练覆盖。
    book_windows = {record["book"]: 0 for record in records}  # 为每本书累计实际训练窗口数。
    full_counts = np.zeros(len(data.tokens), dtype=np.int64)  # 只用本轮训练目标拟合全训练集字频基线。
    seen_windows, seen_positions, completed_steps, last_batch_size = 0, 0, 0, 0  # 初始化整轮覆盖统计。
    loss_sum = 0.0  # 累计各批更新前的损失总和。
    batches = data.iter_batches(data.iter_windows(records, seq_length, seq_length), batch_size)  # 这次不再用 islice 截断训练批次流。
    print("计划整轮窗口数与更新步数：", expected_windows, expected_steps, flush=True)  # 当前应为 76584 和 4787。
    started = perf_counter()  # 开始计时，后续耗时包含周期检查点的评估、生成和保存。

    for step, batch in enumerate(batches, start=1):  # 直到训练窗口生成器自然结束，完整处理一次。
        x, y = batch["x"], batch["y"]  # 读取本批实际行数，最后一批只有八行。
        loss, raw_grads = model.loss_and_grad(x, y)  # 梯度按本批实际位置数平均，不假定始终有 512 个位置。
        assert np.isfinite(loss), "训练损失异常"  # 出现无效数值时停止。
        grads, raw_norm, scale = network.clip_gradients(raw_grads, max_norm)  # 完整反向后执行一次全局裁剪。
        for name in model.params:  # 使用当前批次的完整梯度更新五组参数。
            model.params[name] -= learning_rate * grads[name]  # 下一批沿用更新后的参数。
        assert all(np.isfinite(param).all() for param in model.params.values()), "更新后的参数异常"  # 防止无效权重继续传播。
        full_counts += np.bincount(y.reshape(-1), minlength=len(data.tokens))  # 同时累计全轮训练目标的真实字频。
        seen_windows += x.shape[0]  # 尾批按实际八行统计，不补成十六行。
        seen_positions += y.size  # 累计实际预测位置数。
        loss_sum += loss * y.size  # 按位置数加权，避免尾批影响平均方式。
        completed_steps, last_batch_size = step, x.shape[0]  # 记录最新步数与最后批次大小。
        for chapter_id in batch["chapter_ids"]:  # 将每行训练窗口归入对应书籍。
            book_windows[chapter_to_book[chapter_id]] += 1  # 累计四本书实际提供的数据量。
        if step % 250 == 0 or step == expected_steps:  # 定期显示进度，最后一步也显示。
            print(f"训练 step={step}/{expected_steps} 批次损失={loss:.6f} grad_norm={raw_norm:.6f} scale={scale:.6f}", flush=True)  # 不要求不同批次损失逐行下降。
        if step in checkpoints:  # 在预定检查点记录同一固定验证子集。
            history.append(record_checkpoint(model, data.vocab, subset, step, output_dir))  # 保存当前阶段结果，不用验证梯度训练。
    elapsed = perf_counter() - started  # 结束本轮训练与周期检查点计时，尚未包含末尾完整验证。
    assert completed_steps == expected_steps and seen_windows == expected_windows, "完整一轮覆盖不正确"  # 核对没有提前停止或错误多算窗口。
    assert seen_positions == expected_windows * seq_length == int(full_counts.sum()), "训练目标计数不一致"  # 窗口统计与基线计数应覆盖相同位置。
    assert sum(book_windows.values()) == seen_windows and all(count > 0 for count in book_windows.values()), "四书覆盖不正确"  # 确认所有书籍都实际参与训练。

    frequency_probs = baseline.smoothed_probabilities(full_counts, alpha=1.0)  # 只根据整轮训练目标估计字频，不使用验证答案。
    subset_frequency, subset_positions = baseline.frequency_loss(frequency_probs, subset)  # 在原固定子集上重算对应的全训练集字频分数。
    assert subset_positions == history[-1]["validation_positions"], "子集比较范围不一致"  # RNN 和字频必须比较同一位置集合。
    print("开始完整验证窗口集评估（只前向，不更新参数）", flush=True)  # 提醒接下来还有完整验证计算。
    validation_batches = data.iter_batches(data.iter_windows(validation_records, seq_length, seq_length), batch_size)  # 完整验证窗口按批流式提供，不全部放进内存。
    full_rnn_loss, validation_positions = training.evaluate(model, validation_batches, network)  # 固定最终模型参数，遍历完整验证窗口集。
    validation_batches = data.iter_batches(data.iter_windows(validation_records, seq_length, seq_length), batch_size)  # 新建生成器，避免复用已经耗尽的验证流。
    full_frequency_loss, frequency_positions = baseline.frequency_loss(frequency_probs, validation_batches)  # 使用同一完整验证范围评估字频基线。
    expected_validation_positions = sum(max(0, (len(record["text"]) - seq_length - 1) // seq_length + 1) for record in validation_records) * seq_length  # 按既定丢尾规则独立核对完整验证位置数。
    assert validation_positions == frequency_positions == expected_validation_positions > 0, "完整验证范围不一致"  # 两种模型必须对相同目标评估。
    print("固定验证子集 RNN/字频损失：", history[-1]["validation_loss"], subset_frequency, flush=True)  # 子集只与子集比较。
    print("完整验证窗口集 RNN/字频损失：", full_rnn_loss, full_frequency_loss, flush=True)  # 完整范围只与完整范围比较。
    print("完整验证窗口集预测位置数：", validation_positions, flush=True)  # 该数字不等于原文字符数，首字符和短尾未成为目标。

    report = {  # 记录本轮实际覆盖、固定子集历史与完整验证结果。
        "config": {  # 固定完整一轮训练与生成比较条件。
            "epochs": 1, "batch_size": batch_size, "seq_length": seq_length,  # 记录遍历轮数与批次尺寸。
            "learning_rate": learning_rate, "max_norm": max_norm, "model_seed": 42, "chapter_shuffle_seed": 42,  # 记录更新规则和随机顺序。
            "prompt": "春风", "new_tokens": 40, "temperature": 1.0, "sampling_seed": 42,  # 记录固定生成设置。
            "validation_batches": 3, "frequency_alpha": 1.0,  # 标明中途子集范围和字频平滑。
        },  # 完整配置记录结束。
        "train_sha256": data.vocab["train_sha256"], "expected_steps": expected_steps, "completed_steps": completed_steps,  # 保存数据来源与整轮步数。
        "seen_windows": seen_windows, "seen_positions": seen_positions, "last_batch_size": last_batch_size, "book_windows": book_windows,  # 保存真实覆盖和尾批统计。
        "training_process_loss": loss_sum / seen_positions, "training_and_checkpoint_seconds": elapsed,  # 这是参数不断变化的训练过程均值及含检查点的耗时。
        "history": history, "subset_frequency_loss": subset_frequency,  # 子集指标单独保存，不与完整验证混比。
        "full_validation": {"positions": validation_positions, "rnn_loss": full_rnn_loss, "frequency_loss": full_frequency_loss},  # 两个模型的完整验证比较。
    }  # 不加入验证损失必须下降或 RNN 必须胜出的断言。
    report_path = data.ROOT / "docs" / "training_runs" / "lesson32.json"  # 保存独立报告，不覆盖以前课程记录。
    report_path.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 完整训练与验证完成后保存报告。
    print("四本书实际训练窗口数：", book_windows, flush=True)  # 显示本轮实际覆盖。
    print("整轮步数/窗口数/预测位置数/尾批行数：", completed_steps, seen_windows, seen_positions, last_batch_size, flush=True)  # 当前应为 4787、76584、2450688、8。
    print("训练过程按位置加权平均损失：", report["training_process_loss"], flush=True)  # 不把训练过程均值当作固定最终模型的训练集评估。
    print("训练与周期检查点耗时秒：", elapsed, flush=True)  # 不包含末尾完整验证，也不是纯矩阵计算耗时。
    print("报告路径：", report_path, flush=True)  # 保存的权重可独立加载，但当前脚本尚不自动断点续训。
    print("四书完整一轮、尾批与同范围字频比较检查：通过", flush=True)  # 说明流程与覆盖正确，不保证文字质量达标。


if __name__ == "__main__":  # 直接运行文件时才开始本课完整训练。
    main()  # 从初始模型完整训练一轮并评估。