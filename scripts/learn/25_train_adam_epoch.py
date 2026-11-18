from importlib import import_module  # 复用已经完成的Adam和评估工具。
from pathlib import Path  # 定位项目及独立输出文件。
from time import perf_counter  # 记录含训练、验证及保存的总时间。
import hashlib  # 检查旧模型文件没有被覆盖。
import json  # 读取实验来源并保存报告。
import random  # 固定这一轮的训练章节顺序。
import numpy as np  # 检查参数、损失及前缀复现。

ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于scripts/learn。
adam_tools = import_module("24_adam_experiment")  # 复用Adam类与完整验证和固定样例评估。
network = adam_tools.network  # 保持同一RNN结构。
training = adam_tools.training  # 复用模型保存和加载。
epoch_tools = adam_tools.epoch_tools  # 复用报告保存工具。
LEARNING_RATE = 0.001  # 沿用上一课Adam分支的学习率。

def record_checkpoint(model, vocab, optimizer, data, validation, local_step, start_step, output_dir):  # 保存并评估本轮指定位置。
    assert optimizer.t == local_step, "Adam历史计数不匹配"  # 验证期间不能重建优化器或额外更新。
    row = adam_tools.measure(model, vocab, f"adam_step{local_step:04d}", data, validation)  # 完整验证并保留六段固定生成样例。
    path = output_dir / f"lesson41_step{local_step:04d}.npz"  # 文件名记录本轮局部步数。
    training.save_checkpoint(path, model, vocab, start_step + local_step)  # 权重元数据记录模型累计更新次数。
    assert optimizer.t == local_step, "评估改变了Adam计数"  # 只做前向评估不应更新优化器。
    row.update({"local_step": local_step, "global_step": start_step + local_step, "adam_steps": optimizer.t, "checkpoint": path.name})  # 三种计数的含义分别记录。
    return row  # 不保存m和v，当前检查点用于评估及模型选择。

def main(output_dir=None, report_path=None):  # 完整训练一轮，默认保存正式产物。
    started = perf_counter()  # 耗时包含起点评估、训练、周期验证、生成和保存。
    data = import_module("06_make_windows")  # 沿用原四书训练窗口和词表。
    source = json.loads((ROOT / "docs/training_runs/lesson37.json").read_text(encoding="utf-8"))  # 从完整第四轮最佳模型开始。
    short = json.loads((ROOT / "docs/training_runs/lesson40.json").read_text(encoding="utf-8"))  # 用上一课Adam短分支核对前缀。
    assert source["status"] == "completed" and source["train_sha256"] == short["train_sha256"] == data.vocab["train_sha256"], "来源或数据版本不一致"  # 新语料不自动混入本次对照。
    source_path = ROOT / "model" / source["best_checkpoint"]  # 不加载已经额外训练两百步的Adam分支。
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()  # 留下共同起点校验值。
    assert source_hash == short["starting_sha256"] and short["config"]["adam_lr"] == LEARNING_RATE and short["config"]["chapter_shuffle_seed"] == 46, "短实验条件不一致"  # 前缀复现需要同一起点及设置。
    model, vocab, start_step = training.load_checkpoint(source_path, network)  # 训练模型只加载一次。
    assert vocab == data.vocab and start_step == 19148, "起点模型配置错误"  # 原模型已经训练四轮。
    optimizer = adam_tools.Adam(model.params, learning_rate=LEARNING_RATE)  # 新建一次Adam，m/v为零且t为零，随后全程复用。
    output_dir = ROOT / "model" if output_dir is None else Path(output_dir)  # 老师预跑时可改用临时目录。
    report_path = ROOT / "docs/training_runs/lesson41.json" if report_path is None else Path(report_path)  # 独立报告不覆盖旧实验。
    best_path = output_dir / "lesson41_best.npz"  # 独立最佳权重快照。
    records = list(data.train_records)  # 从相同原始章节顺序复制。
    random.Random(46).shuffle(records)  # 与上一课短实验使用相同顺序。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 仍使用固定完整验证集。
    expected_books = {}  # 独立根据章节长度计算整轮规模。
    for record in records:  # 统计应有的章节内窗口。
        count = max(0, (len(record["text"]) - 33) // 32 + 1)  # 三十三个字符形成三十二个错位目标。
        expected_books[record["book"]] = expected_books.get(record["book"], 0) + count  # 汇总各书窗口数量。
    expected_windows = sum(expected_books.values())  # 得到完整训练窗口数。
    expected_steps = (expected_windows + 15) // 16  # 保留不足十六行的尾批。
    initial = adam_tools.measure(model, vocab, "starting_model", data, validation)  # 起点也参加候选比较。
    assert np.isclose(initial["validation_loss"], source["best_validation_loss"], rtol=1e-6, atol=1e-6), "起点验证不一致"  # 防止错误权重或验证范围。
    initial.update({"local_step": 0, "global_step": start_step, "adam_steps": 0, "checkpoint": source_path.name})  # 旧模型累计次数不等于新Adam历史次数。
    history, best_row = [initial], initial  # 保存所有预定候选并先保留旧模型。
    training.save_checkpoint(best_path, model, vocab, start_step)  # 后续都变差时仍有正确的最佳文件。
    report = {  # 保存运行状态、配置和完整候选轨迹。
        "status": "running", "starting_checkpoint": source_path.name, "starting_sha256": source_hash, "train_sha256": vocab["train_sha256"],  # 记录起点来源。
        "config": {"epoch": 5, "learning_rate": LEARNING_RATE, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "batch_size": 16, "seq_length": 32, "max_norm": 1.0, "chapter_shuffle_seed": 46},  # 全轮使用预先规定条件。
        "history": history, "completed_local_steps": 0, "best_local_step": 0, "best_validation_loss": initial["validation_loss"], "best_checkpoint": best_path.name,  # 记录当前选优结果。
        "optimizer_state_saved": False,  # 目前文件仅保存模型权重，不能精确恢复中断的Adam。
    }  # 不使用测试集，不启用自动早停。
    epoch_tools.save_report(report_path, report)  # 先保留起点状态。
    chapter_to_book = {record["id"]: record["book"] for record in records}  # 窗口来源映射到书籍。
    book_windows = {book: 0 for book in expected_books}  # 训练覆盖从零累计。
    seen_windows, seen_positions, loss_sum = 0, 0, 0.0  # 分别累计窗口、预测位置和训练过程损失。
    batches = data.iter_batches(data.iter_windows(records, 32, 32), 16)  # 使用完整生成器，不限制为两百批。


    for local_step, batch in enumerate(batches, start=1):  # 完整遍历这一轮的训练窗口。
        x, y = batch["x"], batch["y"]  # 读取当前批次实际数据。
        loss, raw_grads = model.loss_and_grad(x, y)  # 使用当前模型求完整梯度。
        assert np.isfinite(loss), "训练损失无效"  # 无效值不能继续更新。
        grads, _, _ = network.clip_gradients(raw_grads, 1.0)  # 所有参数的梯度统一裁剪。
        optimizer.step(model.params, grads)  # 始终使用循环外创建的同一个Adam对象。
        assert optimizer.t == local_step and all(np.isfinite(value).all() for value in model.params.values()), "Adam计数或参数异常"  # 检查历史未被重置及参数有限。
        seen_windows += x.shape[0]  # 尾批按实际行数计入。
        seen_positions += y.size  # 每个实际预测位置都计入。
        loss_sum += loss * y.size  # 训练过程损失按实际位置加权。
        for chapter_id in batch["chapter_ids"]:  # 按行核对来源覆盖。
            book_windows[chapter_to_book[chapter_id]] += 1  # 每本书的窗口都应完整访问。
        if local_step % 500 == 0:  # 定期显示训练进度。
            print(f"Adam local_step={local_step}/{expected_steps} global_step={start_step+local_step} t={optimizer.t} batch_loss={loss:.6f}", flush=True)  # 同时显示局部、累计和Adam历史计数。
        if local_step in (200, 1000, 3000, expected_steps):  # 只在预先规定的位置做完整验证。
            row = record_checkpoint(model, vocab, optimizer, data, validation, local_step, start_step, output_dir)  # 评估后继续当前状态，不重建优化器。
            if local_step == 200:  # 验证短实验前缀得到复现。
                previous = next(item for item in short["history"] if item["label"] == "adam")  # 取得上一课Adam分支记录。
                previous_model, previous_vocab, previous_step = training.load_checkpoint(ROOT / "model" / previous["checkpoint"], network)  # 只加载作比较，不替换当前训练模型。
                assert previous_vocab == vocab and previous_step == start_step + local_step and np.isclose(row["validation_loss"], previous["validation_loss"], rtol=1e-6, atol=1e-6), "短实验前缀指标不一致"  # 起点和前两百批条件相同。
                assert all(np.allclose(value, previous_model.params[name], rtol=1e-6, atol=1e-6) for name, value in model.params.items()), "短实验前缀参数不一致"  # 同时核对实际五组参数。
            history.append(row)  # 保留每个候选结果，无论好坏。
            if row["validation_loss"] < best_row["validation_loss"]:  # 只有严格更低才替换最佳快照。
                best_row = row  # 最佳不一定是最新检查点。
                training.save_checkpoint(best_path, model, vocab, row["global_step"])  # 只保存副本，不把训练模型退回旧状态。
            report.update({"completed_local_steps": local_step, "best_local_step": best_row["local_step"], "best_validation_loss": best_row["validation_loss"]})  # 同步报告进度。
            epoch_tools.save_report(report_path, report)  # 每个检查点后保存一次。

    assert local_step == expected_steps == optimizer.t == 4787 and seen_windows == expected_windows == 76584, "整轮更新或窗口数量错误"  # Adam历史应持续覆盖整轮。
    assert seen_positions == 2450688 and x.shape[0] == 8 and book_windows == expected_books, "目标、尾批或书籍覆盖不一致"  # 不遗漏小尾批或某一本书。
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash, "旧起点权重被覆盖"  # 旧模型保留。
    selected = min(history, key=lambda item: item["validation_loss"])  # 独立核对包括起点的最低分。
    assert selected["local_step"] == best_row["local_step"], "最佳候选错误"  # 不把最后一步自动视为最好。
    best_model, best_vocab, best_step = training.load_checkpoint(best_path, network)  # 训练结束后加载实际最佳文件。
    selected_path = source_path if selected["local_step"] == 0 else output_dir / selected["checkpoint"]  # 被选中的可以是旧模型。
    candidate, candidate_vocab, candidate_step = training.load_checkpoint(selected_path, network)  # 加载对应候选作参数对照。
    assert best_vocab == candidate_vocab == vocab and best_step == candidate_step == selected["global_step"] and all(np.array_equal(value, candidate.params[name]) for name, value in best_model.params.items()), "最佳文件与所选候选不一致"  # 保存内容必须与报告对应。
    state_bytes = sum(value.nbytes for group in (optimizer.m, optimizer.v) for value in group.values())  # 统计全程保留的m/v数组额外内存。
    report.update({  # 全部检查完成后记录整轮结果。
        "status": "completed", "completed_global_steps": start_step + local_step,  # 记录完成状态和累计步数。
        "adam_steps": optimizer.t,  # Adam本次从零开始积累的历史次数。
        "seen_windows": seen_windows, "seen_positions": seen_positions,  # 记录训练覆盖规模。
        "last_batch_size": x.shape[0], "book_windows": book_windows,  # 保留尾批和各书覆盖。
        "training_process_loss": loss_sum / seen_positions,  # 计算按位置加权的训练过程损失。
        "adam_state_bytes": state_bytes,  # 只统计m/v数组的额外内存。
        "run_seconds": perf_counter() - started,  # 包含验证、生成和保存，不是纯训练耗时。
    })  # 不把实际完成位置与选中的最佳位置混淆。
    epoch_tools.save_report(report_path, report)  # 全部核对结束后标记完成。
    print("本轮、累计、Adam历史步数：", local_step, report["completed_global_steps"], optimizer.t, flush=True)  # 预期为4787、23935、4787。
    print("最佳局部步数及完整验证损失：", best_row["local_step"], best_row["validation_loss"], flush=True)  # 最佳来自预定五个候选。
    print("Adam整轮覆盖、前缀复现与最佳参数检查：通过", flush=True)  # 不把验证损失当作句子流畅度评分。
    return report  # 返回本轮完整记录。

if __name__ == "__main__":  # 直接运行时才训练并保存正式产物。
    main()  # 默认写入model和docs/training_runs。
