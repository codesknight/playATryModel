from importlib import import_module  # 复用模型、权重、验证和生成工具。
from pathlib import Path  # 定位项目文件与实验输出。
from time import perf_counter  # 记录含验证与保存的整次运行时间。
import hashlib  # 检查旧最佳文件未被覆盖。
import json  # 读取来源报告并保存新记录。
import random  # 固定第四轮的章节顺序。
import numpy as np  # 检查数值及重新加载后的参数。

ROOT = Path(__file__).resolve().parents[2]  # 本文件放在 scripts/learn。
network = import_module("07_tiny_rnn")  # 继续使用同一个模型结构。
training = import_module("11_train_steps")  # 复用权重保存与加载。
epoch_tools = import_module("17_train_epochs")  # 复用完整验证与报告保存。
decoding = import_module("12_generate_text")  # 固定生成条件比较文字。
LEARNING_RATE = 0.03  # 前三轮使用0.1，本轮降低为0.03。

def record_checkpoint(model, vocab, data, validation, local_step, global_step, output_dir):  # 测量并保存指定训练位置。
    loss, positions = epoch_tools.evaluate_full(model, data, validation)  # 每次都评估相同完整验证窗口集。
    assert positions == 316960, "验证范围变化"  # 不混用子集指标。
    path = output_dir / f"lesson37_step{local_step:04d}.npz"  # 文件名记录本轮局部步数。
    training.save_checkpoint(path, model, vocab, global_step)  # 文件元数据记录累计步数。
    greedy, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32)  # 保持原来的贪心设置。
    print(f"检查点 local_step={local_step} global_step={global_step} full_validation={loss:.6f}", flush=True)  # 立即显示长训练中的验证结果。
    print("贪心：", repr(greedy), flush=True)  # 如实保留重复和不通顺。
    return {"local_step": local_step, "global_step": global_step, "checkpoint": path.name, "validation_loss": loss, "validation_positions": positions, "greedy": greedy}  # 只保存可序列化的记录。

def main(output_dir=None, report_path=None):  # 默认保存正式产物，老师预跑时可指定临时目录。
    started = perf_counter()  # 运行耗时包括验证、生成及保存。
    data = import_module("06_make_windows")  # 复用固定词表与训练窗口。
    source = json.loads((ROOT / "docs/training_runs/lesson33.json").read_text(encoding="utf-8"))  # 从完整三轮的最佳模型出发。
    short_run = json.loads((ROOT / "docs/training_runs/lesson36.json").read_text(encoding="utf-8"))  # 读取上一课短实验用于核对前两百步。
    assert source["status"] == "completed" and source["best_epoch"] == 3 and source["train_sha256"] == short_run["train_sha256"] == data.vocab["train_sha256"], "起点或数据版本不正确"  # 本课明确从三轮完整模型开始。
    source_path = ROOT / "model" / source["best_checkpoint"]  # 不直接从上一课的两百步分支开始。
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()  # 保留旧文件校验值。
    model, vocab, start_step = training.load_checkpoint(source_path, network)  # 训练模型只在循环之前加载一次。
    assert vocab == data.vocab and start_step == 14361, "起点权重配置不一致"  # 起点已经完成三轮。
    output_dir = ROOT / "model" if output_dir is None else Path(output_dir)  # 正式权重默认放入model。
    report_path = ROOT / "docs/training_runs/lesson37.json" if report_path is None else Path(report_path)  # 正式报告与旧实验分开。
    best_path = output_dir / "lesson37_best.npz"  # 最佳文件是独立快照。
    records = list(data.train_records)  # 从原始章节列表复制。
    random.Random(45).shuffle(records)  # 与上一课短实验相同的第四轮顺序。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 全程使用相同完整验证集。
    windows_per_book = {}  # 独立计算本轮应覆盖的窗口数量。
    for record in records:  # 按章节长度计算，避免依赖已训练计数。
        count = max(0, (len(record["text"]) - 33) // 32 + 1)  # 每个窗口需要三十三字才能产生三十二个目标。
        windows_per_book[record["book"]] = windows_per_book.get(record["book"], 0) + count  # 汇总每本书的窗口。
    expected_windows = sum(windows_per_book.values())  # 得到整轮窗口数。
    expected_steps = (expected_windows + 15) // 16  # 保留尾批，批次数向上取整。
    initial_loss, positions = epoch_tools.evaluate_full(model, data, validation)  # 起点也必须参加选优。
    assert positions == 316960 and np.isclose(initial_loss, source["best_validation_loss"], rtol=1e-6, atol=1e-6), "起点验证不一致"  # 核对加载和评估范围。
    history = [{"local_step": 0, "global_step": start_step, "checkpoint": source_path.name, "validation_loss": initial_loss, "validation_positions": positions}]  # 零步表示本轮尚未更新。
    best_row = history[0]  # 如果后续全部变差，仍保留起点。
    training.save_checkpoint(best_path, model, vocab, start_step)  # 初始化独立最佳文件。
    report = {  # 新报告保留配置和全部已评估候选。
        "status": "running",  # 尚未完成整轮训练及最终核对。
        "starting_checkpoint": source_path.name, "starting_sha256": source_hash,  # 记录起点文件及其校验值。
        "train_sha256": vocab["train_sha256"],  # 记录训练数据版本。
        "config": {"epoch": 4, "learning_rate": LEARNING_RATE, "batch_size": 16, "seq_length": 32, "max_norm": 1.0, "chapter_shuffle_seed": 45},  # 固定本轮条件。
        "history": history, "completed_local_steps": 0,  # 保存已评估记录与当前进度。
        "best_local_step": 0, "best_checkpoint": best_path.name,  # 起点暂时是最佳候选。
    }  # 不修改旧课程报告。
    epoch_tools.save_report(report_path, report)  # 先保存起点状态。
    chapter_to_book = {record["id"]: record["book"] for record in records}  # 用于累计实际训练来源。
    book_windows = {book: 0 for book in windows_per_book}  # 实际覆盖从零开始。
    seen_windows, seen_positions, loss_sum = 0, 0, 0.0  # 窗口、目标及训练过程损失分别统计。
    batches = data.iter_batches(data.iter_windows(records, 32, 32), 16)  # 本轮完整遍历，不只取前两百批。

    for local_step, batch in enumerate(batches, start=1):  # 每个批次更新一次，局部步数从一开始。
        x, y = batch["x"], batch["y"]  # 本批实际输入和目标。
        loss, raw_grads = model.loss_and_grad(x, y)  # 在当前参数上完成前向与反向。
        assert np.isfinite(loss), "训练损失无效"  # 无效数值不能继续传播。
        grads, _, _ = network.clip_gradients(raw_grads, 1.0)  # 与之前相同的全局裁剪。
        for name in model.params:  # 完整反向结束后更新五组参数。
            model.params[name] -= LEARNING_RATE * grads[name]  # 本轮所有步骤都使用0.03。
        assert all(np.isfinite(value).all() for value in model.params.values()), "参数无效"  # 及时发现异常更新。
        seen_windows += x.shape[0]  # 按实际批大小累计窗口数。
        seen_positions += y.size  # 尾批只计实际目标。
        loss_sum += loss * y.size  # 训练过程损失按实际位置加权。
        for chapter_id in batch["chapter_ids"]:  # 各行归属对应书籍。
            book_windows[chapter_to_book[chapter_id]] += 1  # 检查完整四书覆盖。
        if local_step % 500 == 0:  # 每五百步显示训练进度。
            print(f"local_step={local_step}/{expected_steps} global_step={start_step+local_step} batch_loss={loss:.6f}", flush=True)  # 不要求不同批次损失单调下降。
        if local_step in (200, 1000, 3000, expected_steps):  # 只在预先规定的位置评估完整验证集。
            row = record_checkpoint(model, vocab, data, validation, local_step, start_step + local_step, output_dir)  # 当前参数固定后评估保存。
            if local_step == 200:  # 相同起点、数据顺序及学习率应复现上一课前缀指标。
                previous = next(item for item in short_run["history"] if item["label"] == "lr003")  # 取上一课0.03分支。
                assert np.isclose(row["validation_loss"], previous["validation_loss"], rtol=1e-6, atol=1e-6), "前两百步与短实验不一致"  # 评估不更新参数，插入检查不应改变训练轨迹。
            history.append(row)  # 所有预定候选都保留。
            if row["validation_loss"] < best_row["validation_loss"]:  # 只在严格改善时替换最佳文件。
                best_row = row  # 保留对应局部和累计步数。
                training.save_checkpoint(best_path, model, vocab, row["global_step"])  # 独立快照不会随后续训练变化。
            report.update({"completed_local_steps": local_step, "best_local_step": best_row["local_step"], "best_validation_loss": best_row["validation_loss"]})  # 同步当前完整检查点进度。
            epoch_tools.save_report(report_path, report)  # 每个检查点后保存报告。

    assert local_step == expected_steps == 4787 and seen_windows == expected_windows == 76584, "整轮覆盖不正确"  # 本轮必须完整训练。
    assert seen_positions == 2450688 and x.shape[0] == 8 and book_windows == windows_per_book, "目标、尾批或书籍覆盖错误"  # 不能漏掉尾批或某一本书。
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash, "旧起点被覆盖"  # 新实验保留旧文件。
    selected = min(history, key=lambda item: item["validation_loss"])  # 包括起点在内独立核对最低分。
    assert selected["local_step"] == best_row["local_step"], "最佳选择不正确"  # 最后一步未必是最佳一步。
    best_model, best_vocab, best_step = training.load_checkpoint(best_path, network)  # 训练结束后重新加载实际最佳文件。
    selected_path = source_path if selected["local_step"] == 0 else output_dir / selected["checkpoint"]  # 最佳也可能仍是旧起点。
    selected_model, _, selected_step = training.load_checkpoint(selected_path, network)  # 加载候选文件作参数对照。
    assert best_vocab == vocab and best_step == selected_step == selected["global_step"] and all(np.array_equal(value, selected_model.params[name]) for name, value in best_model.params.items()), "最佳文件与所选候选不一致"  # 防止最佳报告和实际文件不匹配。
    report.update({  # 最终确认完成后补充整轮统计。
        "status": "completed", "completed_global_steps": start_step + local_step,  # 记录完成状态和累计步数。
        "seen_windows": seen_windows, "seen_positions": seen_positions,  # 记录实际训练规模。
        "last_batch_size": x.shape[0], "book_windows": book_windows,  # 保存尾批及四书覆盖。
        "training_process_loss": loss_sum / seen_positions,  # 按实际位置计算训练过程平均损失。
        "run_seconds": perf_counter() - started,  # 耗时包含验证、生成及保存。
    })  # 这个耗时不能直接当作纯训练耗时。
    epoch_tools.save_report(report_path, report)  # 全部核对完成后标记实验结束。
    print("本轮及累计更新步数：", local_step, report["completed_global_steps"], flush=True)  # 预期为4787和19148。
    print("最佳局部步数及完整验证损失：", best_row["local_step"], best_row["validation_loss"], flush=True)  # 仅在本课已评估候选中选择。
    print("第四轮覆盖、短实验前缀与最佳快照检查：通过", flush=True)  # 不将验证改善等同于流畅生成。
    return report  # 返回结果供检查工具使用。

if __name__ == "__main__":  # 直接运行才训练和保存正式产物。
    main()  # 默认写入model与docs/training_runs。
