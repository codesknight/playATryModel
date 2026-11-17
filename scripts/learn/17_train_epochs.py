from importlib import import_module  # 复用数据、模型、评估、保存和生成工具。
from time import perf_counter  # 记录本次继续训练及验证的耗时。
import json  # 读取上一轮报告，并逐轮保存新实验记录。
import random  # 为每轮建立可复现、不同的章节顺序。
import numpy as np  # 检查数值、比较权重与验证指标。

network = import_module("07_tiny_rnn")  # 复用模型结构和完整梯度裁剪。
training = import_module("11_train_steps")  # 复用加载、保存与固定参数评估。
decoding = import_module("12_generate_text")  # 复用固定设置的文字生成。
BATCH_SIZE, SEQ_LENGTH = 16, 32  # 保持批大小与窗口长度不变。
LEARNING_RATE, MAX_NORM = 0.1, 1.0  # 沿用无动量 SGD 与全局裁剪设置。
TOTAL_EPOCHS = 3  # 总共完成三轮，本脚本从已有第一轮继续完成第二、三轮。

def evaluate_full(model, data, records):  # 在完整验证窗口集上评估当前参数。
    windows = data.iter_windows(records, SEQ_LENGTH, SEQ_LENGTH)  # 每次评估都新建验证窗口流。
    batches = data.iter_batches(windows, BATCH_SIZE)  # 流式处理所有验证批次，不缓存完整分数。
    return training.evaluate(model, batches, network)  # 只前向，已有函数检查评估不改参数。


def save_report(path, report):  # 每轮完成后保存配置、指标与当前最佳选择。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存中文可读的 JSON。

def main():  # 在完整轮边界继续训练，而不是重复初始化模型。
    run_started = perf_counter()  # 总耗时包含起始核验、继续训练、逐轮验证和生成。
    data = import_module("06_make_windows")  # 复用四书训练数据和固定词表，首次导入显示旧数据检查。
    source = json.loads((data.ROOT / "docs" / "training_runs" / "lesson32.json").read_text(encoding="utf-8"))  # 读取已经完成第一轮的报告。
    start_epoch = source["config"]["epochs"]  # 当前起点是完整训练一轮之后。
    assert start_epoch == 1 and source["train_sha256"] == data.vocab["train_sha256"], "起点轮数或数据版本不一致"  # 只接受本课预定的第一轮起点。
    assert source["config"]["batch_size"] == BATCH_SIZE and source["config"]["seq_length"] == SEQ_LENGTH and source["config"]["learning_rate"] == LEARNING_RATE and source["config"]["max_norm"] == MAX_NORM, "继续训练配置不一致"  # 保持与上一轮相同的更新规则。
    steps_per_epoch, windows_per_epoch = source["expected_steps"], source["seen_windows"]  # 使用已核验的一轮规模。
    source_row = source["history"][-1]  # 选择第一轮最后一步，不加载单窗口诊断模型。
    source_path = data.ROOT / "model" / source_row["checkpoint"]  # 旧权重只读，不覆盖。
    model, vocab, global_step = training.load_checkpoint(source_path, network)  # 只在外层循环之前加载一次，后续轮保留更新后的参数。
    assert vocab == data.vocab and model.hidden_size == 64 and global_step == start_epoch * steps_per_epoch == source["completed_steps"], "起点权重配置不一致"  # 确认模型、词表和累计步数对应第一轮结束。
    with (data.ROOT / "data" / "splits" / "validation.jsonl").open(encoding="utf-8") as file:  # 读取相同验证集，不用于参数更新。
        validation_records = [json.loads(line) for line in file]  # 所有轮都评估同一套完整验证窗口。
    initial_loss, positions = evaluate_full(model, data, validation_records)  # 独立核对加载模型的完整验证表现。
    assert positions == source["full_validation"]["positions"] and np.isclose(initial_loss, source["full_validation"]["rnn_loss"], rtol=1e-6, atol=1e-6), "起点验证结果不一致"  # 不拿旧子集分数冒充完整分数。
    output_dir = data.ROOT / "model"  # 本课使用独立的逐轮权重与最佳权重文件名。
    best_path = output_dir / "lesson33_best.npz"  # 最佳文件是独立快照，不是同一模型对象的引用。
    best_epoch, best_loss = start_epoch, initial_loss  # 第一轮也参与比较，后续训练变差时仍可保留它。
    training.save_checkpoint(best_path, model, vocab, global_step)  # 先保存起点，确保最佳文件始终对应一个已评估模型。
    history = [{"epoch": start_epoch, "global_step": global_step, "validation_loss": initial_loss, "validation_positions": positions, "checkpoint": source_path.name, "greedy": source_row["greedy"], "sampled": source_row["sampled"]}]  # 第一轮生成样例沿用对应权重的已有观察。
    report = {  # 新报告与旧第一轮报告分开保存。
        "config": {  # 固定本课从第一轮继续到第三轮的条件。
            "starting_epoch": start_epoch, "total_epochs": TOTAL_EPOCHS, "batch_size": BATCH_SIZE, "seq_length": SEQ_LENGTH,  # 记录训练起点和批次结构。
            "learning_rate": LEARNING_RATE, "max_norm": MAX_NORM, "shuffle_seed_base": 42,  # 保持更新规则并记录按轮种子起点。
            "prompt": "春风", "new_tokens": 40, "temperature": 1.0, "sampling_seed": 42,  # 生成比较条件与之前一致。
        },  # 其余结果记录在下面的历史和统计中。
        "train_sha256": data.vocab["train_sha256"], "starting_checkpoint": source_path.name, "steps_per_epoch": steps_per_epoch,  # 保留起点与数据来源。
        "history": history, "training_stats": [], "completed_epochs": start_epoch, "status": "running",  # 每完成一轮就更新进度记录。
        "best_epoch": best_epoch, "best_validation_loss": best_loss, "best_checkpoint": best_path.name,  # 当前最佳按完整验证损失决定。
        "full_frequency_loss": source["full_validation"]["frequency_loss"],  # 沿用同范围、已核验的全训练集字频参照。
    }  # 没有加入测试集指标或自动提前停止逻辑。
    report_path = data.ROOT / "docs" / "training_runs" / "lesson33.json"  # 独立保存本课继续训练记录。
    save_report(report_path, report)  # 先保存起点，避免只有最后才有记录。
    print(f"起点 epoch={start_epoch} global_step={global_step} 完整验证损失={initial_loss:.6f}", flush=True)  # 明确本次加载的是第一轮结束模型。
    chapter_to_book = {record["id"]: record["book"] for record in data.train_records}  # 后续逐轮核对四书覆盖。
    for epoch in range(start_epoch + 1, TOTAL_EPOCHS + 1):  # 本次依次训练第二轮和第三轮。
        records = list(data.train_records)  # 每轮从原始章节列表重新复制，不沿用上轮乱序列表。
        shuffle_seed = 42 + epoch - 1  # 第一轮为 42，第二轮为 43，第三轮为 44。
        random.Random(shuffle_seed).shuffle(records)  # 每轮可复现且通常不同，只改变章节顺序。
        batches = data.iter_batches(data.iter_windows(records, SEQ_LENGTH, SEQ_LENGTH), BATCH_SIZE)  # 每轮重新创建训练流，避免复用已耗尽的生成器。
        epoch_steps, seen_windows, seen_positions, last_batch_size = 0, 0, 0, 0  # 这些覆盖计数每轮从零开始。
        loss_sum = 0.0  # 每轮单独统计训练过程损失。
        book_windows = {record["book"]: 0 for record in records}  # 每轮都应完整访问四本书。
        started = perf_counter()  # 此计时只覆盖本轮训练与进度输出，不含轮末验证和保存。
        print(f"开始 epoch={epoch} 章节打乱种子={shuffle_seed} 前三章={[record['id'] for record in records[:3]]}", flush=True)  # 展示本轮顺序已经重建。
        for batch in batches:  # 完整处理本轮所有窗口，包括八行尾批。
            x, y = batch["x"], batch["y"]  # 当前批次实际输入与目标。
            loss, raw_grads = model.loss_and_grad(x, y)  # 使用上一轮留下的参数继续求梯度，不重新初始化。
            assert np.isfinite(loss), "训练损失异常"  # 无效损失不能继续更新。
            grads, _, _ = network.clip_gradients(raw_grads, MAX_NORM)  # 完整反向后统一裁剪。
            for name in model.params:  # 用本批完整梯度更新五组参数。
                model.params[name] -= LEARNING_RATE * grads[name]  # 下一批及下一轮都继续使用这些更新。
            assert all(np.isfinite(param).all() for param in model.params.values()), "模型参数异常"  # 及时发现训练中的无效数值。
            epoch_steps += 1  # 本轮步数只统计当前轮。
            global_step += 1  # 累计步数不在轮与轮之间清零。
            seen_windows += x.shape[0]  # 按实际批大小累加窗口。
            seen_positions += y.size  # 按实际预测位置数统计。
            last_batch_size = x.shape[0]  # 记录本轮尾批行数。
            loss_sum += loss * y.size  # 训练过程损失按实际位置数加权。
            for chapter_id in batch["chapter_ids"]:  # 将每行窗口计入对应书籍。
                book_windows[chapter_to_book[chapter_id]] += 1  # 各轮窗口数量应相同，顺序可以不同。
            if epoch_steps % 500 == 0 or epoch_steps == steps_per_epoch:  # 定期输出本轮与累计进度。
                print(f"epoch={epoch} step={epoch_steps}/{steps_per_epoch} global_step={global_step} loss={loss:.6f}", flush=True)  # 不要求不同批次损失单调下降。
        epoch_seconds = perf_counter() - started  # 训练计时结束，接下来保持参数固定评估。
        assert epoch_steps == steps_per_epoch and global_step == epoch * steps_per_epoch and seen_windows == windows_per_epoch, "轮数或更新覆盖错误"  # 核对没有漏掉或重复完整轮。
        assert seen_positions == windows_per_epoch * SEQ_LENGTH and last_batch_size == source["last_batch_size"] and book_windows == source["book_windows"], "本轮数据覆盖不一致"  # 顺序变化不能改变窗口集合与尾批大小
        checkpoint_path = output_dir / f"lesson33_epoch{epoch:02d}.npz"  # 每轮单独保存，不覆盖旧轮文件。
        training.save_checkpoint(checkpoint_path, model, vocab, global_step)  # 元数据记录累计更新步数，不是当前轮内步数。
        validation_loss, current_positions = evaluate_full(model, data, validation_records)  # 所有轮都用完整同一验证窗口集。
        assert current_positions == positions, "验证范围在轮之间改变"  # 不将不同范围的损失用于最佳模型选择。
        greedy, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32)  # 保持原来的贪心生成设置。
        sampled, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32, temperature=1.0, seed=42)  # 固定温度和采样种子。
        history.append({"epoch": epoch, "global_step": global_step, "validation_loss": validation_loss, "validation_positions": current_positions, "checkpoint": checkpoint_path.name, "greedy": greedy, "sampled": sampled})  # 每轮结果都保留，不只留下最好的一轮。
        report["training_stats"].append({"epoch": epoch, "shuffle_seed": shuffle_seed, "first_chapter_ids": [record["id"] for record in records[:3]], "epoch_steps": epoch_steps, "seen_windows": seen_windows, "seen_positions": seen_positions, "last_batch_size": last_batch_size, "book_windows": book_windows, "training_process_loss": loss_sum / seen_positions, "training_seconds": epoch_seconds})  # 记录顺序、覆盖与训练统计。
        if validation_loss < best_loss:  # 只有完整验证损失严格更低才替换最佳，相等时保留较早轮。
            best_epoch, best_loss = epoch, validation_loss  # 更新最佳轮数及对应分数。
            training.save_checkpoint(best_path, model, vocab, global_step)  # 保存独立文件快照，后续训练不会继续修改它。
        report["best_epoch"], report["best_validation_loss"] = best_epoch, best_loss  # 报告同步记录当前选择。
        report["completed_epochs"] = epoch  # 该轮训练和验证完成后才记录为完成。
        save_report(report_path, report)  # 每轮落盘一次，即使后面中断仍保留此前完整轮的记录。
        print(f"轮末 epoch={epoch} 完整验证损失={validation_loss:.6f} 当前最佳轮={best_epoch} 最佳损失={best_loss:.6f}", flush=True)  # 最后轮与最佳轮可能不同。
        print("贪心：", repr(greedy), flush=True)  # 如实输出重复或不通顺。
        print("采样：", repr(sampled), flush=True)  # 固定设置下记录文字变化。
    selected = min(history, key=lambda row: row["validation_loss"])  # 独立从所有已评估轮末记录核对最低完整验证损失。
    assert selected["epoch"] == best_epoch and np.isclose(selected["validation_loss"], best_loss), "最佳选择与记录不一致"  # 不把最后一轮直接当作最好。
    best_model, best_vocab, best_step = training.load_checkpoint(best_path, network)  # 训练结束后重新读取实际最佳文件。
    selected_path = source_path if selected["epoch"] == start_epoch else output_dir / selected["checkpoint"]  # 最佳可能仍是旧第一轮，也可能是新增轮。
    selected_model, _, selected_step = training.load_checkpoint(selected_path, network)  # 加载被选中轮的独立检查点作对照。
    assert best_vocab == vocab and best_step == selected_step == selected["global_step"] and all(np.array_equal(param, selected_model.params[name]) for name, param in best_model.params.items()), "最佳文件与所选轮参数不一致"  # 防止最佳文件其实保存了最后一轮或可变引用。
    best_text, _ = decoding.generate(best_model, best_vocab, "春风", new_tokens=40, context_size=32)  # 结果展示使用最佳文件加载的模型。
    report["status"], report["run_seconds"] = "completed", perf_counter() - run_started  # 只有全部训练和最佳文件核对完成才标记结束。
    save_report(report_path, report)  # 保存最终完成状态与耗时。
    print("总完成轮数与累计更新步数：", report["completed_epochs"], global_step, flush=True)  # 当前应为三轮、14361 步，其中第一轮来自上一课。
    print("选中最佳轮与完整验证损失：", best_epoch, best_loss, flush=True)  # 该选择只在本课观察过的三个轮末之间比较。
    print("同范围字频基线损失：", report["full_frequency_loss"], flush=True)  # 重复遍历同一训练语料不增加独立样本。
    print("加载最佳文件后的贪心：", repr(best_text), flush=True)  # 不把文字样例替代完整验证指标。
    print("最佳权重与报告：", best_path, report_path, flush=True)  # 旧第一轮文件保持不变。
    print("多轮覆盖、逐轮打乱与最佳快照检查：通过", flush=True)  # 不保证增加轮数必然提升可读性。


if __name__ == "__main__":  # 直接运行脚本时才继续训练本课两轮。
    main()  # 从已完成一轮的模型继续到总共三轮。
