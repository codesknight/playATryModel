import argparse  # 在开始前固定额外训练预算。
import hashlib  # 记录源文件和实际数据排列指纹。
import shutil  # 保存最佳候选的独立副本。
from time import perf_counter  # 在启动新批次前检查截止时间。
import bounded_recurrent as base  # 复用已核验的模型、数据和检查点工具。
from bounded_recurrent import np, torch, nn, F, ROOT  # 沿用相同计算实现和项目路径。


def main():  # 延续最佳RNN，不重新初始化Adam。
    parser = argparse.ArgumentParser()  # 训练预算可以明确复现。
    parser.add_argument("--max-steps", type=int, default=2394)  # 本阶段额外更新上限。
    parser.add_argument("--max-epochs", type=int, default=2)  # 最多继续两个轮次，首轮从游标开始。
    parser.add_argument("--max-seconds", type=float, default=360)  # 批次边界截止，收尾核验另计。
    parser.add_argument("--source", type=base.Path, default=ROOT / "model/lesson42/lesson42_rnn_best.pt")  # 明确从哪份完整状态继续。
    parser.add_argument("--output-dir", type=base.Path, default=ROOT / "model/lesson43")  # 不同实验使用独立目录。
    parser.add_argument("--report", type=base.Path, default=ROOT / "docs/training_runs/lesson43.json")  # 不覆盖其他课程的原始记录。
    parser.add_argument("--evaluate-each-epoch", action="store_true")  # 长训练时每轮完整验证并保留固定样例。
    args = parser.parse_args()  # 在看到本次指标前确定配置。
    assert min(args.max_steps, args.max_epochs, args.max_seconds) > 0, "预算必须为正"  # 禁止无限训练。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用现有环境的可用设备。
    torch.set_num_threads(4)  # 限制本进程的CPU线程数。
    torch.use_deterministic_algorithms(True)  # 不静默使用已知非确定性计算。
    torch.backends.cudnn.benchmark = False  # 固定循环层计算路径。
    torch.backends.cudnn.deterministic = True  # CUDA循环层也使用确定性设置。
    torch.backends.cuda.matmul.allow_tf32 = False  # 保持上一课float32精度设置。
    source = args.source.resolve()  # 从上一课实际最佳候选继续。
    model, optimizer, original = base.restore_checkpoint(source, device)  # 恢复参数、Adam动量和随机状态。
    vocab, train, validation = base.load_data(device)  # 数据划分、词表、32字窗口均不变。
    assert vocab == original["vocab"], "恢复时词表改变"  # 编号含义必须一致。
    x, y = train  # 当前训练窗口仍按章节独立构造。
    step, seen = original["step"], original["seen_positions"]  # 保留累计进度。
    initial_step, initial_seen = step, seen  # 额外预算与累计预算分开记录。
    cursor = dict(original["cursor"])  # 下一批位置不能从首轮重新开始。
    order_epoch, order = original["order_epoch"], original["epoch_order"].numpy().copy()  # 恢复尚未完成的排列。
    first_epoch, batch_size = cursor["next_epoch"], original["batch_size"]  # 跨轮边界时按下一轮种子重新排列。
    output = args.output_dir.resolve()  # 旧课产物保留。
    output.mkdir(parents=True, exist_ok=True)  # 创建本阶段输出目录。
    last_path, best_path = output / f"{output.name}_last.pt", output / f"{output.name}_best.pt"  # 最后状态与最佳状态分开保存。
    report_path = args.report.resolve()  # 所有固定样例写入原始报告。
    report = {"status": "running", "config": {key: str(value) if isinstance(value, base.Path) else value for key, value in vars(args).items()}, "source": source.relative_to(ROOT).as_posix(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "initial_step": step, "initial_seen_positions": seen, "learning_rate": optimizer.param_groups[0]["lr"], "device": str(device), "torch_version": str(torch.__version__), "numpy_version": np.__version__, "train_sha256": vocab["train_sha256"], "history": []}  # 保留本阶段可追溯条件。
    best_loss, started, exposure = float("inf"), perf_counter(), hashlib.sha256()  # 起点也参加最佳选择。

    def capture():  # 完整验证后保存同一位置的参数和优化器。
        nonlocal best_loss  # 严格按最低完整验证损失选优。
        row = {"step": step, "additional_steps": step - initial_step, "seen_positions": seen, **base.evaluate(model, validation), "samples": base.samples(model, vocab)}  # 不选择性省略生成样例。
        report["history"].append(row)  # 变差的候选也完整保留。
        state = {**original, "model_state": base.cpu_tree(model.state_dict()), "optimizer_state": base.cpu_tree(optimizer.state_dict()), "step": step, "seen_positions": seen, "cursor": dict(cursor), "order_epoch": order_epoch, "epoch_order": torch.from_numpy(order.copy()), "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}  # 用当前状态替换源文件的进度字段。
        temporary = last_path.with_suffix(".tmp")  # 避免半写入的最后状态。
        torch.save(state, temporary)  # 保存完整恢复所需信息。
        temporary.replace(last_path)  # 写完后再替换正式文件。
        if row["loss"] < best_loss:  # 不因为训练更久就强制替换最佳。
            best_loss = row["loss"]  # 保留未经舍入的指标。
            shutil.copyfile(last_path, best_path)  # 最佳参数和Adam状态属于同一次更新。
            report["best_step"] = step  # 最佳位置可以是本阶段起点。
        report["best_loss"] = best_loss  # 随检查点更新记录。
        base.write_json(report_path, report)  # 训练中断也有可审阅记录。
        print(f"additional={step-initial_step} total={step} validation={row['loss']:.6f} accuracy={row['accuracy']:.2%}", flush=True)  # 当前批损失不用于选优。

    capture()  # 验证实际恢复的起点。
    reason = "epochs"  # 正常完成轮数时的停止原因。
    for epoch in range(first_epoch, first_epoch + args.max_epochs):  # 不额外偷偷扩展训练轮数。
        start_at = cursor["next_start"]  # 首轮可从中途继续，后续轮从零开始。
        if epoch != order_epoch:  # 只有跨轮时才重建排列。
            order = np.random.default_rng(original["data_seed"] + epoch).permutation(len(x)).astype(np.int64)  # 延续原先的逐轮随机规则。
            order_epoch = epoch  # 保存此排列属于哪一轮。
        for start in range(start_at, len(x), batch_size):  # 最后一批按实际窗口数计算。
            if step - initial_step >= args.max_steps or perf_counter() - started >= args.max_seconds:  # 到截止点后不再启动新批次。
                reason = "steps" if step - initial_step >= args.max_steps else "time"  # 如实记录预算停止原因。
                break  # 后续收尾验证仍会执行。
            indices = order[start:start + batch_size]  # 与游标对应的实际下一批。
            selection = torch.as_tensor(indices, device=device)  # 数据已在同一设备上。
            model.train()  # 评估后恢复训练模式。
            optimizer.zero_grad(set_to_none=True)  # 只清梯度，保留Adam历史。
            loss = F.cross_entropy(model(x[selection]).flatten(0, 1), y[selection].flatten())  # 按实际目标数求平均。
            assert torch.isfinite(loss).item(), "损失无效"  # 非有限值不能继续更新。
            loss.backward()  # 自动求导不等于重新建立模型。
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)  # 保持原全局范数裁剪规则。
            optimizer.step()  # 动量和二阶矩延续上一课。
            step, seen = step + 1, seen + len(indices) * 32  # 分别累计更新数和目标数。
            exposure.update(indices.tobytes())  # 只记录本阶段额外训练的数据顺序。
            end = start + len(indices)  # 下一批位置从实际尾批长度计算。
            cursor = {"next_epoch": epoch + int(end == len(x)), "next_start": 0 if end == len(x) else end}  # 整轮结束后推进轮号。
            if (step - initial_step) % 200 == 0:  # 定期展示真实训练进度。
                print(f"additional={step-initial_step} batch_loss={loss.item():.6f}", flush=True)  # 不把单批下降当成泛化改善。
            if not args.evaluate_each_epoch and step - initial_step == 1000:  # 固定的中途完整验证候选。
                capture()  # 不中断优化器历史。
        if args.evaluate_each_epoch and report["history"][-1]["step"] != step:  # 轮末或预算停止处验证，避免重复候选。
            capture()  # 周期保存最佳状态，最后状态可继续恢复。
        if reason != "epochs":  # 任一预算到达后退出全部训练循环。
            break  # 不自动重试更大预算。
    if report["history"][-1]["step"] != step:  # 最后更新后补做完整验证。
        capture()  # 收尾可以超过训练循环的截止时间。
    report.update({"status": "completed", "stop_reason": reason, "completed_steps": step, "additional_steps": step-initial_step, "additional_positions": seen-initial_seen, "seen_positions": seen, "phase_elapsed_seconds": perf_counter()-started, "exposure_sha256": exposure.hexdigest(), "final_cursor": cursor})  # 实际预算与累计进度一同记录。
    report["restore_next_batch_max_abs_diff"] = base.verify_resume(model, optimizer, last_path, train, device)  # 在副本上验证恢复后的一步，正式模型不多训。
    best, _, best_state = base.restore_checkpoint(best_path, device)  # 重新加载实际最佳文件。
    assert best_state["step"] == report["best_step"] and np.isclose(base.evaluate(best, validation)["loss"], best_loss, atol=1e-5), "最佳文件核验失败"  # 核对文件和报告对应。
    report["artifacts"] = {path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (best_path, last_path)}  # 保存体积和内容证据。
    assert all(item["bytes"] < 1_000_000_000 for item in report["artifacts"].values()), "文件超过一GB"  # 完整状态也遵守大小要求。
    for sample in next(row for row in report["history"] if row["step"] == report["best_step"])["samples"]:  # 固定六个样例全部展示。
        print(sample["mode"], repr(sample["text"]), flush=True)  # 不只展示最好看的一段。
    base.write_json(report_path, report)  # 最后记录恢复核验和文件哈希。


if __name__ == "__main__":  # 导入检查工具时不开始训练。
    main()  # 执行明确预算的一次续训。
