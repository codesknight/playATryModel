import argparse  # 在训练前确定轮数、步数和时间预算。
import hashlib  # 核对数据和模型文件。
import json  # 保存全部真实指标与样例。
import shutil  # 独立保存验证最优状态。
from time import perf_counter  # 在批次边界检查预算。
import tiny_transformer as lm  # 使用可逐块阅读的模型实现。
from tiny_transformer import ROOT, DIRECTORY, torch, nn, F, np  # 复用已定义的数据目录与依赖。
from bounded_recurrent import cpu_tree, write_json  # 复用CPU状态保存工具。


def main():  # 本课从零训练新结构，不能直接载入RNN权重。
    parser = argparse.ArgumentParser()  # 所有预算在看到训练结果前确定。
    parser.add_argument("--max-epochs", type=int, default=6)  # 最多六轮混合训练。
    parser.add_argument("--max-steps", type=int, default=5100)  # 最多5100次更新。
    parser.add_argument("--max-seconds", type=float, default=600)  # 训练阶段最多十分钟，收尾验证另计。
    args = parser.parse_args()  # 命令行可缩小训练规模。
    assert min(args.max_epochs, args.max_steps, args.max_seconds) > 0, "训练预算必须为正"  # 禁止隐式无限训练。
    torch.set_num_threads(4)  # 仅限制当前进程线程。
    torch.manual_seed(45)  # 模型初始化和后续随机状态固定。
    torch.use_deterministic_algorithms(True)  # 不允许静默使用非确定性计算。
    torch.backends.cudnn.benchmark = False  # 不改变后端算法选择。
    torch.backends.cuda.matmul.allow_tf32 = False  # 使用float32，便于教学和核对。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 复用已有PyTorch环境。
    vocab = json.loads((DIRECTORY / "vocab.json").read_text(encoding="utf-8"))  # 使用新训练词表，不覆盖旧词表。
    manifest = json.loads((DIRECTORY / "manifest.json").read_text(encoding="utf-8"))  # 记录真实数据版本。
    for name, expected in manifest["files"].items():  # 在训练前核对全部新数据文件的字节版本。
        assert hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() == expected, "语料产物被修改"  # 测试正文仅做文件哈希，不作训练或指标计算。
    train, validation = lm.load_windows(vocab, "train"), lm.load_windows(vocab, "validation")  # 不创建测试窗口。
    model = lm.TinyTransformer(len(vocab["tokens"])).to(device)  # 四层、四头、128维、128字上下文。
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.95), weight_decay=0.01)  # 固定学习率与解耦权重衰减。
    output = ROOT / "model/lesson45"  # 本课独立权重目录。
    output.mkdir(parents=True, exist_ok=True)  # 不覆盖旧RNN或GRU。
    best_path, last_path = output / "lesson45_best.pt", output / "lesson45_last.pt"  # 最佳用于生成，最后用于续训。
    report_path = ROOT / "docs/training_runs/lesson45.json"  # 原始实验报告保存全部候选。
    batch_size, data_seed = 32, 145  # 混合数据每轮按独立种子打乱窗口。
    counts = torch.bincount(train[1][train[1] != -100], minlength=len(vocab["tokens"])).double() + 1  # 只用训练目标拟合加一平滑字频。
    log_probabilities = (counts / counts.sum()).log()  # 字频基线不看前文。
    frequency = {}  # 分来源比较相同目标集合。
    for source in sorted(set(validation[2])):  # 四书与新故事分别报告。
        selection = torch.tensor([item == source for item in validation[2]])  # 只选该来源的验证窗口。
        targets = validation[1][selection]  # 包含尾段和EOS，不含被忽略的补齐。
        targets = targets[targets != -100]  # 真实目标参与分母。
        frequency[source] = {"loss": float(-log_probabilities[targets].mean()), "positions": len(targets)}  # 与模型的分来源口径一致。
    report = {"status": "running", "config": vars(args), "model_config": model.spec, "parameter_count": sum(p.numel() for p in model.parameters()), "batch_size": batch_size, "learning_rate": 0.0003, "data_seed": data_seed, "torch_version": str(torch.__version__), "device": str(device), "data_manifest_sha256": hashlib.sha256((DIRECTORY / "manifest.json").read_text(encoding="utf-8").encode("utf-8")).hexdigest(), "train_windows": len(train[0]), "train_positions_per_epoch": int((train[1] != -100).sum()), "frequency_baseline": frequency, "history": []}  # 新数据新目标不能直接和旧RNN数值做严格对照。
    step, seen, best_loss, started = 0, 0, float("inf"), perf_counter()  # 计时包含本阶段周期验证和保存。
    order_epoch, order = 0, np.random.default_rng(data_seed).permutation(len(train[0])).astype(np.int64)  # 起点就保存下一批真实排列。
    cursor, exposure = {"next_epoch": 0, "next_start": 0}, hashlib.sha256()  # 恢复游标和实际数据顺序证据。

    def capture():  # 完整验证、固定生成、保存同位置的全部训练状态。
        nonlocal best_loss  # 严格按原始完整验证交叉熵选优。
        metrics = lm.evaluate(model, validation)  # 按所有真实目标加权。
        samples = [lm.generate(model, vocab, prompt, greedy=greedy) for prompt in ("从前，有一个小男孩", "唐僧道：", "宝玉听了") for greedy in (True, False)]  # 三开头和两种解码预先固定。
        row = {"step": step, "seen_positions": seen, **metrics, "samples": samples}  # 每个候选都保留完整文字。
        report["history"].append(row)  # 不省略变差的结果。
        state = {"format": "playatry_transformer_v1", "model_config": model.spec, "model_state": cpu_tree(model.state_dict()), "optimizer_state": cpu_tree(optimizer.state_dict()), "vocab": vocab, "learning_rate": 0.0003, "step": step, "seen_positions": seen, "cursor": dict(cursor), "order_epoch": order_epoch, "epoch_order": torch.from_numpy(order.copy()), "batch_size": batch_size, "data_seed": data_seed, "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [], "data_manifest_sha256": report["data_manifest_sha256"]}  # 以后续训需要参数、AdamW历史、数据位置与随机状态。
        temporary = last_path.with_suffix(".tmp")  # 避免留下半写入的正式文件。
        torch.save(state, temporary)  # CPU张量状态可在无GPU环境读取。
        temporary.replace(last_path)  # 原子替换最后进度。
        if metrics["loss"] < best_loss:  # 起点和所有轮末都参加选优。
            best_loss = metrics["loss"]  # 最佳不一定等于最后一轮。
            shutil.copyfile(last_path, best_path)  # 对应优化器状态一起保存。
            report["best_step"] = step  # 保留最佳文件的位置。
        report["best_loss"] = best_loss  # 每次捕获都及时更新记录。
        write_json(report_path, report)  # 中途也能查看实际进度。
        print(f"step={step} validation={metrics['loss']:.6f} accuracy={metrics['accuracy']:.2%} sources={metrics['by_source']}", flush=True)  # 不混淆自由生成质量与逐字预测准确率。

    capture()  # 随机初始化的起点也有同范围验证记录。
    reason = "epochs"  # 正常轮数结束时的停止原因。
    for epoch in range(args.max_epochs):  # 外层最多执行预定轮数。
        order_epoch = epoch  # 保存当前排列所属轮号。
        order = np.random.default_rng(data_seed + epoch).permutation(len(train[0])).astype(np.int64)  # 每轮可复现的混合窗口打乱。
        for start in range(0, len(order), batch_size):  # 保留尾批。
            if step >= args.max_steps or perf_counter() - started >= args.max_seconds:  # 截止后不再启动训练批次。
                reason = "steps" if step >= args.max_steps else "time"  # 如实记录停止原因。
                break  # 收尾验证另计。
            indices = order[start:start + batch_size]  # 该批实际窗口编号。
            inputs, targets = train[0][indices].to(device), train[1][indices].to(device)  # 每批搬运，控制显存占用。
            model.train()  # 生成或验证后显式恢复模式。
            optimizer.zero_grad(set_to_none=True)  # 清梯度而不清历史。
            loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())  # 默认忽略-100补齐目标。
            assert torch.isfinite(loss).item(), "损失无效"  # 不继续传播非有限值。
            loss.backward()  # PyTorch计算完整注意力网络梯度。
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)  # 防止一次更新过大。
            optimizer.step()  # AdamW每批更新一次。
            step, seen = step + 1, seen + int((targets != -100).sum().item())  # 记录真实目标数，不能用填充后大小冒充。
            exposure.update(indices.tobytes())  # 核对实际使用的数据顺序。
            end = start + len(indices)  # 下一批从实际尾批之后开始。
            cursor = {"next_epoch": epoch + int(end == len(order)), "next_start": 0 if end == len(order) else end}  # 保存可继续的数据游标。
            if step % 200 == 0:  # 长训练定期显示进度。
                print(f"step={step} batch_loss={loss.item():.6f} elapsed={perf_counter()-started:.1f}s", flush=True)  # 该批损失不参与最佳选优。
        if report["history"][-1]["step"] != step:  # 轮末或提前停止处完整验证。
            capture()  # 保留最后和最佳两个状态。
        if reason != "epochs":  # 达到时间或步数预算后退出整个训练。
            break  # 不自动加轮数。
    report.update({"status": "completed", "completed_steps": step, "seen_positions": seen, "final_cursor": cursor, "stop_reason": reason, "phase_elapsed_seconds": perf_counter()-started, "exposure_sha256": exposure.hexdigest()})  # 计时截止于训练阶段，不含随后收尾核验。
    restored, _, state = lm.restore(best_path, device)  # 从磁盘重新加载真实最佳模型。
    assert state["step"] == report["best_step"] and np.isclose(lm.evaluate(restored, validation)["loss"], best_loss, atol=1e-5), "最佳文件重载验证失败"  # 保证保存和报告一致。
    report["artifacts"] = {path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (best_path, last_path)}  # 记录完整状态文件体积。
    assert all(item["bytes"] < 1_000_000_000 for item in report["artifacts"].values()), "检查点超过一GB"  # 权重与训练状态都满足小模型要求。
    for sample in next(row for row in report["history"] if row["step"] == report["best_step"])["samples"]:  # 最终展示全部固定样例。
        print(sample["mode"], repr(sample["text"]), flush=True)  # 不挑选最好看的结果。
    write_json(report_path, report)  # 保存包含文件哈希的最终报告。


if __name__ == "__main__":  # 导入测试工具时不自动训练。
    main()  # 运行一次有明确预算的训练。
