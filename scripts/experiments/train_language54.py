import argparse  # 在模型评估前固定新增预算。
import json  # 保存完整混合训练记录。
import shutil  # 最佳与最后相同时保留完全相同的完整文件字节。
from time import perf_counter  # 在更新边界检查900秒上限。
import numpy as np  # 分域窗口排列可重建。
from check_chat_resume import restore  # 恢复第2020步完整预训练参数、AdamW、Scaler和随机状态。
from train_chat_pilot import evaluate, generate, batch_from_stream  # 复用原预训练的实际目标与评估。
from train_chat_sft import evaluate as evaluate_tasks  # 新任务只作固定诊断，不参与本阶段更新。
from chat_model import ChatLM, torch, nn, F  # 同一30M结构。
from sft_batches import load_tokenizer, ROOT  # 旧BPE编号不能改变。
from prepare_chat54 import DIRECTORY, DOMAINS, SCHEDULE, WEIGHTS, LANGUAGE  # 数据及固定4/3/1配比。
from prepare_chat_corpus import sha, dump  # 固定版本摘要。
from bounded_recurrent import cpu_tree  # 完整保存主参数与优化器状态。
OUT = ROOT / "model/lesson54"  # 不覆盖原预训练或SFT文件。


def load_data():  # 训练及CI都从同一冻结版本读取。
    folder = DIRECTORY / LANGUAGE  # 仅使用模型更新前审阅通过的v3。
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))  # 保留原源文件摘要。
    review = json.loads((DIRECTORY / "review.json").read_text(encoding="utf-8"))  # 抽查范围明示，不宣称全库事实审核。
    assert review["approved_language_manifest_sha256"] == sha((folder / "manifest.json").read_bytes()), "审阅不是当前数据版本"  # 不在未确认派生语料上训练。
    for source in manifest["sources"].values():  # 语料出处与冻结编码同样可追溯。
        assert sha((ROOT / source["path"]).read_bytes()) == source["sha256"], "原始划分文件改变"  # 不在不同原文上声称重建一致。
    assert manifest["schedule"] == list(SCHEDULE) and manifest["validation_weights"] == WEIGHTS, "混合配比改变"  # 结果出来后不调整指标权重。
    arrays = {}  # 分域数组保留各自的游标。
    for name, fingerprint in manifest["files"].items():  # 所有实际输入文件均要核验。
        path = folder / name  # 不读测试文件。
        assert sha(path.read_bytes()) == fingerprint, "混合语料编码产物改变"  # 不把不同文件当作同一实验。
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)  # 只读内存映射。
    tokenizer = load_tokenizer()  # 沿用6400词分词器。
    assert manifest["tokenizer_sha256"] == sha((ROOT / "data/chat48/tokenizer/tokenizer.json").read_bytes()), "分词器改变"  # 参数矩阵对应编号固定。
    task_manifest = json.loads((DIRECTORY / "curriculum/manifest.json").read_text(encoding="utf-8"))  # 新任务诊断也固定数据版本。
    task_path = DIRECTORY / "curriculum/validation.jsonl"  # 只读已划分的任务验证。
    assert sha(task_path.read_bytes()) == task_manifest["files"][task_path.name], "任务诊断题改变"  # 不在看到分数后换验证题。
    tasks = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines()]  # 任务验证不加入预训练流。
    return arrays, tokenizer, manifest, tasks  # 统一传给验证及训练。


def main():  # 新数据混合，原预训练目标保持下一token预测。
    parser = argparse.ArgumentParser()  # 新阶段预算不自动扩大。
    parser.add_argument("--max-updates", type=int, default=1000)  # 最多新增一千次成功更新。
    parser.add_argument("--max-seconds", type=float, default=900)  # 累积组边界截止十五分钟。
    args = parser.parse_args()  # 预算写入结果。
    assert 0 < args.max_updates <= 1000 and 0 < args.max_seconds <= 900, "超出本阶段预算"  # 不以效果不足自行增加轮数。
    assert torch.cuda.is_available(), "本阶段使用已验证的CUDA环境"  # 不意外在CPU长训练。
    torch.set_num_threads(4)  # 限制CPU线程。
    torch.backends.cuda.matmul.allow_tf32 = False  # 与原预训练精度路径一致。
    arrays, tokenizer, manifest, tasks = load_data()  # 数据先审阅、冻结，再更新参数。
    source = ROOT / "model/lesson50/last_train.pt"  # 从未做SFT的第2020步分支加强语言能力。
    old = json.loads((ROOT / "docs/training_runs/lesson50.json").read_text(encoding="utf-8"))  # 保存旧最佳作为回退。
    source_hash = sha(source.read_bytes())  # 精确来源证据。
    assert source_hash == old["artifacts"]["last_train.pt"]["sha256"], "不是约定预训练来源"  # 不混入SFT第100步参数。
    model, optimizer, scaler, parent = restore(source)  # 原m、v、t及Scaler保持。
    assert parent["step"] == 2020, "原预训练计数错误"  # 不清空累计历史。
    for group in optimizer.param_groups:  # 新域分布下预先固定较小学习率。
        group["lr"] = 1e-4  # 明示从原0.0003调整，不宣称原轨迹完全不变。
    cursors = {domain: {"epoch": 0, "offset": 0} for domain in DOMAINS}  # 新混合数据游标从0开始，不误用旧单流偏移。
    windows = {domain: manifest["stats"][domain]["train_windows"] for domain in DOMAINS}  # 不同域窗口数量不同。
    step, updates, seen, skipped = 2020, 0, 0, 0  # 旧累计与本阶段新增计数分开。
    domain_positions, exposures, history, training = {d: 0 for d in DOMAINS}, [], [], []  # 完整记录实际混合曝光。
    best_score, best_step = float("inf"), 2020  # 起点也参加最佳选择。
    OUT.mkdir(exist_ok=True)  # 新课保存目录。
    start = perf_counter()  # 阶段计时含起点验证、生成和周期保存。
    def checkpoint():  # 全部指标范围固定，选择只用语言混合分数。
        nonlocal best_score, best_step  # 严格按预先权重选优。
        metrics = {domain: evaluate(model, arrays[f"{domain}_validation_x.npy"], arrays[f"{domain}_validation_y.npy"]) for domain in DOMAINS}  # 三个域全量验证。
        score = sum(WEIGHTS[domain] * metrics[domain]["loss"] for domain in DOMAINS)  # 分域损失固定加权，不受文档长度支配。
        task_metric = evaluate_tasks(model, tasks, tokenizer)  # 只作未训练课程的开发诊断，不参与选优。
        samples = [generate(model, tokenizer, prompt, greedy=greedy) for prompt in ("清晨，小明走进教室，", "学习新知识时，", "春风") for greedy in (True, False)]  # 所有固定续写保留。
        improved = score < best_score  # 不用单条好看生成替换验证选优。
        if improved:  # 起点或新检查点真正改善。
            best_score, best_step = score, step  # 保存累计预训练步数。
        entry = {"step": step, "additional_updates": updates, "domains": metrics, "weighted_score": score, "untrained_curriculum_validation": task_metric, "samples": samples, "elapsed_seconds": perf_counter() - start}  # 显示诊断任务尚未训练。
        history.append(entry)  # 所有候选保留，不只写最佳。
        state = {"format": "chat_mix54_train_v1", "model_config": model.spec, "model_state": cpu_tree(model.state_dict()), "optimizer_state": cpu_tree(optimizer.state_dict()), "scaler_state": scaler.state_dict(), "step": step, "additional_updates": updates, "seen_positions": parent["seen_positions"] + seen, "phase_seen_positions": seen, "domain_positions": dict(domain_positions), "cursors": {d: dict(c) for d, c in cursors.items()}, "data_seed": 654, "schedule": list(SCHEDULE), "microbatch": 2, "accumulation": 4, "tokenizer_sha256": manifest["tokenizer_sha256"], "language_manifest_sha256": sha((DIRECTORY / LANGUAGE / "manifest.json").read_bytes()), "parent_checkpoint_sha256": source_hash, "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(), "best_score": best_score, "best_step": best_step}  # 新混合状态有三个独立游标。
        torch.save(state, OUT / "last_train.pt")  # 完整FP32训练进度。
        if improved:  # 拷贝完整字节，避免可变引用及同内容不同文件前缀。
            shutil.copyfile(OUT / "last_train.pt", OUT / "best_train.pt")  # 最佳等于最后时可在Release复用附件。
        print(f"mixed updates={updates} global={step} score={score:.6f} old={metrics['original']['loss']:.6f} story={metrics['stories']['loss']:.6f} novel={metrics['novels']['loss']:.6f} best={best_step}", flush=True)  # 报告各域改善或回退。
    checkpoint()  # 第一次更新之前验证同一来源。
    while updates < args.max_updates and perf_counter() - start < args.max_seconds:  # 截止后不启动新累积组。
        pending, references, items = {d: dict(c) for d, c in cursors.items()}, [], []  # 成功后才提交正式游标。
        for domain in SCHEDULE:  # 每组固定原始4、故事3、四书1。
            index = DOMAINS.index(domain)  # 域索引同时进入可重建种子。
            cursor = pending[domain]  # 当前域下一未用窗口。
            order = np.random.default_rng(654 + index * 1000 + cursor["epoch"]).permutation(windows[domain])  # 每域每轮独立排列。
            window = int(order[cursor["offset"]])  # 保留真实窗口行号。
            items.append(batch_from_stream(arrays[domain + "_train.npy"], window))  # 下一token输入和尾窗真实目标。
            references.append((index, cursor["epoch"], window))  # 指纹包括来源，防止混合比例伪报。
            cursor["offset"] += 1  # 推进临时域游标。
            if cursor["offset"] == windows[domain]:  # 域内整轮用完。
                pending[domain] = {"epoch": cursor["epoch"] + 1, "offset": 0}  # 不重置模型或优化器。
        count = sum(item[2] for item in items)  # 八窗共同的有效目标分母。
        for attempt in range(3):  # 溢出重试同样的八个窗口。
            optimizer.zero_grad(set_to_none=True)  # 每组开始清空梯度。
            model.train()  # 进入训练模式。
            loss_sum = 0.0  # 记录更新前组损失。
            for i in range(0, 8, 2):  # 微批2、累积4。
                x = torch.cat([item[0] for item in items[i:i + 2]]).cuda()  # 所有窗口长度256。
                y = torch.cat([item[1] for item in items[i:i + 2]]).cuda()  # 尾段用-100忽略。
                with torch.autocast("cuda", dtype=torch.float16):  # FP32主参数加FP16计算。
                    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum") / count  # 不平均微批均值。
                loss_sum += float(loss.item())  # 同一分母可直接累加。
                scaler.scale(loss).backward()  # 四个微批更新前都使用同一参数。
            scaler.unscale_(optimizer)  # 裁剪前还原真实梯度。
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())  # 全组统一裁剪。
            previous_scale = scaler.get_scale()  # 识别被跳过的尝试。
            scaler.step(optimizer)  # 一组只更新一次。
            scaler.update()  # 维护恢复的动态缩放历史。
            if scaler.get_scale() >= previous_scale:  # 成功更新才提交数据位置。
                break  # 不重试已经成功的组。
            skipped += 1  # 溢出不当作新增成功步。
        else:  # 不无限重试溢出。
            raise RuntimeError("混合预训练同组连续三次溢出")  # 保留之前检查点。
        step, updates, seen, cursors = step + 1, updates + 1, seen + count, pending  # 成功后才累计。
        exposures.extend(references)  # 实际曝光指纹源数据。
        for domain, item in zip(SCHEDULE, items):  # 来源目标数随尾窗而变。
            domain_positions[domain] += item[2]  # 不直接假定每窗256。
        training.append({"step": step, "loss": loss_sum, "positions": count, "gradient_norm": norm})  # 不把不同批次损失当固定验证。
        if updates % 100 == 0:  # 定期报告真实更新进展。
            print(f"mixed train {updates}/{args.max_updates} global={step} loss={loss_sum:.6f}", flush=True)  # 不把微批数当step。
        if updates in (250, 500, 750, args.max_updates):  # 起点前固定的验证点。
            checkpoint()  # 周期评估和保存计入阶段耗时。
    if history[-1]["step"] != step:  # 时间上限可能落在固定检查点之间。
        checkpoint()  # 完成必要收尾后不再更新。
    seconds = perf_counter() - start  # 导出及恢复核验另计。
    saved = torch.load(OUT / "best_train.pt", map_location="cpu", weights_only=True)  # 最佳可以是旧起点或中间位置。
    model.load_state_dict(saved["model_state"])  # 对真正最佳做导出。
    state = {"format": "chat_lm48_inference_v1", "model_config": model.spec, "model_state": {name: value.half() for name, value in saved["model_state"].items()}, "step": best_step, "tokenizer_sha256": manifest["tokenizer_sha256"], "language_manifest_sha256": saved["language_manifest_sha256"], "stage": "mixed_language_pretrain_not_sft"}  # 兼容旧独立续写入口。
    torch.save(state, OUT / "base_fp16.pt")  # 推理文件仍约60MB。
    model.load_state_dict(state["model_state"])  # 复算实际FP16舍入后的权重。
    export_domains = {d: evaluate(model, arrays[d + "_validation_x.npy"], arrays[d + "_validation_y.npy"]) for d in DOMAINS}  # 全部三域完整验证。
    ci = {d: evaluate(model, arrays[d + "_validation_x.npy"], arrays[d + "_validation_y.npy"], limit=4) for d in DOMAINS}  # CI明确只复算每域四窗。
    assert sha(source.read_bytes()) == source_hash, "原预训练文件改变"  # 不覆盖2020步来源。
    artifacts = {name: {"bytes": (OUT / name).stat().st_size, "sha256": sha((OUT / name).read_bytes())} for name in ("best_train.pt", "last_train.pt", "base_fp16.pt")}  # 完整产物真实大小和摘要。
    report = {"status": "completed", "stage": "bounded_mixed_language_pretraining", "config": vars(args), "initial_step": 2020, "completed_steps": step, "additional_updates": updates, "phase_seen_positions": seen, "cumulative_seen_positions": parent["seen_positions"] + seen, "domain_positions": domain_positions, "cursors": cursors, "best_step": best_step, "best_score": best_score, "stage_seconds": seconds, "scaler_skipped_updates": skipped, "source_sha256": source_hash, "learning_rate": 1e-4, "microbatch": 2, "accumulation": 4, "data_seed": 654, "schedule": list(SCHEDULE), "validation_weights": WEIGHTS, "model_config": model.spec, "tokenizer_sha256": manifest["tokenizer_sha256"], "language_manifest_sha256": state["language_manifest_sha256"], "history": history, "training": training, "exposure_sha256": sha(np.asarray(exposures, dtype=np.int64).tobytes()), "export_domains": export_domains, "export_ci_subset": ci, "task_data_prepared_not_sft_trained": True, "artifacts": artifacts, "limitations": "fiction language domains and 8 excerpt review, not full factual verification; mixed traditional/simplified; task labels verified algorithmically, task holdout shares templates; no new SFT in this stage"}  # 不声称本课已获得聊天能力。
    (ROOT / "docs/training_runs/lesson54.json").write_bytes(dump(report))  # 旧记录全部保留。
    print("混合语言阶段完成：", updates, "累计：", step, "最佳：", best_step, "秒：", seconds, flush=True)  # 不自动追加预算。


if __name__ == "__main__":  # 导入不启动GPU训练。
    main()  # 显式运行最多新增1000更新或900秒。
