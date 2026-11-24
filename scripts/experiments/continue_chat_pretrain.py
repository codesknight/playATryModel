import argparse  # 训练前明确各阶段预算。
from check_chat_resume import restore  # 恢复已经验证的完整模型、AdamW、Scaler与随机状态。
import json  # 保存真实训练指标和生成文字。
import math  # 计算打包窗口数量。
from pathlib import Path  # 管理本课产物路径。
import shutil  # 将完整最佳训练状态保留为独立文件。
from time import perf_counter  # 在更新边界检查时间预算。
from chat_model import ChatLM, torch, nn, F  # 使用原生实现，不加载既有权重。
import numpy as np  # 复现训练窗口顺序与采样。
from tokenizers import Tokenizer  # 与已冻结的分词器配套解码。
from prepare_chat_corpus import ROOT, DIRECTORY, dump, sha  # 数据来源与固定字节格式。
from bounded_recurrent import cpu_tree  # 保存CPU张量而不依赖GPU设备。


def batch_from_stream(stream, index, context=256):  # 流式打包允许跨文档，文档边界已由BOS/EOS标识。
    chunk = np.asarray(stream[index * context:index * context + context + 1], dtype=np.int64)  # 需要多一项以构造下一token目标。
    x, y = np.zeros(context, dtype=np.int64), np.full(context, -100, dtype=np.int64)  # 最后一窗允许右侧填充。
    length = len(chunk) - 1  # 只计算真实的目标位置。
    x[:length], y[:length] = chunk[:-1], chunk[1:]  # 同一训练流中目标错一位。
    return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y).unsqueeze(0), length  # 本次微批固定1个窗口。


@torch.inference_mode()  # 验证不计算梯度。
def evaluate(model, xs, ys, limit=None):  # 全部文档尾段按真实位置平均。
    model.eval()  # 明确评估行为。
    device = next(model.parameters()).device  # CPU与CUDA使用相同评估入口。
    loss_sum, correct, count = 0.0, 0, 0  # 累计而不平均窗口均值。
    length = len(xs) if limit is None else min(limit, len(xs))  # CI小子集与完整验证明确区分。
    for start in range(0, length, 4):  # 验证批4以控制显存与输出logits大小。
        x = torch.from_numpy(np.array(xs[start:min(start + 4, length)], dtype=np.int64)).to(device)  # 复制只读内存映射数组。
        y = torch.from_numpy(np.array(ys[start:min(start + 4, length)], dtype=np.int64)).to(device)  # 保留-100填充。
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):  # CUDA混合精度，CPU用FP32复核。
            logits = model(x)  # 只进行前向。
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum")  # 忽略补齐目标并求和。
        loss_sum += float(loss.item())  # 所有有效位置累加。
        correct += int(((logits.argmax(-1) == y) & (y != -100)).sum().item())  # 逐token准确率不等于聊天正确率。
        count += int((y != -100).sum().item())  # 分母排除补齐。
    return {"loss": loss_sum / count, "accuracy": correct / count, "positions": count}  # 完整保留评估口径。


@torch.inference_mode()  # 固定续写只做推理，不假装是聊天。
def generate(model, tokenizer, prompt, greedy=False, length=40):  # 每个候选采用相同三个开头。
    model.eval()  # 关闭训练模式。
    device = next(model.parameters()).device  # 匹配设备。
    history = [1] + tokenizer.encode(prompt, add_special_tokens=False).ids  # 预训练正文以BOS开头。
    output, rng = [], np.random.default_rng(42)  # 固定局部采样种子。
    for _ in range(length):  # 最大40个新token，不是40个汉字。
        x = torch.tensor([history[-model.spec["context"]:]], device=device)  # 当前版本不使用KV cache。
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):  # 与验证精度一致。
            scores = model(x)[0, -1].float().cpu().numpy().astype(np.float64)  # 读取最后位置的下一token分数。
        scores[[0, 1, 3, 4, 5, 6, 7]] = -np.inf  # 本阶段只生成普通正文和EOS，不生成聊天角色头。
        candidates = np.argsort(-scores, kind="stable")[:40]  # 固定top-k40。
        weights = np.exp((scores[candidates] - scores[candidates].max()) / 0.7)  # 固定温度0.7。
        token = int(scores.argmax()) if greedy else int(rng.choice(candidates, p=weights / weights.sum()))  # 两种解码都完整记录。
        output.append(token)  # 包括可能出现的EOS。
        if token == 2:  # 预训练文档结束。
            break  # 不把EOS显示为正文。
        history.append(token)  # 每次把预测接回上下文。
    return {"prompt": prompt, "mode": "greedy" if greedy else "topk40_t07", "seed": 42, "generated_ids": output, "text": prompt + tokenizer.decode(output, skip_special_tokens=True)}  # 原始输出不挑选修饰。


def main():  # 从完整检查点继续有界预训练。
    parser = argparse.ArgumentParser()  # 预算在看到结果前确定。
    parser.add_argument("--source", type=Path, default=ROOT / "model/lesson48/last_train.pt")  # 本课固定从1020步的完整状态继续。
    parser.add_argument("--max-updates", type=int, default=1000)  # 额外更新上限，不是重置累计步数。
    parser.add_argument("--max-seconds", type=float, default=1800)  # 阶段更新边界截止，验证收尾另计。
    args = parser.parse_args()  # 本次不增加轮数或其他搜索分支。
    args.source = args.source.resolve()  # 命令行相对路径也统一为项目中的绝对路径。
    assert args.max_updates > 0 and args.max_seconds > 0, "预算必须为正"  # 不允许无限训练。
    assert torch.cuda.is_available(), "本次硬件试跑需要CUDA"  # 不把CPU速度当作3080显卡结果。
    torch.set_num_threads(4)  # 不占用全部CPU线程。
    assert args.source.is_file(), "缺少完整状态，请先从第四十八课Release下载last_train.pt"  # 推理FP16文件不能代替。
    torch.use_deterministic_algorithms(True)  # 不使用已知非确定性算法。
    torch.backends.cudnn.benchmark = False  # 固定后端选择。
    torch.backends.cuda.matmul.allow_tf32 = False  # AMP使用FP16而非额外TF32变量。
    device, folder = torch.device("cuda"), DIRECTORY / "tokenizer"  # 本次使用现有CUDA环境。
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))  # 新BPE独立于此前字符词表。
    token_manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))  # 冻结的编码版本。
    assert sha((DIRECTORY / "manifest.json").read_bytes()) == token_manifest["corpus_manifest_sha256"], "编码来源与当前语料版本不同"  # 不能只校验npy而忽略原始划分版本。
    for name, expected in token_manifest["files"].items():  # 启动前核对全部实际训练输入。
        assert sha((folder / name).read_bytes()) == expected, "分词器或编号产物改变"  # 不只核对词表大小。
    stream = np.load(folder / "pretrain_train.npy", mmap_mode="r", allow_pickle=False)  # 不读取SFT或测试目标。
    xs = np.load(folder / "pretrain_validation_x.npy", mmap_mode="r", allow_pickle=False)  # 全量冻结预训练验证输入。
    ys = np.load(folder / "pretrain_validation_y.npy", mmap_mode="r", allow_pickle=False)  # 含尾段忽略标记。
    source_hash = sha(args.source.read_bytes())  # 核对确实是上一课保存的完整状态。
    previous = json.loads((ROOT / "docs/training_runs/lesson48.json").read_text(encoding="utf-8"))  # 起点报告不改变。
    assert source_hash == previous["artifacts"]["last_train.pt"]["sha256"], "不是本课约定的1020步来源"  # 不把不同起点混入对照。
    model, optimizer, scaler, original = restore(args.source)  # m、v、t、Scaler与随机状态一起恢复。
    assert original["step"] == 1020, "来源累计步数不符"  # 本课学习目标就是从这个位置续训。
    assert all(int(item["step"].item()) == original["step"] for item in optimizer.state.values()), "AdamW累计步数与模型不对应"  # 不能拿新优化器冒充恢复。
    assert sum(p.numel() for p in model.parameters()) == 30025216, "结构改变"  # 本次只增加训练，不扩模型。
    output = ROOT / "model/lesson50"  # 大训练状态本地单独保留。
    output.mkdir(parents=True, exist_ok=True)  # 不覆盖旧权重。
    last_path, best_path = output / "last_train.pt", output / "best_train.pt"  # 完整状态含FP32参数、AdamW与Scaler。
    export_path = output / "pretrain_fp16.pt"  # 仅推理权重可控制在GitHub单文件100MB以内。
    report_path = ROOT / "docs/training_runs/lesson50.json"  # 本课独立记录，不覆盖上一课。
    windows, data_seed, accumulation = math.ceil((len(stream) - 1) / 256), original["data_seed"], original["gradient_accumulation"]  # 每次更新处理8个微批。
    cursor, step, seen, skipped = dict(original["cursor"]), original["step"], original["seen_positions"], 0  # 本轮位置与累计更新分开。
    exposure, best_loss = __import__("hashlib").sha256(), float("inf")  # 实际窗口次序和最低验证损失。
    report = {"status": "running", "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "model_config": model.spec, "parameters": 30025216, "tokenizer_sha256": token_manifest["tokenizer_sha256"], "corpus_manifest_sha256": token_manifest["corpus_manifest_sha256"], "device": torch.cuda.get_device_name(), "torch_version": str(torch.__version__), "micro_batch_size": 1, "gradient_accumulation": accumulation, "data_seed": data_seed, "windows_per_epoch": windows, "train_stream_tokens": len(stream), "history": [], "phases": []}  # 不宣称本次已训练出聊天能力。
    report.update({"source": args.source.relative_to(ROOT).as_posix(), "source_sha256": source_hash, "initial_step": step, "initial_seen_positions": seen, "initial_cursor": dict(cursor), "initial_scaler_scale": scaler.get_scale(), "learning_rates": [group["lr"] for group in optimizer.param_groups]})  # 记录恢复证据而非仅打印已加载。
    counts = np.bincount(np.asarray(stream, dtype=np.int64), minlength=6400).astype(np.float64) + 1  # 只从预训练训练流拟合字频基线。
    targets = np.asarray(ys)[np.asarray(ys) != -100]  # 与完整验证相同的目标集合。
    report["frequency_baseline_loss"] = float(-np.log(counts / counts.sum())[targets].mean())  # 不看验证数据来拟合概率。

    def capture(phase):  # 完整验证、固定续写与配套训练状态。
        nonlocal best_loss  # 严格按完整验证损失选优。
        metrics = evaluate(model, xs, ys)  # 所有388180个目标。
        row = {"phase": phase, "step": step, "seen_positions": seen, **metrics, "samples": [generate(model, tokenizer, prompt, greedy) for prompt in ("春风", "解释什么是梯度。", "你好，请介绍一下自己。") for greedy in (True, False)]}  # 这些是续写测试，不是SFT聊天验收。
        report["history"].append(row)  # 每个候选都保留。
        state = {"format": "chat_lm48_train_v1", "model_config": model.spec, "model_state": cpu_tree(model.state_dict()), "optimizer_state": cpu_tree(optimizer.state_dict()), "scaler_state": scaler.state_dict(), "step": step, "seen_positions": seen, "cursor": dict(cursor), "data_seed": data_seed, "gradient_accumulation": accumulation, "tokenizer_sha256": report["tokenizer_sha256"], "corpus_manifest_sha256": report["corpus_manifest_sha256"], "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(), "parent_checkpoint_sha256": source_hash, "lesson": 50}  # 续训需要全部历史，不只权重。
        temporary = last_path.with_suffix(".tmp")  # 防止覆盖到一半。
        torch.save(state, temporary)  # 完整状态在本地保留，后续发布附件。
        temporary.replace(last_path)  # 写完后替换。
        if metrics["loss"] < best_loss:  # 最佳允许是早期位置。
            best_loss = metrics["loss"]  # 原始未舍入指标。
            shutil.copyfile(last_path, best_path)  # 模型与优化器属于同一时刻。
            exported = {"format": "chat_lm48_inference_v1", "model_config": model.spec, "model_state": {name: value.half() for name, value in cpu_tree(model.state_dict()).items()}, "step": step, "tokenizer_sha256": report["tokenizer_sha256"], "stage": "pretrain_not_sft"}  # FP16导出没有优化器，不能冒充完整续训状态。
            torch.save(exported, export_path)  # 约60MB推理模型符合1GB目标。
            report["best_step"] = step  # 最佳与最后分别跟踪。
        report["best_loss"] = best_loss  # 中途也可以审阅进度。
        report_path.parent.mkdir(parents=True, exist_ok=True)  # 记录目录存在。
        report_path.write_bytes(dump(report))  # 固定LF字节。
        print(f"{phase} step={step} validation={metrics['loss']:.6f} accuracy={metrics['accuracy']:.2%}", flush=True)  # 不将token准确率解释为问答正确率。

    def update():  # 八个微批累计一次优化器更新，按有效目标数统一归一。
        nonlocal cursor, step, seen, skipped  # 只在更新真正成功后推进数据游标。
        local, references, batches = dict(cursor), [], []  # 确定本次更新的输入，溢出时重试同一批。
        for _ in range(accumulation):  # 不是每个微批执行step。
            order = np.random.default_rng(data_seed + local["epoch"]).permutation(windows)  # 可从游标重建任意轮。
            index = int(order[local["offset"]])  # 当前轮的下一个窗口。
            references.append((local["epoch"], index))  # 指纹包含轮号。
            batches.append(batch_from_stream(stream, index))  # 真实尾段可能少于256目标。
            local["offset"] += 1  # 暂存游标尚不提交。
            if local["offset"] == windows:  # 一整轮恰好用完。
                local = {"epoch": local["epoch"] + 1, "offset": 0}  # 新轮不重复最后一批。
        valid = sum(batch[2] for batch in batches)  # 累积组的真实目标总数。
        for _ in range(8):  # 最多八次缩放调整重试，不无限循环。
            model.train()  # 验证后恢复模式。
            optimizer.zero_grad(set_to_none=True)  # 仅在累积组开始时清梯度。
            loss_sum = 0.0  # 记录该次成功更新对应的原始平均损失。
            for x, y, _ in batches:  # 每个微批参与同一组梯度累积。
                with torch.autocast(device_type="cuda", dtype=torch.float16):  # FP32主权重配合FP16矩阵运算。
                    losses = F.cross_entropy(model(x.to(device)).flatten(0, 1), y.to(device).flatten(), reduction="sum")  # 先求和，再除以整个累积组目标数。
                assert torch.isfinite(losses).item(), "前向损失非有限"  # 真正异常时停止，不生成虚假训练记录。
                scaler.scale(losses / valid).backward()  # 不按微批均值无条件除8，兼容尾段。
                loss_sum += float(losses.item())  # 未缩放的loss之和。
            scaler.unscale_(optimizer)  # 裁剪前先还原真实梯度。
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 超限统一缩小，非有限时由Scaler跳过。
            old_scale = scaler.get_scale()  # 检查更新是否被溢出跳过。
            scaler.step(optimizer)  # 有溢出时不会真正调用优化器更新。
            scaler.update()  # 根据溢出情况调整缩放系数。
            if scaler.get_scale() < old_scale:  # 更新被跳过，不能增加真实step。
                skipped += 1  # 记录溢出并重试同一组数据。
                continue  # 不推进正式数据游标。
            assert torch.isfinite(norm).item(), "成功更新后梯度范数无效"  # 确认真实成功更新。
            cursor, step, seen = local, step + 1, seen + valid  # 一组只增加一次累计step。
            exposure.update(np.asarray(references, dtype=np.int64).tobytes())  # 只记录成功更新的数据序列。
            return loss_sum / valid  # 当前组真实位置平均损失。
        raise RuntimeError("连续FP16溢出，停止检查数值设置")  # 不把失败重复计为训练完成。

    capture("restored_start")  # 检查恢复起点的完整验证，不先做一次更新。
    assert np.isclose(report["history"][0]["loss"], previous["history"][-1]["loss"], atol=1e-5), "恢复起点验证不同"  # 不在错误起点上继续。
    for phase, maximum, seconds in (("continuation", args.max_updates, args.max_seconds),):  # 本课不重复硬件试跑。
        start_step, start_positions, started, compute_seconds = step, seen, perf_counter(), 0.0  # 不混淆累计与本段更新。
        torch.cuda.reset_peak_memory_stats()  # 仅统计本进程，不声称代表全卡总使用。
        while step - start_step < maximum and perf_counter() - started < seconds:  # 截止后不启动新的累积组。
            before = perf_counter()  # 计时包含本次数据搬运与八个微批。
            loss = update()  # 确保返回时真实更新已发生。
            torch.cuda.synchronize()  # 吞吐计时不能遗漏异步GPU运算。
            compute_seconds += perf_counter() - before  # 单独累计更新阶段耗时。
            if (step - start_step) % 100 == 0:  # 定期显示真实更新进度。
                print(f"{phase} updates={step-start_step}/{maximum} global={step} loss={loss:.6f}", flush=True)  # 不把微批当作更新step。
            if (step - start_step) % 200 == 0:  # 预定中途完整验证点。
                capture(phase)  # 不重置优化器或数据游标。
        if report["history"][-1]["step"] != step:  # 停止处尚未保存时收尾。
            capture(phase)  # 最后验证和写盘可能超过批次截止时间。
        result = {"phase": phase, "start_step": start_step, "updates": step-start_step, "positions": seen-start_positions, "phase_seconds": perf_counter()-started, "update_seconds": compute_seconds, "valid_tokens_per_update_second": (seen-start_positions)/compute_seconds if compute_seconds else 0.0, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "stop_reason": "steps" if step-start_step == maximum else "time"}  # 这些是实测而非权重字节估算。
        report["phases"].append(result)  # 保存每段实测。
        report_path.write_bytes(dump(report))  # 阶段结束先落盘。
    report.update({"status": "completed", "completed_steps": step, "additional_steps": step-original["step"], "additional_positions": seen-original["seen_positions"], "seen_positions": seen, "cursor": cursor, "scaler_skipped_updates": skipped, "exposure_sha256": exposure.hexdigest()})  # 正式累计更新数。
    state = torch.load(export_path, map_location="cpu", weights_only=True)  # 验证实际导出的推理文件。
    restored = ChatLM(**state["model_config"]).to(device)  # 加载FP16保存、FP32主参数的副本。
    restored.load_state_dict(state["model_state"])  # 量化舍入来自FP16导出，不能混作原FP32检查点。
    report["export_validation"] = evaluate(restored, xs, ys)  # 如实记录导出后的完整验证。
    report["export_ci_subset"] = evaluate(restored, xs, ys, limit=8)  # CI只复算明确记录的八窗，不冒充完整验证。
    report["artifacts"] = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())} for path in (last_path, best_path, export_path)}  # 完整训练状态可能超过GitHub单文件100MB。
    assert report["artifacts"][export_path.name]["bytes"] < 1_000_000_000, "导出权重超过1GB"  # 实际文件而非仅公式验收。
    for sample in next(row for row in report["history"] if row["step"] == report["best_step"])["samples"]:  # 展示最佳模型全部固定续写。
        print(sample["mode"], repr(sample["text"]), flush=True)  # 不声称已完成SFT聊天。
    report_path.write_bytes(dump(report))  # 最终文件证据落盘。


if __name__ == "__main__":  # 导入检查工具时不训练。
    main()  # 运行本课明确预算的真实续训。
