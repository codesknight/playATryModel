import argparse  # 选择核验的完整状态和独立报告路径。
import json  # 保存恢复检查的证据。
from pathlib import Path  # 使用临时目录保存核验副本。
import tempfile  # 不覆盖正式训练检查点。
from chat_model import ChatLM, torch, nn, F  # 使用真实模型和同一梯度计算。
from prepare_chat_corpus import ROOT, DIRECTORY, dump, sha  # 项目目录与固定报告格式。
from train_chat_pilot import batch_from_stream  # 与真实训练采用相同打包窗口。
from bounded_recurrent import cpu_tree  # 完整保存参数和优化器历史。
import numpy as np  # 按数据游标复现相同下一批。


def restore(path):  # 加载完整训练状态，不能拿FP16推理文件代替。
    state = torch.load(path, map_location="cpu", weights_only=True)  # 原文件仅含张量和普通配置。
    assert state["format"] == "chat_lm48_train_v1", "需要完整训练状态"  # 拒绝不含优化器的文件。
    manifest = json.loads((DIRECTORY / "tokenizer/manifest.json").read_text(encoding="utf-8"))  # 恢复时也要校验原数据和编号来源。
    assert state["tokenizer_sha256"] == sha((DIRECTORY / "tokenizer/tokenizer.json").read_bytes()) == manifest["tokenizer_sha256"], "恢复时分词器改变"  # 相同词表大小也不能混用。
    assert state["corpus_manifest_sha256"] == sha((DIRECTORY / "manifest.json").read_bytes()) == manifest["corpus_manifest_sha256"], "恢复时语料版本改变"  # 游标只对应原版本。
    assert sha((DIRECTORY / "tokenizer/pretrain_train.npy").read_bytes()) == manifest["files"]["pretrain_train.npy"], "恢复时训练编号流改变"  # 不在两份错误数据上比较后声称复现。
    model = ChatLM(**state["model_config"]).cuda()  # 核验副本使用同一GPU精度路径。
    model.load_state_dict(state["model_state"])  # 载入FP32主参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.95), weight_decay=0.01)  # 随后恢复原参数组。
    optimizer.load_state_dict(state["optimizer_state"])  # m、v和累计t一起恢复。
    scaler = torch.amp.GradScaler("cuda")  # 先创建再恢复动态缩放历史。
    scaler.load_state_dict(state["scaler_state"])  # 不重置缩放比例。
    torch.set_rng_state(state["torch_rng_state"])  # CPU随机状态也属于完整进度。
    torch.cuda.set_rng_state_all(state["cuda_rng_states"])  # 恢复GPU随机状态。
    return model, optimizer, scaler, state  # 保留下一批数据位置。


def update(model, optimizer, scaler, cursor, stream, seed):  # 只对核验副本执行一步，不更新正式产物。
    cursor, batches = dict(cursor), []  # 输入游标按值复制。
    windows = int(np.ceil((len(stream) - 1) / 256))  # 与真实训练一致的尾窗数量。
    for _ in range(8):  # 八微批按有效目标数累计。
        order = np.random.default_rng(seed + cursor["epoch"]).permutation(windows)  # 同一轮同一排列。
        batches.append(batch_from_stream(stream, int(order[cursor["offset"]])))  # 不重复上次已更新窗口。
        cursor["offset"] += 1  # 暂存下一微批位置。
        if cursor["offset"] == windows:  # 轮末进入下一轮。
            cursor = {"epoch": cursor["epoch"] + 1, "offset": 0}  # 重建下一轮排列。
    valid = sum(item[2] for item in batches)  # 统一分母兼容尾窗。
    model.train()  # 同正式更新模式。
    optimizer.zero_grad(set_to_none=True)  # 只清本组旧梯度。
    for x, y, _ in batches:  # 累积同一个优化器更新。
        with torch.autocast(device_type="cuda", dtype=torch.float16):  # 与原训练一致。
            loss = F.cross_entropy(model(x.cuda()).flatten(0, 1), y.cuda().flatten(), reduction="sum") / valid  # 不是直接平均微批均值。
        scaler.scale(loss).backward()  # 真实自动求导。
    scaler.unscale_(optimizer)  # 先还原再裁剪。
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)  # 核验时异常必须显式失败。
    scale = scaler.get_scale()  # 不把被跳过的更新当作成功。
    scaler.step(optimizer)  # 更新核验副本。
    scaler.update()  # 更新对应缩放历史。
    assert scaler.get_scale() >= scale, "核验更新溢出被跳过"  # 这次比较必须是真实更新。
    return cursor, valid  # 用于下一批核验。


def main():  # 真实30M加载、再保存和下一批轨迹比较。
    parser = argparse.ArgumentParser()  # 默认仍核验第四十八课。
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model/lesson48/last_train.pt")  # 也可指定后续课程完整状态。
    parser.add_argument("--report", type=Path)  # 默认由检查点课程目录决定，避免覆盖旧课证据。
    args = parser.parse_args()  # 开始前明确本次范围。
    if args.report is None:  # 同课默认报告可直接复现，外部路径需要显式指定。
        assert args.checkpoint.parent.name in ("lesson48", "lesson50"), "其他目录请明确指定--report"  # 不将新课结果误写回第四十八课。
        args.report = ROOT / "docs/training_runs" / f"{args.checkpoint.parent.name}_resume.json"  # 依据来源课程选择独立文件。
    torch.set_num_threads(4)  # 不占满CPU。
    torch.use_deterministic_algorithms(True)  # 比较相同数值路径。
    torch.backends.cuda.matmul.allow_tf32 = False  # 与训练设置一致。
    path = args.checkpoint  # 正式文件只读。
    reference, reference_optimizer, reference_scaler, state = restore(path)  # 从实际完整状态开始。
    stream = np.load(DIRECTORY / "tokenizer/pretrain_train.npy", mmap_mode="r", allow_pickle=False)  # 同一训练流。
    cursor, valid = update(reference, reference_optimizer, reference_scaler, state["cursor"], stream, state["data_seed"])  # 在副本先更新一次建立待保存状态。
    with tempfile.TemporaryDirectory() as name:  # 临时文件不会发布为新训练成果。
        temporary = Path(name) / "resume.pt"  # 检查中断时刻的完整状态。
        snapshot = {**state, "model_state": cpu_tree(reference.state_dict()), "optimizer_state": cpu_tree(reference_optimizer.state_dict()), "scaler_state": reference_scaler.state_dict(), "cursor": cursor, "step": state["step"] + 1, "seen_positions": state["seen_positions"] + valid, "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all()}  # 参数、历史与数据位置属于同一步。
        torch.save(snapshot, temporary)  # 实际写入后重新读取。
        recovered, recovered_optimizer, recovered_scaler, recovered_state = restore(temporary)  # 另一份恢复副本。
        update(reference, reference_optimizer, reference_scaler, cursor, stream, state["data_seed"])  # 不间断参考继续下一步。
        update(recovered, recovered_optimizer, recovered_scaler, recovered_state["cursor"], stream, state["data_seed"])  # 恢复副本执行完全相同下一步。
        difference = max(float((a - b).abs().max().item()) for a, b in zip(reference.parameters(), recovered.parameters()))  # 比较全部参数。
        assert difference <= 1e-6, "恢复后的参数轨迹不同"  # 不只核对前向预测。
        for left, right in zip(reference_optimizer.state.values(), recovered_optimizer.state.values()):  # 同时核对AdamW状态。
            assert all(torch.allclose(left[key], right[key], atol=1e-6, rtol=1e-6) for key in ("step", "exp_avg", "exp_avg_sq")), "恢复后的优化器状态不同"  # m、v、t均应相符。
        assert reference_scaler.state_dict() == recovered_scaler.state_dict(), "恢复后的Scaler不同"  # 混合精度历史也要延续。
    result = {"formal_checkpoint_step": state["step"], "verification_only_updates": 3, "max_parameter_abs_diff": difference, "adamw_and_scaler_equal": True, "scope": "actual checkpoint -> one copy update -> temporary save -> same next update against uninterrupted copy; formal checkpoint unchanged"}  # 三次副本更新不计入正式训练预算。
    args.report.parent.mkdir(parents=True, exist_ok=True)  # 确保独立报告目录存在。
    args.report.write_bytes(dump(result))  # 单独记录核验，不修改原训练报告。
    print(json.dumps(result, ensure_ascii=False, indent=2))  # 展示真实最大误差。


if __name__ == "__main__":  # 导入脚本不会修改模型。
    main()  # 直接运行才执行副本核验。
