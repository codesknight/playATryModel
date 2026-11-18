import argparse  # 从命令行读取明确的训练预算。
import contextlib  # 隐藏旧数据模块导入时的重复演示输出。
import copy  # 在核验副本上比较不中断和恢复后的下一步。
import hashlib  # 固定数据和公共初始参数的指纹。
from importlib import import_module  # 复用原四书窗口规则。
import io  # 接收旧模块的演示输出。
import json  # 持续保存实验配置、指标和文字。
import os  # 在加载CUDA前设置确定性配置。
from pathlib import Path  # 所有输出都定位到项目目录。
import shutil  # 复制最佳检查点，避免随后被覆盖。
import sys  # 设置已有学习模块的导入路径。
from time import perf_counter  # 在批次边界检查时间预算。

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # CUDA确定性矩阵计算需要在导入前配置。
import numpy as np  # 复现窗口打乱及温度采样。
import torch  # 使用自动求导和可用GPU进行真实训练。
from torch import nn  # 搭建表示表、循环层和输出层。
from torch.nn import functional as F  # 计算稳定的交叉熵。

ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于scripts/experiments。
sys.path.insert(0, str(ROOT / "scripts/learn"))  # 导入已有数据处理模块。
PROMPTS = ("春风", "唐僧道：", "宝玉听了")  # 沿用固定开头，不挑选最好看的例子。


class RecurrentLM(nn.Module):  # 两个模型仅切换循环层类型，公共维度相同。
    def __init__(self, vocab_size, kind, width=128, layers=2):  # 默认仍是远小于一GB的小模型。
        super().__init__()  # 注册子模块和可学习参数。
        assert kind in ("rnn", "gru"), "未知循环层类型"  # 不接受不在本实验中的结构。
        self.spec = {"vocab_size": vocab_size, "kind": kind, "width": width, "layers": layers}  # 保存重建模型所需配置。
        self.embedding = nn.Embedding(vocab_size, width)  # 字符编号转换为向量。
        self.head = nn.Linear(width, vocab_size)  # 先创建公共输出层，确保两组同种子时公共初始化相同。
        cell = nn.RNN if kind == "rnn" else nn.GRU  # RNN用tanh，GRU使用门控状态更新。
        self.recurrent = cell(width, width, num_layers=layers, batch_first=True, dropout=0.0)  # 使用相同层数和宽度，不额外引入dropout变量。

    def forward(self, ids):  # 每个窗口独立从零隐藏状态开始。
        states, _ = self.recurrent(self.embedding(ids))  # 输出形状为批大小、窗口长度、隐藏维度。
        return self.head(states)  # 为每个位置产生词表分数。


def cpu_tree(value):  # 将优化器状态中的张量递归移到CPU后保存。
    if isinstance(value, torch.Tensor):  # 参数、动量和随机状态都可能是张量。
        return value.detach().cpu().clone()  # 文件不依赖当前GPU存储。
    if isinstance(value, dict):  # 优化器状态包含多层字典。
        return {key: cpu_tree(item) for key, item in value.items()}  # 保留原键和结构。
    if isinstance(value, (list, tuple)):  # 保留参数组和系数序列类型。
        return type(value)(cpu_tree(item) for item in value)  # 递归处理各元素。
    return value  # 普通整数、浮点数和字符串直接保存。


def write_json(path, report):  # 检查点后及时保存可审阅记录。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保留中文文字和真实数值。


def load_data(device):  # 原始四书数据划分和词表保持不变。
    with contextlib.redirect_stdout(io.StringIO()):  # 不重复打印以前课程的演示。
        data = import_module("06_make_windows")  # 使用已核验的章节内窗口。
    records = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 只读取验证集，不读测试集。
    assert not ({row["id"] for row in data.train_records} & {row["id"] for row in records}), "训练验证章节重叠"  # 保留整章划分。
    tensors = []  # 分别构造训练和验证的输入目标。
    for source in (data.train_records, records):  # 不额外引入外部语料。
        windows = list(data.iter_windows(source, 32, 32))  # 与旧RNN评估使用相同窗口长度和步长。
        x = torch.as_tensor(np.asarray([row["x"] for row in windows], dtype=np.int64), device=device)  # GPU预算允许时将小数据一次送入显存。
        y = torch.as_tensor(np.asarray([row["y"] for row in windows], dtype=np.int64), device=device)  # 保存错位目标编号。
        assert torch.equal(x[:, 1:], y[:, :-1]), "输入目标未正确错位"  # 保留原预测任务定义。
        tensors.append((x, y))  # 后续只打乱窗口次序。
    assert tensors[0][0].shape == (76584, 32) and tensors[1][0].numel() == 316960, "窗口覆盖变化"  # 确保与旧指标可比。
    return data.vocab, tensors[0], tensors[1]  # 返回固定词表及两个数据流的张量。


@torch.inference_mode()  # 验证不构建反向计算图。
def evaluate(model, validation):  # 所有候选都评估相同完整验证窗口。
    was_training = model.training  # 验证后恢复原模式。
    model.eval()  # 明确进入评估模式。
    x, y = validation  # 输入目标包含全部验证窗口。
    total, correct = 0.0, 0  # 按位置累计损失和最高分命中数。
    for start in range(0, len(x), 64):  # 保留验证尾批。
        target = y[start:start + 64]  # 使用当前实际目标数。
        logits = model(x[start:start + 64])  # 只前向计算分数。
        total += float(F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum").item())  # 全部位置加总，最后统一平均。
        correct += int((logits.argmax(-1) == target).sum().item())  # 逐位置统计命中数。
    model.train(was_training)  # 不改变后续训练模式。
    return {"loss": total / y.numel(), "positions": y.numel(), "correct": correct, "accuracy": correct / y.numel()}  # 记录同一评估范围的指标。


@torch.inference_mode()  # 生成不计算梯度。
def samples(model, vocab):  # 每个候选都保留固定六段文字。
    model.eval()  # 固定生成时的行为。
    tokens = vocab["tokens"]  # 保存编号对应的字符。
    char_to_id = {char: index for index, char in enumerate(tokens)}  # 使用原词表编码开头。
    device = next(model.parameters()).device  # 输入与模型保持同一设备。
    rows = []  # 不只保留好看的例子。
    for prompt in PROMPTS:  # 三个开头与旧实验一致。
        for mode in ("greedy", "topk40_t07"):  # 固定贪心和top-k采样两种设置。
            history = [char_to_id.get(char, vocab["unk_id"]) for char in prompt]  # 不扩充词表。
            generated, rng = [], np.random.default_rng(42)  # 每段固定同一个采样种子。
            for _ in range(60):  # 每段续写六十个编号。
                scores = model(torch.tensor([history[-32:]], device=device))[0, -1].cpu().numpy()  # 与旧模型一样只看最近三十二个输入。
                if mode == "greedy":  # 每次选最高分。
                    index = int(scores.argmax())  # 不引入随机抽样。
                else:  # 在最高分的四十个候选内以温度0.7采样。
                    candidates = np.argsort(-scores, kind="stable")[:40]  # 稳定排序处理并列分数。
                    weights = np.exp((scores[candidates].astype(np.float64) - scores[candidates].max()) / 0.7)  # 双精度稳定计算抽样权重。
                    index = int(rng.choice(candidates, p=weights / weights.sum()))  # 重新归一化候选概率。
                history.append(index)  # 将新字接回上下文。
                generated.append(index)  # 单独记录续写编号。
            pieces = {tuple(generated[start:start + 4]) for start in range(57)}  # 重复率只统计续写部分。
            rows.append({"prompt": prompt, "mode": mode, "seed": 42, "text": prompt + "".join(tokens[index] for index in generated), "generated_ids": generated, "repeat4": 1 - len(pieces) / 57})  # 原始输出完整保留。
    model.train()  # 后续训练继续正常运行。
    return rows  # 人工阅读与重复统计分开判断。


def restore_checkpoint(path, device):  # 恢复模型和对应Adam历史，不只加载权重。
    state = torch.load(path, map_location="cpu", weights_only=True)  # 只加载张量和普通配置，不反序列化自定义类。
    assert state["format"] == "playatry_torch_recurrent_v1", "检查点格式不正确"  # 不混用旧NumPy权重。
    model = RecurrentLM(**state["model_config"]).to(device)  # 按保存的结构重建模型。
    model.load_state_dict(state["model_state"])  # 完整恢复参数。
    model.recurrent.flatten_parameters()  # 恢复后整理cuDNN权重布局，不改变参数数值。
    optimizer = torch.optim.Adam(model.parameters(), lr=state["learning_rate"])  # 创建对应优化器后恢复状态。
    optimizer.load_state_dict(state["optimizer_state"])  # 恢复每个参数的动量和步数。
    torch.set_rng_state(state["torch_rng_state"])  # 恢复CPU随机状态。
    if device.type == "cuda":  # 同机CUDA恢复时还原GPU随机状态。
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])  # 不把GPU状态当成模型参数。
    return model, optimizer, state  # 数据游标和当前排列也保留在state中。


def verify_resume(live_model, live_optimizer, last_path, train, device):  # 只在副本上验证恢复后的下一批一致。
    reference = copy.deepcopy(live_model)  # 保留不中断训练时的参数副本，不改正式模型。
    reference.recurrent.flatten_parameters()  # 深复制后重新整理连续权重存储。
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=live_optimizer.param_groups[0]["lr"])  # 为副本重建同类优化器。
    reference_optimizer.load_state_dict(cpu_tree(live_optimizer.state_dict()))  # 延续真实运行中的历史。
    recovered, recovered_optimizer, state = restore_checkpoint(last_path, device)  # 从磁盘恢复另一份副本。
    x, y = train  # 使用相同训练窗口。
    epoch, start = state["cursor"]["next_epoch"], state["cursor"]["next_start"]  # 恢复下一批游标。
    order = state["epoch_order"].numpy() if epoch == state["order_epoch"] else np.random.default_rng(state["data_seed"] + epoch).permutation(len(x))  # 跨轮边界时重建下一轮排列。
    indices = torch.as_tensor(order[start:start + state["batch_size"]], device=device)  # 两个副本使用同一批编号。
    assert len(indices) > 0, "恢复的数据游标无有效下一批"  # 防止错误游标悄悄跳过数据。
    for model, optimizer in ((reference, reference_optimizer), (recovered, recovered_optimizer)):  # 各在核验副本上执行一步，不计入正式训练产物。
        model.train()  # 与真实训练模式一致。
        optimizer.zero_grad(set_to_none=True)  # 清梯度而不清动量。
        loss = F.cross_entropy(model(x[indices]).flatten(0, 1), y[indices].flatten())  # 同一下一批的平均损失。
        loss.backward()  # 完成副本上的反向。
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)  # 保持相同裁剪规则。
        optimizer.step()  # 正式检查点文件不会被写回。
    difference = max(float((left - right).abs().max().item()) for left, right in zip(reference.parameters(), recovered.parameters()))  # 比较实际下一步后的全部参数。
    assert difference <= 1e-6, "恢复后的下一批参数轨迹不一致"  # 同机核对包含Adam历史的恢复能力。
    for left, right in zip(reference_optimizer.state.values(), recovered_optimizer.state.values()):  # 同时核对更新后的历史状态。
        assert all(torch.allclose(left[key], right[key], rtol=1e-6, atol=1e-6) for key in ("step", "exp_avg", "exp_avg_sq")), "恢复后的优化器历史不同"  # 不只比较前向预测。
    return difference  # 核验副本随后释放，不作为新的训练结果。


def train_kind(kind, vocab, train, validation, args, report, report_path):  # 独立训练一个结构，遵守相同步数与时间预算。
    torch.manual_seed(args.seed)  # 两组公共表示表与输出层从相同随机状态创建。
    model = RecurrentLM(len(vocab["tokens"]), kind, args.width, args.layers).to(args.device)  # 本组不加载旧模型作为不对等起点。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # 两组使用同一Adam配置。
    public_hash = hashlib.sha256()  # 核对公共初始参数确实相同。
    for name in ("embedding.weight", "head.weight", "head.bias"):  # 循环层因结构不同而不要求同形。
        public_hash.update(model.state_dict()[name].detach().cpu().numpy().tobytes())  # 保留初始化证据。
    result = {"kind": kind, "parameter_count": sum(value.numel() for value in model.parameters()), "public_initial_sha256": public_hash.hexdigest(), "history": [], "status": "running"}  # 逐检查点更新记录。
    report["models"].append(result)  # 提前落盘，避免只有最后才有记录。
    x, y = train  # 使用相同训练窗口集合。
    started, step, seen_positions = perf_counter(), 0, 0  # 时间预算包含本组周期评估，收尾另计。
    exposure, best_loss = hashlib.sha256(), float("inf")  # 跟踪真实数据顺序与最低验证损失。
    order = np.random.default_rng(args.data_seed).permutation(len(x)).astype(np.int64)  # 起点也保存真正将使用的首轮排列。
    order_epoch = 0  # 记录保存的排列属于哪一轮。
    cursor = {"next_epoch": 0, "next_start": 0}  # 保存下一批应该从哪里开始。
    last_path = args.output_dir / f"lesson42_{kind}_last.pt"  # 最后进度用于恢复训练。
    best_path = args.output_dir / f"lesson42_{kind}_best.pt"  # 最佳文件用于选优。

    def capture():  # 固定参数评估并保存完整状态。
        nonlocal best_loss  # 更新本组当前最低分。
        metrics = evaluate(model, validation)  # 完整验证损失与原验证范围一致。
        row = {"step": step, "seen_positions": seen_positions, **metrics, "samples": samples(model, vocab)}  # 全部固定文字保留。
        result["history"].append(row)  # 不丢弃变差的检查点。
        state = {"format": "playatry_torch_recurrent_v1", "model_config": model.spec, "vocab": vocab, "model_state": cpu_tree(model.state_dict()), "optimizer_state": cpu_tree(optimizer.state_dict()), "learning_rate": args.learning_rate, "step": step, "seen_positions": seen_positions, "cursor": dict(cursor), "order_epoch": order_epoch, "epoch_order": torch.from_numpy(order.copy()), "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all() if args.device.type == "cuda" else [], "data_seed": args.data_seed, "batch_size": args.batch_size}  # 恢复需要参数、优化器、数据位置及随机状态。
        temporary = last_path.with_suffix(".tmp")  # 先写临时文件避免覆盖到一半。
        torch.save(state, temporary)  # 保存张量与普通配置。
        temporary.replace(last_path)  # 完成后替换本组最后状态。
        if metrics["loss"] < best_loss:  # 严格改善才替换最佳文件。
            best_loss = metrics["loss"]  # 保留最低原始验证损失。
            shutil.copyfile(last_path, best_path)  # 参数与Adam状态来自同一位置。
            result["best_step"] = step  # 最佳位置与停止位置分开。
        result.update({"best_loss": best_loss, "best_checkpoint": best_path.name, "last_checkpoint": last_path.name, "completed_steps": step, "seen_positions": seen_positions})  # 保存当前进度。
        write_json(report_path, report)  # 每次完整验证后落盘。
        print(f"{kind} step={step} validation={metrics['loss']:.6f} accuracy={metrics['accuracy']:.2%}", flush=True)  # 不把准确率当作自由生成质量。

    capture()  # 未训练起点也参加本组最佳选择。
    reason = "epochs"  # 正常达到轮数预算时的停止原因。
    for epoch in range(args.max_epochs):  # 轮数也有预先规定的上限。
        order_epoch = epoch  # 将当前排列与所属轮数一起保存。
        order = np.random.default_rng(args.data_seed + epoch).permutation(len(x)).astype(np.int64)  # 两组每轮使用同一窗口排列。
        for start in range(0, len(x), args.batch_size):  # 尾批按实际大小处理。
            if step >= args.max_steps or perf_counter() - started >= args.max_seconds:  # 到截止点不再启动新训练批次。
                reason = "steps" if step >= args.max_steps else "time"  # 如实区分停止原因。
                break  # 最后的验证和保存仍需完成。
            indices = order[start:start + args.batch_size]  # 当前实际训练窗口编号。
            selection = torch.as_tensor(indices, device=args.device)  # 在GPU上索引小批次。
            optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度，不清空Adam历史。
            logits = model(x[selection])  # 当前批次只进行一次前向。
            loss = F.cross_entropy(logits.flatten(0, 1), y[selection].flatten())  # 按实际预测位置平均。
            assert torch.isfinite(loss).item(), "训练损失无效"  # 无效计算不能继续更新。
            loss.backward()  # 自动求导按计算图完成反向传播。
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)  # 两组相同的全局范数裁剪。
            optimizer.step()  # 同一优化器跨批次保留状态。
            step += 1  # 一批仅计一次更新。
            seen_positions += len(indices) * 32  # 记录实际目标预算。
            exposure.update(indices.tobytes())  # 完全相同前缀应产生相同指纹。
            end = start + len(indices)  # 计算下一批的数据位置。
            cursor = {"next_epoch": epoch + (end == len(x)), "next_start": 0 if end == len(x) else end}  # 不把最后一批重复训练。
            if step % 200 == 0:  # 长训练时显示真实进度。
                print(f"{kind} step={step}/{args.max_steps} loss={loss.item():.6f} elapsed={perf_counter()-started:.1f}s", flush=True)  # 当前批损失不用于选优。
            if step == 1000:  # 预先指定的中途完整验证点。
                capture()  # 验证后继续同一个模型和Adam。
        if reason != "epochs":  # 达到时间或步数上限后停止整个外层循环。
            break  # 不自动延长训练预算。
    if result["history"][-1]["step"] != step:  # 停止位置尚未验证时统一收尾。
        capture()  # 最终完整验证和保存可略超出训练循环截止时间。
    result.update({"status": "completed", "stop_reason": reason, "elapsed_seconds": perf_counter() - started, "exposure_sha256": exposure.hexdigest()})  # 此时间含周期评估和保存，不含随后恢复核验收尾。
    result["restore_next_batch_max_abs_diff"] = verify_resume(model, optimizer, last_path, train, args.device)  # 额外副本检查不改变正式训练步数或文件。
    model, _, restored = restore_checkpoint(best_path, args.device)  # 重新加载实际最佳状态。
    check = evaluate(model, validation)  # 核对保存重载后完整验证指标。
    assert np.isclose(check["loss"], best_loss, rtol=1e-5, atol=1e-5), "重载最佳指标不一致"  # 允许平台小浮点差。
    assert restored["step"] == result["best_step"], "最佳文件位置不一致"  # 不能误把最后状态当成最佳状态。
    result["reload_verified"] = True  # 记录实际文件加载检查通过。
    for sample in next(row for row in result["history"] if row["step"] == result["best_step"])["samples"]:  # 展示预定全部六段。
        print(kind, sample["mode"], repr(sample["text"]), flush=True)  # 不只展示最好看的一句。
    write_json(report_path, report)  # 保存该组最终状态。
    del model, optimizer  # 释放本组GPU对象。
    if args.device.type == "cuda":  # 两组之间清理空闲缓存。
        torch.cuda.empty_cache()  # 不结束其他用户GPU进程。


def main():  # 固定预算运行两个独立模型，不自动扩数据或调参。
    parser = argparse.ArgumentParser()  # 命令行预算清楚可审阅。
    parser.add_argument("--max-steps", type=int, default=2394)  # 每组最多两整轮的更新数。
    parser.add_argument("--max-epochs", type=int, default=2)  # 每组最多遍历两轮。
    parser.add_argument("--max-seconds", type=float, default=360)  # 截止后不再启动新批次，收尾验证另计。
    parser.add_argument("--batch-size", type=int, default=64)  # 两组相同批大小。
    parser.add_argument("--width", type=int, default=128)  # 两组相同表示与隐藏维度。
    parser.add_argument("--layers", type=int, default=2)  # 两组相同循环层数。
    parser.add_argument("--learning-rate", type=float, default=0.001)  # 两组相同Adam学习率。
    parser.add_argument("--seed", type=int, default=42)  # 两组相同公共初始化种子。
    parser.add_argument("--data-seed", type=int, default=47)  # 每轮窗口排列独立于模型初始化。
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")  # 可在CPU复现小预算检查。
    parser.add_argument("--threads", type=int, default=4)  # 限制CPU线程避免过度争用。
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model/lesson42")  # 不覆盖已有NumPy模型。
    parser.add_argument("--report", type=Path, default=ROOT / "docs/training_runs/lesson42.json")  # 独立保存本次实验记录。
    args = parser.parse_args()  # 预算在看到结果前确定。
    assert args.max_steps > 0 and args.max_epochs > 0 and args.max_seconds > 0 and args.batch_size > 0, "预算必须为正"  # 不接受无限训练设置。
    args.device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)  # 选择实际可用设备。
    torch.set_num_threads(args.threads)  # 只影响本进程。
    torch.use_deterministic_algorithms(True)  # 不允许静默使用已知非确定性算法。
    torch.backends.cudnn.benchmark = False  # 不为本次数据反复选择不同算法。
    torch.backends.cudnn.deterministic = True  # 固定cuDNN循环计算路径。
    torch.backends.cuda.matmul.allow_tf32 = False  # 使用float32对照，不引入不同精度变量。
    args.output_dir.mkdir(parents=True, exist_ok=True)  # 创建本实验独立输出目录。
    vocab, train, validation = load_data(args.device)  # 原数据不重分、不扩词表。
    legacy = json.loads((ROOT / "docs/training_runs/lesson41.json").read_text(encoding="utf-8"))  # 保留旧模型已复核的分数作为参考。
    assert legacy["train_sha256"] == vocab["train_sha256"], "旧参考模型数据版本不同"  # 同验证范围才可比较。
    config = {key: str(value) if isinstance(value, (Path, torch.device)) else value for key, value in vars(args).items()}  # 只保存普通配置，不序列化命令行对象。
    report = {"status": "running", "config": config, "torch_version": str(torch.__version__), "numpy_version": np.__version__, "device": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else "cpu", "train_sha256": vocab["train_sha256"], "validation_positions": validation[1].numel(), "legacy_reference_loss": legacy["best_validation_loss"], "models": []}  # 旧模型不是两组随机初始化的公平起点。
    write_json(args.report, report)  # 训练开始前保存预算。
    for kind in ("rnn", "gru"):  # 两组都从头训练相同预算。
        train_kind(kind, vocab, train, validation, args, report, args.report)  # 不因第一组结果临时调整第二组预算。
    left, right = report["models"]  # 核对实际公平条件。
    assert left["public_initial_sha256"] == right["public_initial_sha256"], "公共初始化不同"  # 表示表与输出层应完全相同。
    report["matched_exposure"] = left["exposure_sha256"] == right["exposure_sha256"]  # 如果时间上限导致预算不同则明确记录。
    selected = min(report["models"], key=lambda row: row["best_loss"])  # 先记录两组同预算模型中的最低分。
    keep_legacy = legacy["best_validation_loss"] <= selected["best_loss"]  # 新模型都更差时仍保留旧模型。
    report.update({"status": "completed", "paired_selected_kind": selected["kind"], "selected_kind": "legacy_numpy_rnn" if keep_legacy else selected["kind"], "selected_loss": min(legacy["best_validation_loss"], selected["best_loss"])})  # 明确区分公平对照赢家和最终使用候选。
    artifact_names = {row[key] for row in report["models"] for key in ("best_checkpoint", "last_checkpoint")}  # 两个结构各保留最佳和最后状态。
    report["artifacts"] = {name: {"bytes": (args.output_dir / name).stat().st_size, "sha256": hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()} for name in sorted(artifact_names)}  # 保存二进制文件的大小与来源指纹。
    assert all(item["bytes"] < 1_000_000_000 for item in report["artifacts"].values()), "单个检查点超过项目大小要求"  # 权重和Adam状态合在一起仍须小于一GB。
    write_json(args.report, report)  # 保存最终完整结果。
    print(json.dumps({"selected": report["selected_kind"], "loss": report["selected_loss"], "matched_exposure": report["matched_exposure"]}), flush=True)  # 方便核对实验结论。


if __name__ == "__main__":  # 直接运行才开始预算内训练。
    main()  # 导入该文件不会启动实验。
