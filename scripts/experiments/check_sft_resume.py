import json  # 保存正式SFT完整状态的恢复核验。
import tempfile  # 核验副本不覆盖正式文件。
from pathlib import Path  # 定位临时保存文件。
import numpy as np  # 重建下一组会话顺序。
from train_chat_sft import ChatLM, torch, nn, F, load_data, tensors, ROOT, OUT  # 复用实际模型和批次规则。
from prepare_chat_corpus import sha, dump  # 核对正式文件没有被改写。
from bounded_recurrent import cpu_tree  # 比较CPU上的完整参数与优化器历史。


def branch(path, rows, tokenizer):  # 每条核验分支只在独立副本上做下一次更新。
    saved = torch.load(path, map_location="cpu", weights_only=True)  # 必须是完整SFT状态。
    assert saved["format"] == "chat_sft53_train_v1", "不是完整SFT断点"  # 推理文件不能恢复AdamW。
    model = ChatLM(**saved["model_config"]).cuda()  # 不更新正式文件。
    model.load_state_dict(saved["model_state"])  # FP32主参数准确恢复。
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.01)  # 随后恢复保存的参数组与历史。
    optimizer.load_state_dict(saved["optimizer_state"])  # m、v和t都不能重置。
    scaler = torch.amp.GradScaler("cuda")  # 使用相同AMP路径。
    scaler.load_state_dict(saved["scaler_state"])  # 恢复scale与增长计数。
    torch.set_rng_state(saved["torch_rng_state"])  # 恢复CPU随机状态。
    torch.cuda.set_rng_state_all(saved["cuda_rng_states"])  # 恢复GPU随机状态。
    cursor, references = dict(saved["cursor"]), []  # 原cursor仍指向下一条尚未处理的会话。
    for _ in range(8):  # 下一正式累积组有八条会话。
        order = np.random.default_rng(saved["data_seed"] + cursor["epoch"]).permutation(len(rows))  # 与训练一致的逐轮排列。
        references.append(int(order[cursor["offset"]]))  # 记录真实行号。
        cursor["offset"] += 1  # 推进副本游标。
        if cursor["offset"] == len(rows):  # 允许累积组跨轮。
            cursor = {"epoch": cursor["epoch"] + 1, "offset": 0}  # 不重置优化器历史。
    batches = [tensors([rows[index] for index in references[i:i + 2]], tokenizer, "cuda") for i in range(0, 8, 2)]  # 与正式训练一样分四微批。
    positions = sum(int((y != -100).sum().item()) for _, y in batches)  # 整组真实助手目标分母。
    optimizer.zero_grad(set_to_none=True)  # 每组开始清空旧梯度。
    model.train()  # 同一训练模式。
    for x, y in batches:  # 同样的四次反向累积。
        with torch.autocast("cuda", dtype=torch.float16):  # 主参数仍FP32。
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum") / positions  # 不平均微批均值。
        scaler.scale(loss).backward()  # 使用已恢复的动态缩放。
    scaler.unscale_(optimizer)  # 裁剪前还原真实梯度。
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)  # 异常必须显式失败。
    old_scale = scaler.get_scale()  # 比较的更新不能被跳过。
    scaler.step(optimizer)  # 核验分支唯一一次更新。
    scaler.update()  # 更新对应缩放历史。
    assert scaler.get_scale() >= old_scale, "核验更新溢出跳过"  # 否则不能声称比较了下一步。
    return cpu_tree(model.state_dict()), cpu_tree(optimizer.state_dict()), scaler.state_dict(), cursor  # GPU对象离开函数后释放。


def equal(left, right):  # 深层比较AdamW和Scaler结构。
    if isinstance(left, torch.Tensor):  # 张量逐元素比较。
        return torch.equal(left, right)  # 不只比较平均范数。
    if isinstance(left, dict):  # 键和值都应一致。
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)  # 递归比对全部历史。
    if isinstance(left, (tuple, list)):  # 参数组内列表保持相同顺序。
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))  # 不忽略某个参数状态。
    return left == right  # 普通整数与浮点配置直接比较。


def main():  # 保存重载后下一步应与直接恢复分支一致。
    torch.set_num_threads(4)  # 与正式训练相同线程设置。
    torch.backends.cuda.matmul.allow_tf32 = False  # 维持数值路径。
    rows, tokenizer, manifest = load_data()  # 不改变冻结数据。
    path = OUT / "last_train.pt"  # 核验最后正式进度。
    before = sha(path.read_bytes())  # 源文件不能被核验更新改写。
    saved = torch.load(path, map_location="cpu", weights_only=True)  # 用普通配置和张量恢复。
    assert saved["sft_manifest_sha256"] == sha((ROOT / "data/chat53/frozen_v1/manifest.json").read_bytes()) and saved["tokenizer_sha256"] == manifest["tokenizer_sha256"], "恢复数据版本错误"  # 游标只适用于同一版本。
    assert {int(item["step"].item()) for item in saved["optimizer_state"]["state"].values()} == {saved["step"]}, "AdamW与模型累计步数不同"  # 新建空优化器不能通过。
    with tempfile.TemporaryDirectory() as directory:  # 大文件仅在核验期间存在。
        temporary = Path(directory) / "roundtrip.pt"  # 不覆盖任何正式状态。
        torch.save(saved, temporary)  # 真正保存完整模型、优化器、Scaler和游标。
        first = branch(path, rows["train"], tokenizer)  # 直接恢复后做同一个下一步。
        torch.cuda.empty_cache()  # 释放第一分支缓存以减小峰值。
        second = branch(temporary, rows["train"], tokenizer)  # 保存重载后再做相同下一步。
    error = max(float((first[0][key] - second[0][key]).abs().max().item()) for key in first[0])  # 全部矩阵逐元素核对。
    same = equal(first[1], second[1]) and equal(first[2], second[2]) and first[3] == second[3]  # 完整历史与下一位置一起核对。
    assert error <= 1e-6 and same and sha(path.read_bytes()) == before, "SFT恢复不一致或正式文件被改写"  # 核验不增加正式500步。
    result = {"formal_checkpoint_step": saved["step"], "pretrain_step": saved["pretrain_step"], "verification_copy_updates": 2, "max_parameter_abs_diff": error, "adamw_scaler_cursor_equal": same, "formal_checkpoint_sha256_unchanged": before}  # 验证副本更新单独记账。
    (ROOT / "docs/training_runs/lesson53_resume.json").write_bytes(dump(result))  # 与正式训练报告分开。
    print(json.dumps(result, ensure_ascii=False, indent=2))  # 只打印核验结果。


if __name__ == "__main__":  # 导入不自动更新副本。
    main()  # 仅显式命令运行两次核验更新。
