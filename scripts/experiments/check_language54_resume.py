import json  # 保存混合语言完整状态恢复证据。
from pathlib import Path  # 临时文件明确放在新课目录。
import numpy as np  # 重建三个域的下一组窗口。
from train_language54 import load_data, OUT, DIRECTORY, LANGUAGE, DOMAINS, SCHEDULE  # 复用正式数据配比。
from train_chat_pilot import batch_from_stream  # 与实际训练相同的尾窗规则。
from chat_model import ChatLM, torch, F, nn  # 使用真实30M自动求导。
from bounded_recurrent import cpu_tree  # 比较完整CPU状态。
from check_sft_resume import equal  # 深层比较张量、字典与参数组。
from prepare_chat_corpus import ROOT, sha, dump  # 校验文件未改写。


def restore_mixed(path):  # 恢复参数、AdamW、Scaler和三个域游标。
    saved = torch.load(path, map_location="cpu", weights_only=True)  # 只接受完整混合状态。
    assert saved["format"] == "chat_mix54_train_v1", "不是完整混合预训练状态"  # 不把推理文件当断点。
    assert saved["language_manifest_sha256"] == sha((DIRECTORY / LANGUAGE / "manifest.json").read_bytes()), "混合数据版本改变"  # 游标不能用在不同语料。
    model = ChatLM(**saved["model_config"]).cuda()  # 独立核验副本。
    model.load_state_dict(saved["model_state"])  # 载入FP32主参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)  # 随后恢复实际参数组。
    optimizer.load_state_dict(saved["optimizer_state"])  # m、v和t一并恢复。
    scaler = torch.amp.GradScaler("cuda")  # 先创建再恢复历史。
    scaler.load_state_dict(saved["scaler_state"])  # 保留动态缩放比例与增长计数。
    torch.set_rng_state(saved["torch_rng_state"])  # 恢复CPU随机状态。
    torch.cuda.set_rng_state_all(saved["cuda_rng_states"])  # 恢复GPU随机状态。
    return model, optimizer, scaler, saved  # 三个游标在saved中保留。


def branch(path, arrays, manifest):  # 只给核验副本做相同的下一次更新。
    model, optimizer, scaler, saved = restore_mixed(path)  # 不修改正式文件。
    cursors, items = {d: dict(c) for d, c in saved["cursors"].items()}, []  # 游标逐层复制。
    for domain in SCHEDULE:  # 同样的4/3/1八个窗口。
        index, cursor = DOMAINS.index(domain), cursors[domain]  # 定位当前来源。
        windows = manifest["stats"][domain]["train_windows"]  # 各域窗口数不同。
        order = np.random.default_rng(saved["data_seed"] + index * 1000 + cursor["epoch"]).permutation(windows)  # 与正式更新相同。
        items.append(batch_from_stream(arrays[domain + "_train.npy"], int(order[cursor["offset"]])))  # 保留真实尾段长度。
        cursor["offset"] += 1  # 只推进副本游标。
        if cursor["offset"] == windows:  # 域内跨轮处理。
            cursors[domain] = {"epoch": cursor["epoch"] + 1, "offset": 0}  # 不清空优化器。
    positions = sum(item[2] for item in items)  # 共享有效目标分母。
    optimizer.zero_grad(set_to_none=True)  # 一组开始前清梯度。
    model.train()  # 与正式训练模式一致。
    for i in range(0, 8, 2):  # 微批2、累积4。
        x = torch.cat([item[0] for item in items[i:i + 2]]).cuda()  # 输入全部长256。
        y = torch.cat([item[1] for item in items[i:i + 2]]).cuda()  # 标签可能包含-100。
        with torch.autocast("cuda", dtype=torch.float16):  # 与正式AMP路径一致。
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum") / positions  # 不分别平均微批。
        scaler.scale(loss).backward()  # 累积同一套参数的梯度。
    scaler.unscale_(optimizer)  # 裁剪前还原缩放。
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)  # 核验不接受非有限梯度。
    previous_scale = scaler.get_scale()  # 更新必须真实发生。
    scaler.step(optimizer)  # 副本唯一一次step。
    scaler.update()  # 保留对应增长历史。
    assert scaler.get_scale() >= previous_scale, "核验溢出被跳过"  # 否则不算下一步复现。
    return cpu_tree(model.state_dict()), cpu_tree(optimizer.state_dict()), scaler.state_dict(), cursors  # 主文件保持不变。


def main():  # 比较直接恢复与临时保存重载后的下一步。
    torch.set_num_threads(4)  # 与正式线程设置一致。
    torch.backends.cuda.matmul.allow_tf32 = False  # 数值配置一致。
    arrays, tokenizer, manifest, tasks = load_data()  # 核对全部冻结数组。
    path, temporary = OUT / "last_train.pt", OUT / "resume_verification_copy.pt"  # 临时副本不会覆盖正式状态。
    before = sha(path.read_bytes())  # 用摘要确认源文件未变。
    saved = torch.load(path, map_location="cpu", weights_only=True)  # 真实累计状态。
    assert {int(value["step"].item()) for value in saved["optimizer_state"]["state"].values()} == {saved["step"]}, "AdamW累计步数与模型不同"  # 原2020步历史已延续。
    assert not temporary.exists(), "临时核验文件已存在"  # 不覆盖未知本地文件。
    try:  # 无论核验成功与否只清理自己新建的明确文件。
        torch.save(saved, temporary)  # 实际保存完整状态再加载。
        first = branch(path, arrays, manifest)  # 直接恢复后一步。
        torch.cuda.empty_cache()  # 减少两条分支的显存峰值。
        second = branch(temporary, arrays, manifest)  # 保存重载后同一步。
    finally:  # 临时文件为本函数确定创建。
        temporary.unlink(missing_ok=True)  # 单文件删除，不递归操作目录。
    error = max(float((first[0][key] - second[0][key]).abs().max().item()) for key in first[0])  # 全参数逐元素比较。
    same = equal(first[1], second[1]) and equal(first[2], second[2]) and first[3] == second[3]  # 优化器、Scaler和三游标一起核验。
    assert error <= 1e-6 and same and sha(path.read_bytes()) == before, "混合状态恢复不一致"  # 两次副本更新不计正式训练。
    result = {"formal_checkpoint_step": saved["step"], "verification_copy_updates": 2, "max_parameter_abs_diff": error, "optimizer_scaler_cursors_equal": same, "formal_checkpoint_sha256_unchanged": before}  # 明示独立核验计数。
    (ROOT / "docs/training_runs/lesson54_resume.json").write_bytes(dump(result))  # 不改正式训练报告。
    print(json.dumps(result, ensure_ascii=False, indent=2))  # 只输出核验结论。


if __name__ == "__main__":  # 导入恢复函数不启动训练。
    main()  # 显式运行才对副本做两次核验更新。
