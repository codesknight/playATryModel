import hashlib  # 独立核对产物和窗口顺序。
import json  # 读取本阶段原始记录。
from pathlib import Path  # 定位项目文件。
import sys  # 导入实验中的实际恢复工具。
ROOT = Path(__file__).resolve().parents[2]  # 项目根目录。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 不执行训练入口。
import bounded_recurrent as base  # 复用读取与评估定义。
from bounded_recurrent import np, torch  # 在CPU复核GPU产物。
torch.set_num_threads(4)  # 避免CPU线程过度争用。
report = json.loads((ROOT / "docs/training_runs/lesson43.json").read_text(encoding="utf-8"))  # 读取真实报告。
source = ROOT / report["source"]  # 原始起点仍可核对。
assert hashlib.sha256(source.read_bytes()).hexdigest() == report["source_sha256"], "源检查点改变"  # 不将随机初始化称为续训。
original = torch.load(source, map_location="cpu", weights_only=True)  # 源文件包含Adam历史和游标。
vocab, train, validation = base.load_data(torch.device("cpu"))  # 只读取训练与验证。
assert report["status"] == "completed" and report["initial_step"] == original["step"] == 2394, "起点错误"  # 分开统计额外更新。
assert report["additional_steps"] == 2394 and report["additional_steps"] <= report["config"]["max_steps"], "更新预算错误"  # 本次实际完成两轮。
assert report["additional_positions"] == 2 * len(train[0]) * 32 == 4901376, "目标数错误"  # 尾批按实际长度核对。
assert report["train_sha256"] == vocab["train_sha256"] and report["learning_rate"] == 0.001, "数据或学习率改变"  # 本课仅增加训练预算。
exposure = hashlib.sha256()  # 独立重建接下来的两轮顺序。
for epoch in (2, 3):  # 上一课已完成第零和第一轮。
    order = np.random.default_rng(original["data_seed"] + epoch).permutation(len(train[0])).astype(np.int64)  # 延续原种子规则。
    exposure.update(order.tobytes())  # 分批或整轮拼接的字节序列一致。
assert exposure.hexdigest() == report["exposure_sha256"], "续训数据顺序错误"  # 不能重复第一轮冒充下一轮。
assert report["final_cursor"] == {"next_epoch": 4, "next_start": 0}, "尾批游标错误"  # 恢复下一轮不会漏样本。
assert [row["additional_steps"] for row in report["history"]] == [0, 1000, 2394], "验证候选改变"  # 固定中途和最终评估点。
assert report["best_loss"] == min(row["loss"] for row in report["history"]), "最佳选择错误"  # 旧起点也参与选优。
assert report["restore_next_batch_max_abs_diff"] <= 1e-6, "恢复轨迹错误"  # 实际GPU恢复核验有记录。
for name, artifact in report["artifacts"].items():  # 同时验证最佳和最后状态。
    path = ROOT / "model/lesson43" / name  # 不依赖本机绝对输出路径。
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"] and path.stat().st_size == artifact["bytes"] < 1_000_000_000, "产物哈希或大小错误"  # 保存全部状态仍小于一GB。
    model, optimizer, state = base.restore_checkpoint(path, torch.device("cpu"))  # 恢复正式保存的优化器。
    assert state["step"] == report["best_step" if "best" in name else "completed_steps"], "文件位置错误"  # 防止混淆最佳和最后。
    assert all(int(item["step"].item()) == state["step"] for item in optimizer.state.values()), "Adam历史被清零"  # 动量步数必须包含上一课的2394次。
    if "best" in name:  # 完整验证只重算最佳文件，避免重复相同计算。
        metrics = base.evaluate(model, validation)  # CPU上独立重算全部316960目标。
        assert metrics["positions"] == 316960 and np.isclose(metrics["loss"], report["best_loss"], atol=2e-5), "重载指标不一致"  # 允许CPU和GPU的浮点差。
        print(f"CPU best loss={metrics['loss']:.6f}")  # 输出实际核验结果。
for row in report["history"]:  # 检查固定样例完整且解码一致。
    assert len(row["samples"]) == 6, "固定样例不完整"  # 三开头各两种解码方式。
    for sample in row["samples"]:  # 每个字符均可追溯到保存编号。
        assert len(sample["generated_ids"]) == 60 and sample["text"] == sample["prompt"] + "".join(vocab["tokens"][index] for index in sample["generated_ids"]), "样例解码不一致"  # 不人工修改生成文本。
print("续训预算、Adam历史、数据顺序、完整验证及固定样例检查：通过")  # 不启动真实续训入口。
