import hashlib  # 独立验证来源与产物指纹。
import json  # 读取离线数据和真实训练记录。
from pathlib import Path  # 定位仓库。
import sys  # 导入实验模块，不执行训练。
ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于scripts/ci。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 使用实际模型实现。
import tiny_transformer as lm  # 复用因果前向与窗口格式。
from tiny_transformer import torch, np  # CPU核验不依赖CUDA。
from prepare_transformer_data import combined, read_jsonl, DIRECTORY  # 检查文档级数据归属。
torch.set_num_threads(4)  # 限制CI线程数。
manifest = json.loads((DIRECTORY / "manifest.json").read_text(encoding="utf-8"))  # 读取固定语料清单。
report = json.loads((ROOT / "docs/training_runs/lesson45.json").read_text(encoding="utf-8"))  # 读取实际实验预算和指标。
assert hashlib.sha256((DIRECTORY / "stories.csv").read_bytes()).hexdigest() == manifest["raw_sha256"], "原始语料改变"  # 原文件也随仓库保存。
for name, expected in manifest["files"].items():  # 所有划分和词表都有确定的内容版本。
    assert hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() == expected, "衍生产物改变"  # 测试正文仅用于数据完整性检查，不评测模型。
parts = [read_jsonl(DIRECTORY / f"{split}.jsonl") for split in ("train", "validation", "test")]  # 检查组归属不能跨集合。
assert sum(map(len, parts)) == manifest["retained_stories"] == 2277, "故事覆盖错误"  # 不丢弃来源行。
for left in range(3):  # 两两检查来源编号、主题组和规范化正文。
    for right in range(left + 1, 3):  # 验证与测试也应互斥。
        for key in ("id", "group", "text_sha256"):  # 相同主题不能因为人物改名而跨集合。
            assert not ({row[key] for row in parts[left]} & {row[key] for row in parts[right]}), "划分有交集"  # 检查元数据泄漏。
for split, expected in manifest["old_split_sha256"].items():  # 原四书归属不能重新打乱。
    assert hashlib.sha256((ROOT / f"data/splits/{split}.jsonl").read_bytes()).hexdigest() == expected, "旧划分改变"  # 新结构保持旧章节归属。
vocab = json.loads((DIRECTORY / "vocab.json").read_text(encoding="utf-8"))  # 新编号与旧RNN分开。
assert vocab["tokens"] == ["<UNK>", "<BOS>", "<EOS>"] + sorted({char for row in combined("train") for char in row["text"]}), "词表使用了非训练正文"  # 独立验证训练词表来源。
train, validation = lm.load_windows(vocab, "train"), lm.load_windows(vocab, "validation")  # 不创建任何测试窗口。
for split, data in (("train", train), ("validation", validation)):  # 分母必须含真实尾段而不含补齐。
    assert int((data[1] != -100).sum()) == sum(len(row["text"]) + 1 for row in combined(split)), "目标覆盖错误"  # 每文档预测全部字符及一个EOS。
    valid = data[1][:, 1:] != -100  # 两个相邻位置均有效时，下一个输入必须等于前一个目标。
    assert torch.equal(data[0][:, 1:][valid], data[1][:, :-1][valid]), "输入目标错位错误"  # 包括跨小窗口边界的切片规则。
torch.manual_seed(7)  # 独立固定结构检查。
small = lm.TinyTransformer(19, width=16, heads=4, layers=2, context=8)  # 用微型配置核对因果性。
x = torch.tensor([[1, 3, 4, 5, 6, 7]])  # 两个序列前半部分完全相同。
changed = x.clone()  # 防止修改原输入影响对照。
changed[:, 3:] = 9  # 只改变未来输入。
assert torch.allclose(small(x)[:, :3], small(changed)[:, :3], atol=1e-7), "未来字符泄漏到前文"  # 多层堆叠也应满足因果性。
assert not torch.allclose(small(x)[:, 3:], small(changed)[:, 3:]), "因果测试没有改变有效输入"  # 防止无效的恒定输出通过测试。
loss = lm.F.cross_entropy(small(x).flatten(0, 1), x.flatten())  # 核对所有参数梯度可达。
loss.backward()  # 自动求导走完全部注意力层。
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters()), "存在缺失或非有限梯度"  # 不只检查前向形状。
assert report["status"] == "completed" and 0 < report["completed_steps"] <= report["config"]["max_steps"], "更新预算错误"  # 实际训练遵守上限。
assert report["best_loss"] == min(row["loss"] for row in report["history"]), "最佳候选选错"  # 初始化起点也参与选择。
assert hashlib.sha256((DIRECTORY / "manifest.json").read_bytes()).hexdigest() == report["data_manifest_sha256"], "训练数据版本不同"  # 保留训练来源证据。
exposure, remaining, positions = hashlib.sha256(), report["completed_steps"], 0  # 独立重建实际窗口前缀。
for epoch in range(report["config"]["max_epochs"]):  # 不能超过预定轮数。
    order = np.random.default_rng(report["data_seed"] + epoch).permutation(len(train[0])).astype(np.int64)  # 每轮复现同一打乱。
    for start in range(0, len(order), report["batch_size"]):  # 处理实际尾批。
        if remaining == 0:  # 不计算未实际训练的后续批次。
            break  # 保留实际前缀。
        indices = order[start:start + report["batch_size"]]  # 对应实际窗口编号。
        exposure.update(indices.tobytes())  # 字节序列与训练记录一致。
        positions += int((train[1][indices] != -100).sum())  # 只计真实预测位置。
        remaining -= 1  # 每批一次参数更新。
    if remaining == 0:  # 足够覆盖实际训练前缀就结束。
        break  # 不把未训练样本计入预算。
assert remaining == 0 and positions == report["seen_positions"] and exposure.hexdigest() == report["exposure_sha256"], "数据预算或顺序错误"  # 核对训练覆盖证据。
for name, artifact in report["artifacts"].items():  # 最佳和最后文件都应可加载。
    path = ROOT / "model/lesson45" / name  # 使用仓库相对位置。
    assert path.stat().st_size == artifact["bytes"] < 1_000_000_000 and hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"], "状态文件改变或过大"  # 完整状态也应小于一GB。
    model, optimizer, state = lm.restore(path, torch.device("cpu"))  # 在CPU上恢复GPU训练结果。
    assert sum(p.numel() for p in model.parameters()) == report["parameter_count"] == 1680000, "参数量计算错误"  # 输出层共享权重不能重复计数。
    assert state["step"] == report["best_step" if "best" in name else "completed_steps"], "文件对应的更新步数错误"  # 最佳与最后允许不同。
    assert all(int(item["step"].item()) == state["step"] for item in optimizer.state.values()), "AdamW历史步数不匹配"  # 参数与历史需属于同一位置。
    assert state["vocab"] == vocab, "词表与权重不对应"  # 新词表不能使用旧RNN解码。
    if "best" in name:  # 完整验证只复算最佳文件一次。
        metrics = lm.evaluate(model, validation)  # 含四书和新故事的完整真实目标。
        assert np.isclose(metrics["loss"], report["best_loss"], atol=3e-5), "完整验证复算不一致"  # 接受CPU与CUDA的浮点误差。
        print("CPU Transformer validation:", metrics)  # 输出实际复算指标。
for row in report["history"]:  # 逐一核验所有固定样例。
    assert len(row["samples"]) == 6, "固定样例缺失"  # 三开头各两种模式。
    for sample in row["samples"]:  # EOS只允许在最后且不输出成普通文字。
        ids = sample["generated_ids"]  # 实际保存的生成编号。
        assert 0 < len(ids) <= 60 and all(2 <= index < len(vocab["tokens"]) for index in ids), "生成编号非法"  # 不生成UNK或BOS。
        assert 2 not in ids[:-1] and sample["text"] == sample["prompt"] + "".join(vocab["tokens"][index] for index in ids if index != 2), "EOS或解码错误"  # 不人工修改样例文字。
print("新语料分组、训练词表、尾段覆盖、因果性、梯度、预算和完整验证检查：通过")  # 不重新执行真实训练。
