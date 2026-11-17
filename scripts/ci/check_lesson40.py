from collections import Counter  # 独立统计保存样例的重复片段。
from importlib import import_module  # 复用模型、评估与Adam实现。
from itertools import islice  # 重建相同两百批训练前缀。
from pathlib import Path  # 定位仓库文件。
import hashlib  # 核对共同起点和检查前后的文件哈希。
import json  # 读取已完成的对照实验。
import random  # 复现预先规定的章节顺序。
import sys  # 设置学习模块的导入路径。
import numpy as np  # 核对手算状态、参数与浮点指标。

ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于scripts/ci。
sys.path.insert(0, str(ROOT / "scripts/learn"))  # 允许导入数字开头的学习模块。


def check_adam_history(lesson):  # 检查变化梯度下的历史累积，而不只核对第一步。
    lesson.check_first_step()  # 核对两组参数共用一次批次计数及零梯度示例。
    params = {"weight": np.array([1.0])}  # 使用双精度标量参数方便手算。
    optimizer = lesson.Adam(params)  # 从全零优化器历史开始。
    optimizer.step(params, {"weight": np.array([0.4])})  # 第一批梯度为0.4。
    after_first = params["weight"].copy()  # 保留第一步参数供第二步公式核对。
    optimizer.step(params, {"weight": np.array([0.2])})  # 第二批改用0.2，检查历史是否延续。
    expected_m, expected_v = 0.056, 0.00019984  # 手算0.9乘0.04加0.1乘0.2，以及对应平方历史。
    expected = after_first - 0.001 * (expected_m / (1 - 0.9 ** 2)) / (np.sqrt(expected_v / (1 - 0.999 ** 2)) + 1e-8)  # 使用第二步偏差修正计算参数。
    assert optimizer.t == 2 and np.allclose(optimizer.m["weight"], expected_m, rtol=0, atol=1e-14) and np.allclose(optimizer.v["weight"], expected_v, rtol=0, atol=1e-14), "Adam未正确累积历史"  # t与两个状态必须都匹配手算。
    assert np.allclose(params["weight"], expected, rtol=0, atol=1e-12), "Adam第二步参数更新不正确"  # 防止使用错误的修正指数或稳定项位置。
    before_zero = params["weight"].copy()  # 留下已有非零历史时的参数。
    optimizer.step(params, {"weight": np.array([0.0])})  # 当前梯度为零，但之前仍有方向历史。
    assert optimizer.t == 3 and params["weight"][0] < before_zero[0], "Adam历史在零梯度时被重置"  # 已有m历史仍可能使参数移动。


def main():  # 只读核验对照产物，不重新训练两个分支。
    path = ROOT / "docs/training_runs/lesson40.json"  # 定位正式报告。
    saved_bytes = path.read_bytes()  # 留下报告原字节检查文件未变。
    report = json.loads(saved_bytes)  # 读取配置、指标和固定样例。
    lesson = import_module("24_adam_experiment")  # 不进入训练和保存报告入口。
    check_adam_history(lesson)  # 先检查优化器最关键的状态累积行为。
    data = import_module("06_make_windows")  # 使用相同词表及训练窗口规则。
    config = {"steps_per_branch": 200, "sgd_lr": 0.03, "adam_lr": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "max_norm": 1.0, "batch_size": 16, "seq_length": 32, "chapter_shuffle_seed": 46}  # 明确两个预先指定的配置。
    assert report["config"] == config and report["optimizer_state_saved"] is False, "对照配置或恢复能力记录错误"  # 本课权重没有保存Adam历史。
    assert report["train_sha256"] == data.vocab["train_sha256"] and report["starting_checkpoint"] == "lesson37_best.npz", "共同起点来源错误"  # 不混用其他阶段模型。
    source_path = ROOT / "model" / report["starting_checkpoint"]  # 定位来源权重。
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == report["starting_sha256"], "共同起点文件被覆盖"  # 两组应保留相同旧模型。
    records = list(data.train_records)  # 从原始训练章节顺序复制。
    random.Random(46).shuffle(records)  # 复现本次短前缀顺序。
    books = {record["book"]: 0 for record in records}  # 独立累计各书实际窗口数。
    chapter_to_book = {record["id"]: record["book"] for record in records}  # 将每行来源映射到书名。
    batches = data.iter_batches(data.iter_windows(records, 32, 32), 16)  # 只重建数据，不更新参数。
    positions, steps = 0, 0  # 统计实际目标数量及批次数。
    for batch in islice(batches, 200):  # 只访问两组共同使用的两百批。
        steps += 1  # 每个批次对应一次预算更新。
        positions += batch["y"].size  # 按实际数量统计预测目标。
        for chapter_id in batch["chapter_ids"]:  # 每行是一个章节内窗口。
            books[chapter_to_book[chapter_id]] += 1  # 独立复算前缀书籍覆盖。
    assert steps == 200 and positions == report["training_positions_per_branch"] == 102400 and books == report["book_windows"] and sum(books.values()) == 3200, "训练预算或覆盖不一致"  # 不把短前缀误记为完整一轮。
    history = report["history"]  # 三个候选必须都保留。
    assert [row["label"] for row in history] == ["starting_model", "sgd", "adam"], "候选记录不完整"  # 起点必须参加选优。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 相同完整验证章节。
    expected_cases = {(prompt, mode) for prompt in ("春风", "唐僧道：", "宝玉听了") for mode in ("greedy", "topk40_t07")}  # 每个候选都有同样六段观察。
    for index, row in enumerate(history):  # 独立加载各候选并复算完整验证。
        expected_name = "lesson37_best.npz" if index == 0 else f"lesson40_{row['label']}.npz"  # 两个分支使用独立文件名。
        assert row["checkpoint"] == expected_name, "候选文件名错误"  # 防止两种配置错误绑定到同一文件。
        weight_path = ROOT / "model" / expected_name  # 定位实际保存的模型。
        before_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()  # 留下只读复核前校验值。
        assert weight_path.stat().st_size < 1_000_000_000, "模型文件超过大小要求"  # 每份模型小于一GB。
        model, vocab, step = lesson.training.load_checkpoint(weight_path, lesson.network)  # 校验权重格式与参数类型。
        assert vocab == data.vocab and model.hidden_size == 64 and step == row["completed_steps"] == 19148 + (200 if index else 0), "模型元数据错误"  # 模型累计更新次数与Adam历史分开。
        loss, count = lesson.epoch_tools.evaluate_full(model, data, validation)  # 只前向评估并检查参数未变。
        assert count == row["validation_positions"] == 316960 and np.isclose(loss, row["validation_loss"], rtol=1e-6, atol=1e-6), "完整验证结果不一致"  # 允许微小平台浮点误差。
        assert len(row["samples"]) == 6 and {(sample["prompt"], sample["mode"]) for sample in row["samples"]} == expected_cases, "固定样例缺失或重复"  # 不只提交最好看的句子。
        for sample in row["samples"]:  # 核对保存文字和辅助统计。
            ids = sample["generated_ids"]  # 只取续写编号。
            assert sample["seed"] == 42 and len(ids) == 60 and all(isinstance(value, int) and 0 <= value < model.vocab_size for value in ids), "样例编号或种子错误"  # 所有候选观察条件相同。
            assert sample["text"] == sample["prompt"] + "".join(vocab["tokens"][value] for value in ids), "样例文字与编号不一致"  # 不允许手工润色模型原始输出。
            counts = Counter(tuple(ids[start:start + 4]) for start in range(57))  # 独立计数重叠四元组。
            assert np.isclose(sample["repeat4"], sum(value - 1 for value in counts.values()) / 57), "样例重复统计不一致"  # 低重复不等于通顺，此处只验证统计。
        if index:  # 两个训练分支还需核对更新预算和额外状态记录。
            assert row["learning_rate"] == (0.03 if index == 1 else 0.001) and row["optimizer_steps"] == 200 and row["adam_state_steps"] == (None if index == 1 else 200), "优化器配置或历史计数错误"  # Adam历史只从本次切换时开始。
            state_bytes = 0 if index == 1 else 2 * sum(value.nbytes for value in model.params.values())  # 两个同形状态数组的额外内存。
            assert row["optimizer_state_bytes"] == state_bytes and np.isclose(row["change_from_start"], row["validation_loss"] - history[0]["validation_loss"]), "额外状态或损失变化记录错误"  # 不把Adam状态当作可学习参数。
            with np.load(weight_path, allow_pickle=False) as archive:  # 核对文件确实仅保存模型和配置。
                assert set(archive.files) == set(model.params) | {"metadata"}, "本课权重内容与恢复能力说明不符"  # 不能把模型权重声称为完整Adam断点。
        assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == before_hash, "只读检查修改了模型文件"  # 来源和分支文件均保持不变。
        print(f"lesson40 {row['label']} full_validation={loss:.6f} passed", flush=True)  # 显示逐候选复核进度。
    selected = min(history, key=lambda row: row["validation_loss"])  # 全部候选中包括起点。
    assert report["selected_label"] == selected["label"] and report["selected_checkpoint"] == selected["checkpoint"], "候选选优错误"  # 两个分支都变差时应保留旧模型。
    assert path.read_bytes() == saved_bytes, "正式报告被覆盖"  # 自动检查不写回实验记录。
    print("lesson40 Adam history, fixed budget and checkpoint comparison passed", flush=True)  # 全程不重训四百步。


if __name__ == "__main__":  # 直接运行检查脚本才复核。
    main()  # 不调用实验训练入口，也不新增权重或报告。
