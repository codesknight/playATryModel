from collections import Counter  # 独立核对固定样例的重复统计。
from importlib import import_module  # 复用只读加载和评估工具。
from pathlib import Path  # 定位仓库文件。
import hashlib  # 检查来源及候选权重哈希。
import json  # 读取完整训练与短实验报告。
import sys  # 设置学习模块的导入路径。
import numpy as np  # 比较模型参数与浮点指标。

ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于scripts/ci。
sys.path.insert(0, str(ROOT / "scripts/learn"))  # 允许导入数字开头的学习模块。


def main():  # 只读核对完整Adam训练产物，不重训一整轮。
    path = ROOT / "docs/training_runs/lesson41.json"  # 定位正式报告。
    saved_bytes = path.read_bytes()  # 保存原字节以核对报告未变。
    report = json.loads(saved_bytes)  # 读取候选、指标与覆盖统计。
    short = json.loads((ROOT / "docs/training_runs/lesson40.json").read_text(encoding="utf-8"))  # 读取相同前两百批的短实验。
    lesson = import_module("25_train_adam_epoch")  # 只导入函数，不进入训练入口。
    data = import_module("06_make_windows")  # 使用同一四书数据和词表。
    config = {"epoch": 5, "learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "batch_size": 16, "seq_length": 32, "max_norm": 1.0, "chapter_shuffle_seed": 46}  # 核对预先规定的整轮配置。
    assert report["status"] == "completed" and report["config"] == config and report["optimizer_state_saved"] is False, "配置、完成状态或恢复能力记录错误"  # 本课仅保存可评估权重。
    assert report["train_sha256"] == short["train_sha256"] == data.vocab["train_sha256"] and report["starting_checkpoint"] == "lesson37_best.npz", "起点或数据版本错误"  # 新语料没有混入。
    source_path = ROOT / "model" / report["starting_checkpoint"]  # 定位第四轮最佳模型。
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == report["starting_sha256"] == short["starting_sha256"], "来源文件与短实验不一致"  # 不接受已经被覆盖的起点。
    expected_books = {}  # 独立计算整轮应有的各书窗口。
    for record in data.train_records:  # 顺序变化不影响完整窗口集合。
        count = max(0, (len(record["text"]) - 33) // 32 + 1)  # 需要三十三个字符形成三十二个目标。
        expected_books[record["book"]] = expected_books.get(record["book"], 0) + count  # 逐书累计窗口数量。
    assert report["book_windows"] == expected_books and report["seen_windows"] == sum(expected_books.values()) == 76584, "完整训练覆盖不一致"  # 不只核对总数。
    assert report["seen_positions"] == 2450688 and report["last_batch_size"] == 8, "预测位置或尾批错误"  # 最后八行仍须参与更新。
    assert report["completed_local_steps"] == report["adam_steps"] == 4787 and report["completed_global_steps"] == 19148 + 4787 == 23935, "局部、累计或Adam历史计数错误"  # 本次Adam历史从零开始。
    assert np.isfinite(report["training_process_loss"]) and report["training_process_loss"] >= 0 and report["run_seconds"] > 0, "训练统计无效"  # 不比较跨机器实际耗时。
    history = report["history"]  # 起点及四个预定检查点均需保留。
    assert [row["local_step"] for row in history] == [0, 200, 1000, 3000, 4787], "检查点候选不完整"  # 不遗漏变差的检查点。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 读取同一完整验证集。
    cases = {(prompt, mode) for prompt in ("春风", "唐僧道：", "宝玉听了") for mode in ("greedy", "topk40_t07")}  # 每个候选预定六段文字。
    for row in history:  # 独立加载每个候选模型复算验证。
        local_step = row["local_step"]  # 当前候选在本轮的位置。
        expected_name = "lesson37_best.npz" if local_step == 0 else f"lesson41_step{local_step:04d}.npz"  # 文件名保存局部步数。
        assert row["checkpoint"] == expected_name and row["adam_steps"] == local_step, "候选名称或优化器计数错误"  # 检查未把旧SGD次数当作Adam历史。
        weight_path = ROOT / "model" / expected_name  # 定位实际权重文件。
        before_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()  # 留下只读检查前校验值。
        assert weight_path.stat().st_size < 1_000_000_000, "模型文件超过大小要求"  # 每份模型小于一GB。
        model, vocab, step = lesson.training.load_checkpoint(weight_path, lesson.network)  # 校验结构、类型和有限参数。
        assert vocab == data.vocab and model.hidden_size == 64 and step == row["global_step"] == 19148 + local_step, "模型元数据不一致"  # 文件内保存的是累计步数。
        loss, positions = lesson.epoch_tools.evaluate_full(model, data, validation)  # 只前向重算，并检查参数未变。
        assert positions == row["validation_positions"] == 316960 and np.isclose(loss, row["validation_loss"], rtol=1e-6, atol=1e-6), "完整验证损失不一致"  # 允许平台浮点微差。
        assert report["adam_state_bytes"] == 2 * sum(value.nbytes for value in model.params.values()) == 5757784, "额外Adam状态内存记录错误"  # m/v与参数同形同类型。
        assert len(row["samples"]) == 6 and {(sample["prompt"], sample["mode"]) for sample in row["samples"]} == cases, "固定样例缺失或重复"  # 所有候选都有相同观察条件。
        for sample in row["samples"]:  # 核对原始文字与重复统计。
            ids = sample["generated_ids"]  # 只取新增的六十个编号。
            assert sample["seed"] == 42 and len(ids) == 60 and all(isinstance(value, int) and 0 <= value < model.vocab_size for value in ids), "样例条件或编号错误"  # 不通过换种子挑选结果。
            assert sample["text"] == sample["prompt"] + "".join(vocab["tokens"][value] for value in ids), "样例文本与编号不一致"  # 不接受人工润色代替模型输出。
            counts = Counter(tuple(ids[start:start + 4]) for start in range(57))  # 独立计算重叠四元组次数。
            assert np.isclose(sample["repeat4"], sum(value - 1 for value in counts.values()) / 57), "样例重复比例错误"  # 只验证统计，不当作流畅度结论。
        if local_step == 200:  # 对照上一课相同训练前缀。
            previous = next(item for item in short["history"] if item["label"] == "adam")  # 取得短Adam分支。
            previous_model, previous_vocab, previous_step = lesson.training.load_checkpoint(ROOT / "model" / previous["checkpoint"], lesson.network)  # 独立加载旧分支作比较。
            assert previous_vocab == vocab and previous_step == step and np.isclose(loss, previous["validation_loss"], rtol=1e-6, atol=1e-6), "短前缀指标不一致"  # 必须对应同一累计位置。
            assert all(np.allclose(value, previous_model.params[name], rtol=1e-6, atol=1e-6) for name, value in model.params.items()), "短前缀参数不一致"  # 验证前两百步轨迹得到复现。
        with np.load(weight_path, allow_pickle=False) as archive:  # 当前格式只支持模型权重保存。
            assert set(archive.files) == set(model.params) | {"metadata"}, "权重内容与恢复能力说明不符"  # 没有伪称保存完整Adam状态。
        assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == before_hash, "验证修改了权重文件"  # 所有候选检查均只读。
        print(f"lesson41 local={local_step} global={step} full_validation={loss:.6f} passed", flush=True)  # 显示复核进度。
    selected = min(history, key=lambda row: row["validation_loss"])  # 包含旧模型在内取最低分。
    assert report["best_local_step"] == selected["local_step"] and np.isclose(report["best_validation_loss"], selected["validation_loss"]), "最佳候选选择错误"  # 最后一个候选不一定最好。
    assert report["best_checkpoint"] == "lesson41_best.npz", "最佳文件名错误"  # 最佳快照单独保存。
    best, best_vocab, best_step = lesson.training.load_checkpoint(ROOT / "model" / report["best_checkpoint"], lesson.network)  # 独立加载实际最佳文件。
    candidate, candidate_vocab, candidate_step = lesson.training.load_checkpoint(ROOT / "model" / selected["checkpoint"], lesson.network)  # 加载所选候选文件作对照。
    assert best_vocab == candidate_vocab == data.vocab and best_step == candidate_step == selected["global_step"] and all(np.array_equal(value, candidate.params[name]) for name, value in best.params.items()), "最佳文件与所选候选不一致"  # 五组参数必须完全对应。
    assert path.read_bytes() == saved_bytes, "正式报告被覆盖"  # 检查不写回正式记录。
    print("lesson41 complete coverage, Adam prefix and best snapshot passed", flush=True)  # 不重训一整轮。


if __name__ == "__main__":  # 直接运行才检查产物。
    main()  # 不调用训练入口、不保存新模型或报告。
