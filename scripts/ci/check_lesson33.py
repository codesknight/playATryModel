import json  # 读取逐轮实验报告。
import random  # 独立重建每轮章节顺序。
import sys  # 设置学习模块的导入路径。
from importlib import import_module  # 复用模型加载与只读评估。
from pathlib import Path  # 根据脚本位置定位仓库。
import numpy as np  # 核对指标及最佳文件的全部参数。

ROOT = Path(__file__).resolve().parents[2]  # 向上两级找到项目根目录。
sys.path.insert(0, str(ROOT / "scripts" / "learn"))  # 允许导入数字开头的学习模块。


def main():  # 检查已保存产物，不重新训练两轮。
    data = import_module("06_make_windows")  # 复用相同数据版本与窗口规则。
    network = import_module("07_tiny_rnn")  # 导入完整模型实现。
    training = import_module("11_train_steps")  # 复用权重加载及参数固定的评估。
    decoding = import_module("12_generate_text")  # 检查独立加载后的生成流程。
    report = json.loads((ROOT / "docs/training_runs/lesson33.json").read_text(encoding="utf-8"))  # 读取学习者实际实验记录。
    source = json.loads((ROOT / "docs/training_runs/lesson32.json").read_text(encoding="utf-8"))  # 读取第一轮起点记录。
    config = report["config"]  # 获取本课控制条件。
    assert config == {"starting_epoch": 1, "total_epochs": 3, "batch_size": 16, "seq_length": 32, "learning_rate": 0.1, "max_norm": 1.0, "shuffle_seed_base": 42, "prompt": "春风", "new_tokens": 40, "temperature": 1.0, "sampling_seed": 42}, "实验配置不匹配"  # 防止无意改变实验条件。
    assert report["status"] == "completed" and report["completed_epochs"] == 3 and report["run_seconds"] > 0, "实验未完成"  # 耗时只检查有效性，不比较跨机器数值。
    assert report["train_sha256"] == source["train_sha256"] == data.vocab["train_sha256"], "数据版本不一致"  # 检查训练来源对应相同词表。
    assert report["starting_checkpoint"] == "lesson32_step4787.npz" and report["steps_per_epoch"] == 4787, "训练起点不正确"  # 本次从第一轮继续训练。
    assert np.isclose(report["full_frequency_loss"], source["full_validation"]["frequency_loss"]), "基线范围不一致"  # 字频分数已由上一课检查独立重算。
    expected_books = {}  # 独立根据章节长度计算各书窗口数。
    for record in data.train_records:  # 不依赖报告内已有的覆盖计数。
        count = max(0, (len(record["text"]) - 33) // 32 + 1)  # 每个窗口需要三十三个字符以形成错位目标。
        expected_books[record["book"]] = expected_books.get(record["book"], 0) + count  # 汇总该书的窗口。
    assert sum(expected_books.values()) == 76584, "完整窗口数量不正确"  # 核对当前语料的固定规模。
    assert [row["epoch"] for row in report["training_stats"]] == [2, 3], "新增轮记录不完整"  # 第一轮沿用旧实验而不重复训练。
    for row in report["training_stats"]:  # 检查两轮各自的顺序、覆盖及统计。
        records = list(data.train_records)  # 每次从相同原始顺序复制章节。
        seed = 42 + row["epoch"] - 1  # 第二、三轮分别使用四十三和四十四。
        random.Random(seed).shuffle(records)  # 独立重建预定章节顺序。
        assert row["shuffle_seed"] == seed and row["first_chapter_ids"] == [record["id"] for record in records[:3]], "按轮顺序不匹配"  # 核对种子与记录中的前三章。
        assert row["epoch_steps"] == 4787 and row["seen_windows"] == 76584 and row["seen_positions"] == 2450688, "轮内覆盖不正确"  # 每轮都应遍历完整训练窗口。
        assert row["last_batch_size"] == 8 and row["book_windows"] == expected_books, "尾批或四书覆盖不正确"  # 总数正确之外还需逐书正确。
        assert np.isfinite(row["training_process_loss"]) and row["training_process_loss"] >= 0 and row["training_seconds"] > 0, "训练统计无效"  # 不要求不同批次的损失单调下降。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 读取固定完整验证章节。
    assert [row["epoch"] for row in report["history"]] == [1, 2, 3], "候选轮不完整"  # 第一轮同样参与最佳选择。
    assert np.isclose(report["history"][0]["validation_loss"], source["full_validation"]["rnn_loss"], rtol=1e-6, atol=1e-6), "起点指标不一致"  # 禁止混用旧验证子集指标。
    for row in report["history"]:  # 独立加载三个候选轮末模型。
        expected_name = "lesson32_step4787.npz" if row["epoch"] == 1 else f"lesson33_epoch{row['epoch']:02d}.npz"  # 明确每轮对应的文件。
        assert row["checkpoint"] == expected_name, "候选权重文件名不匹配"  # 避免不同轮引用同一个文件。
        path = ROOT / "model" / expected_name  # 定位已提交权重。
        assert path.stat().st_size < 1_000_000_000, "模型超过大小要求"  # 每个权重文件均须小于一 GB。
        model, vocab, step = training.load_checkpoint(path, network)  # 校验格式、参数维度及有限数值。
        assert vocab == data.vocab and model.hidden_size == 64 and step == row["global_step"] == row["epoch"] * 4787, "累计步数或模型配置不匹配"  # 累计步数不能每轮清零。
        batches = data.iter_batches(data.iter_windows(validation, 32, 32), 16)  # 为当前模型创建新的完整验证流。
        loss, positions = training.evaluate(model, batches, network)  # 只前向重算指标，并检查参数未变。
        assert positions == row["validation_positions"] == 316960 and np.isclose(loss, row["validation_loss"], rtol=1e-6, atol=1e-6), "完整验证指标不一致"  # 允许不同平台的微小浮点误差。
        for temperature in (None, 1.0):  # 分别检查贪心及固定温度采样能运行。
            text, ids = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32, temperature=temperature, seed=42)  # 使用相同上下文和采样条件。
            assert text.startswith("春风") and len(ids) == 40 and all(0 <= index < model.vocab_size for index in ids), "生成输出无效"  # 不把整段随机文字作为跨平台黄金值。
        print(f"lesson33 epoch={row['epoch']} full_validation={loss:.6f} passed", flush=True)  # 显示逐轮只读检查进度。
    selected = min(report["history"], key=lambda row: row["validation_loss"])  # 相等时保留记录中较早的一轮。
    assert selected["epoch"] == report["best_epoch"] and np.isclose(selected["validation_loss"], report["best_validation_loss"]), "最佳轮选择错误"  # 不默认最后一轮一定最好。
    assert report["best_checkpoint"] == "lesson33_best.npz", "最佳快照文件名不匹配"  # 最佳参数需要保存在独立文件中。
    best, best_vocab, best_step = training.load_checkpoint(ROOT / "model" / report["best_checkpoint"], network)  # 独立加载最佳快照。
    selected_model, selected_vocab, selected_step = training.load_checkpoint(ROOT / "model" / selected["checkpoint"], network)  # 加载所选轮末文件作对照。
    assert best_vocab == selected_vocab == data.vocab and best_step == selected_step == selected["global_step"], "最佳快照元数据不一致"  # 词表与步数必须同步。
    assert all(np.array_equal(value, selected_model.params[name]) for name, value in best.params.items()), "最佳快照参数不一致"  # 五组参数必须完全匹配被选中轮。
    print("lesson33 epoch coverage, full validation and best snapshot passed", flush=True)  # 全部检查通过，全程不更新参数或写文件。


if __name__ == "__main__":  # 直接运行时才检查学习产物。
    main()  # 只加载已有文件，不重演两轮训练。
