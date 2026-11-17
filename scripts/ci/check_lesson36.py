import hashlib  # 核对训练共同起点的文件哈希。
import json  # 读取已提交的实验记录。
import random  # 独立重建本课训练章节顺序。
import sys  # 设置学习模块导入路径。
from importlib import import_module  # 复用模型加载与只前向评估。
from itertools import islice  # 只重建本实验使用的两百批数据。
from pathlib import Path  # 根据检查脚本位置确定根目录。
import numpy as np  # 核对参数与浮点指标。

ROOT = Path(__file__).resolve().parents[2]  # 本文件位于 scripts/ci。
sys.path.insert(0, str(ROOT / "scripts/learn"))  # 允许导入数字开头的学习模块。


def main():  # 只读核对已保存产物，不重训两个分支。
    path = ROOT / "docs/training_runs/lesson36.json"  # 定位学习者实际报告。
    saved_bytes = path.read_bytes()  # 保留原字节检查报告没有被覆盖。
    report = json.loads(saved_bytes)  # 读取实验条件与结果。
    data = import_module("06_make_windows")  # 使用相同数据及窗口规则。
    network = import_module("07_tiny_rnn")  # 导入模型实现。
    training = import_module("11_train_steps")  # 复用加载和固定参数评估。
    decoding = import_module("12_generate_text")  # 检查独立加载后的生成流程。
    assert report["config"] == {"steps_per_branch": 200, "learning_rates": [0.1, 0.03], "batch_size": 16, "seq_length": 32, "max_norm": 1.0, "chapter_shuffle_seed": 45}, "训练配置不匹配"  # 检查预先规定的对照条件。
    assert report["train_sha256"] == data.vocab["train_sha256"], "训练数据版本不一致"  # 编码来源必须相同。
    assert report["starting_checkpoint"] == "lesson33_best.npz", "共同起点错误"  # 两组应从三轮最佳检查点出发。
    source_path = ROOT / "model" / report["starting_checkpoint"]  # 定位原始模型文件。
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == report["starting_sha256"], "起点权重哈希不一致"  # 不接受被覆盖的起点。
    initial, vocab, start_step = training.load_checkpoint(source_path, network)  # 保存起点对象供分支参数比较。
    assert start_step == 14361 and vocab == data.vocab, "起点步数或词表不正确"  # 两个分支都以相同阶段为起点。
    records = list(data.train_records)  # 从原始顺序重新复制训练章节。
    random.Random(45).shuffle(records)  # 按报告指定种子重建章节顺序。
    chapter_to_book = {record["id"]: record["book"] for record in records}  # 将窗口来源对应到书名。
    books = {record["book"]: 0 for record in records}  # 各书从零计数。
    positions, xy_bytes, batch_count = 0, 0, 0  # 累计实际训练目标、数组内存与批次数。
    batches = data.iter_batches(data.iter_windows(records, 32, 32), 16)  # 只构造训练数据，不更新任何模型。
    for batch in islice(batches, 200):  # 两组约定使用同一训练前缀。
        batch_count += 1  # 每个批次对应一次更新预算。
        positions += batch["y"].size  # 累计实际目标数量。
        xy_bytes += batch["x"].nbytes + batch["y"].nbytes  # 只核对输入目标数组内存。
        for chapter_id in batch["chapter_ids"]:  # 每行对应一个训练窗口。
            books[chapter_to_book[chapter_id]] += 1  # 独立重建各书实际覆盖。
    assert batch_count == 200 and positions == report["training_positions_per_branch"] == 102400, "训练预算不一致"  # 不把短实验误记成完整一轮。
    assert xy_bytes == report["cached_xy_bytes"] == 1638400 and books == report["book_windows"] and sum(books.values()) == 3200, "训练覆盖或缓存字节不一致"  # 检查不均衡前缀的实际组成。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 读取相同完整验证章节。
    history = report["history"]  # 起点及两组最终文件都应保留。
    assert [row["label"] for row in history] == ["starting_model", "lr010", "lr003"], "候选不完整"  # 起点也必须参与选择。
    for index, row in enumerate(history):  # 独立加载并验证三个候选模型。
        expected_name = "lesson33_best.npz" if index == 0 else f"lesson36_{row['label']}.npz"  # 各分支使用不同文件名。
        assert row["checkpoint"] == expected_name, "候选文件名不一致"  # 防止把同一模型错误绑定为两种学习率。
        weight_path = ROOT / "model" / expected_name  # 定位保存的权重。
        before_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()  # 保留只读检查前的文件哈希。
        assert weight_path.stat().st_size < 1_000_000_000, "模型超过大小要求"  # 每个模型文件都小于一 GB。
        model, current_vocab, step = training.load_checkpoint(weight_path, network)  # 校验文件格式及参数维度类型。
        assert current_vocab == vocab and model.hidden_size == 64 and step == row["completed_steps"] == start_step + (0 if index == 0 else 200), "候选元数据错误"  # 两个分支都是累计一万四千五百六十一步。
        batches = data.iter_batches(data.iter_windows(validation, 32, 32), 16)  # 每个模型使用新的完整验证流。
        loss, count = training.evaluate(model, batches, network)  # 只前向复算，并检查参数没有变化。
        assert count == report["validation_positions"] == 316960 and np.isclose(loss, row["validation_loss"], rtol=1e-6, atol=1e-6), "完整验证指标不一致"  # 浮点指标允许微小平台差异。
        if index > 0:  # 额外核对训练分支的学习率和变化量。
            assert row["learning_rate"] == (0.1 if index == 1 else 0.03), "分支学习率记录错误"  # 两组分别对应预定设置。
            assert np.isclose(row["change_from_start"], row["validation_loss"] - history[0]["validation_loss"]), "变化量错误"  # 负值表示比起点更低。
            assert any(not np.array_equal(value, initial.params[name]) for name, value in model.params.items()), "分支参数没有变化"  # 保存文件应包含训练后的新参数。
            text, ids = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32)  # 检查已保存分支可以独立生成。
            assert text.startswith("春风") and len(ids) == 40 and all(0 <= value < model.vocab_size for value in ids), "生成输出异常"  # 不把完整字符串作为跨平台黄金值。
        assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == before_hash, "验证改变权重文件"  # 所有候选检查均只读。
        print(f"lesson36 {row['label']} full_validation={loss:.6f} passed", flush=True)  # 显示核对进度。
    selected = min(history, key=lambda row: row["validation_loss"])  # 在包括起点的全部候选中选最低分。
    assert report["selected_label"] == selected["label"] and report["selected_checkpoint"] == selected["checkpoint"], "选择未覆盖起点或选错候选"  # 禁止只在新分支之间比较。
    assert path.read_bytes() == saved_bytes, "正式报告被覆盖"  # 检查不应写回报告。
    print("lesson36 fixed data budget, checkpoint metrics and candidate selection passed", flush=True)  # 不重演训练轨迹，只验证已有产物。


if __name__ == "__main__":  # 直接运行检查脚本时才复核本课。
    main()  # 不训练、不创建权重、不覆盖报告。
