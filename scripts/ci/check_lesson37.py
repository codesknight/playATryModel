import hashlib  # 核对旧起点文件的完整哈希。
import json  # 读取整轮训练与短实验报告。
import sys  # 设置学习模块的导入路径。
from importlib import import_module  # 复用模型加载与固定参数评估。
from pathlib import Path  # 定位仓库文件。
import numpy as np  # 比较完整验证损失和最佳文件参数。

ROOT = Path(__file__).resolve().parents[2]  # 本文件位于 scripts/ci。
sys.path.insert(0, str(ROOT / "scripts/learn"))  # 允许导入数字开头的学习模块。


def main():  # 只读检查五个候选和最佳快照，不重新训练第四轮。
    path = ROOT / "docs/training_runs/lesson37.json"  # 定位学习者的正式报告。
    saved_bytes = path.read_bytes()  # 留下报告原字节，确保检查不覆盖它。
    report = json.loads(saved_bytes)  # 读取已完成训练的结果。
    short = json.loads((ROOT / "docs/training_runs/lesson36.json").read_text(encoding="utf-8"))  # 读取前两百步对照记录。
    data = import_module("06_make_windows")  # 复用同一数据版本与窗口规则。
    network = import_module("07_tiny_rnn")  # 导入模型结构。
    training = import_module("11_train_steps")  # 复用加载及固定参数评估。
    assert report["status"] == "completed" and report["config"] == {"epoch": 4, "learning_rate": 0.03, "batch_size": 16, "seq_length": 32, "max_norm": 1.0, "chapter_shuffle_seed": 45}, "本轮配置或状态错误"  # 固定预先规定的第四轮安排。
    assert report["train_sha256"] == short["train_sha256"] == data.vocab["train_sha256"], "训练数据版本不一致"  # 不比较不同词表来源的指标。
    assert report["starting_checkpoint"] == "lesson33_best.npz", "训练起点错误"  # 不应从短实验分支重复喂整轮。
    source_path = ROOT / "model" / report["starting_checkpoint"]  # 定位三轮最佳模型。
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == report["starting_sha256"], "起点权重被覆盖"  # 核对保存时记录的哈希。
    expected_books = {}  # 独立按章节长度计算整轮窗口覆盖。
    for record in data.train_records:  # 打乱顺序不改变整轮窗口总量。
        count = max(0, (len(record["text"]) - 33) // 32 + 1)  # 产生一个窗口需要三十三个字符。
        expected_books[record["book"]] = expected_books.get(record["book"], 0) + count  # 累计每本书的窗口数。
    assert report["seen_windows"] == sum(expected_books.values()) == 76584 and report["book_windows"] == expected_books, "四书完整覆盖不一致"  # 不只比较总量。
    assert report["seen_positions"] == 2450688 and report["last_batch_size"] == 8, "目标数量或尾批错误"  # 保留实际八行尾批。
    assert report["completed_local_steps"] == 4787 and report["completed_global_steps"] == 14361 + 4787 == 19148, "局部或累计步数错误"  # 两种步数的起点不同。
    assert np.isfinite(report["training_process_loss"]) and report["training_process_loss"] >= 0 and report["run_seconds"] > 0, "训练统计无效"  # 不把耗时作为跨机器黄金值。
    history = report["history"]  # 起点及四个预定检查点都应保留。
    assert [row["local_step"] for row in history] == [0, 200, 1000, 3000, 4787], "候选检查点不完整"  # 不默认最后一步最佳。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 读取固定完整验证章节。
    for row in history:  # 分别加载每个候选，只前向复算指标。
        local_step = row["local_step"]  # 获取本轮局部位置。
        expected_name = "lesson33_best.npz" if local_step == 0 else f"lesson37_step{local_step:04d}.npz"  # 局部步数决定文件名。
        assert row["checkpoint"] == expected_name, "候选文件名错误"  # 不把不同位置绑定到同一个文件。
        model_path = ROOT / "model" / expected_name  # 定位正式权重。
        before_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()  # 保留只读复算前的文件哈希。
        assert model_path.stat().st_size < 1_000_000_000, "模型文件超过大小要求"  # 每份模型小于一 GB。
        model, vocab, step = training.load_checkpoint(model_path, network)  # 校验参数形状、类型和有限数值。
        assert vocab == data.vocab and model.hidden_size == 64 and step == row["global_step"] == 14361 + local_step, "候选累计步数或配置错误"  # 文件元数据必须记录累计步数。
        batches = data.iter_batches(data.iter_windows(validation, 32, 32), 16)  # 每个模型需要新的未耗尽验证流。
        loss, positions = training.evaluate(model, batches, network)  # 评估函数同时检查参数未变。
        assert positions == row["validation_positions"] == 316960 and np.isclose(loss, row["validation_loss"], rtol=1e-6, atol=1e-6), "完整验证指标不一致"  # 允许平台浮点误差。
        if local_step == 200:  # 前缀应与上一课低学习率分支一致。
            previous = next(item for item in short["history"] if item["label"] == "lr003")  # 取已核验的短实验分支。
            previous_model, _, previous_step = training.load_checkpoint(ROOT / "model" / previous["checkpoint"], network)  # 独立加载旧分支用于参数对照。
            assert step == previous_step and np.isclose(loss, previous["validation_loss"], rtol=1e-6, atol=1e-6), "短实验前缀不一致"  # 相同起点和前缀应给出相同指标。
            assert all(np.allclose(value, previous_model.params[name], rtol=1e-6, atol=1e-6) for name, value in model.params.items()), "前缀参数不一致"  # 不只检查损失偶然接近。
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == before_hash, "验证修改权重文件"  # 所有模型文件保持只读。
        print(f"lesson37 local={local_step} global={step} full_validation={loss:.6f} passed", flush=True)  # 显示较长复核任务的进度。
    selected = min(history, key=lambda row: row["validation_loss"])  # 从包含起点的所有候选中取最低分。
    assert report["best_local_step"] == selected["local_step"] and np.isclose(report["best_validation_loss"], selected["validation_loss"]), "最佳候选选择错误"  # 完成整轮和选择模型是两件事。
    assert report["best_checkpoint"] == "lesson37_best.npz", "最佳文件名错误"  # 最佳快照独立保存。
    best, best_vocab, best_step = training.load_checkpoint(ROOT / "model" / report["best_checkpoint"], network)  # 加载实际最佳文件。
    candidate, candidate_vocab, candidate_step = training.load_checkpoint(ROOT / "model" / selected["checkpoint"], network)  # 加载被选中的原候选文件。
    assert best_vocab == candidate_vocab == data.vocab and best_step == candidate_step == selected["global_step"], "最佳文件元数据错误"  # 词表和累计步数同步。
    assert all(np.array_equal(value, candidate.params[name]) for name, value in best.params.items()), "最佳文件参数与候选不一致"  # 五组参数必须逐项完全相同。
    assert path.read_bytes() == saved_bytes, "正式报告被覆盖"  # 自动检查不写回学习者报告。
    print("lesson37 complete epoch, prefix reproduction and best snapshot passed", flush=True)  # 全程没有训练或保存操作。


if __name__ == "__main__":  # 直接运行时才检查本课产物。
    main()  # 不重训一整轮，只加载保存的检查点。
