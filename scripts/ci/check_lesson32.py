import json  # 读取已完成整轮实验的报告。
import sys  # 为数字开头的学习模块设置导入路径。
from importlib import import_module  # 复用已有模型、评估和生成代码。
from itertools import islice  # 缓存与实验相同的验证前三批。
from pathlib import Path  # 根据检查脚本位置定位仓库。
import numpy as np  # 统计目标字频并在容差内核对损失。

ROOT = Path(__file__).resolve().parents[2]  # 当前文件在 scripts/ci，向上两级得到根目录。
sys.path.insert(0, str(ROOT / "scripts" / "learn"))  # 允许导入同一仓库的学习模块。


def main():  # 只读检查整轮产物，不执行完整训练。
    data = import_module("06_make_windows")  # 复用固定数据与窗口函数，首次导入仍会显示旧检查。
    network = import_module("07_tiny_rnn")  # 导入 RNN 实现，不执行旧演示。
    training = import_module("11_train_steps")  # 复用保存权重加载与固定参数评估。
    decoding = import_module("12_generate_text")  # 复用贪心及采样生成。
    baseline = import_module("14_frequency_baseline")  # 复用训练字频估计与损失函数。
    report = json.loads((ROOT / "docs" / "training_runs" / "lesson32.json").read_text(encoding="utf-8"))  # 读取已提交实验报告。
    config = report["config"]  # 获取窗口、评估与生成的控制条件。
    batch_size, seq_length = config["batch_size"], config["seq_length"]  # 使用报告对应的批次尺寸。
    assert config["epochs"] == 1 and batch_size == 16 and seq_length == 32, "本课训练配置不匹配"  # 本课明确检查固定的一轮配置。
    assert report["train_sha256"] == data.vocab["train_sha256"], "训练数据来源不一致"  # 不允许错误词表数据版本进入比较。
    by_book = {}  # 独立按章节长度计算每本书的窗口数量。
    for record in data.train_records:  # 只读取训练章节，不执行更新。
        count = max(0, (len(record["text"]) - seq_length - 1) // seq_length + 1)  # 使用与训练相同的章节内窗口规则。
        by_book[record["book"]] = by_book.get(record["book"], 0) + count  # 汇总各本书应被遍历的窗口数。
    expected_windows = sum(by_book.values())  # 得到全轮窗口数。
    expected_steps = (expected_windows + batch_size - 1) // batch_size  # 尾批保留，因此向上取整。
    assert report["book_windows"] == by_book and report["seen_windows"] == expected_windows == 76584, "训练窗口覆盖不正确"  # 按书逐项验证覆盖，而非只检查总和。
    assert report["expected_steps"] == report["completed_steps"] == expected_steps == 4787, "整轮步数不正确"  # 完整一轮应有四千七百八十七次更新。
    assert report["seen_positions"] == expected_windows * seq_length == 2450688, "训练位置数不正确"  # 每个完整窗口有三十二个目标。
    assert report["last_batch_size"] == (expected_windows % batch_size or batch_size) == 8, "尾批大小不正确"  # 不能把八行尾批补计成十六行。
    assert [row["step"] for row in report["history"]] == [0, 200, 1000, 3000, expected_steps], "检查点记录不完整"  # 保留预先指定的五个阶段。
    assert np.isfinite(report["training_process_loss"]) and report["training_process_loss"] >= 0 and report["training_and_checkpoint_seconds"] > 0, "训练统计不合法"  # 耗时不作跨机器黄金值比较。
    with (ROOT / "data" / "splits" / "validation.jsonl").open(encoding="utf-8") as file:  # 只读取独立验证章节。
        validation_records = [json.loads(line) for line in file]  # 保留训练时采用的验证顺序。
    subset = list(islice(data.iter_batches(data.iter_windows(validation_records, seq_length, seq_length), batch_size), config["validation_batches"]))  # 构造相同中途验证子集。
    assert len(subset) == config["validation_batches"] == 3, "验证子集大小不匹配"  # 防止部分验证流意外缺失。
    for row in report["history"]:  # 独立加载每份检查点，不重演训练轨迹。
        expected_name = f"lesson32_step{row['step']:04d}.npz"  # 根据步数核对独立文件名。
        assert row["checkpoint"] == expected_name, "检查点文件名不正确"  # 避免把不同步数绑定到同一权重。
        path = ROOT / "model" / expected_name  # 定位已提交的模型文件。
        assert path.stat().st_size < 1_000_000_000, "单个权重超过项目大小要求"  # 检查实际保存文件小于一 GB。
        model, vocab, step = training.load_checkpoint(path, network)  # 校验权重格式、维度、类型与数值。
        assert step == row["step"] and vocab == data.vocab and model.hidden_size == 64, "权重配置不一致"  # 恢复的模型必须与报告对应。
        loss, positions = training.evaluate(model, subset, network)  # 参数固定，重算相同子集损失。
        assert positions == row["validation_positions"] == 1536 and np.isclose(loss, row["validation_loss"], rtol=1e-6, atol=1e-6), "检查点子集指标不一致"  # 允许跨平台微小浮点误差。
        for temperature in (None, config["temperature"]):  # 确认贪心和采样都能使用这个文件。
            text, ids = decoding.generate(model, vocab, config["prompt"], new_tokens=config["new_tokens"], context_size=seq_length, temperature=temperature, seed=config["sampling_seed"])  # 依照记录设置生成。
            assert text.startswith(config["prompt"]) and len(ids) == config["new_tokens"] and all(0 <= index < model.vocab_size for index in ids), "生成编号或开头异常"  # 不将整段随机文字作为跨平台黄金值。
        print(f"lesson32 step={step}: checkpoint and subset metrics passed", flush=True)  # 及时显示各检查点验证进度。
    counts = np.zeros(len(data.tokens), dtype=np.int64)  # 重建字频基线的训练目标计数，不更新 RNN。
    for batch in data.iter_batches(data.iter_windows(data.train_records, seq_length, seq_length), batch_size):  # 遍历全部训练目标，顺序不影响总计数。
        counts += np.bincount(batch["y"].reshape(-1), minlength=len(data.tokens))  # 累计重复目标编号的出现次数。
    assert int(counts.sum()) == report["seen_positions"], "基线计数范围不一致"  # 字频和 RNN 应使用相同训练目标范围。
    probabilities = baseline.smoothed_probabilities(counts, config["frequency_alpha"])  # 只根据训练计数固定字频概率。
    subset_loss, subset_positions = baseline.frequency_loss(probabilities, subset)  # 重算子集字频分数。
    assert subset_positions == 1536 and np.isclose(subset_loss, report["subset_frequency_loss"], rtol=1e-6, atol=1e-6), "子集字频损失不一致"  # 不与完整验证分数混比。
    print("Checking complete validation windows without training...", flush=True)  # 提示接下来只做完整验证前向。
    validation_batches = data.iter_batches(data.iter_windows(validation_records, seq_length, seq_length), batch_size)  # 重新构造完整验证流。
    rnn_loss, positions = training.evaluate(model, validation_batches, network)  # 此时 model 是最后检查点的独立加载对象。
    validation_batches = data.iter_batches(data.iter_windows(validation_records, seq_length, seq_length), batch_size)  # 字频评估需要新的未耗尽流。
    frequency_loss, frequency_positions = baseline.frequency_loss(probabilities, validation_batches)  # 在完整相同位置上测量字频。
    expected_positions = sum(max(0, (len(record["text"]) - seq_length - 1) // seq_length + 1) for record in validation_records) * seq_length  # 独立按章节长度计算验证目标数。
    full = report["full_validation"]  # 读取完整验证记录，不使用子集结果替代。
    assert positions == frequency_positions == expected_positions == full["positions"] == 316960, "完整验证覆盖不正确"  # 尾批也必须纳入实际平均。
    assert np.isclose(rnn_loss, full["rnn_loss"], rtol=1e-6, atol=1e-6) and np.isclose(frequency_loss, full["frequency_loss"], rtol=1e-6, atol=1e-6), "完整验证指标不一致"  # 比较同范围结果，不断言模型必须赢过基线。
    print("lesson32 complete epoch coverage and full validation metrics passed", flush=True)  # 只读检查全部通过。


if __name__ == "__main__":  # 直接运行检查脚本时才验证本课产物。
    main()  # 全程不调用损失反向或参数更新，也不保存新模型。
