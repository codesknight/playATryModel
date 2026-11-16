import json  # 读取已提交的实验配置与观察记录。
import sys  # 将学习脚本目录加入模块搜索路径。
from importlib import import_module  # 导入数字开头的学习模块。
from itertools import islice  # 只构造实验使用的固定验证子集。
from pathlib import Path  # 从检查脚本位置定位仓库根目录。
import numpy as np  # 在浮点容差内比较验证损失。

ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于 scripts/ci，向上两级得到仓库根目录。
sys.path.insert(0, str(ROOT / "scripts" / "learn"))  # 允许独立调用已有学习模块。


def main():  # 验证实验产物，不重新执行两百步训练。
    data = import_module("06_make_windows")  # 复用固定词表和窗口函数，首次导入会显示已有数据检查。
    network = import_module("07_tiny_rnn")  # 加载模型结构，不执行旧模型演示。
    training = import_module("11_train_steps")  # 复用权重加载与只前向评估。
    decoding = import_module("12_generate_text")  # 复用已验证的生成函数。
    report = json.loads((ROOT / "docs" / "training_runs" / "lesson29.json").read_text(encoding="utf-8"))  # 读取本课已保存的实验记录。
    config = report["config"]  # 读取训练与生成时固定的条件。
    assert [row["step"] for row in report["history"]] == [0, 5, 50, 200], "检查点记录不完整"  # 本课应保留四个时间点。
    assert config["steps"] == 200 and report["train_sha256"] == data.vocab["train_sha256"], "配置或训练来源不一致"  # 防止使用错误的数据版本。
    assert set(report["book_windows"]) == {record["book"] for record in data.train_records}, "书籍覆盖记录不一致"  # 记录应包含当前训练集的四本书。
    assert all(count > 0 for count in report["book_windows"].values()) and sum(report["book_windows"].values()) == 3200, "实际训练窗口统计异常"  # 两百个完整批次合计三千二百个窗口。
    with (ROOT / "data" / "splits" / "validation.jsonl").open(encoding="utf-8") as file:  # 只读取验证章节，不参与参数更新。
        records = [json.loads(line) for line in file]  # 保留实验中使用的验证章节顺序。
    windows = data.iter_windows(records, config["seq_length"], config["seq_length"])  # 沿用实验的窗口长度与步长。
    batches = list(islice(data.iter_batches(windows, config["batch_size"]), config["validation_batches"]))  # 固定缓存相同的前三个验证批次。
    assert len(batches) == config["validation_batches"], "验证子集不完整"  # 不允许悄悄缩小评估范围。
    for row in report["history"]:  # 分别检查四个独立保存的模型文件。
        expected_name = f"lesson29_step{row['step']:03d}.npz"  # 根据检查点步数得到预期文件名。
        assert row["checkpoint"] == expected_name, "检查点文件名与步数不符"  # 不将错误权重绑定到观察记录。
        model, vocab, step = training.load_checkpoint(ROOT / "model" / expected_name, network)  # 只从已提交的权重恢复模型。
        assert step == row["step"] and vocab == data.vocab and model.hidden_size == 64, "权重配置不一致"  # 核对步数、词表顺序和隐藏维度。
        loss, positions = training.evaluate(model, batches, network)  # 使用固定参数重新测量相同验证子集。
        assert positions == row["validation_positions"] == 1536, "评估位置数不一致"  # 本课前三个完整批次应包含一千五百三十六个位置。
        assert np.isclose(loss, row["validation_loss"], rtol=1e-6, atol=1e-6), "权重与记录中的验证损失不一致"  # 允许不同计算平台的微小浮点误差。
        for temperature in (None, config["temperature"]):  # 检查贪心与固定温度采样都能使用保存的权重。
            text, ids = decoding.generate(model, vocab, config["prompt"], new_tokens=config["new_tokens"], context_size=config["seq_length"], temperature=temperature, seed=config["sampling_seed"])  # 用记录中的固定条件生成编号。
            assert text.startswith(config["prompt"]) and len(ids) == config["new_tokens"] and all(0 <= index < model.vocab_size for index in ids), "生成结果结构异常"  # 检查合法编号，不把可能受浮点边界影响的整段文字作为跨平台黄金值。
        assert row["greedy"].startswith(config["prompt"]) and row["sampled"].startswith(config["prompt"]), "记录中的开头不一致"  # 原始观察文字仍须保留指定开头。
        print(f"lesson29 step={step}: saved weights, validation loss and generation passed")  # 显示当前检查点通过。


if __name__ == "__main__":  # 直接运行检查脚本时才验证本课产物。
    main()  # 只读检查权重和记录，不训练或保存新模型。
