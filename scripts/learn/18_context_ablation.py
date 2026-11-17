from importlib import import_module  # 复用模型、权重加载与验证函数。
from pathlib import Path  # 从脚本位置确定项目根目录。
import json  # 读取上一课报告并保存本课实验记录。
import numpy as np  # 检查参数独立性和数值结果。

ROOT = Path(__file__).resolve().parents[2]  # 本文件放在 scripts/learn，向上两级得到根目录。
network = import_module("07_tiny_rnn")  # 导入已经完成的 RNN。
training = import_module("11_train_steps")  # 复用权重加载与只前向评估。


def evaluate_full(model, data, records):  # 每次使用相同的完整验证窗口。
    windows = data.iter_windows(records, 32, 32)  # 创建新的窗口流，避免生成器已经耗尽。
    batches = data.iter_batches(windows, 16)  # 使用十六行批次并保留尾批。
    return training.evaluate(model, batches, network)  # 按实际位置加权，并检查参数没有变化。

def main():  # 比较原模型与切断循环连接的独立副本。
    data = import_module("06_make_windows")  # 复用训练词表和相同窗口规则，导入时会显示旧数据检查。
    source = json.loads((ROOT / "docs/training_runs/lesson33.json").read_text(encoding="utf-8"))  # 读取上一课选出的最佳模型信息。
    assert source["status"] == "completed" and source["train_sha256"] == data.vocab["train_sha256"], "起点未完成或数据版本不一致"  # 不接受错误来源的模型。
    path = ROOT / "model" / source["best_checkpoint"]  # 始终使用上一课验证选出的最佳权重。
    original, vocab, step = training.load_checkpoint(path, network)  # 第一次加载原模型。
    ablated, other_vocab, other_step = training.load_checkpoint(path, network)  # 第二次加载独立副本，不能写成 ablated = original。
    assert vocab == other_vocab == data.vocab and step == other_step, "模型配置不一致"  # 两个模型必须从同一检查点开始。
    assert all(np.array_equal(value, ablated.params[name]) and not np.shares_memory(value, ablated.params[name]) for name, value in original.params.items()), "模型参数不相同或共享内存"  # 相同数值也必须存放在独立数组中。
    saved = {name: value.copy() for name, value in original.params.items()}  # 留下原模型参数，结束时核对它没有变化。
    ablated.params["recurrent"].fill(0.0)  # 只在副本中把循环矩阵置零，切断来自前一步状态的贡献。
    assert all(np.array_equal(value, saved[name]) for name, value in ablated.params.items() if name != "recurrent"), "其他参数被改变"  # 表示表、偏移与输出层保持原值。
    records = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 两种条件使用相同验证章节。
    preview = next(data.iter_batches(data.iter_windows(records, 32, 32), 16))  # 取一个批次检查切断连接的直接影响。
    original_logits = original.forward(preview["x"])  # 原模型在这个批次上的全部分数。
    ablated_logits = ablated.forward(preview["x"])  # 副本在相同输入上的全部分数。
    assert np.allclose(original_logits[:, 0, :], ablated_logits[:, 0, :]), "首位置分数应相同"  # 每个窗口的初始状态为零，首位置原本也没有前文贡献。
    later_difference = float(np.max(np.abs(original_logits[:, 1:, :] - ablated_logits[:, 1:, :])))  # 测量后续位置分数的最大变化。
    assert later_difference > 0, "后续位置分数没有变化"  # 确认这次干预确实影响了递推计算。

    original_loss, positions = evaluate_full(original, data, records)  # 在完整验证窗口上测量正常模型。
    print("正常模型完整验证损失：", original_loss, flush=True)  # 先显示第一个条件的结果。
    ablated_loss, other_positions = evaluate_full(ablated, data, records)  # 同样位置上测量切断连接后的副本。
    assert positions == other_positions == 316960, "验证范围不一致"  # 两个平均数必须覆盖相同预测位置。
    assert np.isclose(original_loss, source["best_validation_loss"], rtol=1e-6, atol=1e-6), "正常模型与上课指标不一致"  # 先确认没有加载错文件或改变验证范围。
    assert all(np.array_equal(value, saved[name]) for name, value in original.params.items()), "原模型被修改"  # 副本干预和全部评估都不能改变原模型。
    delta = ablated_loss - original_loss  # 正数表示切断前文连接后损失上升。
    report = {  # 保存实验条件与结果，不保存干预后的模型权重。
        "checkpoint": path.name, "completed_steps": step, "train_sha256": vocab["train_sha256"],  # 记录模型和数据来源。
        "config": {"batch_size": 16, "seq_length": 32, "intervention": "zero_recurrent_without_retraining"},  # 明确这是不重新训练的干预。
        "positions": positions, "original_loss": original_loss, "ablated_loss": ablated_loss,  # 两种条件都覆盖完整验证窗口集。
        "loss_increase": delta, "full_frequency_loss": source["full_frequency_loss"],  # 字频是另一个已独立拟合的参照，不能与副本混为一谈。
        "preview_later_max_logit_difference": later_difference,  # 保留预览批次后续位置确实改变的证据。
    }  # 这份报告不使用测试集，也没有参数更新。
    print("切断循环连接后的损失：", ablated_loss, flush=True)  # 如实显示结果，不预先断言必须更差。
    print("切断后损失增加量：", delta, flush=True)  # 比较按位置平均的自然对数损失差。
    print("同范围字频基线：", report["full_frequency_loss"], flush=True)  # 保留上一课已经核对的参照值。
    print("验证预测位置数：", positions, flush=True)  # 明确当前分数的覆盖规模。
    print("独立副本、首位置相同与原模型保留检查：通过", flush=True)  # 检查计算和控制条件，而非断言实验结论。
    return report  # 返回普通字典，交给入口保存。

if __name__ == "__main__":  # 只有直接运行本脚本时才执行实验并保存报告。
    result = main()  # 运行两个条件的完整验证，不训练模型。
    report_path = ROOT / "docs/training_runs/lesson34.json"  # 把本课结果放入独立记录文件。
    report_path.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存可读的中文 JSON。
    print("实验报告：", report_path)  # 告诉你检查结果文件的位置。
