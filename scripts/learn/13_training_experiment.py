from importlib import import_module  # 复用已有数据、模型、评估、保存与生成代码。
from itertools import islice  # 只取本次实验需要的前 200 个训练批次。
import json  # 将实验配置、验证损失和生成文字保存成可复查的记录。
import random  # 用固定种子打乱章节顺序，不打乱章内字符。
import numpy as np  # 检查损失和参数是否为有限数值。

network = import_module("07_tiny_rnn")  # 导入模型与梯度裁剪，不执行旧模型演示。
training = import_module("11_train_steps")  # 复用固定参数评估和权重保存，不执行旧训练入口。
decoding = import_module("12_generate_text")  # 复用贪心与采样生成，不执行旧生成演示。

def inspect_checkpoint(model, vocab, validation_batches, step, output_dir):  # 对当前参数做固定评估、生成和保存。
    validation_loss, positions = training.evaluate(model, validation_batches, network)  # 始终评估同一份固定验证子集，不更新参数。
    greedy, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32)  # 固定开头和生成长度，观察贪心结果。
    sampled, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32, temperature=1.0, seed=42)  # 固定温度和抽样种子，减少比较中的随机干扰。
    checkpoint_path = output_dir / f"lesson29_step{step:03d}.npz"  # 各步使用不同文件名，不覆盖原五步学习权重。
    training.save_checkpoint(checkpoint_path, model, vocab, step)  # 保存当前参数，以后可以独立重载这个检查点。
    print(f"检查点 step={step} 固定验证损失={validation_loss:.6f}")  # 显示同一子集上的模型表现。
    print("贪心：", repr(greedy))  # 如实保留重复或不通顺的输出。
    print("采样：", repr(sampled))  # 同步记录概率采样结果，不凭一条样例判断总体质量。
    return {"step": step, "validation_loss": validation_loss, "validation_positions": positions, "greedy": greedy, "sampled": sampled, "checkpoint": checkpoint_path.name}  # 返回可写入 JSON 的本次观察。

def main():  # 从头训练一个模型，比较不同训练步数的检查点。
    data = import_module("06_make_windows")  # 读取训练集和固定词表，首次导入仍会显示旧数据检查。
    batch_size = 16  # 每个完整训练批次使用 16 个窗口。
    seq_length = 32  # 窗口长度和步长均为 32。
    max_steps = 200  # 本次先连续更新 200 步，仍未完成整个训练集的一轮。
    learning_rate = 0.1  # 保持普通梯度下降学习率不变。
    max_norm = 1.0  # 保持全局梯度裁剪阈值不变。
    train_records = list(data.train_records)  # 复制章节列表，不改变原数据模块的章节顺序。
    random.Random(42).shuffle(train_records)  # 固定随机打乱章节顺序，章内字序保持原样。
    with (data.ROOT / "data" / "splits" / "validation.jsonl").open(encoding="utf-8") as file:  # 读取独立验证集。
        validation_records = [json.loads(line) for line in file]  # 按原顺序保留验证章节。
    validation_batches = list(islice(data.iter_batches(data.iter_windows(validation_records, seq_length, seq_length), batch_size), 3))  # 沿用之前固定的前三个验证批次。
    assert len(validation_batches) == 3, "固定验证子集不足三批"  # 确认本次评估范围与之前课程一致。
    model = network.TinyRNN(len(data.tokens), hidden_size=64, seed=42)  # 只创建一次模型，之后在同一训练轨迹上比较步数。
    output_dir = data.ROOT / "model"  # 将四个独立检查点写入模型目录。
    history = [inspect_checkpoint(model, data.vocab, validation_batches, 0, output_dir)]  # 更新之前先记录零步基线。
    chapter_to_book = {record["id"]: record["book"] for record in train_records}  # 根据章节标识追踪训练窗口来自哪本书。
    book_windows = {record["book"]: 0 for record in train_records}  # 为四本书分别累计实际参与训练的窗口数。
    batches = data.iter_batches(data.iter_windows(train_records, seq_length, seq_length), batch_size)  # 按本次固定章节顺序提供连续训练批次。
    completed_steps = 0  # 累计实际完成的更新次数。

    for step, batch in enumerate(islice(batches, max_steps), start=1):  # 连续处理前 200 个批次，每批更新一次。
        x, y = batch["x"], batch["y"]  # 取真实输入和下一字目标编号。
        loss, raw_grads = model.loss_and_grad(x, y)  # 用当前参数完成本批前向与完整反向。
        assert np.isfinite(loss), "训练损失异常"  # 不让无效损失继续进入训练记录。
        grads, raw_norm, scale = network.clip_gradients(raw_grads, max_norm)  # 汇总全部梯度后，再按全局范数裁剪。
        for name in model.params:  # 使用本批完整梯度更新五组参数。
            model.params[name] -= learning_rate * grads[name]  # 下一批继续使用本次更新后的参数。
        assert all(np.isfinite(param).all() for param in model.params.values()), "模型参数异常"  # 发现无效参数时停止。
        for chapter_id in batch["chapter_ids"]:  # 按实际窗口来源统计四本书的训练覆盖。
            book_windows[chapter_to_book[chapter_id]] += 1  # 本批每行窗口为对应书籍增加一次计数。
        completed_steps = step  # 记录已经完成的更新步数。
        if step % 25 == 0:  # 每隔 25 步显示一次进度，不逐步刷屏。
            print(f"训练 step={step} 更新前批次损失={loss:.6f} grad_norm={raw_norm:.6f} scale={scale:.6f}")  # 不要求不同批次损失单调下降。
        if step in (5, 50, 200):  # 在预先选定的步数评估同一个模型的当前参数。
            history.append(inspect_checkpoint(model, data.vocab, validation_batches, step, output_dir))  # 记录固定验证损失和相同设置下的生成结果。

    assert completed_steps == max_steps and all(count > 0 for count in book_windows.values()), "本次实验步数或四书覆盖不符合预期"  # 当前数据和固定顺序应完成 200 步并接触四本书。
    report = {  # 汇总本次实验的配置、数据来源和观察结果。
        "config": {  # 固定比较条件，方便以后复现实验。
            "steps": max_steps, "batch_size": batch_size, "seq_length": seq_length,  # 记录训练量与批次尺寸。
            "learning_rate": learning_rate, "max_norm": max_norm, "model_seed": 42, "chapter_shuffle_seed": 42,  # 记录更新规则与初始化顺序。
            "prompt": "春风", "new_tokens": 40, "temperature": 1.0, "sampling_seed": 42, "validation_batches": 3,  # 记录固定生成和验证条件。
        },  # 配置记录结束。
        "train_sha256": data.vocab["train_sha256"],  # 记录固定词表对应的训练文件指纹。
        "book_windows": book_windows,  # 记录四本书实际提供的训练窗口数。
        "history": history,  # 保存四个检查点的验证损失和生成样例。
    }  # 完整实验记录组装结束。
    report_path = data.ROOT / "docs" / "training_runs" / "lesson29.json"  # 为本次实验保留独立的可复查记录。
    report_path.parent.mkdir(parents=True, exist_ok=True)  # 创建记录目录，已经存在时继续使用。
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 写入中文可读的 JSON，不覆盖学习笔记。
    print("四本书实际训练窗口数：", book_windows)  # 确认四本书都实际提供了本次训练数据。
    print("完成更新步数：", completed_steps)  # 应为 200，而不是 200 个 epoch。
    print("实验记录路径：", report_path)  # 显示配置、验证损失和生成样例的保存位置。
    print("多步训练、固定验证与生成对照记录：完成")  # 只说明流程完成，不保证所有模型指标都更好。


if __name__ == "__main__":  # 直接运行脚本时才开始本次训练实验。
    main()  # 训练并记录零步、五步、五十步和两百步的模型。
