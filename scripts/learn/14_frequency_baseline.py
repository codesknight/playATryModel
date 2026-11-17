from importlib import import_module  # 复用固定词表、窗口、模型和评估接口。
from itertools import islice  # 构造与上一课完全相同的固定验证子集。
import json  # 读取实验配置，并保存本课诊断记录。
import random  # 恢复上一课固定的训练章节顺序。
import numpy as np  # 按编号计数、计算概率和交叉熵。


def smoothed_probabilities(counts, alpha=1.0):  # 给每个编号加虚拟计数，再计算字符频率概率。
    counts = np.asarray(counts, dtype=np.float64)  # 用双精度计算概率，避免整数除法问题。
    assert counts.ndim == 1 and counts.size > 0 and np.isfinite(counts).all() and np.all(counts >= 0), "计数不合法"  # 每个词表编号对应一个非负计数。
    assert np.isfinite(alpha) and alpha > 0, "平滑系数必须是有限正数"  # 正平滑保证没出现的编号也有非零概率。
    return (counts + alpha) / (counts.sum() + alpha * counts.size)  # 所有编号的概率总和为 1。


def frequency_loss(probabilities, batches):  # 不看输入前文，只根据正确目标编号计算字频基线损失。
    assert probabilities.ndim == 1 and np.isfinite(probabilities).all() and np.all(probabilities > 0) and np.isclose(probabilities.sum(), 1.0), "概率分布不合法"  # 对数需要正概率。
    log_probs = np.log(probabilities)  # 同一分布用于每个位置，只需计算一次对数。
    loss_sum = 0.0  # 累计全部预测位置的负对数概率。
    positions = 0  # 累计实际评估位置数。
    for batch in batches:  # 遍历与 RNN 相同的验证批次。
        loss_sum += float((-log_probs[batch["y"]]).sum())  # 每个位置查正确答案对应概率，再取负对数。
        positions += batch["y"].size  # 尾批也按实际位置数统计。
    assert positions > 0, "没有可评估的位置"  # 不对空子集计算平均。
    return loss_sum / positions, positions  # 返回按位置平均的交叉熵及评估规模。

def main():  # 建立两条训练集字频基线，诊断已保存的两百步 RNN。
    data = import_module("06_make_windows")  # 复用训练数据、固定词表及窗口函数，首次导入会显示旧检查。
    network = import_module("07_tiny_rnn")  # 只导入模型结构，不重新训练。
    training = import_module("11_train_steps")  # 复用保存权重的加载和固定参数评估。
    experiment = json.loads((data.ROOT / "docs" / "training_runs" / "lesson29.json").read_text(encoding="utf-8"))  # 读取上一课实际实验条件。
    config = experiment["config"]  # 保持训练顺序、批大小和窗口长度与上一课一致。
    assert experiment["train_sha256"] == data.vocab["train_sha256"], "训练数据版本不一致"  # 不能将不同数据版本的基线直接比较。
    records = list(data.train_records)  # 复制训练章节列表，原数据顺序保持不变。
    random.Random(config["chapter_shuffle_seed"]).shuffle(records)  # 恢复上一课训练时的同一章节顺序。
    vocab_size = len(data.tokens)  # 每个词表编号都要有一个计数和概率。
    full_counts = np.zeros(vocab_size, dtype=np.int64)  # 统计全部训练窗口目标字符。
    seen_counts = np.zeros_like(full_counts)  # 另统计 RNN 前两百步实际见过的目标字符。
    batches = data.iter_batches(data.iter_windows(records, config["seq_length"], config["seq_length"]), config["batch_size"])  # 重走训练窗口流，只计数不更新模型。
    for batch_index, batch in enumerate(batches, start=1):  # 按相同顺序遍历全部训练批次。
        batch_counts = np.bincount(batch["y"].reshape(-1), minlength=vocab_size)  # 同一编号出现多次就增加多次计数。
        full_counts += batch_counts  # 全训练集基线累加每个批次的目标出现次数。
        if batch_index <= config["steps"]:  # 只给匹配训练范围的基线累计前两百批。
            seen_counts += batch_counts  # 与 RNN 相同的目标出现范围，包含重复字符。
    assert int(seen_counts.sum()) == 102400 and int(full_counts.sum()) == 2450688, "本课目标计数与实验范围不符"  # 确认匹配范围和全训练范围。
    alpha = 1.0  # 预先固定加一平滑，不用验证集选择该数值。
    seen_probs = smoothed_probabilities(seen_counts, alpha)  # 得到匹配两百步目标范围的频率模型。
    full_probs = smoothed_probabilities(full_counts, alpha)  # 得到使用全部训练目标的频率模型。
    with (data.ROOT / "data" / "splits" / "validation.jsonl").open(encoding="utf-8") as file:  # 验证集仅用于测量，不参与计数拟合。
        validation_records = [json.loads(line) for line in file]  # 保留原验证章节顺序。
    validation_batches = list(islice(data.iter_batches(data.iter_windows(validation_records, config["seq_length"], config["seq_length"]), config["batch_size"]), config["validation_batches"]))  # 使用与上一课相同的前三批。
    seen_loss, seen_positions = frequency_loss(seen_probs, validation_batches)  # 测量匹配训练范围的字频模型。
    full_loss, full_positions = frequency_loss(full_probs, validation_batches)  # 测量看过更多训练目标的全训练集字频模型。
    reference = next(row for row in experiment["history"] if row["step"] == config["steps"])  # 找到上一课两百步检查点记录。
    model, vocab, step = training.load_checkpoint(data.ROOT / "model" / reference["checkpoint"], network)  # 加载已训练模型，不增加训练步数。
    assert vocab == data.vocab and step == config["steps"], "权重与实验配置不一致"  # 词表顺序和训练步数都要匹配。
    rnn_loss, rnn_positions = training.evaluate(model, validation_batches, network)  # 固定 RNN 参数，重新评估同一子集。
    assert seen_positions == full_positions == rnn_positions == reference["validation_positions"] == 1536, "评估范围不一致"  # 四种预测方式必须比较相同位置。
    assert np.isclose(rnn_loss, reference["validation_loss"], rtol=1e-6, atol=1e-6), "RNN 与历史记录不一致"  # 先确认加载结果可复核。
    metrics = {"uniform": float(np.log(vocab_size)), "matched_frequency": seen_loss, "full_frequency": full_loss, "rnn200": rnn_loss}  # 均匀预测的交叉熵为 ln(V)，其余使用同一验证子集。
    print("匹配前两百步的目标字符数：", int(seen_counts.sum()))  # 应为 102400。
    print("全部训练窗口的目标字符数：", int(full_counts.sum()))  # 应为 2450688，区别于原文字符数。
    for name, loss in metrics.items():  # 列出四种预测方式的同范围损失。
        print(f"{name} 固定验证交叉熵：{loss:.6f}")  # 交叉熵越低，在当前目标上的平均对数概率越好。
    top_frequency = []  # 保留全训练集最常见的五个编号，供解释与记录。
    for index in np.argsort(full_probs)[-5:][::-1]:  # 按概率从高到低取前五项。
        item = {"token": data.tokens[index], "count": int(full_counts[index]), "probability": float(full_probs[index])}  # 记录真实计数和加一平滑概率。
        top_frequency.append(item)  # 将观察保存在本课报告中。
        print("高频字符：", repr(item["token"]), "次数：", item["count"], "平滑概率：", round(item["probability"], 6))  # 用 repr 显示标点和可能的换行。
    most_frequent = data.tokens[int(np.argmax(full_probs))]  # 字频模型不看前文，所以每步的最高概率编号都相同。
    print("全训练集字频基线贪心：", repr("春风" + most_frequent * 40))  # 演示简单字频模型也会连续重复最高频字符。

    observations = []  # 记录不同输入下 RNN 的实际下一字分布摘要。
    comma_id = data.char_to_id["，"]  # 找到当前词表中中文逗号的编号。
    for text in ("春风", "秋风", "唐僧", "曹操", "宝玉", "春风，", "春风，，"):  # 固定若干输入，包含相同末字与逐步追加逗号的对照。
        x = np.array([[data.char_to_id.get(char, data.unk_id) for char in text]], dtype=np.int64)  # 按同一词表编码每条输入。
        probabilities = network.softmax(model.forward(x)[0, -1, :])  # 只前向，查看读完输入后的下一字概率。
        next_id = int(np.argmax(probabilities))  # 观察贪心选择，不改变模型参数。
        item = {"input": text, "top_token": data.tokens[next_id], "top_probability": float(probabilities[next_id]), "comma_probability": float(probabilities[comma_id])}  # 保存最高候选和逗号概率。
        observations.append(item)  # 将不同输入下的观察保留在报告中。
        print(f"输入={text!r} 最高候选={item['top_token']!r} 最高概率={item['top_probability']:.6f} 逗号概率={item['comma_probability']:.6f}")  # 不能把最高候选误认为概率接近百分之百。
    result = {  # 汇总本课可复查的诊断结果。
        "alpha": alpha, "train_sha256": data.vocab["train_sha256"],  # 记录平滑设置与训练来源。
        "matched_steps": config["steps"], "matched_target_count": int(seen_counts.sum()), "full_target_count": int(full_counts.sum()),  # 标明两条基线不同的计数范围。
        "validation_positions": rnn_positions, "reference_checkpoint": reference["checkpoint"],  # 记录共同评估范围和被诊断的 RNN。
        "metrics": metrics, "top_frequency": top_frequency, "context_observations": observations,  # 保留损失、字频和真实输入对照。
    }  # 本课只写诊断记录，不保存或更新新的 RNN 权重。
    report_path = data.ROOT / "docs" / "training_runs" / "lesson30.json"  # 与上一课实验记录分开保存。
    report_path.parent.mkdir(parents=True, exist_ok=True)  # 确保记录目录存在。
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存中文可读的诊断结果。
    print("字频基线诊断记录：", report_path)  # 显示本课记录文件位置。
    print("训练集计数、同范围验证与字频基线检查：通过")  # 表示比较流程正确，不强制 RNN 必须胜出。


if __name__ == "__main__":  # 直接运行文件时才开始基线诊断。
    main()  # 只统计训练目标并评估，不执行参数更新。
