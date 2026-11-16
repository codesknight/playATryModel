from importlib import import_module  # 按字符串导入已有数据与模型模块。
from itertools import islice  # 从批次生成器中只取指定数量的批次。
import numpy as np  # 检查批次数组与训练损失。
import json  # 导入 JSON 工具，读取独立的验证章节文件。
from pathlib import Path  # 导入路径工具，创建权重目录并检查文件大小。

def evaluate(model, batches, network):  # 固定参数，只前向计算给定批次的平均损失。
    saved_params = {name: param.copy() for name, param in model.params.items()}  # 保存评估前的参数，用来检查评估没有更新模型。
    loss_sum = 0.0  # 累计所有预测位置的损失总和。
    position_count = 0  # 累计实际参与评估的预测位置数。
    for batch in batches:  # 依次评估固定子集中的各个批次。
        x, y = batch["x"], batch["y"]  # 取验证输入与正确的下一字编号。
        loss, _ = network.cross_entropy(model.forward(x), y)  # 只做前向和损失计算，不调用反向传播。
        assert np.isfinite(loss), "验证损失不是有限数"  # 出现无效结果时停止检查。
        loss_sum += loss * y.size  # 按本批实际位置数加权，不能把大小不同的批次简单等权平均。
        position_count += y.size  # 累计包括小尾批在内的实际位置数。
    assert position_count > 0, "没有可评估的位置"  # 防止空子集导致除以零。
    assert all(np.array_equal(param, saved_params[name]) for name, param in model.params.items()), "评估修改了参数"  # 参数必须与评估之前完全一致。
    return loss_sum / position_count, position_count  # 返回固定参数下的加权平均损失和评估规模。

def save_checkpoint(path, model, vocab, completed_steps):  # 将模型参数、词表、结构配置和已完成步数保存到一个文件。
    path = Path(path)  # 把传入路径统一转换为 Path 对象。
    assert path.suffix == ".npz", "本课权重文件使用 .npz 后缀"  # 避免保存工具自动补后缀导致路径不一致。
    assert len(vocab["tokens"]) == model.vocab_size, "词表与模型大小不一致"  # 每个输入输出编号都必须对应保存的词表。
    assert isinstance(completed_steps, int) and completed_steps >= 0, "训练步数不合法"  # 已完成更新次数必须是非负整数。
    assert all(param.dtype == np.float32 and np.isfinite(param).all() for param in model.params.values()), "本课只保存有限的 float32 参数"  # 与当前正式模型的类型保持一致。
    metadata = {"format": "tiny_rnn_char_v1", "hidden_size": model.hidden_size, "completed_steps": completed_steps, "vocab": vocab}  # 保存重建模型和解释编号所需的信息。
    metadata_text = json.dumps(metadata, ensure_ascii=False)  # 将配置转成普通 JSON 字符串，保留中文字符。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保权重文件的父目录存在。
    np.savez_compressed(path, metadata=np.array(metadata_text), **model.params)  # 将字符串配置和五组参数压缩到同一个 npz 文件。

def load_checkpoint(path, network):  # 从权重文件创建一个带已训练参数的新模型。
    with np.load(path, allow_pickle=False) as archive:  # 只读取数组和字符串，不依赖 Python 对象反序列化。
        metadata = json.loads(str(archive["metadata"].item()))  # 将保存的单个字符串数组还原为配置字典。
        assert metadata["format"] == "tiny_rnn_char_v1", "权重格式与本模型不匹配"  # 不把其他结构的权重当成本课模型。
        vocab = metadata["vocab"]  # 使用权重文件内保存的词表，保留原来的编号顺序。
        model = network.TinyRNN(len(vocab["tokens"]), hidden_size=metadata["hidden_size"], seed=0)  # 按保存的维度创建模型，随机初值稍后全部被覆盖。
        for name, expected in model.params.items():  # 逐一加载模型必须具有的五组参数。
            value = archive[name]  # 取出当前参数数组。
            assert value.shape == expected.shape and value.dtype == expected.dtype and np.isfinite(value).all(), name + "权重形状、类型或数值不正确"  # 防止错误数组进入模型。
            model.params[name] = value.copy()  # 复制已训练参数，文件关闭后新模型仍可独立使用。
    return model, vocab, metadata["completed_steps"]  # 返回新模型、对应词表和保存时的训练步数。

def main():  # 定义连续训练少量批次的演示入口。
    data = import_module("06_make_windows")  # 复用真实训练章节和批次函数，首次导入会执行旧数据检查。
    network = import_module("07_tiny_rnn")  # 复用模型、完整反向和梯度裁剪。
    batch_size = 16  # 每个批次最多包含 16 个窗口。
    seq_length = 32  # 每个窗口预测 32 个位置，窗口步长也取 32。
    max_steps = 5  # 本课只更新五个批次，不训练完整一轮。
    learning_rate = 0.1  # 每一步使用相同的普通梯度下降学习率。
    max_norm = 1.0  # 每一步都对完整梯度使用相同的全局范数上限。
    model = network.TinyRNN(len(data.tokens), hidden_size=64, seed=42)  # 只在循环外创建一次模型，之后持续更新同一份参数。

    validation_path = data.ROOT / "data" / "splits" / "validation.jsonl"  # 读取此前按完整章节划分的验证集。
    with validation_path.open(encoding="utf-8") as file:  # 以 UTF-8 打开验证章节文件。
        validation_records = [json.loads(line) for line in file]  # 逐行还原验证章节，保留原有顺序。
    assert not ({record["id"] for record in data.train_records} & {record["id"] for record in validation_records}), "训练与验证章节标识重叠"  # 确认评估章节没有进入训练集。
    validation_windows = data.iter_windows(validation_records, seq_length, seq_length)  # 继续使用训练词表编码验证文字，未见字符映射到已有未知编号。
    validation_batches = list(islice(data.iter_batches(validation_windows, batch_size), 3))  # 固定保存前三个验证批次，前后评估复用相同数据。
    assert len(validation_batches) == 3, "本课验证数据不足三批"  # 当前数据应能提供三批窗口。
    validation_before, validation_positions = evaluate(model, validation_batches, network)  # 在任何训练更新之前评估初始模型。
    print("固定验证子集批次数：", len(validation_batches))  # 应为 3，不能称为整个验证集。
    print("固定验证子集预测位置数：", validation_positions)  # 当前三批共有 1536 个位置。
    print("训练前固定验证子集损失：", validation_before)  # 记录未训练模型在固定验证子集上的损失。

    windows = data.iter_windows(data.train_records, seq_length, seq_length)  # 只创建一次窗口生成器，使窗口位置持续向后移动。
    batches = data.iter_batches(windows, batch_size)  # 把这一个窗口流打包为连续批次。
    completed_steps = 0  # 累计真正完成的参数更新次数。
    seen_windows = 0  # 累计真正用于更新的窗口数。
    seen_positions = 0  # 累计真正参与损失计算的预测位置数。
    loss_sum = 0.0  # 累计各批平均损失乘其位置数，准备计算加权平均。

    for batch in islice(batches, max_steps):  # 顺着同一个批次生成器取前五批，每批更新一次。
        x, y = batch["x"], batch["y"]  # 取本批输入和正确的下一字编号。
        assert x.shape == y.shape and 0 < x.shape[0] <= batch_size and x.shape[1] == seq_length, "批次尺寸错误"  # 按实际行数检查，也允许未来遇到不足一批的尾批。
        assert x.dtype == y.dtype == np.int64 and np.array_equal(x[:, 1:], y[:, :-1]), "批次类型或错位错误"  # 保证训练格式仍然正确。
        loss, raw_grads = model.loss_and_grad(x, y)  # 用当前参数计算本批更新前的平均损失与完整梯度。
        assert np.isfinite(loss), "训练损失不是有限数"  # 发现异常数值就停止，不继续污染参数。
        grads, raw_norm, scale = network.clip_gradients(raw_grads, max_norm)  # 每步完整反向结束后再做一次全局裁剪。
        for name in model.params:  # 使用同一批次算出的五组梯度统一更新参数。
            model.params[name] -= learning_rate * grads[name]  # 本步新参数会被下一批直接使用。
        assert all(np.isfinite(param).all() for param in model.params.values()), "更新后参数异常"  # 确认连续更新没有产生无效参数。
        completed_steps += 1  # 五组参数更新完成，记作一个 step。
        seen_windows += x.shape[0]  # 统计实际行数，不假定永远是完整批次。
        seen_positions += y.size  # 统计本批实际预测位置数。
        loss_sum += loss * y.size  # 将平均损失还原为本批损失总和，再跨批累计。
        print(f"step={completed_steps} loss={loss:.6f} grad_norm={raw_norm:.6f} scale={scale:.6f} 来源={batch['chapter_ids'][0]} 起点={batch['starts'][0]}")  # 显示每步更新前的损失和首行窗口来源。

    assert completed_steps == max_steps and seen_positions > 0, "当前训练数据不足以完成本课五步"  # 本例训练集应能提供足够批次。
    windows_per_epoch = sum(max(0, (len(record["text"]) - seq_length - 1) // seq_length + 1) for record in data.train_records)  # 根据当前窗口长度与步长，独立计算完整一轮的窗口数。
    steps_per_epoch = (windows_per_epoch + batch_size - 1) // batch_size  # 尾批也更新一次，用向上取整计算一轮步数。
    print("已完成更新步数：", completed_steps)  # 应为 5，而不是五组参数乘五步。
    print("已训练窗口数：", seen_windows)  # 当前五个完整批次共 80 个窗口。
    print("已训练预测位置数：", seen_positions)  # 当前共 2560 个预测位置。
    print("训练过程按位置加权的平均损失：", loss_sum / seen_positions)  # 汇总各步更新前的损失，模型参数在这些步骤间持续变化。
    print("完整一轮需要的更新步数：", steps_per_epoch)  # 当前配置为 4787 步，包含最后八行的尾批。
    print("本次是否遍历完整训练集：", seen_windows == windows_per_epoch)  # 本课只取五批，因此应为 False。
    print("同一模型连续多批更新检查：通过")  # 所有本课断言通过后显示。

    validation_after, after_positions = evaluate(model, validation_batches, network)  # 五步训练结束后，用同一子集评估最终参数。
    assert after_positions == validation_positions, "前后评估位置数不一致"  # 前后比较必须使用相同的数据范围。
    print("训练后固定验证子集损失：", validation_after)  # 不要求它一定低于训练前，验证结果需要如实记录。
    print("固定验证子集损失变化：", validation_after - validation_before)  # 负数表示该子集损失降低，正数表示升高。
    print("验证只前向、参数不变与固定子集检查：通过")  # 正确性检查通过不代表验证效果一定改善。

    checkpoint_path = data.ROOT / "model" / "learn_rnn_step005.npz"  # 将本课五步训练成果保存在独立文件，不覆盖原示例权重。
    save_checkpoint(checkpoint_path, model, data.vocab, completed_steps)  # 保存训练后的五组参数与完整词表配置。
    loaded_model, loaded_vocab, loaded_steps = load_checkpoint(checkpoint_path, network)  # 从磁盘加载成一个新的模型对象。
    assert loaded_model is not model and loaded_vocab == data.vocab and loaded_steps == completed_steps, "加载配置或训练步数不一致"  # 新对象应保留原词表和步数。
    assert all(np.array_equal(param, loaded_model.params[name]) for name, param in model.params.items()), "加载后的参数发生变化"  # 无损保存后五组参数应逐元素完全一致。
    probe_x = validation_batches[0]["x"]  # 固定同一个验证输入，比较加载前后的预测分数。
    logits_before = model.forward(probe_x)  # 用内存中的训练模型计算分数。
    logits_loaded = loaded_model.forward(probe_x)  # 用磁盘重新加载的模型计算相同输入的分数。
    assert np.allclose(logits_before, logits_loaded, rtol=1e-5, atol=1e-6), "加载前后预测不一致"  # 在浮点容差内检查预测计算一致。
    loaded_loss, loaded_positions = evaluate(loaded_model, validation_batches, network)  # 重新评估加载模型，不进行训练。
    assert loaded_positions == after_positions and np.isclose(loaded_loss, validation_after, rtol=1e-6, atol=1e-7), "加载前后验证损失不一致"  # 使用相同固定子集比较损失。
    assert checkpoint_path.stat().st_size < 1_000_000_000, "权重文件超过本项目的 1 GB 上限"  # 检查实际保存文件的字节数。
    print("权重保存路径：", checkpoint_path)  # 显示可供以后独立加载的文件位置。
    print("权重文件字节数：", checkpoint_path.stat().st_size)  # 文件包含压缩参数与配置，大小不等于裸参数字节数。
    print("加载后训练步数：", loaded_steps)  # 应为 5，这只是保存的记录，不会自动执行继续训练。
    print("加载前后验证损失：", validation_after, loaded_loss)  # 两个值应在容差内一致。
    print("模型保存、重载与预测一致性检查：通过")  # 配置、参数、预测与损失检查全部通过后显示。


if __name__ == "__main__":  # 只有直接运行文件时才开始训练演示。
    main()  # 创建一次模型并连续执行五步更新。
