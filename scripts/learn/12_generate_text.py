from importlib import import_module  # 按字符串导入已有模型和权重加载函数。
from pathlib import Path  # 根据脚本位置找到已保存的权重文件。
import numpy as np  # 创建输入编号数组，并选择分数最高的候选。

def sampling_probabilities(scores, temperature):  # 用温度把最后位置的分数转换成抽样概率。
    scores = np.asarray(scores, dtype=np.float64)  # 用双精度计算概率，供随机抽样使用。
    assert scores.ndim == 1 and scores.size > 0 and np.isfinite(scores).all(), "需要有限的一维候选分数"  # 每次只处理最后一个位置。
    assert np.isfinite(temperature) and temperature > 0, "采样温度必须是有限正数"  # 温度参与除法，不能为零或负数。
    shifted = (scores - scores.max()) / temperature  # 先统一平移，再按温度缩放分数差距。
    weights = np.exp(shifted)  # 将缩放后的分数转换为非负权重。
    return weights / weights.sum()  # 除以权重总和，得到概率总和为 1 的分布。

def generate(model, vocab, prompt, new_tokens=40, context_size=32, temperature=None, seed=42):  # 默认贪心，指定正温度时按概率采样。
    assert isinstance(prompt, str) and len(prompt) > 0, "开头文字不能为空"  # 至少提供一个字符，模型才能预测后续。
    assert isinstance(new_tokens, int) and new_tokens >= 0 and isinstance(context_size, int) and context_size > 0, "生成步数或上下文长度不合法"  # 允许生成零步，但上下文长度必须为正。
    tokens = vocab["tokens"]  # 使用权重文件内保存的词表顺序。
    char_to_id = {char: index for index, char in enumerate(tokens)}  # 根据保存词表恢复字符到编号的映射。
    token_ids = [char_to_id.get(char, vocab["unk_id"]) for char in prompt]  # 未见字符使用已有未知编号，不扩充词表。
    generated_ids = []  # 只记录新生成的编号，便于检查生成步数。
    rng = np.random.default_rng(seed)  # 每次生成创建一个局部随机数生成器，在整个生成循环中持续使用。
    for _ in range(new_tokens):  # 每次循环选出一个新编号。
        x = np.asarray([token_ids[-context_size:]], dtype=np.int64)  # 只取最近的上下文，整理成一条二维输入。
        logits = model.forward(x)  # 只做前向，得到 (1,上下文长度,词表大小) 的分数。
        # next_id = int(np.argmax(logits[0, -1, :]))  # 最后位置预测下一个编号，贪心选择分数最高的一项。
        scores = logits[0, -1, :]  # 读完当前上下文后，用最后位置的分数预测下一字。
        if temperature is None:  # 未指定采样温度时保留上一课的贪心方式。
            next_id = int(np.argmax(scores))  # 直接选择分数最高的编号。
        else:  # 指定正温度后，允许按概率选择其他候选。
            probabilities = sampling_probabilities(scores, temperature)  # 温度调整概率的集中程度，不修改模型参数。
            next_id = int(rng.choice(len(tokens), p=probabilities))  # 根据每个候选的概率抽取一个合法编号。
        token_ids.append(next_id)  # 将本次结果加入输入，下一轮就能读到它。
        generated_ids.append(next_id)  # 单独保存本次生成的编号。
    text = prompt + "".join(tokens[index] for index in generated_ids)  # 保留原始开头，将新编号按保存词表解码。
    return text, generated_ids  # 返回文字和新编号，特殊未知符号也按词表显示。

def main():  # 独立加载权重并生成文字，不运行训练脚本的 main。
    network = import_module("07_tiny_rnn")  # 取得前向模型实现。
    checkpoints = import_module("11_train_steps")  # 只复用加载函数，模块的训练入口不会被执行。
    root = Path(__file__).resolve().parents[2]  # 从当前脚本位置找到项目根目录。
    checkpoint_path = root / "model" / "learn_rnn_step005.npz"  # 使用上一课已经保存并上传的学习权重。
    model, vocab, completed_steps = checkpoints.load_checkpoint(checkpoint_path, network)  # 从文件恢复参数和配套词表。
    prompt = "春风"  # 给定希望模型继续写下去的开头。
    saved_params = {name: param.copy() for name, param in model.params.items()}  # 保存副本，检查生成过程没有修改权重。
    text, generated_ids = generate(model, vocab, prompt, new_tokens=40, context_size=32)  # 连续选择四十个新编号。
    repeat_text, repeat_ids = generate(model, vocab, prompt, new_tokens=40, context_size=32)  # 同参数同输入再次贪心生成，核对确定性。
    assert len(generated_ids) == 40 and all(0 <= index < len(vocab["tokens"]) for index in generated_ids), "生成编号数量或范围不正确"  # 每步恰好生成一个合法编号。
    assert text.startswith(prompt) and text == repeat_text and generated_ids == repeat_ids, "开头或重复生成结果不一致"  # 贪心生成没有随机抽样。
    assert all(np.array_equal(param, saved_params[name]) for name, param in model.params.items()), "生成过程修改了参数"  # 生成只前向，不训练。
    print("加载权重的训练步数：", completed_steps)  # 当前权重只训练了五步。
    print("输入开头：", repr(prompt))  # 清楚显示开头中的特殊字符。
    print("新生成编号数：", len(generated_ids))  # 应为 40，不包含开头编号。
    print("生成结果：", repr(text))  # 用 repr 保留换行等特殊字符，方便核对输出。
    print("权重独立加载、贪心生成与参数不变检查：通过")  # 正确性检查通过不代表文字质量已经良好。
    demo_scores = np.log(np.array([0.6, 0.3, 0.1], dtype=np.float64))  # 构造温度为 1 时恰好得到这三个概率的分数。
    for temperature in (0.5, 1.0, 2.0):  # 比较较低、标准和较高温度。
        probabilities = sampling_probabilities(demo_scores, temperature)  # 对同一组分数只改变温度。
        assert np.isfinite(probabilities).all() and np.all(probabilities >= 0) and np.isclose(probabilities.sum(), 1.0), "采样概率不合法"  # 核对抽样分布。
        print(f"温度 {temperature} 的演示概率：", np.round(probabilities, 6))  # 只为展示而四舍五入，抽样不使用截短概率。
    sampled_text, sampled_ids = generate(model, vocab, prompt, new_tokens=40, context_size=32, temperature=1.0, seed=42)  # 用保存权重按原始概率连续抽样四十步。
    repeated_sample, repeated_ids = generate(model, vocab, prompt, new_tokens=40, context_size=32, temperature=1.0, seed=42)  # 同参数、温度、开头和采样种子再次生成。
    assert sampled_text == repeated_sample and sampled_ids == repeated_ids, "固定采样种子没有复现结果"  # 局部随机数状态应从同一种子出发。
    assert len(sampled_ids) == 40 and all(0 <= index < len(vocab["tokens"]) for index in sampled_ids), "采样编号数量或范围错误"  # 每一步必须得到一个合法编号。
    assert all(np.array_equal(param, saved_params[name]) for name, param in model.params.items()), "采样生成修改了参数"  # 改变解码策略不改变权重。
    print("温度 1.0、采样种子 42 的结果：", repr(sampled_text))  # 观察概率抽样与贪心结果的区别。
    print("温度概率、固定种子采样与参数不变检查：通过")  # 检查流程正确，不判断文字质量。

if __name__ == "__main__":  # 直接运行新文件时才开始生成演示。
    main()  # 加载磁盘权重并生成，不执行训练更新。
