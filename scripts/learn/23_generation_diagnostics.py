from importlib import import_module  # 复用模型、权重加载和温度概率函数。
from pathlib import Path  # 从脚本位置找到项目目录。
import hashlib  # 检查生成前后权重文件没有变化。
import json  # 读取模型来源并保存诊断记录。
import numpy as np  # 选择候选编号和执行概率采样。

ROOT = Path(__file__).resolve().parents[2]  # 本文件位于scripts/learn。
network = import_module("07_tiny_rnn")  # 使用已经训练好的模型结构。
training = import_module("11_train_steps")  # 复用权重加载函数。
decoding = import_module("12_generate_text")  # 复用稳定的温度概率计算。
PROMPTS = ("春风", "唐僧道：", "宝玉听了")  # 预先固定三个开头，避免只展示最好看的一个。
MODES = (("greedy", None, None), ("sample_t1", 1.0, None), ("sample_t07", 0.7, None), ("topk40_t07", 0.7, 40))  # 分别比较贪心、温度变化及候选限制。


def repeat_rate(ids, n=4):  # 统计新生成编号中扣除首次出现后的重复片段比例。
    assert n >= 1, "片段长度必须为正"  # 本课默认连续四个编号为一个片段。
    pieces = [tuple(ids[start:start + n]) for start in range(len(ids) - n + 1)]  # 用步长一取出可重叠的片段。
    return 0.0 if not pieces else 1.0 - len(set(pieces)) / len(pieces)  # 片段越重复，比例通常越高，但不直接衡量通顺程度。

def generate_text(model, vocab, prompt, temperature=None, top_k=None, seed=42, new_tokens=60):  # 固定最近三十二字上下文，支持候选限制采样。
    tokens = vocab["tokens"]  # 使用权重保存的词表顺序。
    char_to_id = {char: index for index, char in enumerate(tokens)}  # 恢复字符到编号的对应关系。
    assert isinstance(prompt, str) and prompt and isinstance(new_tokens, int) and new_tokens >= 0, "开头或长度无效"  # 至少提供一个输入字。
    assert top_k is None or isinstance(top_k, int) and 1 <= top_k <= len(tokens), "候选数量无效"  # 候选数量不能超出词表。
    assert temperature is None or np.isfinite(temperature) and temperature > 0, "温度无效"  # 采样温度必须为正。
    assert temperature is not None or top_k is None, "贪心模式不需要top-k"  # 本课只在采样模式限制候选。
    history = [char_to_id.get(char, vocab["unk_id"]) for char in prompt]  # 未见输入字符使用固定未知编号。
    generated = []  # 只保存续写编号，不把开头计入重复率。
    rng = np.random.default_rng(seed)  # 随机生成器在整个生成循环中持续使用。
    for _ in range(new_tokens):  # 每次生成一个编号并接回上下文。
        x = np.asarray([history[-32:]], dtype=np.int64)  # 与旧生成函数使用同样的上下文长度。
        scores = model.forward(x)[0, -1, :]  # 最后位置给出下一字的分数。
        if temperature is None:  # 贪心每次直接选最高分。
            next_id = int(np.argmax(scores))  # 不进行随机抽样。
        else:  # 指定温度时才使用概率采样。
            candidates = np.arange(len(tokens)) if top_k is None or top_k == len(tokens) else np.argsort(-scores, kind="stable")[:top_k]  # 全词表或最高分的k个编号作为候选。
            probabilities = decoding.sampling_probabilities(scores[candidates], temperature)  # 只在保留候选内重新归一化。
            next_id = int(rng.choice(candidates, p=probabilities))  # 按候选编号及其对应概率抽样。
        history.append(next_id)  # 新编号会成为下一步的输入。
        generated.append(next_id)  # 单独记录续写部分。
    return prompt + "".join(tokens[index] for index in generated), generated  # 返回完整文字及续写编号。

def main():  # 固定权重，比较四种解码设置的生成结果。
    source = json.loads((ROOT / "docs/training_runs/lesson37.json").read_text(encoding="utf-8"))  # 使用完整第四轮选出的最佳模型。
    assert source["status"] == "completed", "第四轮尚未完成"  # 不使用未核验训练结果。
    model_path = ROOT / "model" / source["best_checkpoint"]  # 指向保存的最佳权重。
    before_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()  # 留下原权重文件校验值。
    model, vocab, step = training.load_checkpoint(model_path, network)  # 只加载，不训练。
    assert vocab["train_sha256"] == source["train_sha256"], "词表来源不一致"  # 保证输入编号含义正确。
    saved = {name: value.copy() for name, value in model.params.items()}  # 结束时检查内存参数未变。
    rows = []  # 保存所有结果，不只保留看起来最好的一段。
    for prompt in PROMPTS:  # 每种设置都使用同样的三个开头。
        for mode, temperature, top_k in MODES:  # 成对比较温度和top-k的影响。
            for seed in ((42,) if temperature is None else (42, 43)):  # 贪心不需要多个种子，采样固定两个种子观察波动。
                text, ids = generate_text(model, vocab, prompt, temperature, top_k, seed)  # 每段固定续写六十个编号。
                assert len(ids) == 60 and all(0 <= index < len(vocab["tokens"]) for index in ids), "生成编号无效"  # 检查生成长度及范围。
                row = {"prompt": prompt, "mode": mode, "temperature": temperature, "top_k": top_k, "seed": seed, "text": text, "generated_ids": ids, "repeat4": repeat_rate(ids), "unknown_count": ids.count(vocab["unk_id"])}  # 记录解码条件、全文及辅助统计。
                rows.append(row)  # 全量保留，避免只挑选好看的例子。
                print(f"开头={prompt!r} mode={mode} seed={seed} repeat4={row['repeat4']:.1%}", flush=True)  # 显示重复片段比例。
                print(repr(text), flush=True)  # 用repr保留换行及特殊字符。


    summaries = []  # 汇总各设置在这些固定样例上的平均重复比例。
    for mode, _, _ in MODES:  # 每种设置单独汇总。
        group = [row for row in rows if row["mode"] == mode]  # 收集对应三个开头及预定种子的结果。
        average = sum(row["repeat4"] for row in group) / len(group)  # 各段续写长度相同，按样例平均。
        summaries.append({"mode": mode, "samples": len(group), "mean_repeat4": average})  # 重复统计不能当作语言质量分数。
        print(f"汇总 {mode}: samples={len(group)} mean_repeat4={average:.1%}", flush=True)  # 先看统计，再人工阅读文字。
    old_text, old_ids = decoding.generate(model, vocab, PROMPTS[0], new_tokens=60)  # 核对新函数没有改变原贪心行为。
    assert rows[0]["text"] == old_text and rows[0]["generated_ids"] == old_ids, "贪心实现与旧版不一致"  # 新增采样功能不应破坏贪心。
    first_text, first_ids = generate_text(model, vocab, PROMPTS[0], 0.7, 40, 42)  # 再次用相同种子生成top-k样例。
    repeated_text, repeated_ids = generate_text(model, vocab, PROMPTS[0], 0.7, 40, 42)  # 核对当前环境中可复现。
    assert first_text == repeated_text and first_ids == repeated_ids, "固定种子未复现"  # 不比较不同平台的整段黄金文字。
    assert all(np.array_equal(value, saved[name]) for name, value in model.params.items()) and hashlib.sha256(model_path.read_bytes()).hexdigest() == before_hash, "生成修改了权重"  # 内存参数和磁盘文件都应不变。
    return {"checkpoint": model_path.name, "checkpoint_sha256": before_hash, "completed_steps": step, "train_sha256": vocab["train_sha256"], "config": {"new_tokens": 60, "context_size": 32, "sampling_seeds": [42, 43], "repeat_ngram": 4}, "rows": rows, "summaries": summaries}  # 保存全部原始结果，人工评价另行记录。


if __name__ == "__main__":  # 直接运行时才生成诊断报告。
    result = main()  # 本课只前向生成，不进行训练。
    path = ROOT / "docs/training_runs/lesson39.json"  # 保存独立生成对照报告。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存全部设置、统计和文本。
    print("生成诊断报告：", path)  # 输出结果位置。
