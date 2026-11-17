from collections import Counter  # 独立复算已保存片段的重复次数。
from importlib import import_module  # 导入生成工具而不触发保存报告入口。
from pathlib import Path  # 根据检查脚本确定项目位置。
import hashlib  # 检查来源权重文件不变。
import json  # 读取正式生成报告。
import sys  # 设置学习模块导入路径。
import numpy as np  # 检查候选排名与统计数值。

ROOT = Path(__file__).resolve().parents[2]  # 本文件位于scripts/ci。
sys.path.insert(0, str(ROOT / "scripts/learn"))  # 允许导入数字开头的学习模块。


def main():  # 只读核对样例、候选约束与生成行为，不训练。
    path = ROOT / "docs/training_runs/lesson39.json"  # 定位学习者实际保存的诊断报告。
    saved_bytes = path.read_bytes()  # 留下原报告以检查文件未变。
    report = json.loads(saved_bytes)  # 读取全部样例及汇总。
    lesson = import_module("23_generation_diagnostics")  # 导入函数，不进入报告保存入口。
    assert report["config"] == {"new_tokens": 60, "context_size": 32, "sampling_seeds": [42, 43], "repeat_ngram": 4}, "生成配置不一致"  # 确保各段长度和上下文条件固定。
    assert lesson.PROMPTS == ("春风", "唐僧道：", "宝玉听了") and lesson.MODES == (("greedy", None, None), ("sample_t1", 1.0, None), ("sample_t07", 0.7, None), ("topk40_t07", 0.7, 40)), "对照设置不一致"  # 不用临时开头或参数替换预先规定的实验。
    source = json.loads((ROOT / "docs/training_runs/lesson37.json").read_text(encoding="utf-8"))  # 核对真实训练来源。
    assert report["checkpoint"] == source["best_checkpoint"] == "lesson37_best.npz", "模型来源错误"  # 所有设置应使用同一最佳文件。
    weight_path = ROOT / "model" / report["checkpoint"]  # 定位来源权重。
    before_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()  # 保存模型文件校验值。
    assert before_hash == report["checkpoint_sha256"], "权重文件与报告不一致"  # 不接受其他模型生成的报告。
    model, vocab, step = lesson.training.load_checkpoint(weight_path, lesson.network)  # 校验文件并加载当前模型。
    assert step == report["completed_steps"] == 19148 and vocab["train_sha256"] == report["train_sha256"] == source["train_sha256"], "步数或数据版本不一致"  # 检查模型阶段与编码来源。
    tokens = vocab["tokens"]  # 保存编号解释所需的完整词表。
    char_to_id = {char: index for index, char in enumerate(tokens)}  # 根据模型词表编码开头。
    expected_cases = {(prompt, mode, seed) for prompt in lesson.PROMPTS for mode, temperature, _ in lesson.MODES for seed in ((42,) if temperature is None else (42, 43))}  # 三个开头共二十一种固定组合。
    rows = report["rows"]  # 所有原始生成样例都需保留。
    assert len(rows) == 21 and {(row["prompt"], row["mode"], row["seed"]) for row in rows} == expected_cases, "样例缺失或重复"  # 不只保存好看的结果。
    settings = {mode: (temperature, top_k) for mode, temperature, top_k in lesson.MODES}  # 按名称核对实际记录参数。
    for row in rows:  # 独立验证保存文本、统计与候选约束。
        ids = row["generated_ids"]  # 只检查续写，不把开头计入统计。
        assert (row["temperature"], row["top_k"]) == settings[row["mode"]], "模式参数不匹配"  # 同一名称必须对应相同生成策略。
        assert len(ids) == 60 and all(isinstance(index, int) and 0 <= index < len(tokens) for index in ids), "编号数量或范围错误"  # 每次续写六十个合法编号。
        assert row["text"] == row["prompt"] + "".join(tokens[index] for index in ids), "文本与编号不一致"  # 不允许手动润色保存的模型输出。
        counts = Counter(tuple(ids[start:start + 4]) for start in range(57))  # 独立累计五十七个重叠四元组。
        repeated = sum(count - 1 for count in counts.values()) / 57  # 每种片段首次出现不算重复。
        assert np.isclose(repeated, row["repeat4"]) and row["unknown_count"] == ids.count(vocab["unk_id"]), "重复或未知编号统计错误"  # 核对报告内统计而非强制某个质量结论。
        if row["mode"] in ("greedy", "topk40_t07"):  # 这两种策略对每个保存编号有明确排名约束。
            history = [char_to_id.get(char, vocab["unk_id"]) for char in row["prompt"]]  # 重新编码保存的开头。
            for index in ids:  # 沿保存的轨迹逐步核对候选，不重新抽取替代文字。
                scores = model.forward(np.asarray([history[-32:]], dtype=np.int64))[0, -1, :]  # 使用相同最近上下文算下一字分数。
                threshold = scores.max() if row["mode"] == "greedy" else np.partition(scores, -40)[-40]  # 贪心取最高分，top-k取第四十名边界。
                assert float(scores[index]) >= float(threshold) - 1e-6, "保存编号不属于规定候选"  # 为接近边界的跨平台浮点误差保留微小容差。
                history.append(index)  # 继续核对该保存轨迹的下一位置。
    assert [row["mode"] for row in report["summaries"]] == [mode for mode, _, _ in lesson.MODES], "汇总模式不完整"  # 四种设置都应有汇总。
    for summary in report["summaries"]:  # 独立从原始样例计算平均重复比例。
        group = [row for row in rows if row["mode"] == summary["mode"]]  # 获取该组全部固定样例。
        assert summary["samples"] == len(group) and np.isclose(summary["mean_repeat4"], sum(row["repeat4"] for row in group) / len(group)), "汇总不一致"  # 不要求重复率最小的策略语言质量最好。
    actual = lesson.main()  # 重跑当前环境的生成及固定种子复现、旧贪心兼容、参数未变检查。
    assert actual["config"] == report["config"] and len(actual["rows"]) == 21, "重新生成配置错误"  # 不将整段随机文字当成跨平台黄金值。
    greedy = lesson.generate_text(model, vocab, "春风", new_tokens=12)  # 取得当前模型贪心短续写。
    top_one = lesson.generate_text(model, vocab, "春风", temperature=0.7, top_k=1, seed=43, new_tokens=12)  # 只保留一个候选时应退化为贪心。
    assert greedy == top_one, "top-k为一没有退化为贪心"  # 确认候选范围限制在抽样时确实生效。
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == before_hash and path.read_bytes() == saved_bytes, "来源权重或报告被修改"  # 自动检查保持文件只读。
    print("lesson39 saved samples, candidate constraints and generation invariants passed", flush=True)  # 验证流程正确，不声明文字已经通顺。


if __name__ == "__main__":  # 直接运行检查脚本时才复核。
    main()  # 不训练，不新增模型或覆盖报告。
