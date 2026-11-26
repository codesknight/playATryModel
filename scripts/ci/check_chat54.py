import copy  # 在内存篡改标签验证检查器真的拒绝错误。
import json  # 读取来源、训练和审阅证据。
import sys  # 导入实际训练数据函数。
from pathlib import Path  # 定位仓库。
import numpy as np  # 重建全部成功更新的数据位置。
ROOT = Path(__file__).resolve().parents[2]  # 项目根目录。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 不调用训练入口。
from train_language54 import load_data, DOMAINS, SCHEDULE, WEIGHTS, DIRECTORY, LANGUAGE  # 实际固定配比和文件。
from train_chat_pilot import batch_from_stream, evaluate  # 尾窗计数及CPU明确子集复核。
from prepare_chat54 import verify_task  # 正式任务标签检查器。
from prepare_chat54 import build_catalog  # 重建一般问答与新课程的独立组件目录。
from chat_model import ChatLM, torch  # CPU载入推理文件。
from prepare_chat_corpus import sha  # 内容摘要。
torch.set_num_threads(2)  # 不在CI启动长训练或CUDA。
arrays, tokenizer, language, tasks = load_data()  # 核对全部实际输入和审阅版本。
assert build_catalog()["used_for_sft_this_stage"] is False, "本阶段目录用途改变"  # 不把课程准备记成已经微调。
curriculum = json.loads((DIRECTORY / "curriculum/manifest.json").read_text(encoding="utf-8"))  # 800条程序化课程，不是800条人工标注。
groups, example = {}, {}  # 记录原输入组及每类损坏测试样例。
for split in ("train", "validation"):  # 保持已发布的组级划分。
    path = DIRECTORY / "curriculum" / (split + ".jsonl")  # 完整可验证问答。
    assert sha(path.read_bytes()) == curriculum["files"][path.name], "课程文件摘要改变"  # 不静默换题。
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]  # 只读当前两个划分。
    groups[split] = {r["group_id"] for r in rows}  # 同有序输入跨题型也保持组。
    for row in rows:  # 所有正标签均核验。
        verify_task(row)  # 逆运算、集合与单调性检查。
        example.setdefault(row["family"], row)  # 保存第一条用于反例测试。
assert not groups["train"] & groups["validation"] and len(example) == 10, "任务组泄漏或覆盖不足"  # 不把模板共享误称未见模板测试。
for family, row in example.items():  # 每类都应拒绝一个有效格式但错误的标签。
    bad = copy.deepcopy(row)  # 不改正式课程数据。
    content = bad["messages"][-1]["content"]  # 原正确答案。
    if family in ("add", "subtract", "multiply", "gcd", "count"):  # 有效整数改成错误值。
        content = str(int(content) + 1)  # 不依赖解析失败来通过测试。
    elif family == "reverse":  # 仍为合法文本但不是倒序。
        content += "错"  # 反转恢复不应通过。
    else:  # 数组仍是合法JSON。
        value = json.loads(content)  # 保留原数组元素类型。
        content = json.dumps(value + [value[0]])  # 排序破坏重数，去重增加重复，极值变三项。
    bad["messages"][-1]["content"] = content  # 将错误值送正式检查器。
    try:  # 不能让错误监督标签混入。
        verify_task(bad)  # 与实际审核相同入口。
    except AssertionError:  # 正确拒绝。
        pass  # 继续下一类。
    else:  # 任一漏放就使CI失败。
        raise AssertionError("错误课程标签被放行：" + family)  # 不仅测试生成器自己算出的值。
report = json.loads((ROOT / "docs/training_runs/lesson54.json").read_text(encoding="utf-8"))  # 实际混合预训练记录。
old = json.loads((ROOT / "docs/training_runs/lesson50.json").read_text(encoding="utf-8"))  # 原预训练来源固定。
assert report["source_sha256"] == old["artifacts"]["last_train.pt"]["sha256"] and report["completed_steps"] == 2020 + report["additional_updates"] and 0 < report["additional_updates"] <= 1000, "来源或预算计数错误"  # 不含原SFT500步。
assert report["config"]["max_seconds"] == 900 and report["task_data_prepared_not_sft_trained"] and report["schedule"] == list(SCHEDULE), "本阶段范围改变"  # 任务课程没有暗中参加本阶段梯度更新。
assert abs(report["history"][0]["domains"]["original"]["loss"] - old["history"][-1]["loss"]) < 1e-5, "旧验证起点没有复现"  # 三域比较不能换掉旧尺子。
for row in report["history"]:  # 每个固定检查点使用相同域及范围。
    assert all(row["domains"][d]["positions"] == language["stats"][d]["validation_positions"] for d in DOMAINS), "分域验证范围改变"  # 不挑容易的窗口。
    assert abs(row["weighted_score"] - sum(WEIGHTS[d] * row["domains"][d]["loss"] for d in DOMAINS)) < 1e-8, "加权分数错误"  # 权重在起点评估前固定。
assert report["best_score"] == min(row["weighted_score"] for row in report["history"]), "没有选择真正最佳"  # 起点也须参与比较。
cursors, positions, references = {d: {"epoch": 0, "offset": 0} for d in DOMAINS}, {d: 0 for d in DOMAINS}, []  # 新流游标不误用旧流epoch1/offset4306。
for _ in range(report["additional_updates"]):  # 重建每次成功更新的八个真实引用。
    for domain in SCHEDULE:  # 固定4/3/1。
        index, cursor = DOMAINS.index(domain), cursors[domain]  # 独立域游标。
        windows = language["stats"][domain]["train_windows"]  # 域内真实尾窗数。
        order = np.random.default_rng(654 + index * 1000 + cursor["epoch"]).permutation(windows)  # 与正式训练一致。
        window = int(order[cursor["offset"]])  # 不能重复上一已完成位置。
        positions[domain] += batch_from_stream(arrays[domain + "_train.npy"], window)[2]  # 不盲目乘256。
        references.append((index, cursor["epoch"], window))  # 指纹包含来源。
        cursor["offset"] += 1  # 当前窗口完成。
        if cursor["offset"] == windows:  # 保留跨域跨轮处理。
            cursors[domain] = {"epoch": cursor["epoch"] + 1, "offset": 0}  # 下一轮重新排列。
assert positions == report["domain_positions"] and sum(positions.values()) == report["phase_seen_positions"] and cursors == report["cursors"] and sha(np.asarray(references, dtype=np.int64).tobytes()) == report["exposure_sha256"], "混合比例、尾窗或游标记录错误"  # 不把尝试次数当成功曝光。
file = ROOT / "model/lesson54/base_fp16.pt"  # 实际导出的最佳基础模型。
assert sha(file.read_bytes()) == report["artifacts"][file.name]["sha256"] and file.stat().st_size < 1_000_000_000, "推理文件不正确或过大"  # 真实文件满足1GB。
state = torch.load(file, map_location="cpu", weights_only=True)  # CPU只复算固定子集。
assert state["step"] == report["best_step"] and state["stage"] == "mixed_language_pretrain_not_sft", "阶段或最佳位置错误"  # 不将基础模型说成新聊天微调。
model = ChatLM(**state["model_config"])  # 与旧30M参数结构相同。
model.load_state_dict(state["model_state"])  # FP16导出载入FP32。
for domain in DOMAINS:  # 每域四窗，明确不是全量CPU验证。
    metric = evaluate(model, arrays[domain + "_validation_x.npy"], arrays[domain + "_validation_y.npy"], limit=4)  # 仅12个窗口。
    assert metric["positions"] == report["export_ci_subset"][domain]["positions"] and abs(metric["loss"] - report["export_ci_subset"][domain]["loss"]) < 0.003, "分域CPU导出核验失败"  # 容许平台精度差。
    print("CPU四窗：", domain, metric)  # 显示真实范围。
resume = json.loads((ROOT / "docs/training_runs/lesson54_resume.json").read_text(encoding="utf-8"))  # 完整状态实际保存重载后的下一步证据。
assert resume["formal_checkpoint_step"] == report["completed_steps"] and resume["max_parameter_abs_diff"] <= 1e-6 and resume["optimizer_scaler_cursors_equal"], "三域完整恢复不一致"  # 两次核验副本不算正式新增更新。
print("800任务标签及反例、组隔离、三域曝光/验证、导出与完整恢复检查：通过")  # 不在CI重新训练。
