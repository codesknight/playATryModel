from pathlib import Path  # 从当前脚本位置找到项目根目录。
import json  # 读取旧验证记录，并保存本课模拟结果。
import math  # 检查损失和阈值是否为有限数。

ROOT = Path(__file__).resolve().parents[2]  # 本文件放在 scripts/learn。


def simulate_early_stopping(history, patience, min_delta):  # 按时间顺序模拟早停，不训练模型。
    assert isinstance(patience, int) and patience >= 1, "耐心次数至少为一"  # patience按验证检查次数计数。
    assert math.isfinite(min_delta) and min_delta >= 0, "改善阈值不能为负"  # min_delta使用交叉熵的绝对差值。
    assert len(history) >= 1, "没有验证记录"  # 至少需要一条起点记录。
    best = dict(history[0])  # 起点也是已经观察过的最佳候选。
    reference_loss = best["validation_loss"]  # 早停参考值只在改善足够明显时更新。
    bad_checks = 0  # 连续没有明显改善的检查次数。
    actions = []  # 记录每次实际访问的验证检查和决定。

    for row in history[1:]:  # 从起点之后的第一条验证记录开始逐条观察。
        loss = row["validation_loss"]  # 此时只能看到当前分数，不能提前使用未来记录。
        new_best = loss < best["validation_loss"]  # 任何严格更低的损失都值得保留快照。
        if new_best:  # 更新用于最终模型选择的最低分记录。
            best = dict(row)  # 复制普通记录，不修改原始历史。
        significant = reference_loss - loss > min_delta  # 本课规定改善量必须严格大于阈值。
        if significant:  # 相比上次有效参考值已有足够改善。
            reference_loss = loss  # 更新早停参考值。
            bad_checks = 0  # 有明显改善，连续未改善次数清零。
        else:  # 变差、持平或改善不足阈值都计入等待。
            bad_checks += 1  # 不是训练步数加一，而是验证检查次数加一。
        should_stop = bad_checks >= patience  # 达到预先规定的等待次数就触发停止。
        actions.append({"local_step": row["local_step"], "validation_loss": loss, "new_best": new_best, "significant": significant, "reference_loss": reference_loss, "bad_checks": bad_checks, "should_stop": should_stop})  # 记录判断依据。
        if should_stop:  # 真实训练将在这里跳出训练过程。
            break  # 本模拟也不再读取后面的验证分数。
    last_seen = history[len(actions)]  # 起点不计入actions，因此此下标对应最后观察的位置。
    return {  # 记录停止位置与选中模型，二者可以不同。
        "patience": patience, "min_delta": min_delta,  # 保留本组规则。
        "stop_triggered": bool(actions and actions[-1]["should_stop"]),  # 是否触发停止，而非宣称实际撤销了训练。
        "last_seen_local_step": last_seen["local_step"], "best_local_step": best["local_step"],  # 区分停止位置和最低分位置。
        "best_checkpoint": best["checkpoint"], "best_validation_loss": best["validation_loss"],  # 模拟中应恢复的最佳候选。
        "reference_loss": reference_loss, "actions": actions,  # 同时保留早停参考值和逐次决定。
    }  # 模型选择的最低分和早停的改善阈值分别管理。

def main():  # 使用已经完成的第四轮报告回放三组早停规则。
    source_path = ROOT / "docs/training_runs/lesson37.json"  # 只读取已有验证记录。
    source_bytes = source_path.read_bytes()  # 保留原报告以检查文件未变。
    source = json.loads(source_bytes)  # 将JSON还原成字典。
    assert source["status"] == "completed", "上一课尚未完成"  # 本课分析已完成的轨迹。
    history = source["history"]  # 沿用原来的验证时间顺序，不按损失排序。
    assert history and history[0]["local_step"] == 0, "缺少起点"  # 第一条记录用于初始化参考值。
    assert all(math.isfinite(row["validation_loss"]) and row["validation_loss"] >= 0 and row["validation_positions"] == 316960 for row in history), "验证记录无效"  # 所有分数来自同一完整验证范围。
    assert all(left["local_step"] < right["local_step"] for left, right in zip(history, history[1:])), "记录不是时间顺序"  # 防止把未来最低分提前放到前面。
    results = []  # 保存三种教学规则的模拟结果。
    for patience, min_delta in ((1, 0.0), (2, 0.0), (2, 0.005)):  # 用不同规则理解等待次数和改善阈值，不据此宣称找到最佳策略。
        result = simulate_early_stopping(history, patience, min_delta)  # 每组都从起点重新模拟。
        results.append(result)  # 保留该组完整判断过程。
        print(f"patience={patience} min_delta={min_delta}", flush=True)  # 显示本组规则。
        for action in result["actions"]:  # 只打印这组实际观察过的记录。
            print(f"  local_step={action['local_step']} loss={action['validation_loss']:.6f} new_best={action['new_best']} significant={action['significant']} bad_checks={action['bad_checks']} stop={action['should_stop']}", flush=True)  # 展示一次次决定，而非只看结论。
        print("触发停止：", result["stop_triggered"], "最后观察位置：", result["last_seen_local_step"], "应加载最佳位置：", result["best_local_step"], flush=True)  # 停止时不一定使用最后观察到的模型。
    assert source_path.read_bytes() == source_bytes, "旧报告被修改"  # 模拟不能覆盖真实训练记录。
    return {"source_report": source_path.name, "train_sha256": source["train_sha256"], "mode": "replay_without_training", "results": results}  # 明确这只是回放，没有实际节省已发生的训练。

if __name__ == "__main__":  # 直接运行时才模拟并保存本课结果。
    report = main()  # 不加载模型、不更新参数、不写权重。
    path = ROOT / "docs/training_runs/lesson38.json"  # 保存独立的教学模拟报告。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存中文可读JSON。
    print("早停模拟报告：", path)  # 输出本课报告路径。
