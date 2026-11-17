from copy import deepcopy  # 检查回放没有修改来源记录。
from importlib import import_module  # 导入模拟函数，不执行保存报告入口。
from pathlib import Path  # 定位仓库及报告文件。
import json  # 读取学习者提交的模拟结果。
import sys  # 设置学习模块导入路径。

ROOT = Path(__file__).resolve().parents[2]  # 当前文件放在scripts/ci。
sys.path.insert(0, str(ROOT / "scripts/learn"))  # 允许导入数字开头的学习模块。


def main():  # 只读复算早停回放，不训练、不覆盖报告。
    path = ROOT / "docs/training_runs/lesson38.json"  # 定位本课正式模拟报告。
    saved_bytes = path.read_bytes()  # 保留原字节检查文件未变。
    expected = json.loads(saved_bytes)  # 读取已提交结果。
    source_path = ROOT / "docs/training_runs/lesson37.json"  # 来源必须是上一课真实训练轨迹。
    source_bytes = source_path.read_bytes()  # 保留真实训练记录原字节。
    history = json.loads(source_bytes)["history"]  # 读取按时间排列的验证检查点。
    original_history = deepcopy(history)  # 独立复制来源供参数检查。
    lesson = import_module("22_early_stopping_demo")  # 只定义函数，不保存新的报告。
    actual = lesson.main()  # 重新回放三组规则。
    assert actual == expected, "早停回放与正式报告不一致"  # 只做相同输入上的基础算术，完整结构应一致。
    assert expected["mode"] == "replay_without_training" and expected["source_report"] == "lesson37.json", "来源或实验模式错误"  # 不能把回放误记为真实停止训练。
    first, second, third = actual["results"]  # 分别核对三组教学规则。
    assert first["stop_triggered"] and first["last_seen_local_step"] == 1000 and first["best_local_step"] == 200, "停止位置与最佳位置混淆"  # 提前停止后应选已观察到的最低损失文件。
    assert not second["stop_triggered"] and second["best_local_step"] == 4787, "等待改善的计数错误"  # 第二组能等待后续再次改善。
    assert not third["stop_triggered"] and third["actions"][-1]["new_best"] and not third["actions"][-1]["significant"] and third["actions"][-1]["bad_checks"] == 1, "最低损失和改善阈值混淆"  # 小改善可刷新最佳但不清零等待。
    poisoned_future = history[:3] + [None]  # 用不可用记录代替触发停止之后的未来，验证算法不会偷看。
    prefix_result = lesson.simulate_early_stopping(poisoned_future, 1, 0.0)  # 应在触及不可用未来前停止。
    assert prefix_result == first, "提前停止后的结果依赖未来记录"  # 停止决定和最佳选择只能依赖已观察前缀。
    assert history == original_history, "回放修改原始记录"  # 原来的检查点顺序和内容均保留。
    assert source_path.read_bytes() == source_bytes and path.read_bytes() == saved_bytes, "报告文件被覆盖"  # 自动检查保持两份报告只读。
    print("lesson38 patience, improvement threshold and no future access passed", flush=True)  # 全部回放及前缀隔离检查通过。


if __name__ == "__main__":  # 直接运行检查脚本时才开始复核。
    main()  # 不加载模型、不训练、不保存报告。
