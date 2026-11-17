import hashlib  # 对照实验前后权重文件的完整哈希。
import json  # 读取已保存的实验报告。
import sys  # 设置学习脚本导入路径。
from importlib import import_module  # 导入实验函数而不触发报告保存入口。
from pathlib import Path  # 从检查脚本定位仓库根目录。
import numpy as np  # 在浮点容差内比较复算结果。

ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于 scripts/ci。
sys.path.insert(0, str(ROOT / "scripts/learn"))  # 允许导入数字开头的学习模块。


def main():  # 只读复算两个验证条件，不训练或保存权重。
    report_path = ROOT / "docs/training_runs/lesson34.json"  # 定位学习者提交的正式报告。
    saved_bytes = report_path.read_bytes()  # 保留报告原字节用于检查文件未变。
    expected = json.loads(saved_bytes)  # 读取已记录的实验条件与指标。
    weight_path = ROOT / "model" / expected["checkpoint"]  # 定位本实验使用的最佳模型。
    before_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()  # 留下原权重文件校验值。
    lesson = import_module("18_context_ablation")  # 导入只定义函数，不执行文件保存入口。
    actual = lesson.main()  # 重新检查独立副本、首位置相同、完整验证与原参数不变。
    assert actual.keys() == expected.keys(), "报告字段不一致"  # 避免漏掉实验条件或指标。
    for key in expected:  # 分别核对来源、配置、覆盖数量与浮点结果。
        if isinstance(expected[key], float):  # 损失和分数差使用适当容差。
            assert np.isclose(actual[key], expected[key], rtol=1e-6, atol=1e-6), key + "复算不一致"  # 允许跨平台计算产生微小浮点差。
        else:  # 文件名、整数、配置与哈希要求精确一致。
            assert actual[key] == expected[key], key + "记录不一致"  # 核对实验确实使用相同条件。
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == before_hash, "原权重文件被改变"  # 副本干预不能写回最佳文件。
    assert report_path.read_bytes() == saved_bytes, "正式报告被改变"  # 自动检查不覆盖学习者实验记录。
    print("lesson34 ablation metrics, independent models and source files passed", flush=True)  # 全部复算与文件保留检查通过。


if __name__ == "__main__":  # 直接运行时才执行只读检查。
    main()  # 不执行训练，也不生成新的模型或报告。
