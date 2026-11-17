import hashlib  # 检查评估前后模型文件保持不变。
import json  # 读取已提交的评估报告。
import sys  # 设置学习模块的导入路径。
from importlib import import_module  # 复用评估入口而不执行保存报告的代码。
from pathlib import Path  # 定位仓库文件。
import numpy as np  # 在容差内比较复算的浮点指标。

ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于 scripts/ci。
sys.path.insert(0, str(ROOT / "scripts/learn"))  # 允许导入数字开头的学习模块。


def main():  # 只读复算正式报告，不训练或保存文件。
    path = ROOT / "docs/training_runs/lesson35.json"  # 定位本课评估报告。
    saved_bytes = path.read_bytes()  # 保留原报告字节以核对文件未变。
    expected = json.loads(saved_bytes)  # 读取学习者实际记录的指标。
    weight_path = ROOT / "model" / expected["checkpoint"]  # 找到本课使用的最佳权重文件。
    before_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()  # 对权重文件计算完整校验值。
    lesson = import_module("19_evaluation_metrics")  # 导入函数，不进入保存报告的脚本入口。
    actual = lesson.main()  # 重建训练字频并复算同一验证范围的指标。
    assert actual.keys() == expected.keys(), "报告字段不一致"  # 防止漏掉来源或统计指标。
    for key in expected.keys() - {"rnn", "frequency"}:  # 来源、配置及验证规模必须精确一致。
        assert actual[key] == expected[key], key + "记录不一致"  # 避免错误模型或评估范围进入比较。
    for name in ("rnn", "frequency"):  # 分别检查两组评估结果。
        saved, measured = expected[name], actual[name]  # 获取已提交值与重新计算值。
        assert measured.keys() == saved.keys(), "指标字段不一致"  # 损失、困惑度、准确率及命中数都需保留。
        assert measured["correct"] == saved["correct"], name + "命中数不一致"  # 命中数应对应保存的同一检查点。
        for metric in ("loss", "perplexity", "accuracy"):  # 其余指标按浮点数比较。
            assert np.isclose(measured[metric], saved[metric], rtol=1e-6, atol=1e-6), name + metric + "复算不一致"  # 允许浮点计算的微小平台差异。
        assert np.isclose(saved["perplexity"], np.exp(saved["loss"])) and np.isclose(saved["accuracy"], saved["correct"] / expected["positions"]), "报告内指标不自洽"  # 核对报告中的公式关系。
    assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == before_hash, "权重文件被修改"  # 整个评估必须保留原权重。
    assert path.read_bytes() == saved_bytes, "评估报告被覆盖"  # 自动检查不能覆盖学习者报告。
    print("lesson35 full validation loss, perplexity, accuracy and source files passed", flush=True)  # 只读复算全部通过。


if __name__ == "__main__":  # 直接运行检查脚本时才开始复核。
    main()  # 不调用训练或报告保存入口。
