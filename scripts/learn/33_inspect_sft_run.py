import json  # 读取正式SFT数据与实际训练报告。
import sys  # 导入同一编码及SHA工具。
from pathlib import Path  # 定位项目文件。
ROOT = Path(__file__).resolve().parents[2]  # 不依赖终端当前目录。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 不调用训练入口。
from train_chat_sft import load_data  # 核对数据文件、原组隔离与BPE。
from prepare_chat_corpus import sha  # 核对已交付推理文件。


def main():  # 只读实操，不重新训练或生成随机样例。
    rows, tokenizer, manifest = load_data()  # 数据版本和词表不能改变。
    path = ROOT / "docs/training_runs/lesson53.json"  # 正式训练必须先完成。
    report = json.loads(path.read_text(encoding="utf-8"))  # 所有检查点和原始输出均保留。
    assert report["status"] == "completed" and report["sft_manifest_sha256"] == sha((ROOT / "data/chat53/frozen_v1/manifest.json").read_bytes()), "训练未完成或数据版本不一致"  # 不把副本单步记录当正式SFT。
    assert 0 < report["completed_steps"] <= report["config"]["max_updates"] <= 500 and report["config"]["max_seconds"] <= 900, "正式SFT超预算"  # 只读检查不增加累计步数。
    assert report["best_loss"] == min(item["loss"] for item in report["history"]), "最佳检查点选择错误"  # 包括未经SFT的起点。
    model = ROOT / "model/lesson53/chat_fp16.pt"  # 真实最佳推理文件。
    assert model.stat().st_size == report["artifacts"][model.name]["bytes"] < 1_000_000_000 and sha(model.read_bytes()) == report["artifacts"][model.name]["sha256"], "推理文件内容或大小错误"  # 不只根据参数数估算1GB限制。
    for split in ("train", "validation"):  # 显示独立组及多轮变体数量。
        print(split, "会话：", len(rows[split]), "独立组：", manifest["stats"][split]["groups"], "助手目标：", manifest["stats"][split]["assistant_targets"])  # 不把变体当独立新知识。
    print("预训练来源更新：", report["pretrain_step"], "本阶段SFT更新：", report["completed_steps"])  # 分阶段计数，不重写预训练历史。
    for item in report["history"]:  # 所有固定验证候选都展示。
        print("SFT step：", item["step"], "验证损失：", round(item["loss"], 6), "助手目标：", item["positions"])  # 目标范围应完全一致。
    print("最佳SFT步数：", report["best_step"], "最佳验证损失：", report["best_loss"])  # 最佳不一定是最后。
    print("推理文件字节数：", model.stat().st_size)  # 文件实际不超过1GB。
    print("正式中文SFT数据、预算、固定验证与最佳权重检查：通过；本脚本没有训练模型。")  # 成功不代表所有聊天回答正确。


if __name__ == "__main__":  # 导入不加载权重或启动训练。
    main()  # 学习者运行只读检查入口。
