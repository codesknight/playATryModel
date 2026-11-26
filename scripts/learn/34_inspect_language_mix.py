import json  # 查看混合语言训练与可验证课程。
import sys  # 导入真实编码及验证函数。
from pathlib import Path  # 定位仓库。
ROOT = Path(__file__).resolve().parents[2]  # 不依赖终端工作目录。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 不调用训练main。
from train_language54 import load_data, DOMAINS  # 核对冻结语言数组及审阅版本。
from prepare_chat54 import DIRECTORY, verify_task  # 全量检查程序化任务标签。
from prepare_chat_corpus import sha  # 核对数据及推理权重。


def main():  # 学习者只读验收，不追加模型更新。
    arrays, tokenizer, language, tasks = load_data()  # 词表与旧通用验证不变。
    curriculum = json.loads((DIRECTORY / "curriculum/manifest.json").read_text(encoding="utf-8"))  # 记录800条的审核方法。
    for split in ("train", "validation"):  # 标签是程序校验而非800次人工标注。
        path = DIRECTORY / "curriculum" / (split + ".jsonl")  # 已冻结任务划分。
        assert sha(path.read_bytes()) == curriculum["files"][path.name], "课程任务版本改变"  # 不静默换题。
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()):  # 每条任务都核验。
            verify_task(row)  # 检查算术、单调性、元素守恒等不变量。
        print("任务", split, "条数/组数：", curriculum["stats"][split]["rows"], curriculum["stats"][split]["groups"], "类别数：", len(curriculum["stats"][split]["families"]))  # 不把已知模板的新输入当通用聊天测评。
    report = json.loads((ROOT / "docs/training_runs/lesson54.json").read_text(encoding="utf-8"))  # 正式新增训练记录。
    assert report["status"] == "completed" and report["language_manifest_sha256"] == sha((DIRECTORY / "language_v3/manifest.json").read_bytes()), "阶段未完成或语料改变"  # 不读取临时半成品。
    assert 0 < report["additional_updates"] <= report["config"]["max_updates"] <= 1000 and report["config"]["max_seconds"] <= 900 and report["completed_steps"] == 2020 + report["additional_updates"], "预算或累计计数错误"  # 不把SFT500步混入预训练轨迹。
    for domain in DOMAINS:  # 新增语料与旧语料分别显示。
        print(domain, "训练文档：", language["stats"][domain]["train_documents"], "训练token：", language["stats"][domain]["train_tokens"])  # 真实来源规模。
    print("混合配比：", "旧通用4窗、故事3窗、四书1窗；每次更新8窗")  # 独立域游标避免短数据被长文淹没。
    print("新增/累计预训练更新：", report["additional_updates"], report["completed_steps"])  # 不与SFT500步混为一条轨迹。
    for item in report["history"]:  # 所有固定验证候选都显示。
        print("累计步：", item["step"], "加权分数：", round(item["weighted_score"], 6), "分域损失：", {domain: round(item["domains"][domain]["loss"], 6) for domain in DOMAINS})  # 不只展示改进最大的域。
    file = ROOT / "model/lesson54/base_fp16.pt"  # 新最佳仍为语言模型，不是新SFT聊天模型。
    assert sha(file.read_bytes()) == report["artifacts"][file.name]["sha256"] and file.stat().st_size < 1_000_000_000, "新推理文件错误"  # 实际大小与摘要均检查。
    print("最佳累计步：", report["best_step"], "推理文件字节数：", file.stat().st_size)  # 最佳与最后可不同。
    print("800条新任务已进行SFT：", not report["task_data_prepared_not_sft_trained"])  # 准备数据与训练能力分开验收。
    print("任务全量标签、数据版本、分域验证与预算检查：通过；本脚本没有训练模型。")  # 不把准备课程数据说成已完成新SFT。


if __name__ == "__main__":  # 导入不执行检查。
    main()  # 显式运行仅核对现有文件。
