import json  # 为过拟合判断补充完整训练集指标。
from train_chat_sft import ChatLM, torch, load_data, evaluate, ROOT, OUT  # 复用固定模板与无梯度评估。
from prepare_chat_corpus import sha, dump  # 核对源状态没有被改写。


def main():  # 只评估最佳和最后状态，不执行参数更新。
    torch.set_num_threads(4)  # 限制CPU线程。
    rows, tokenizer, manifest = load_data()  # 原数据范围不改变。
    report = json.loads((ROOT / "docs/training_runs/lesson53.json").read_text(encoding="utf-8"))  # 正式训练已结束。
    results = {}  # 同一完整训练集对照两个检查点。
    for name in ("best_train.pt", "last_train.pt"):  # 不能只用最后一批训练损失代表全训练集。
        path = OUT / name  # 真实完整FP32参数来源。
        assert sha(path.read_bytes()) == report["artifacts"][name]["sha256"], "完整状态改变"  # 不修改正式文件。
        saved = torch.load(path, map_location="cpu", weights_only=True)  # 仅取模型参数，不做优化器step。
        model = ChatLM(**saved["model_config"]).cuda()  # 与正式验证相同设备路径。
        model.load_state_dict(saved["model_state"])  # 加载相应检查点矩阵。
        results[name] = {"sft_step": saved["step"], "full_train": evaluate(model, rows["train"], tokenizer)}  # 全117会话2496助手目标。
        print(name, results[name], flush=True)  # 如实展示全部训练集结果。
        del model, saved  # 释放当前检查点副本。
        torch.cuda.empty_cache()  # 不让缓存累积占用显存。
    path = ROOT / "docs/training_runs/lesson53_analysis.json"  # 在独立诊断报告追加指标。
    analysis = json.loads(path.read_text(encoding="utf-8"))  # 保留已有固定错误分析。
    analysis["full_training_set_assessment"] = results  # 不覆盖固定验证指标或正式计数。
    path.write_bytes(dump(analysis))  # 没有生成任何新权重。


if __name__ == "__main__":  # 导入不会评估或加载GPU模型。
    main()  # 显式运行仅做完整训练集前向复算。
