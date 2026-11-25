import json  # 核对正式SFT报告及审核文件。
import sys  # 导入真实评估和聊天函数。
from pathlib import Path  # 定位仓库。
import numpy as np  # 重建跨轮样本引用。
ROOT = Path(__file__).resolve().parents[2]  # 项目根目录。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 不调用main训练入口。
from train_chat_sft import ChatLM, torch, load_data, evaluate  # CPU只复算明确验证子集。
from prepare_chat_corpus import sha  # 核对交付数据与权重摘要。
torch.set_num_threads(2)  # CI不启动CUDA长训练。
rows, tokenizer, manifest = load_data()  # 文件摘要及组隔离均应通过。
directory = ROOT / "data/chat53"  # 本课新数据版本。
audit = json.loads((directory / "reviews.json").read_text(encoding="utf-8"))  # 逐条决定不由模型损失产生。
assert sha((directory / "reviews.json").read_bytes()) == manifest["reviews_sha256"] and sha((directory / "queue_v2.json").read_bytes()) == audit["queue_sha256"] == manifest["queue_sha256"], "审核版本错误"  # 已批准内容不能静默变更。
assert sha((directory / "raw/alpaca_gpt4_data_zh.json").read_bytes()) == manifest["source_sha256"] and sha((directory / "raw/README.md").read_bytes()) == manifest["source_card_sha256"], "原始来源快照错误"  # 精确固定提交与使用声明。
assert manifest["stats"]["train"]["groups"] == 91 and manifest["stats"]["validation"]["groups"] == 23, "独立组数量改变"  # 复制追问不增加独立组数。
report = json.loads((ROOT / "docs/training_runs/lesson53.json").read_text(encoding="utf-8"))  # 实际GPU训练结果。
assert report["status"] == "completed" and 0 < report["completed_steps"] <= report["config"]["max_updates"] <= 500 and report["config"]["max_seconds"] <= 900, "正式SFT预算错误"  # 不把副本核验更新计入正式阶段。
old = json.loads((ROOT / "docs/training_runs/lesson50.json").read_text(encoding="utf-8"))  # 旧预训练不覆盖。
assert report["source_sha256"] == old["artifacts"]["last_train.pt"]["sha256"] and report["pretrain_step"] == 2020, "来源主参数错误"  # 新阶段不继承旧优化器轨迹。
assert all(item["positions"] == manifest["stats"]["validation"]["assistant_targets"] for item in report["history"]) and report["best_loss"] == min(item["loss"] for item in report["history"]), "验证范围或最佳选择错误"  # 不挑更容易的题来制造进步。
cursor, positions, references = {"epoch": 0, "offset": 0}, 0, []  # 从本阶段零游标核对成功更新。
from sft_batches import encode_dialogue  # 目标数量按实际BPE及回答标记计算。
sizes = [sum(value != -100 for value in encode_dialogue(row["messages"], tokenizer)[1]) for row in rows["train"]]  # 不以固定序列长度乘更新数。
for _ in range(report["completed_steps"] * 8):  # 每次成功更新八条会话。
    order = np.random.default_rng(553 + cursor["epoch"]).permutation(len(rows["train"]))  # 与真实轮次顺序一致。
    index = int(order[cursor["offset"]])  # 当前尚未处理的会话。
    positions += sizes[index]  # 有效助手目标数随会话变化。
    references.append((cursor["epoch"], index))  # 指纹保留轮号。
    cursor["offset"] += 1  # 当前会话完成。
    if cursor["offset"] == len(rows["train"]):  # 轮末进入新排列。
        cursor = {"epoch": cursor["epoch"] + 1, "offset": 0}  # 不重置模型参数。
assert cursor == report["cursor"] and positions == report["seen_positions"] and sha(np.asarray(references, dtype=np.int64).tobytes()) == report["exposure_sha256"], "SFT曝光顺序或累计目标数错误"  # 不将跳过溢出计成成功更新。
file = ROOT / "model/lesson53/chat_fp16.pt"  # 真正发布的推理文件。
assert file.stat().st_size == report["artifacts"][file.name]["bytes"] < 1_000_000_000 and sha(file.read_bytes()) == report["artifacts"][file.name]["sha256"], "推理文件未通过摘要或大小检查"  # 不依赖名义精度估计。
state = torch.load(file, map_location="cpu", weights_only=True)  # CPU读取不启动训练。
assert state["format"] == "chat_sft53_inference_v1" and state["step"] == report["best_step"], "导出的不是SFT最佳"  # 不把最后文件冒充最佳。
model = ChatLM(**state["model_config"])  # 模型结构与预训练相同。
model.load_state_dict(state["model_state"])  # FP16文件载入FP32主存储。
metrics = evaluate(model, rows["validation"], tokenizer, limit=8)  # 只复算前八会话，不冒充完整验证。
assert metrics["positions"] == report["export_ci_subset"]["positions"] and abs(metrics["loss"] - report["export_ci_subset"]["loss"]) < 0.003, "导出权重CPU复算失败"  # 容许GPU与CPU舍入差。
assert len(report["export_samples"]) == 4 and sum(len(case) for case in report["export_samples"]) == 5, "固定开发回答不完整"  # 多轮不能只展示成功的第一轮。
resume = json.loads((ROOT / "docs/training_runs/lesson53_resume.json").read_text(encoding="utf-8"))  # 完整SFT状态另做实际恢复核验。
assert resume["formal_checkpoint_step"] == report["completed_steps"] and resume["max_parameter_abs_diff"] <= 1e-6 and resume["adamw_scaler_cursor_equal"] and resume["formal_checkpoint_sha256_unchanged"] == report["artifacts"]["last_train.pt"]["sha256"], "SFT完整恢复失败"  # 核对m、v、t、Scaler及游标。
print("SFT CPU八会话复算：", metrics)  # 显示明确范围。
print("来源、审核、组隔离、预算计数、曝光指纹和最佳SFT导出：通过")  # 不从token损失宣称可用聊天。
