import hashlib  # 检查续训数据与权重内容。
import json  # 读取旧起点和新的独立报告。
from pathlib import Path  # 定位仓库。
import sys  # 导入实际模型实现。
ROOT = Path(__file__).resolve().parents[2]  # 项目根目录。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 不执行训练入口。
from chat_model import ChatLM, torch  # CPU重载推理权重。
from prepare_chat_corpus import DIRECTORY, sha  # 原冻结语料仍在chat48。
from train_chat_pilot import evaluate, batch_from_stream  # 复用相同评估口径与打包规则。
import numpy as np  # 重建数据游标和有效目标数。
torch.set_num_threads(2)  # CI不启动长训练或占用显卡。
old = json.loads((ROOT / "docs/training_runs/lesson48.json").read_text(encoding="utf-8"))  # 本课约定的起点证据。
report = json.loads((ROOT / "docs/training_runs/lesson50.json").read_text(encoding="utf-8"))  # 正式续训结果。
assert report["status"] == "completed" and report["initial_step"] == old["completed_steps"] == 1020, "续训起点错误"  # 不从随机参数重新开始。
assert report["source_sha256"] == old["artifacts"]["last_train.pt"]["sha256"], "来源完整状态错误"  # FP16推理文件不能冒充完整来源。
assert report["model_config"] == old["model_config"] and report["tokenizer_sha256"] == old["tokenizer_sha256"] and report["corpus_manifest_sha256"] == old["corpus_manifest_sha256"], "续训改变了结构或数据"  # 本课只延长训练。
assert report["learning_rates"] == [0.0003] and report["gradient_accumulation"] == 8 and report["initial_scaler_scale"] == 1024.0, "优化器或Scaler设置未延续"  # 起点缩放和学习率也记录。
assert report["initial_cursor"] == old["cursor"] == {"epoch": 0, "offset": 8160}, "数据游标不是下一组"  # 不重复第1020次更新的数据。
assert 0 < report["additional_steps"] <= report["config"]["max_updates"] == 1000 and report["config"]["max_seconds"] == 1800, "续训预算错误"  # 实际停止位置允许时间截止。
assert report["completed_steps"] == 1020 + report["additional_steps"] and report["initial_seen_positions"] == old["seen_positions"], "累计计数被重置"  # 与本阶段新增计数分开。
assert np.isclose(report["history"][0]["loss"], old["history"][-1]["loss"], atol=1e-5), "恢复起点指标不一致"  # 先验证再更新。
assert all(row["positions"] == 388180 for row in report["history"]), "验证范围改变"  # 同一冻结验证集比较。
assert report["best_loss"] == min(row["loss"] for row in report["history"]), "最佳未包含旧起点"  # 更久训练不强制成为最佳。
for row in report["history"]:  # 全部固定生成样例都应保留。
    assert len(row["samples"]) == 6 and [item["prompt"] for item in row["samples"]] == [p for p in ("春风", "解释什么是梯度。", "你好，请介绍一下自己。") for _ in range(2)], "固定开头或样例数改变"  # 不能只展示最好看的回答。
stream = np.load(DIRECTORY / "tokenizer/pretrain_train.npy", mmap_mode="r", allow_pickle=False)  # 未更换训练编号流。
epoch, offset = report["initial_cursor"]["epoch"], report["initial_cursor"]["offset"]  # 从旧状态的下一组继续。
windows, positions, exposure = report["windows_per_epoch"], 0, hashlib.sha256()  # 本课只记录新增曝光。
for _ in range(report["additional_steps"]):  # 每个真实更新8微批。
    references = []  # 指纹按每组写入顺序重建。
    for _ in range(8):  # 窗口编号必须与游标一致。
        order = np.random.default_rng(report["data_seed"] + epoch).permutation(windows)  # 新轮使用新轮号派生的种子。
        index = int(order[offset])  # 真实下一个窗口，而非上一批。
        references.append((epoch, index))  # 保存轮号以区分跨轮。
        positions += batch_from_stream(stream, index)[2]  # 只数真实目标，尾窗不足256。
        offset += 1  # 当前微批已经被使用。
        if offset == windows:  # 整轮用完再推进。
            epoch, offset = epoch + 1, 0  # 不清空模型或优化器。
    exposure.update(np.asarray(references, dtype=np.int64).tobytes())  # 与实际训练日志比较。
assert positions == report["additional_positions"] and report["seen_positions"] == report["initial_seen_positions"] + positions, "新增目标数错误"  # 不重复计旧目标。
assert report["cursor"] == {"epoch": epoch, "offset": offset} and report["exposure_sha256"] == exposure.hexdigest(), "跨轮次序或游标错误"  # 本课会跨越第一轮尾部。
path = ROOT / "model/lesson50/pretrain_fp16.pt"  # 新权重独立保存。
artifact = report["artifacts"][path.name]  # 实际文件证据。
assert path.stat().st_size == artifact["bytes"] < 1_000_000_000 and sha(path.read_bytes()) == artifact["sha256"], "新推理文件改变或过大"  # 1GB要求针对实际导出文件。
assert sha((ROOT / "model/lesson48/pretrain_fp16.pt").read_bytes()) == old["artifacts"][path.name]["sha256"], "旧模型被覆盖"  # 旧最佳独立保留。
state = torch.load(path, map_location="cpu", weights_only=True)  # 无CUDA也可检查。
assert state["step"] == report["best_step"] and state["stage"] == "pretrain_not_sft", "导出位置或阶段错误"  # 续训不自动获得SFT聊天能力。
model = ChatLM(**state["model_config"])  # CPU侧FP32参数存储。
model.load_state_dict(state["model_state"])  # 加载FP16导出的矩阵。
xs = np.load(DIRECTORY / "tokenizer/pretrain_validation_x.npy", mmap_mode="r", allow_pickle=False)  # 同一验证输入。
ys = np.load(DIRECTORY / "tokenizer/pretrain_validation_y.npy", mmap_mode="r", allow_pickle=False)  # 同一目标与尾段规则。
metrics = evaluate(model, xs, ys, limit=8)  # CI明确只复算八窗，不冒充完整GPU验证。
assert metrics["positions"] == report["export_ci_subset"]["positions"] and np.isclose(metrics["loss"], report["export_ci_subset"]["loss"], atol=0.003), "新导出指标复核失败"  # 允许FP16与FP32平台误差。
resume = json.loads((ROOT / "docs/training_runs/lesson50_resume.json").read_text(encoding="utf-8"))  # 正式文件不参与副本更新。
assert resume["formal_checkpoint_step"] == report["completed_steps"] and resume["max_parameter_abs_diff"] <= 1e-6 and resume["adamw_and_scaler_equal"], "新完整状态恢复失败"  # m、v、t与Scaler均验证。
release = json.loads((ROOT / "docs/training_runs/lesson50_assets.json").read_text(encoding="utf-8"))  # 已发布附件的远端回执。
assert release["draft"] is False and release["tag"] == "lesson50-pretrain-pilot", "完整状态尚未发布"  # 克隆Git之外还应有可下载断点。
for asset in release["assets"]:  # 最佳与最后相同时允许内容别名。
    assert asset["sha256"] == report["artifacts"][asset["local_name"]]["sha256"] and asset["bytes"] == report["artifacts"][asset["local_name"]]["bytes"], "附件与报告不一致"  # 核对逻辑文件证据。
    if "same_as" not in asset:  # 实际上传的文件有服务器摘要。
        assert asset["remote_digest"] == "sha256:" + asset["sha256"] and "/download/lesson50-pretrain-pilot/" in asset["url"], "远端摘要或永久链接错误"  # 不保留草稿临时链接。
print("续训CPU八窗复核：", metrics)  # 显示实际范围。
print("来源、1020步起点、跨轮游标、累计预算、独立导出及恢复核验：通过")  # 不在CI重新训练。
