import hashlib  # 核对真正使用的数据和权重。
import json  # 读取训练报告。
from pathlib import Path  # 适配Windows与Linux。
import sys  # 导入实际实现，不调用训练入口。
ROOT = Path(__file__).resolve().parents[2]  # 项目根目录。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 查找本课模块。
from prepare_chat_corpus import DIRECTORY, sha  # 复用固定数据目录。
from chat_model import ChatLM, torch, F  # CPU核对RoPE和GQA。
from tokenizers import Tokenizer  # 与真实训练采用同一切分器。
import numpy as np  # 核对编号数组和实际窗口顺序。
from train_chat_pilot import evaluate, batch_from_stream  # 导入不会启动训练。
torch.set_num_threads(2)  # CI只用小子集复核，不重跑大模型训练。
manifest = json.loads((DIRECTORY / "manifest.json").read_text(encoding="utf-8"))  # 本次来源与划分。
token_manifest = json.loads((DIRECTORY / "tokenizer/manifest.json").read_text(encoding="utf-8"))  # 分词器和编号版本。
report = json.loads((ROOT / "docs/training_runs/lesson48.json").read_text(encoding="utf-8"))  # 实际两段训练结果。
for block in manifest["blocks"]:  # 原始来源块也要随仓库保持完整。
    path = DIRECTORY / "raw" / f"{block['kind']}_{block['block']:02d}.bin"  # 与固定范围采集文件对应。
    assert path.stat().st_size == block["bytes"] and sha(path.read_bytes()) == block["sha256"], "来源字节块改变"  # 不只检查清洗后的文件。
assert sha((DIRECTORY / "manifest.json").read_bytes()) == token_manifest["corpus_manifest_sha256"] == report["corpus_manifest_sha256"], "数据版本不匹配"  # LF字节清单不随平台改变。
for name, expected in manifest["files"].items():  # 检查三个集合和映射的字节。
    assert sha((DIRECTORY / name).read_bytes()) == expected, "语料被修改"  # 不会偷偷重划分。
for name, expected in token_manifest["files"].items():  # 编码产物也应匹配分词器。
    assert sha((DIRECTORY / "tokenizer" / name).read_bytes()) == expected, "分词器或编号被修改"  # 防止旧规则套新权重。
group_sets, fingerprints = {}, {}  # 跨阶段检查组互斥和规范化去重。
for split in ("train", "validation", "test"):  # 只检查测试数据归属，不测模型测试成绩。
    group_sets[split], fingerprints[split] = set(), set()  # 同集合可能有多个同组成员。
    for kind in ("pretrain", "sft"):  # 预训练与SFT联合分组。
        rows = [json.loads(line) for line in (DIRECTORY / f"{kind}_{split}.jsonl").read_text(encoding="utf-8").splitlines()]  # 完整文档。
        assert len(rows) == manifest["stats"][f"{kind}_{split}"]["records"], "记录数量错误"  # 文件与统计一致。
        group_sets[split].update(row["group_id"] for row in rows)  # 同组不能跨集合。
        fingerprints[split].update(row["text_sha256"] for row in rows)  # 完整正文重复也不能跨集合。
for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):  # 检查所有成对交集。
    assert not group_sets[left] & group_sets[right] and not fingerprints[left] & fingerprints[right], "数据泄漏"  # 启发式分组不保证发现所有语义改写，已在清单注明。
tokenizer = Tokenizer.from_file(str(DIRECTORY / "tokenizer/tokenizer.json"))  # 真实新词表。
for text in token_manifest["roundtrip_probes"]:  # 罕见字和表情也必须往返。
    ids = tokenizer.encode(text, add_special_tokens=False).ids  # 直接编码原始正文。
    assert 7 not in ids and tokenizer.decode(ids, skip_special_tokens=False) == text, "BPE丢字"  # 不能用未知符掩盖未见字。
small = ChatLM(vocab_size=32, width=32, layers=2, heads=4, kv_heads=2, context=16)  # 低成本核对结构性质。
x = torch.tensor([[1, 8, 9, 10, 11]])  # 固定输入。
changed = x.clone()  # 后文变化不能修改原输入。
changed[:, 3:] = 12  # 只改未来位置。
assert torch.allclose(small(x)[:, :3], small(changed)[:, :3], atol=1e-6), "未来输入泄漏"  # 多层RoPE/GQA仍应保持因果性。
assert not torch.allclose(small(x)[:, 3:], small(changed)[:, 3:]), "输入变化没有影响有效输出"  # 防止常数模型通过因果性检查。
F.cross_entropy(small(x).flatten(0, 1), x.flatten()).backward()  # 检查所有参数梯度可达。
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters()), "梯度缺失或无效"  # 不只检查前向形状。
assert report["status"] == "completed" and report["parameters"] == 30025216, "训练状态或参数量错误"  # 真实模型已创建并训练。
profile, pilot = report["phases"]  # 本次先试跑，再继续首段预训练。
assert profile["updates"] == 20 and 0 < pilot["updates"] <= 1000 and pilot["start_step"] == 20, "两段更新预算错误"  # 不能把微批当step。
assert report["completed_steps"] == profile["updates"] + pilot["updates"], "累计更新错误"  # 优化器跨阶段保留。
assert report["best_loss"] == min(row["loss"] for row in report["history"]), "最佳位置错误"  # 起点也参加选优。
assets = json.loads((ROOT / "docs/training_runs/lesson48_assets.json").read_text(encoding="utf-8"))  # 大训练状态在Release，不强行放入普通Git。
assert assets["draft"] is False and assets["tag"] == "lesson48-pretrain-pilot", "完整训练状态尚未发布"  # 草稿上传不能算作已交付。
for item in assets["assets"]:  # 最佳与最后相同时允许复用附件。
    assert item["sha256"] == report["artifacts"][item["local_name"]]["sha256"], "附件与真实训练状态不对应"  # 指纹与原训练报告一致。
    if "remote_digest" in item:  # 本次GitHub返回了服务器端SHA-256。
        assert item["remote_digest"] == "sha256:" + item["sha256"] and "/untagged-" not in item["url"], "附件指纹或下载地址不正确"  # 不保留草稿临时地址。
resume = json.loads((ROOT / "docs/training_runs/lesson48_resume.json").read_text(encoding="utf-8"))  # GPU实测的恢复副本比较。
assert resume["formal_checkpoint_step"] == report["completed_steps"] and resume["max_parameter_abs_diff"] <= 1e-6 and resume["adamw_and_scaler_equal"], "完整恢复核验不匹配"  # 核验副本更新不计训练预算。
stream = np.load(DIRECTORY / "tokenizer/pretrain_train.npy", mmap_mode="r", allow_pickle=False)  # 按实际流重建训练前缀。
positions, exposure, offset = 0, hashlib.sha256(), 0  # 检查真实目标数与样本顺序。
windows, epoch = report["windows_per_epoch"], 0  # 本次未假定必须完成一整轮。
for _ in range(report["completed_steps"]):  # 每个成功更新有8个微批。
    references = []  # 与训练指纹的每组布局一致。
    for _ in range(8):  # 明确区分微批和优化器更新。
        order = np.random.default_rng(report["data_seed"] + epoch).permutation(windows)  # 同轮排列可重建。
        index = int(order[offset])  # 当前真实窗口。
        positions += batch_from_stream(stream, index)[2]  # 尾窗按有效目标数计。
        references.append((epoch, index))  # 指纹含轮号。
        offset += 1  # 推进数据游标。
        if offset == windows:  # 完整遍历后才进入新一轮。
            epoch, offset = epoch + 1, 0  # 不重复最后一窗。
    exposure.update(np.asarray(references, dtype=np.int64).tobytes())  # 和实际训练记录比较。
assert positions == report["seen_positions"] and exposure.hexdigest() == report["exposure_sha256"], "实际数据预算或顺序错误"  # 不是用step乘固定数掩盖尾段。
path = ROOT / "model/lesson48/pretrain_fp16.pt"  # 仓库中的实际推理权重。
assert path.stat().st_size == report["artifacts"][path.name]["bytes"] < 1_000_000_000 and sha(path.read_bytes()) == report["artifacts"][path.name]["sha256"], "推理权重改变或过大"  # 用文件实数验收1GB目标。
state = torch.load(path, map_location="cpu", weights_only=True)  # 可在无CUDA环境读取。
assert state["stage"] == "pretrain_not_sft" and state["tokenizer_sha256"] == sha((DIRECTORY / "tokenizer/tokenizer.json").read_bytes()), "模型阶段或词表错误"  # 不把预训练权重说成已会聊天。
model = ChatLM(**state["model_config"])  # CPU模型参数采用FP32。
model.load_state_dict(state["model_state"])  # 加载保存的FP16推理矩阵。
assert sum(p.numel() for p in model.parameters()) == 30025216, "实际参数量与预算不同"  # 共享权重不重复计数。
xs = np.load(DIRECTORY / "tokenizer/pretrain_validation_x.npy", mmap_mode="r", allow_pickle=False)  # 冻结验证输入。
ys = np.load(DIRECTORY / "tokenizer/pretrain_validation_y.npy", mmap_mode="r", allow_pickle=False)  # 冻结目标含-100。
metrics = evaluate(model, xs, ys, limit=8)  # CI仅复算报告中明确的八窗子集，不冒充全量验证。
assert metrics["positions"] == report["export_ci_subset"]["positions"] and np.isclose(metrics["loss"], report["export_ci_subset"]["loss"], atol=0.003), "CPU小子集与导出指标不一致"  # FP16 CUDA与FP32 CPU存在小数值差。
print("CPU导出权重八窗检查：", metrics)  # 报告实际复核范围。
print("正式数据、分词器、因果梯度、30M预算与推理导出检查：通过；没有在CI重训。")  # 全量验证由真实GPU实验记录。
