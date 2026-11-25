import json  # 保存一次接线检查的实际结果。
from time import perf_counter  # 限制本次诊断训练预算。
from sft_batches import ROOT, reviewed_rows, load_tokenizer, make_batch  # 复用经过检查的真实训练批次。
from prepare_chat_corpus import sha, dump  # 使用固定摘要和跨平台JSON字节。
from chat_model import ChatLM, torch, F  # 与预训练相同的30M模型和交叉熵。


def main():  # 只更新一个独立副本，不输出可恢复的正式SFT检查点。
    torch.set_num_threads(2)  # CPU备用路径限制线程数量。
    torch.manual_seed(251)  # 固定模型构造和诊断用随机状态。
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 有GPU就做一次真实GPU更新。
    source = ROOT / "model/lesson50/last_train.pt"  # 使用真正FP32主参数，不从FP16导出舍入重建。
    old = json.loads((ROOT / "docs/training_runs/lesson50.json").read_text(encoding="utf-8"))  # 来源参数有实际训练证据。
    source_hash = sha(source.read_bytes())  # 保留正式文件的原始摘要。
    assert source_hash == old["artifacts"]["last_train.pt"]["sha256"], "来源不是本课预训练状态"  # 不从未知文件试验。
    saved = torch.load(source, map_location="cpu", weights_only=True)  # 读取完整来源但只取其中模型参数。
    assert saved["step"] == 2020, "来源累计步数错误"  # SFT新阶段不改写原预训练计数。
    model = ChatLM(**saved["model_config"]).to(device)  # 新对象和原文件相互独立。
    model.load_state_dict(saved["model_state"])  # 保留已学语言参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.01)  # 新SFT诊断阶段显式重置历史，不冒充原预训练轨迹续跑。
    assert not optimizer.state, "新阶段应从空优化器状态开始"  # 这里是有意改变训练目标和优化器状态。
    rows, audit = reviewed_rows()  # 不使用全部未审核对话。
    tokenizer = load_tokenizer()  # 沿用同一BPE编号。
    xs, ys, encoded = make_batch(rows, tokenizer, saved["model_config"]["context"])  # 两条完整多轮，不拆断答案。
    x, y = torch.tensor(xs, device=device), torch.tensor(ys, device=device)  # 输入和目标均为整数编号。
    denominator = int((y != -100).sum().item())  # 本批真实助手目标249个。
    embeddings = []  # 保存中间激活用于展示上下文仍有梯度。
    def capture(module, inputs, output):  # 捕获查表后的激活，而非共享权重的累计梯度。
        output.retain_grad()  # 非叶子激活默认不保留梯度，这里为教学显式保留。
        embeddings.append(output)  # 只在此次前向中记录。
    hook = model.embedding.register_forward_hook(capture)  # 不改变模型计算结果。
    started = perf_counter()  # 文件读取和模型构造之外，训练计算最多120秒。
    model.train()  # 进入训练模式；当前模型没有dropout。
    optimizer.zero_grad(set_to_none=True)  # 新诊断批次开始前清空梯度。
    logits = model(x)  # 形状为2、197、6400。
    logits.retain_grad()  # 观察被忽略位置的直接分数梯度。
    loss = F.cross_entropy(logits.reshape(-1, 6400), y.reshape(-1), ignore_index=-100, reduction="sum") / denominator  # 按249个有效目标求平均。
    loss.backward()  # 回答损失沿因果注意力传播到真实用户上下文。
    hook.remove()  # 后续评估不再保存激活梯度。
    masked_max = float(logits.grad[y == -100].abs().max().item())  # 被忽略目标位置对logits的直接梯度应为0。
    user_gradient = float(embeddings[0].grad[0, 2].norm().item())  # 第0条BOS、user之后的首个用户正文token仍可接收梯度。
    assert encoded[0][2][1] == 4 and encoded[0][2][2] >= 8 and masked_max == 0 and user_gradient > 0, "损失掩码与上下文传播混淆"  # 不计直接损失不等于输入读不到。
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())  # 更新前统一裁剪完整梯度。
    assert torch.isfinite(torch.tensor(norm)) and perf_counter() - started < 120, "梯度异常或预算耗尽，未执行更新"  # 超过训练预算不再启动step。
    before = model.embedding.weight.detach().clone()  # 留一份参数快照确认实际更新。
    optimizer.step()  # 本实验唯一一次参数更新。
    optimizer_steps = sorted({int(value["step"].item()) for value in optimizer.state.values()})  # 新SFT诊断AdamW步数应为1。
    change = float((model.embedding.weight.detach() - before).abs().max().item())  # 不是只计算梯度而忘记step。
    model.eval()  # 对同一批次做更新后的诊断复算。
    with torch.no_grad():  # 不为复算创建第二份反向图。
        after_logits = model(x)  # 当前参数对应的新预测。
        after = float(F.cross_entropy(after_logits.reshape(-1, 6400), y.reshape(-1), ignore_index=-100).item())  # 与训练相同的有效目标均值。
    assert change > 0 and optimizer_steps == [1] and sha(source.read_bytes()) == source_hash, "参数未更新或来源被修改"  # 副本学习不改写正式2020步状态。
    result = {"format": "sft_wiring51_v1", "stage": "disposable_one_step_probe_not_chat_acceptance", "source": str(source.relative_to(ROOT)).replace("\\", "/"), "source_sha256": source_hash, "pretrain_step_unchanged": 2020, "probe_updates": 1, "max_updates": 1, "max_compute_seconds": 120, "compute_seconds": perf_counter() - started, "device": device, "precision": "float32_no_amp", "optimizer": "new_stage_AdamW", "learning_rate": 1e-5, "optimizer_steps": optimizer_steps, "rows": [row["id"] for row in rows], "audit_sha256": sha((ROOT / "data/chat51/audit.json").read_bytes()), "tokenizer_sha256": saved["tokenizer_sha256"], "batch_shape": list(x.shape), "positions": denominator, "loss_before": float(loss.item()), "loss_after": after, "masked_logit_gradient_max": masked_max, "user_embedding_activation_gradient_norm": user_gradient, "gradient_norm_before_clip": norm, "embedding_max_parameter_change": change, "weights_saved": False}  # 单批下降不能代替验证集或聊天评分。
    path = ROOT / "docs/training_runs/lesson51.json"  # 与预训练报告分开保存。
    path.write_bytes(dump(result))  # 只写诊断数据，不写新模型或覆盖旧模型。
    print(json.dumps(result, ensure_ascii=False, indent=2))  # 所有结果如实展示。


if __name__ == "__main__":  # 导入工具不启动训练。
    main()  # 明确运行才更新这份临时副本。
