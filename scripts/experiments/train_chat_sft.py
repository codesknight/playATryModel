import argparse  # 固定正式SFT的更新数和时间预算。
import json  # 保存完整实验记录。
from time import perf_counter  # 在每个累积组前检查时间。
import numpy as np  # 用轮号派生可重建的样本顺序。
from chat_model import ChatLM, torch, F, nn  # 复用30M Transformer。
from sft_batches import ROOT, make_batch, load_tokenizer  # 使用正式BPE、回答标签与右填充。
from prepare_chat_corpus import dump, sha  # 清单和文件字节固定。
from prepare_chat_sft_data import DIRECTORY, SYSTEM  # 训练与聊天使用同一系统提示。
from bounded_recurrent import cpu_tree  # 保存FP32参数和完整AdamW状态到CPU。
OUT = ROOT / "model/lesson53"  # 不覆盖第五十课预训练最佳。
DEV = [["你好，请简单介绍你能做什么。"], ["用一句话解释什么是梯度。"], ["把今天很开心改写得更正式一些。"], ["请记住：我叫小雨。", "我叫什么名字？"]]  # 固定开发问题，不称作最终测试。


def load_data():  # 训练前核对已冻结的正式SFT范围。
    path = DIRECTORY / "frozen_v1/manifest.json"  # 不重新审核或划分数据。
    manifest = json.loads(path.read_text(encoding="utf-8"))  # 所有候选和审核摘要都已保存。
    assert manifest["formal_sft_ready"] and manifest["stats"]["train"]["groups"] >= 64 and manifest["stats"]["validation"]["groups"] >= 16, "SFT数据未满足启动条件"  # 独立组不能用复制行凑数。
    rows = {}  # 训练与验证分开读取，不打开旧测试正文。
    for split in ("train", "validation"):  # 仅使用本课冻结的两个集合。
        file = path.parent / (split + ".jsonl")  # 原始及追问变体均已同组划分。
        assert sha(file.read_bytes()) == manifest["files"][file.name], "冻结SFT文件改变"  # 不在训练途中换验证尺子。
        rows[split] = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]  # 完整会话一行一条。
    assert not {r["group_id"] for r in rows["train"]} & {r["group_id"] for r in rows["validation"]}, "SFT组交集非空"  # 变体不能跨集合。
    tokenizer = load_tokenizer()  # 与预训练模型同词表。
    assert manifest["tokenizer_sha256"] == sha((ROOT / "data/chat48/tokenizer/tokenizer.json").read_bytes()), "词表版本错误"  # 不重训BPE。
    return rows, tokenizer, manifest  # 供训练、独立聊天和CI共用。


def tensors(rows, tokenizer, device):  # 把一个微批放到指定设备。
    x, y, _ = make_batch(rows, tokenizer)  # 只监督助手正文及结束符。
    return torch.tensor(x, device=device), torch.tensor(y, device=device)  # 输入和目标均为int64。


@torch.no_grad()  # 验证不会建立反向图或更新参数。
def evaluate(model, rows, tokenizer, limit=None):  # 用有效助手目标数加权，而非平均各微批均值。
    model.eval()  # 当前结构没有dropout，仍明确切换模式。
    device = next(model.parameters()).device  # 支持GPU训练与CPU验收。
    selected, summed, positions, correct = rows[:limit] if limit else rows, 0.0, 0, 0  # CI可明确只复算固定子集。
    for start in range(0, len(selected), 4):  # 小微批降低验证内存。
        x, y = tensors(selected[start:start + 4], tokenizer, device)  # 与训练相同模板和掩码。
        logits = model(x).float()  # 验证使用FP32前向与交叉熵，减少跨平台误差。
        summed += float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum").item())  # 先累计逐目标损失。
        positions += int((y != -100).sum().item())  # 不计PAD及用户目标。
        correct += int(((logits.argmax(-1) == y) & (y != -100)).sum().item())  # token准确率不是聊天评分。
    return {"loss": summed / positions, "positions": positions, "correct": correct, "accuracy": correct / positions}  # 所有验证候选均用相同口径。


@torch.no_grad()  # 生成不做梯度计算。
def reply(model, tokenizer, messages, max_new=48, sample=False, seed=53):  # 保留完整短历史，遇到结束符停止。
    model.eval()  # 使用当前参数预测下一token。
    ids = [1]  # BOS编号固定。
    for message in messages:  # 系统、用户及已有助手消息按顺序编码。
        role = {"system": 3, "user": 4, "assistant": 5}[message["role"]]  # 不把角色文本拆成普通字。
        ids.extend([role, *tokenizer.encode(message["content"], add_special_tokens=False).ids, 6])  # 历史消息保留结束边界。
    ids.append(5)  # 等待生成的助手角色头没有预先填入答案。
    assert len(ids) + max_new <= model.spec["context"], "聊天历史超长，请清空或缩短输入"  # 不静默丢掉用户要求记住的信息。
    device = next(model.parameters()).device  # 模型可在CPU独立运行。
    generator = torch.Generator(device=device).manual_seed(seed)  # 固定采样种子供对照。
    generated, stopped = [], "max_new_tokens"  # 不能把长度截断伪装成正常结束。
    for _ in range(max_new):  # 生成预算固定。
        x = torch.tensor([ids], device=device)  # 每次将刚生成的token接回真实上下文。
        scores = model(x)[0, -1].float()  # 最后位置预测下一个token。
        scores[[0, 1, 3, 4, 5, 7]] = float("-inf")  # 回复正文不生成PAD、BOS、角色头和UNK，允许END/EOS。
        if sample:  # 固定top-k20、温度0.7作为可选解码。
            values, indices = scores.topk(20)  # 只在二十个候选内采样。
            next_id = int(indices[torch.multinomial(torch.softmax(values / 0.7, -1), 1, generator=generator)].item())  # 随机种子不改变训练权重。
        else:  # 默认贪心便于比较训练前后。
            next_id = int(scores.argmax().item())  # 不用真实答案替代模型生成。
        if next_id in (6, 2):  # 助手结束符或EOS正常停止。
            stopped = "end" if next_id == 6 else "eos"  # 显示真正停止原因。
            break  # 不继续生成下一轮用户消息。
        generated.append(next_id)  # 只返回助手正文。
        ids.append(next_id)  # 下一步读入自己的预测。
    return {"text": tokenizer.decode(generated, skip_special_tokens=False), "tokens": len(generated), "stop_reason": stopped}  # 包括空回答及失败，不筛选漂亮样例。


def development(model, tokenizer):  # 完整保留四组固定开发对话。
    output = []  # 首轮和追问都用模型真实答案。
    for prompts in DEV:  # 开发集不参与梯度更新。
        messages, turns = [{"role": "system", "content": SYSTEM}], []  # 每个案例从独立对话开始。
        for prompt in prompts:  # 多轮案例会复用本模型之前的实际输出。
            messages.append({"role": "user", "content": prompt})  # 追加新的用户问题。
            answer = reply(model, tokenizer, messages)  # 不喂入标准助手答案。
            turns.append({"prompt": prompt, **answer})  # 不删除失败、截断或错误结果。
            messages.append({"role": "assistant", "content": answer["text"]})  # 让下一轮读取真实生成历史。
        output.append(turns)  # 保留所有五个回答。
    return output  # 仅用于观察，不人工挑句子选检查点。


def main():  # 从预训练主参数开始正式SFT新阶段。
    parser = argparse.ArgumentParser()  # 本课预算上限不自动扩大。
    parser.add_argument("--max-updates", type=int, default=500)  # 本阶段最多五百次成功更新。
    parser.add_argument("--max-seconds", type=float, default=900)  # 累积组边界检查十五分钟预算。
    args = parser.parse_args()  # 实际预算写入报告。
    assert 0 < args.max_updates <= 500 and 0 < args.max_seconds <= 900, "超过本课SFT预算"  # 超限需另开阶段而非静默延长。
    assert torch.cuda.is_available(), "本次正式SFT需使用已试跑的CUDA环境"  # 不意外在CPU启动长训练。
    torch.set_num_threads(4)  # 限制CPU线程。
    torch.manual_seed(553)  # 模型构造与随机状态固定。
    torch.backends.cuda.matmul.allow_tf32 = False  # 保存相同数值配置。
    rows, tokenizer, manifest = load_data()  # 数据冻结后才创建训练器。
    source = ROOT / "model/lesson50/last_train.pt"  # 使用完整FP32预训练参数。
    previous = json.loads((ROOT / "docs/training_runs/lesson50.json").read_text(encoding="utf-8"))  # 来源证据独立保留。
    source_hash = sha(source.read_bytes())  # 来源文件不覆盖。
    assert source_hash == previous["artifacts"][source.name]["sha256"], "预训练来源不匹配"  # 不用FP16导出冒充主参数。
    saved = torch.load(source, map_location="cpu", weights_only=True)  # 新阶段只取模型参数，优化器明确重置。
    model = ChatLM(**saved["model_config"]).cuda()  # 30M结构与词表不变。
    model.load_state_dict(saved["model_state"])  # 继承累计2020次预训练学到的矩阵。
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.01)  # 新SFT目标使用新AdamW历史。
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)  # FP16训练使用动态损失缩放。
    OUT.mkdir(exist_ok=True)  # 检查点保存在新课目录。
    cursor, step, seen, skipped = {"epoch": 0, "offset": 0}, 0, 0, 0  # SFT独立计数，不把预训练2020清零覆盖。
    history, batch_log, exposure = [], [], []  # 固定验证和实际微批引用都保存。
    best_loss, best_step = float("inf"), 0  # 起点也参加最佳选择。
    start = perf_counter()  # 本阶段计时含起点验证、生成及周期保存。
    def checkpoint():  # 每次验证后保存完整训练状态。
        nonlocal best_loss, best_step  # 严格按固定验证损失选优。
        metrics = evaluate(model, rows["validation"], tokenizer)  # 完整26条验证，573个助手目标。
        samples = development(model, tokenizer)  # 固定五个回答，真实多轮历史。
        improved = metrics["loss"] < best_loss  # 不因训练更久强制成为最佳。
        if improved:  # 保存真正改善的检查点。
            best_loss, best_step = metrics["loss"], step  # 记录SFT步数而非预训练步数。
        entry = {"step": step, **metrics, "samples": samples, "elapsed_seconds": perf_counter() - start}  # 验证范围、生成及时间一起保存。
        history.append(entry)  # 不只保留最低损失那次。
        state = {"format": "chat_sft53_train_v1", "model_config": model.spec, "model_state": cpu_tree(model.state_dict()), "optimizer_state": cpu_tree(optimizer.state_dict()), "scaler_state": scaler.state_dict(), "step": step, "pretrain_step": 2020, "seen_positions": seen, "cursor": dict(cursor), "data_seed": 553, "microbatch": 2, "accumulation": 4, "tokenizer_sha256": manifest["tokenizer_sha256"], "sft_manifest_sha256": sha((DIRECTORY / "frozen_v1/manifest.json").read_bytes()), "parent_checkpoint_sha256": source_hash, "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all(), "best_loss": best_loss, "best_step": best_step, "elapsed_seconds": perf_counter() - start}  # 下次恢复需同时恢复所有历史与游标。
        torch.save(state, OUT / "last_train.pt")  # 保存最近进度，不覆盖旧模型。
        if improved:  # 最佳可能与最后不同。
            torch.save(state, OUT / "best_train.pt")  # 使用完整快照，不保存可被后续修改的引用。
        print(f"SFT step={step} validation={metrics['loss']:.6f} accuracy={metrics['accuracy']:.2%} best={best_step}", flush=True)  # 只报告固定验证口径。
        for case in samples:  # 所有样例全部显示。
            for turn in case:  # 包括多轮追问。
                print(repr(turn["prompt"]), "=>", repr(turn["text"]), turn["stop_reason"], flush=True)  # 失败也不删除。
    checkpoint()  # 先测未SFT的共同起点。
    while step < args.max_updates and perf_counter() - start < args.max_seconds:  # 不在预算截止后启动新的累积组。
        pending, indices = dict(cursor), []  # 正式游标仅在成功更新后推进。
        for _ in range(8):  # 四个微批，每批两条完整会话。
            order = np.random.default_rng(553 + pending["epoch"]).permutation(len(rows["train"]))  # 每轮重新确定性打乱。
            indices.append((pending["epoch"], int(order[pending["offset"]])))  # 记录轮号及真实行号。
            pending["offset"] += 1  # 推进临时游标。
            if pending["offset"] == len(rows["train"]):  # 用完一轮再进入下一轮。
                pending = {"epoch": pending["epoch"] + 1, "offset": 0}  # 模型和优化器不重置。
        batches = [tensors([rows["train"][index] for _, index in indices[i:i + 2]], tokenizer, "cuda") for i in range(0, 8, 2)]  # 八条会话分四批右填。
        positions = sum(int((y != -100).sum().item()) for _, y in batches)  # 累积组共享真实助手目标分母。
        for attempt in range(3):  # 溢出重试同一组，不前进数据位置。
            model.train()  # 每组更新前回到训练模式。
            optimizer.zero_grad(set_to_none=True)  # 只在组开始清梯度。
            loss_sum = 0.0  # 显示更新前的本组目标均值。
            for x, y in batches:  # 累积四次反向传播。
                with torch.autocast("cuda", dtype=torch.float16):  # 主参数仍是FP32。
                    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum") / positions  # 微批不分别平均。
                loss_sum += float(loss.item())  # 同一分母的贡献可直接相加。
                scaler.scale(loss).backward()  # 回答损失可沿上下文反传。
            scaler.unscale_(optimizer)  # 裁剪前先还原真实梯度。
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())  # 整组统一范数裁剪。
            old_scale = scaler.get_scale()  # 用缩放变化识别被跳过的更新。
            scaler.step(optimizer)  # 每组最多一次真实参数更新。
            scaler.update()  # 成功增长或溢出降低动态比例。
            if scaler.get_scale() >= old_scale:  # 没有因溢出跳过更新。
                break  # 提交本组数据位置。
            skipped += 1  # 不把溢出尝试记作成功step。
        else:  # 连续溢出不无限重试。
            raise RuntimeError("同一SFT累积组连续三次溢出")  # 保留此前已保存断点。
        step, seen, cursor = step + 1, seen + positions, pending  # 成功更新才累计正式计数。
        exposure.extend(indices)  # 保存实际成功读取的顺序指纹来源。
        batch_log.append({"step": step, "loss": loss_sum, "positions": positions, "gradient_norm": norm})  # 不将不同批次训练损失当固定验证。
        if step % 50 == 0:  # 定期显示进度避免长时间无输出。
            print(f"SFT train step={step}/{args.max_updates} loss={loss_sum:.6f} targets={positions}", flush=True)  # 用户可看到真实训练进展。
        if step in (100, 200, 350, args.max_updates):  # 预先固定检查点位置。
            checkpoint()  # 验证、生成及保存计入阶段时间。
    if history[-1]["step"] != step:  # 时间上限可能落在周期之间。
        checkpoint()  # 截止后的必要收尾允许完成，不再启动更新。
    stage_seconds = perf_counter() - start  # 导出和后续恢复检查另计。
    best = torch.load(OUT / "best_train.pt", map_location="cpu", weights_only=True)  # 最佳并不一定是最后。
    model.load_state_dict(best["model_state"])  # 对最佳参数做独立导出。
    exported = {"format": "chat_sft53_inference_v1", "model_config": model.spec, "model_state": {name: tensor.half() for name, tensor in best["model_state"].items()}, "step": best_step, "pretrain_step": 2020, "tokenizer_sha256": manifest["tokenizer_sha256"], "sft_manifest_sha256": best["sft_manifest_sha256"], "system": SYSTEM, "stage": "bounded_sft_pilot_not_chat_acceptance"}  # 推理文件不含优化器状态。
    torch.save(exported, OUT / "chat_fp16.pt")  # 约60MB，满足不超过1GB的目标。
    model.load_state_dict(exported["model_state"])  # 复算真实舍入后的导出文件。
    export_metrics = evaluate(model, rows["validation"], tokenizer)  # 全部验证，不把子集冒充全量。
    ci_metrics = evaluate(model, rows["validation"], tokenizer, limit=8)  # CI明确只复算前八条。
    export_samples = development(model, tokenizer)  # 固定问答对照真实导出文件。
    assert sha(source.read_bytes()) == source_hash, "预训练源文件被改写"  # 原正式2020步权重保持独立。
    artifacts = {name: {"bytes": (OUT / name).stat().st_size, "sha256": sha((OUT / name).read_bytes())} for name in ("last_train.pt", "best_train.pt", "chat_fp16.pt")}  # 交付文件真实大小与摘要。
    report = {"status": "completed", "stage": "formal_bounded_sft_pilot", "config": vars(args), "completed_steps": step, "pretrain_step": 2020, "seen_positions": seen, "best_step": best_step, "best_loss": best_loss, "cursor": cursor, "skipped_updates": skipped, "stage_seconds": stage_seconds, "microbatch": 2, "gradient_accumulation": 4, "learning_rate": 5e-5, "optimizer": "new_stage_AdamW_beta0.9_0.95_decay0.01", "source_sha256": source_hash, "sft_manifest_sha256": best["sft_manifest_sha256"], "tokenizer_sha256": manifest["tokenizer_sha256"], "model_config": model.spec, "data_stats": manifest["stats"], "history": history, "batch_log": batch_log, "exposure_sha256": sha(np.asarray(exposure, dtype=np.int64).tobytes()), "export_validation": export_metrics, "export_ci_subset": ci_metrics, "export_samples": export_samples, "dev_prompts": DEV, "artifacts": artifacts, "limitations": "small short Chinese curriculum; synthetic repeat followups; same-source validation, 23 groups only; known training substring check is not complete semantic contamination audit; no old test read; token loss is not chat quality"}  # 如实报告训练规模与局限。
    (ROOT / "docs/training_runs/lesson53.json").write_bytes(dump(report))  # 不覆盖预训练或副本诊断报告。
    print("正式SFT结束：", step, "最佳：", best_step, best_loss, "阶段秒：", stage_seconds, flush=True)  # 不以模型会说话作为未经检验的结论。


if __name__ == "__main__":  # 导入评估和生成函数不启动训练。
    main()  # 显式命令才运行最多500更新/900秒。
