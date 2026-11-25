import json  # 读取审核决定并保存冻结清单。
from collections import Counter  # 汇总接纳、拒绝与待核实数量。
from prepare_sft_review import ROOT, DIRECTORY, build_queue  # 重建确定性候选并定位本课目录。
from sft_batches import load_tokenizer, encode_dialogue  # 编码规则与未来训练相同。
from prepare_chat_corpus import sha, dump  # 所有清单采用固定LF字节与SHA。
POLICY = {"min_train_conversations": 64, "min_validation_conversations": 16, "min_cjk_conversation_fraction": 0.5, "cjk_character_ratio_threshold": 0.2}  # 本课程启动门槛，不是能chat的充分条件。


def validate_reviews(queue, review, queue_hash):  # 在生成任何子集前拒绝漏审、重复和版本错配。
    assert review["queue_sha256"] == queue_hash, "逐条审核属于其他队列"  # 错误版本不能发布。
    decisions = {(row["split"], row["id"]): row for row in review["reviews"]}  # 同一标识必须保留原划分。
    items = {(row["split"], row["id"]): row for row in queue["items"]}  # 每个候选均需一份明确决定。
    assert len(decisions) == len(review["reviews"]) and set(decisions) == set(items), "存在重复、缺漏或额外审核"  # 未审不能默认接纳。
    for key, decision in decisions.items():  # 决定必须对应审核时的原正文。
        assert decision["text_sha256"] == items[key]["text_sha256"], "正文与审核版本不一致"  # 不允许审核后替换答案。
        assert decision["decision"] in ("accept", "reject", "hold") and decision["reason"] and decision["reason_code"], "审核决定或理由不完整"  # 机器不能自行猜测逐条结论。
    return decisions  # 只有通过检查的决定才能生成冻结子集。


def materialize():  # 计算应冻结的字节；本函数不写文件、不训练模型。
    queue_path, review_path = DIRECTORY / "queue.json", DIRECTORY / "reviews.json"  # 队列与逐条决定独立版本化。
    queue = build_queue()  # 从原始训练和验证文件重新计算候选。
    assert queue_path.read_bytes() == dump(queue), "候选队列版本不一致"  # 不能审核后悄悄替换候选。
    review = json.loads(review_path.read_text(encoding="utf-8"))  # 每条决定必须显式存在。
    decisions = validate_reviews(queue, review, sha(queue_path.read_bytes()))  # 审核不全时先失败，不输出半成品。
    tokenizer, files, statistics, accepted_groups = load_tokenizer(), {}, {}, {}  # 不打开测试集正文。
    for split in ("train", "validation"):  # 保留原划分身份，不能从验证借样本凑训练数。
        path = ROOT / queue["sources"][split]["path"]  # 来源摘要已由队列重建验证。
        source = {row["id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}  # 原始完整会话不做文本改写。
        selected, groups, counts, targets, cjk_count = [], set(), Counter(), 0, 0  # 全部统计来自显式决定。
        for item in (row for row in queue["items"] if row["split"] == split):  # 按冻结队列的确定顺序遍历。
            decision = decisions[(split, item["id"])]  # 没有匹配决定会在前面失败。
            assert decision["decision"] in ("accept", "reject", "hold") and decision["reason"] and decision["reason_code"], "审核决定或理由不完整"  # 待核实不等于已证明错误。
            row = source[item["id"]]  # 同一标识必须仍属于原划分。
            assert decision["text_sha256"] == item["text_sha256"] == row["text_sha256"], "正文与审核版本不一致"  # 保持审核所见答案。
            counts[decision["decision"]] += 1  # 无论是否接纳都纳入统计。
            if decision["decision"] != "accept":  # 拒绝和待核实都不进入训练或验证文件。
                continue  # 不以数量不足为由绕过质量决定。
            x, y, _, _ = encode_dialogue(row["messages"], tokenizer)  # 再验证完整性及256长度约束。
            assert len(x) == item["input_tokens"], "编码长度改变"  # 词表版本或模板不能偷偷改变。
            selected.append(row)  # 保存原始会话及来源字段，不摘取局部好答案。
            groups.add(row["group_id"])  # 沿用原分组，避免训练与验证同组。
            targets += sum(value != -100 for value in y)  # 助手目标按真实BPE token计数。
            body = "".join(message["content"] for message in row["messages"])  # 语言覆盖只做粗略字符统计。
            ratio = sum("\u4e00" <= char <= "\u9fff" for char in body) / len(body)  # 常用汉字比例不是精确语言识别。
            cjk_count += int(ratio >= POLICY["cjk_character_ratio_threshold"])  # 记录中文覆盖，不宣称质量评分。
        data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected).encode("utf-8")  # 完整原会话按确定顺序输出JSONL。
        files[f"{split}.jsonl"] = data  # 只是审核子集，不替换原3206条训练/验证文件。
        statistics[split] = {"reviewed": sum(counts.values()), "decisions": dict(counts), "accepted_conversations": len(selected), "groups": len(groups), "assistant_targets": targets, "cjk_conversations": cjk_count, "cjk_conversation_fraction": cjk_count / len(selected) if selected else 0.0}  # 空集也安全报告。
        accepted_groups[split] = groups  # 下一步检查原组隔离。
    assert not accepted_groups["train"] & accepted_groups["validation"], "训练与验证出现同组会话"  # 不重新洗牌或交换样本。
    blockers = []  # 门槛不足时阻止下一阶段正式SFT。
    for split in ("train", "validation"):  # 两个集合都需数量与中文覆盖检查。
        if statistics[split]["accepted_conversations"] < POLICY[f"min_{split}_conversations"]:  # 数量阈值是课程策略，不是科学保证。
            blockers.append(f"{split}: insufficient reviewed conversations")  # 明确哪项不足。
        if statistics[split]["cjk_conversation_fraction"] < POLICY["min_cjk_conversation_fraction"]:  # 用户目标是中文聊天。
            blockers.append(f"{split}: insufficient Chinese coverage")  # 不拿全英文小样本冒充中文SFT准备完成。
    manifest = {"format": "sft_review52_frozen_v1", "queue_sha256": sha(queue_path.read_bytes()), "reviews_sha256": sha(review_path.read_bytes()), "tokenizer_sha256": sha((ROOT / "data/chat48/tokenizer/tokenizer.json").read_bytes()), "sources": queue["sources"], "inventory": queue["inventory"], "statistics": statistics, "policy": POLICY, "formal_sft_ready": not blockers, "blockers": blockers, "files": {name: {"sha256": sha(data), "bytes": len(data)} for name, data in files.items()}, "test_content_opened": False, "model_updates": 0, "limitations": "source-block quota sample of short full conversations, not a representative quality estimate; CJK ratio is a proxy; inherited grouping is heuristic; accepted sample is not a sufficient chat benchmark"}  # 不把局部审核当全库质量认证。
    files["manifest.json"] = dump(manifest)  # 清单的摘要覆盖其余冻结文件。
    return files, manifest  # 调用者选择初次写入或只读核验。


def main():  # 首次冻结，后续只核对版本不覆盖。
    files, manifest = materialize()  # 所有审核齐全才产生可保存结果。
    target = DIRECTORY / "frozen_v1"  # 保留独立版本目录。
    target.mkdir(exist_ok=True)  # 不覆盖旧课语料。
    for name, data in files.items():  # 逐个文件检查固定字节。
        path = target / name  # 路径只来自显式生成的文件名。
        if path.exists():  # 已冻结版本变化必须新建版本。
            assert path.read_bytes() == data, "冻结文件改变，请另建版本"  # 不静默覆盖审核范围。
        else:  # 初次运行才保存。
            path.write_bytes(data)  # 固定UTF-8与LF字节。
    print(json.dumps(manifest["statistics"], ensure_ascii=False, indent=2))  # 显示实际接纳数量而非夸大训练规模。
    print("允许正式SFT：", manifest["formal_sft_ready"], "原因：", manifest["blockers"])  # 不满足门槛不启动训练。


if __name__ == "__main__":  # 导入不改文件或启动训练。
    main()  # 运行只冻结审核子集。
