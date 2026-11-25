import argparse  # 显式区分下载、候选准备和冻结。
import json  # 保存原始记录与审核决定。
import random  # 固定新数据的组级划分。
import re  # 标记数字模板及排除本轮高风险主题。
import urllib.request  # 仅显式下载固定提交文件。
from collections import Counter  # 保留技术筛查原因及轮次统计。
from sft_batches import ROOT, load_tokenizer, encode_dialogue  # 沿用6400词BPE与256上下文。
from prepare_chat_corpus import normalize, sha, dump  # 固定去重规则和LF字节。
DIRECTORY = ROOT / "data/chat53"  # 新阶段数据不覆盖旧版本。
REVISION = "f39db019a94f8dbea48ab30d2bdc090703284559"  # 固定中文指令数据提交。
BASE = f"https://huggingface.co/datasets/shibing624/alpaca-zh/resolve/{REVISION}/"  # 不依赖浮动main。
SOURCE_HASH = "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"  # 完整原始JSON摘要。
SYSTEM = "你是一个中文助手，请准确、简洁地回答问题。"  # 训练与聊天共用系统提示。


def persist(name, data):  # 冻结版本只初次保存，重复运行核对字节。
    path = DIRECTORY / name  # 只写本课目录。
    path.parent.mkdir(parents=True, exist_ok=True)  # 建立明确的父目录。
    if path.exists():  # 版本已存在时禁止静默覆盖。
        assert path.read_bytes() == data, f"版本改变，请另建版本：{name}"  # 不重划分验证集。
    else:  # 初次构建时保存。
        path.write_bytes(data)  # 不进行Windows换行转换。


def prefix_statistics(tokenizer):  # 旧长会话只在完整轮次边界保留前缀，不拿未审答案训练。
    result = {}  # 训练与验证分别统计，不读取测试。
    for split in ("train", "validation"):  # 保持旧的分组归属。
        counts = Counter()  # 记录完整、可用前缀及首轮超长。
        path = ROOT / f"data/chat48/sft_{split}.jsonl"  # 不改变原文件。
        for line in path.read_text(encoding="utf-8").splitlines():  # 原多轮会话按顺序处理。
            row, prefix, best = json.loads(line), [], []  # 每条记录从原开头开始。
            for message in row["messages"]:  # 不把后面的依赖性追问当成独立首问。
                prefix.append(message)  # 逐条积累原消息。
                if message["role"] != "assistant":  # 必须等到一个完整回答结束。
                    continue  # 不将用户前缀当作完整SFT样本。
                try:  # 只尝试完整问答轮次边界。
                    encode_dialogue(prefix, tokenizer)  # 超长就不能继续纳入此版前缀。
                except AssertionError:  # 不截断正文以凑长度。
                    break  # 已保留的best仍是原开头的完整前缀。
                best = list(prefix)  # 只有编码通过的完整回答才提交。
            counts["full_conversation" if len(best) == len(row["messages"]) else "complete_prefix" if best else "first_answer_over_context_or_invalid"] += 1  # 前缀技术可用不代表内容已审。
        result[split] = dict(counts)  # 本课不发布这些未审核前缀作为训练数据。
    return result  # 数据报告保留实际处理结果。


def candidates():  # 从新来源构造短中文候选，随后仍需逐条审阅。
    path = DIRECTORY / "raw/alpaca_gpt4_data_zh.json"  # 完整来源约35MB。
    assert sha(path.read_bytes()) == SOURCE_HASH, "中文源文件版本错误"  # 不在不同版本上复用行号。
    source = json.loads(path.read_text(encoding="utf-8"))  # 原结构为instruction/input/output。
    tokenizer = load_tokenizer()  # 不另建聊天词表。
    known = "\n".join(normalize(json.loads(line)["text"]) for line in (ROOT / "data/chat48/pretrain_train.jsonl").read_text(encoding="utf-8").splitlines())  # 仅检查实际预训练见过的正文。
    eligible, seen_prompts, seen_answers, counts = [], set(), set(), Counter()  # 保守合并同数字模板或完全相同答案。
    banned = re.compile("医|药|病|健康|治疗|心理|法律|律师|股票|投资|总统|战争|历史|年份|诺贝尔|首都|发明|政治|自杀|武器")  # 本轮避免需额外专业核实的主题，不等于完整质量分类器。
    for index, item in enumerate(source):  # 行号标识可追溯到完整固定源文件。
        prompt = item["instruction"].strip() + ("\n\n" + item["input"].strip() if item["input"].strip() else "")  # 保留非空任务输入。
        answer = item["output"].strip()  # 不自动改写生成答案。
        normalized = normalize(prompt)  # 已知训练文本匹配使用同一字符规范化。
        key, answer_key = re.sub(r"\d+", "#", normalized), normalize(answer)  # 数字模板相同的题不拆组。
        if banned.search(prompt + answer) or len(normalized) < 8:  # 主题和过短提示先作技术候选限制。
            counts["topic_or_short_prompt"] += 1  # 不声称它们均是错误数据。
            continue  # 后续可另建有专业审核的数据版本。
        if len(tokenizer.encode(prompt).ids) > 32 or not 4 <= len(tokenizer.encode(answer).ids) <= 32:  # 本次限定短回答以便完整逐条审阅。
            counts["outside_short_review_budget"] += 1  # 长答案仍可能有价值。
            continue  # 不截断原答案。
        if sum("\u4e00" <= c <= "\u9fff" for c in prompt + answer) / len(prompt + answer) < 0.2:  # 当前目标是中文聊天。
            counts["insufficient_cjk"] += 1  # 字符代理不是精确语言识别。
            continue  # 不以全英文样本补中文配额。
        if normalized in known or len(answer_key) >= 8 and answer_key in known:  # 保守排除与旧预训练正文完全子串匹配的问答。
            counts["known_pretrain_exact_substring"] += 1  # 不能保证排除所有语义改写污染。
            continue  # 新验证至少排除已知的直接文本重复。
        if key in seen_prompts or answer_key in seen_answers:  # 同模板或同答案只保留首条。
            counts["duplicate_prompt_or_answer"] += 1  # 不靠重复内容增加独立样本量。
            continue  # 原行仍留在原始文件中。
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]  # 与后续聊天模板一致。
        try:  # 再使用真实训练编码器检查完整性。
            encode_dialogue(messages, tokenizer)  # 模板中保留完整回答及结束符。
        except AssertionError:  # 格式异常明确计数。
            counts["invalid_format"] += 1  # 不执行数据中的任何指令或代码。
            continue  # 不把结构异常送入训练。
        seen_prompts.add(key)  # 记录数字模板键。
        seen_answers.add(answer_key)  # 记录规范化答案键。
        eligible.append({"id": f"alpaca53:{index:05d}", "source_index": index, "group_id": sha(key.encode("utf-8")), "source_record_sha256": sha(dump(item)), "messages": messages})  # 保存源行、原记录摘要与组身份。
    eligible.sort(key=lambda row: sha(("lesson53:" + row["id"]).encode("utf-8")))  # 固定候选顺序，不按模型效果挑题。
    selected = eligible[:160]  # 同一固定顺序扩大审核到160条，不更换已见候选。
    assert len(selected) == 160, "候选不足，不自动改变采样条件"  # 不临时放宽内容标准。
    queue = {"format": "chat_sft53_queue_v2", "source_records": len(source), "eligible_short_records": len(eligible), "filters": dict(counts), "source_sha256": SOURCE_HASH, "revision": REVISION, "items": selected, "old_prefix_statistics": prefix_statistics(tokenizer)}  # 旧112条队列保留，扩审另建版本。
    return queue  # 冻结后由助手逐条阅读并记录决定。


def freeze():  # 明确审核完成后才生成正式SFT数据。
    queue_path, reviews_path = DIRECTORY / "queue_v2.json", DIRECTORY / "reviews.json"  # 扩审版本独立留存。
    queue = json.loads(queue_path.read_text(encoding="utf-8"))  # 正式训练依赖已发布的候选。
    assert dump(candidates()) == queue_path.read_bytes(), "候选无法重建"  # 防止审核后更换来源或筛查条件。
    audit = json.loads(reviews_path.read_text(encoding="utf-8"))  # 逐条阅读后的显式决定。
    assert audit["queue_sha256"] == sha(queue_path.read_bytes()), "审核队列不匹配"  # 旧112条审核不能代替160条。
    decisions = {row["id"]: row for row in audit["reviews"]}  # 每条原记录只允许一个决定。
    assert len(decisions) == len(audit["reviews"]) == len(queue["items"]) and set(decisions) == {row["id"] for row in queue["items"]}, "缺漏或重复审核"  # 不能默许未审样本。
    accepted = []  # 仅接纳显式批准的原问答。
    for row in queue["items"]:  # 在完整阅读后固定决定。
        decision = decisions[row["id"]]  # 对齐原记录摘要。
        assert decision["source_record_sha256"] == row["source_record_sha256"] and decision["decision"] in ("accept", "reject", "hold") and decision["reason"], "审核版本或理由错误"  # 不悄悄替换正文。
        if decision["decision"] == "accept":  # hold与reject不训练。
            accepted.append(row)  # 保留原来源信息与消息。
    groups = sorted({row["group_id"] for row in accepted})  # 数字模板键组成独立组。
    random.Random(53).shuffle(groups)  # 内容审核完成后按固定组顺序划分。
    mapping = {group: "train" if index < len(groups) * 8 // 10 else "validation" for index, group in enumerate(groups)}  # 留出20%做SFT验证，不新增被调参使用的测试集。
    rows = {"train": [], "validation": []}  # 原始问答和同组追问变体不能拆分。
    for index, row in enumerate(accepted):  # 不因模型分数决定划分。
        split = mapping[row["group_id"]]  # 完全由固定组映射决定。
        rows[split].append({**row, "variant": "original"})  # 单轮原始样本不改答案。
        if index % 4 == 0:  # 有界增加完整多轮复述任务。
            followup = [{"role": "user", "content": "请复述你刚才的回答。"}, {"role": "assistant", "content": row["messages"][-1]["content"]}]  # 复制正确上文，不编造新事实。
            augmented = {**row, "id": row["id"] + ":repeat", "variant": "synthetic_followup_copy", "messages": row["messages"] + followup}  # 变体沿用同组和来源。
            encode_dialogue(augmented["messages"], load_tokenizer())  # 仍需完整装入256上下文。
            rows[split].append(augmented)  # 两个助手回答均监督，报告将披露重复目标。
    tokenizer, stats, output = load_tokenizer(), {}, {}  # 词表不随新语料重训。
    for split, records in rows.items():  # 分别冻结训练和留出验证。
        targets = sum(sum(y != -100 for y in encode_dialogue(row["messages"], tokenizer)[1]) for row in records)  # 多轮仍按实际目标计数。
        stats[split] = {"records": len(records), "groups": len({r["group_id"] for r in records}), "assistant_targets": targets, "multiturn_records": sum(row["variant"] != "original" for row in records), "cjk_conversation_fraction": 1.0}  # 候选已按汉字比例过滤。
        output[split + ".jsonl"] = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records).encode("utf-8")  # 完整消息与来源一并保存。
    assert stats["train"]["groups"] >= 64 and stats["validation"]["groups"] >= 16, "独立组数量不足，不启动SFT"  # 沿用上课初步数量门槛，变体不凑组数。
    assert not {row["group_id"] for row in rows["train"]} & {row["group_id"] for row in rows["validation"]}, "组交集不为空"  # 原问答及追问不跨集合。
    manifest = {"format": "chat_sft53_data_v1", "dataset": "shibing624/alpaca-zh", "revision": REVISION, "source_sha256": SOURCE_HASH, "source_card_sha256": sha((DIRECTORY / "raw/README.md").read_bytes()), "credit": "Shibing Chen; Instruction Tuning with GPT-4 / Baolin Peng and coauthors", "terms": "card metadata says cc-by-4.0 but body explicitly requires CC-BY-NC-4.0 and research-only model use; follow restrictive body; previous pretraining mixed terms also apply", "scope": "noncommercial educational research pilot, not a commercially licensed chat model", "queue_sha256": sha(queue_path.read_bytes()), "reviews_sha256": sha(reviews_path.read_bytes()), "review_counts": dict(Counter(row["decision"] for row in audit["reviews"])), "seed": 53, "tokenizer_sha256": sha((ROOT / "data/chat48/tokenizer/tokenizer.json").read_bytes()), "system": SYSTEM, "context": 256, "stats": stats, "formal_sft_ready": True, "augmentation": "every fourth accepted source gets same-group literal-repeat followup; both assistant turns supervised; not independent new knowledge", "grouping": "numeric-normalized exact prompt templates and exact answers deduplicated before review; semantic paraphrases can remain", "known_pretrain_overlap_check": "normalized prompt and answer exact substrings excluded from actual old pretrain_train text only; not a semantic contamination guarantee; old test contents unopened", "files": {name: sha(data) for name, data in output.items()}}  # 不把小规模人工审核等同通用聊天能力。
    for name, data in output.items():  # 固定语料版本，不改旧文件。
        persist("frozen_v1/" + name, data)  # 后续只读校验字节。
    persist("frozen_v1/manifest.json", dump(manifest))  # 清单完整记录来源与筛查范围。
    print("正式SFT数据：", stats, "审核：", manifest["review_counts"])  # 输出真实独立组与变体数。
    return rows, manifest  # 训练器使用同一份冻结数据。


def main():  # 默认离线核对队列，不启动训练。
    parser = argparse.ArgumentParser()  # 下载必须显式选择。
    parser.add_argument("--download", action="store_true")  # 避免CI触发外网下载。
    parser.add_argument("--freeze", action="store_true")  # 审核存在后才允许冻结正式数据。
    args = parser.parse_args()  # 读取本次操作。
    if args.download:  # 固定两个来源文件，总上限40MiB。
        remaining = 40 * 1024 * 1024  # 两个文件共享总预算，而非各自40MiB。
        for name in ("README.md", "alpaca_gpt4_data_zh.json"):  # 保留使用声明及完整源数据。
            with urllib.request.urlopen(BASE + name, timeout=60) as response:  # 请求固定提交。
                data = response.read(remaining + 1)  # 只读取剩余下载预算。
            assert len(data) <= remaining, "下载超过预算"  # 超预算不写入。
            remaining -= len(data)  # 下一文件继续使用剩余总预算。
            persist("raw/" + name, data)  # 原字节快照不转换换行。
    queue = candidates()  # 重建技术候选和旧完整轮次统计。
    persist("queue_v2.json", dump(queue))  # 扩审另建版本，不覆盖旧112条队列。
    print("来源记录：", queue["source_records"], "短中文可送审：", queue["eligible_short_records"], "本次待审：", len(queue["items"]))  # 不宣称112条已合格。
    print("旧会话完整轮次统计：", queue["old_prefix_statistics"])  # 只统计，未审前缀不训练。
    if args.freeze:  # 显式执行审核后的组级划分。
        freeze()  # 仍不启动神经网络训练。


if __name__ == "__main__":  # 导入不下载或写文件。
    main()  # 显式运行才准备候选。
