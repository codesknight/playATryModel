import json  # 读取冻结训练和验证对话，不读取测试正文。
from collections import Counter  # 汇总长度与格式筛查结果。
from sft_batches import ROOT, load_tokenizer, encode_dialogue  # 沿用256上下文及真实BPE。
from prepare_chat_corpus import sha, dump  # 固定输出字节与版本摘要。
DIRECTORY = ROOT / "data/chat52"  # 不覆盖第五十一课的审核记录。
SEED = "lesson52:20261125"  # 仅用于固定审核队列，不改原划分。


def build_queue():  # 结构筛查与逐条内容审核严格区分。
    tokenizer = load_tokenizer()  # 不训练新词表。
    queue, sources, inventory = [], {}, {}  # 保存候选、输入指纹及全量结构统计。
    for split, quota in (("train", 4), ("validation", 2)):  # 每个非空来源块限量取样，测试集不打开。
        path = ROOT / f"data/chat48/sft_{split}.jsonl"  # 训练与验证身份始终不变。
        sources[split] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path.read_bytes())}  # 固定原文件版本。
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]  # 原记录含来源组标识。
        counts, eligible = Counter(), []  # 自动筛查只决定能否送逐条审核，不批准训练。
        for row in rows:  # 全量检查完整会话是否能装入当前模型。
            counts["total"] += 1  # 原划分记录总数。
            try:  # 沿用正式训练编码的角色和长度条件。
                x, y, _, _ = encode_dialogue(row["messages"], tokenizer)  # 只允许完整会话，不截断回答。
            except AssertionError as error:  # 明确保存不能送审的技术原因。
                counts["over_context" if "超长" in str(error) else "invalid_format"] += 1  # 超长不是内容质量差。
                continue  # 不把技术拒绝的样本强行凑入短上下文。
            counts["eligible_for_review"] += 1  # 这只是技术可用，不是内容已通过。
            eligible.append({"split": split, "id": row["id"], "group_id": row["group_id"], "block": row["block"], "text_sha256": row["text_sha256"], "input_tokens": len(x), "assistant_targets": sum(value != -100 for value in y)})  # 不复制或改写原答案。
        for block in sorted({row["block"] for row in eligible}):  # 来源块独立取样，避免只检查文件开头。
            candidates = [row for row in eligible if row["block"] == block]  # 当前来源块符合技术约束的会话。
            candidates.sort(key=lambda row: sha(f"{SEED}:{split}:{row['id']}".encode("utf-8")))  # 稳定摘要排序，不依赖Python进程hash。
            queue.extend(candidates[:quota])  # 不足配额时全部入队，不跨块复制补足。
        inventory[split] = dict(counts)  # 记录全量结构统计与候选数。
    return {"format": "sft_review52_queue_v1", "seed": SEED, "context": 256, "policy": "complete conversations; unchanged original groups; technical eligibility is not semantic approval; per-block fixed quotas train4 validation2; test unopened", "sources": sources, "inventory": inventory, "items": queue}  # 所有候选仍需逐条内容审核。


def main():  # 初次写队列，后续运行只核对，不覆盖逐条决定。
    queue = build_queue()  # 由原冻结文件确定唯一候选队列。
    DIRECTORY.mkdir(exist_ok=True)  # 只建立本课独立目录。
    path = DIRECTORY / "queue.json"  # 队列版本与审核决定分开存。
    if path.exists():  # 已发布队列不能静默改变。
        assert path.read_bytes() == dump(queue), "候选队列改变，需要另建数据版本"  # 不能用同名文件悄悄更换样本。
    else:  # 仅首次写入确定性候选。
        path.write_bytes(dump(queue))  # 统一UTF-8及LF。
    print("技术筛查：", queue["inventory"])  # 显示全部训练及验证记录的结构统计。
    print("待逐条内容审核：", dict(Counter(row["split"] for row in queue["items"])))  # 不声称队列已可用于训练。


if __name__ == "__main__":  # 导入不写文件或自动审核。
    main()  # 运行只准备队列，不训练模型。
