import io  # 将紧凑整数数组保存为固定字节。
import json  # 读取旧冻结语料并保存新版本。
import math  # 验证可精确计算的课程答案。
import random  # 固定任务数据及划分。
import re  # 本轮排除含诊断或治疗主题的生成故事。
from collections import Counter  # 汇总任务覆盖与文档筛查。
import numpy as np  # 保存BPE编号。
from sft_batches import ROOT, load_tokenizer, encode_dialogue  # 不改变旧模型词表与上下文。
from prepare_chat_corpus import sha, dump, normalize  # 复用来源摘要与字符规范化。
DIRECTORY = ROOT / "data/chat54"  # 新阶段不覆盖旧课数据。
DOMAINS = ("original", "stories", "novels")  # 旧通用文本、中文故事、四书章节。
SCHEDULE = ("original", "stories", "original", "stories", "original", "novels", "original", "stories")  # 每次更新八窗，固定4/3/1来源配比。
WEIGHTS = {"original": 0.5, "stories": 0.375, "novels": 0.125}  # 起点前固定的分域验证权重。
LANGUAGE = "language_v3"  # 模型更新前另立清除生成元数据及已知坏开头的版本，保留抽查历史。
SYSTEM = "你是一个中文助手，请准确、简洁地回答问题。"  # 课程数据仍使用已验证的SFT模板。


def persist(name, data):  # 新版数据只初次写入，之后逐字节核对。
    path = DIRECTORY / name  # 文件只在新课目录。
    path.parent.mkdir(parents=True, exist_ok=True)  # 不移动任何旧文件。
    if path.exists():  # 冻结版本不可静默改写。
        assert path.read_bytes() == data, f"新数据版本变化：{name}"  # 修改逻辑需另建版本。
    else:  # 初次生成保存UTF-8或二进制原字节。
        path.write_bytes(data)  # 不做换行替换。


def npy_bytes(values, dtype):  # 生成可移植整数数组文件。
    buffer = io.BytesIO()  # 内存缓冲避免临时文件副作用。
    np.save(buffer, np.asarray(values, dtype=dtype), allow_pickle=False)  # 不保存可执行Python对象。
    return buffer.getvalue()  # 固定文件摘要以实际字节计算。


def source_rows(domain, split):  # 只读训练及验证，不打开测试正文。
    path = ROOT / {"original": f"data/chat48/pretrain_{split}.jsonl", "stories": f"data/transformer45/{split}.jsonl", "novels": f"data/splits/{split}.jsonl"}[domain]  # 沿用既有独立划分。
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]  # 每条均为完整文档。
    return rows, {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path.read_bytes())}  # 保留来源证据。


def fragments(text):  # 保守检查直接重复片段，不宣称语义去重。
    normalized = normalize(text)  # 仅用于查重，不改训练正文。
    return {sha(normalized[i:i + 64].encode("utf-8")) for i in range(0, len(normalized) - 63, 32)}  # 每32字符采一个64字符片段。


def build_language():  # 为同一30M模型增加原先未用的中文语言域。
    tokenizer, sources, rows = load_tokenizer(), {}, {}  # 词表6400保持固定。
    protected = set()  # 所有验证片段不进入新的训练混合。
    topic = re.compile("自闭|自閉|過動|过动|心理學|心理学|醫|医|治療|治疗|催眠|藥|药|症|診|诊")  # 抽查发现治疗与诅咒混写，本版保守排除此主题，非医学事实判定器。
    for domain in DOMAINS:  # 每个域保留原始分组身份。
        rows[domain] = {}  # 训练与验证分开。
        for split in ("train", "validation"):  # 不读取测试文件。
            records, source = source_rows(domain, split)  # 使用旧课已保存文件。
            sources[f"{domain}_{split}"] = source  # 记录所有实际输入版本。
            prepared = []  # 保留原文摘要及明确变换记录。
            for record in records:  # 清洗在模型更新前固定，对训练和新域验证应用相同规则。
                if domain == "stories" and (topic.search(record["text"]) or record["id"] == "tiny_stories:00469"):  # 排除主题及片段抽查已发现的损坏开头。
                    continue  # 被拒绝片段保留在v2审阅记录，不隐藏问题。
                text = re.sub(r"\s*[（(]\s*(?:Word count|字数|字數)\s*[:：]?\s*\d+\s*[)）]\s*$", "", record["text"], flags=re.I) if domain == "stories" else record["text"]  # 去掉故事末尾生成字数元数据，不改写剧情。
                prepared.append({**record, "text": text, "_raw_text_sha256": sha(record["text"].encode("utf-8")), "_metadata_cleaned": text != record["text"]})  # 原始来源与训练派生文本可分别核对。
            rows[domain][split] = prepared  # 原文件保持不动。
            if split == "validation":  # 查重保护所有分域验证正文。
                for row in rows[domain][split]:  # 不把保留验证内容当语言训练素材。
                    protected.update(fragments(row["text"]))  # 只保留片段摘要供训练筛查。
    stats, files, audit = {}, {}, []  # 保存实际新增数据与抽查范围。
    for domain in DOMAINS:  # 每域独立训练流，避免小说长度决定全部采样概率。
        stream, xs, ys, counts, retained_ids, audit_candidates = [], [], [], Counter(), [], []  # 记录训练曝光来源及可分层抽查文档。
        for split in ("train", "validation"):  # 统一编码器与尾段规则。
            source_count = len(source_rows(domain, split)[0])  # 内容排除数量与最终保留数分开。
            counts[split + "_policy_excluded"] = source_count - len(rows[domain][split])  # 主题排除及已知损坏样例合计，不代表全部为事实错误。
            for row in rows[domain][split]:  # 完整原文不重新分组。
                text = row["text"]  # 保留简繁原文，不声称做过繁简转换。
                if split == "train" and fragments(text) & protected:  # 任一直接片段匹配就保守排除整篇训练文档。
                    counts["train_direct_fragment_overlap_removed"] += 1  # 这不是完整语义污染检测。
                    continue  # 不影响任何原验证文档。
                if split == "train" and (not text.strip() or "\ufffd" in text or any(ord(char) < 32 and char not in "\n\r\t" for char in text)):  # 排除明显编码或控制字符损坏。
                    counts["train_invalid_text_removed"] += 1  # 不将乱码当语言规律。
                    continue  # 只在新版本排除，不改旧文件。
                ids = [1] + tokenizer.encode(text, add_special_tokens=False).ids + [2]  # 每篇保留BOS/EOS边界。
                assert 7 not in ids, "ByteLevel正文出现UNK"  # 保持全字节覆盖。
                counts[split + "_documents"] += 1  # 实际保留文档数。
                counts[split + "_characters"] += len(text)  # 字符与token分开记录。
                counts[split + "_metadata_cleaned"] += int(row["_metadata_cleaned"])  # 明确报告实际清除的生成尾注数量。
                if split == "train":  # 训练按固定来源记录顺序连接。
                    stream.extend(ids)  # 后续按窗口随机打乱，不按文档长度取整丢弃尾部。
                    retained_ids.append(row["id"])  # 可以追溯被保留的原始文档。
                    if domain != "original":  # 新增语言域需要片段抽查。
                        audit_candidates.append({"domain": domain, "id": row["id"], "book": row.get("book"), "source_text_sha256": row["_raw_text_sha256"], "training_text_sha256": sha(text.encode("utf-8")), "scope": "first100_and_last100_characters_only", "excerpt": text[:100] + " … " + text[-100:]})  # 不声称读完全部文档。
                else:  # 验证每篇独立开窗，不跨文档。
                    for start in range(0, len(ids) - 1, 256):  # 完整保留尾窗目标。
                        chunk = ids[start:start + 257]  # 输入和下一token目标错位。
                        xs.append(chunk[:-1] + [0] * (257 - len(chunk)))  # 输入PAD补到256。
                        ys.append(chunk[1:] + [-100] * (257 - len(chunk)))  # 尾段不计虚假目标。
        payloads = {f"{domain}_train.npy": npy_bytes(stream, np.uint16), f"{domain}_validation_x.npy": npy_bytes(xs, np.uint16), f"{domain}_validation_y.npy": npy_bytes(ys, np.int32)}  # 同一词表可用uint16紧凑存储。
        for name, data in payloads.items():  # 来源快照不重复复制，保存实际编码产物。
            persist(LANGUAGE + "/" + name, data)  # 明确新版本路径。
            files[name] = sha(data)  # 后续训练逐文件核对。
        persist(f"{LANGUAGE}/{domain}_retained_ids.json", dump(retained_ids))  # 不把排除文档算成实际训练量。
        if domain == "stories":  # 使用固定摘要顺序，不只检查文件开头。
            audit.extend(sorted(audit_candidates, key=lambda r: sha(("54:" + r["id"]).encode("utf-8")))[:4])  # 保留所有选定样例，不因措辞好看更换。
        elif domain == "novels":  # 四本书各选一个章节片段。
            for book in sorted({r["book"] for r in audit_candidates}):  # 不能让抽查全落在三国演义。
                audit.append(min((r for r in audit_candidates if r["book"] == book), key=lambda r: sha(("54:" + r["id"]).encode("utf-8"))))  # 每书固定选择。
        stats[domain] = {**dict(counts), "train_tokens": len(stream), "train_windows": (len(stream) - 2) // 256 + 1, "validation_windows": len(xs), "validation_positions": int((np.asarray(ys) != -100).sum())}  # 以真实目标计数。
    old = ROOT / "data/chat48/tokenizer"  # 旧通用验证保持完全相同的尺子。
    assert files["original_validation_x.npy"] == sha((old / "pretrain_validation_x.npy").read_bytes()) and files["original_validation_y.npy"] == sha((old / "pretrain_validation_y.npy").read_bytes()), "旧验证编码未复现"  # 不暗中更换旧验证指标。
    manifest = {"format": "mixed_language54_v3", "sources": sources, "stats": stats, "files": files, "tokenizer_sha256": sha((old / "tokenizer.json").read_bytes()), "context": 256, "schedule": list(SCHEDULE), "validation_weights": WEIGHTS, "filtering": "before model updates, same diagnosis/treatment regex excludes stories from train and new-domain validation; trailing Word count metadata removed; known malformed story00469 excluded after excerpt review; sampled normalized64-character overlap removes training docs; not semantic or factual review", "review_scope": "fiction sources inherited; 8 stratified training excerpts sampled, not full factual review; stories GPT3.5 synthetic, CC-BY-4.0 as source card; novels user-supplied classical text", "traditional_simplified": "original scripts retained, no conversion", "test_contents_opened": False}  # 不将基础文学文本称为已核实事实问答。
    persist(LANGUAGE + "/manifest.json", dump(manifest))  # 数据与采样配比冻结。
    persist(LANGUAGE + "/excerpt_queue.json", dump(audit))  # 待助手实际阅读的片段。
    return manifest  # 不训练模型。


def build_curriculum():  # 原创模板审阅加全量可验证标签，不冒充逐条人工审阅。
    tokenizer, rng, rows, seen = load_tokenizer(), random.Random(654), [], set()  # 固定全部任务输入。
    families = ("add", "subtract", "multiply", "gcd", "ascending", "descending", "reverse", "count", "unique", "minmax")  # 十类确定性中文任务。
    for family in families:  # 每类80个不同输入，共800条原始问答。
        made = 0  # 本类独立输入计数。
        while made < 80:  # 重复输入不会增加样本数。
            if family in families[:4]:  # 算术任务的两个操作数。
                payload = [rng.randint(1, 99), rng.randint(1, 99)]  # 排除乘法除零核验歧义。
                a, b = payload  # 明确两个输入。
                group = "numbers:" + json.dumps(payload)  # 相同操作数跨题型也保留同组。
                if family == "add":  # 加法任务使用整数求和。
                    value = a + b  # 保存数学结果，不拼接字符串冒充相加。
                elif family == "subtract":  # 减法允许负数。
                    value = a - b  # 运算顺序必须与问题一致。
                elif family == "multiply":  # 乘法生成精确整数。
                    value = a * b  # 不使用浮点近似。
                else:  # 第四类为最大公约数。
                    value = math.gcd(a, b)  # 生成时用标准库，验证时另枚举约数。
                prompt = {"add": f"计算{a}与{b}的和，只给出结果。", "subtract": f"计算{a}减去{b}，只给出结果。", "multiply": f"计算{a}乘以{b}，只给出结果。", "gcd": f"求{a}与{b}的最大公约数，只给出结果。"}[family]  # 模板含清楚约束。
                answer = str(value)  # 不夹带无法核实的解释。
            elif family in ("reverse", "count"):  # 按Unicode码点而非BPE token操作。
                payload = "".join(rng.choice(("春风", "白云", "小河", "树林", "星光", "花朵", "清晨", "雨滴")) for _ in range(rng.randint(2, 6)))  # 无组合字符或代理对歧义。
                group = "text:" + payload  # 同一文本的反转和计数不拆组。
                prompt = f"将文字逐字倒序，只给出结果：{payload}" if family == "reverse" else f"计算文字的汉字个数，只给出整数：{payload}"  # 明确计数对象。
                answer = payload[::-1] if family == "reverse" else str(len(payload))  # 每个字符都是单个汉字码点。
            else:  # 数列排序、保持顺序去重及极值。
                payload = [rng.randint(1, 30) for _ in range(rng.randint(4, 7))]  # 允许重复值以检查重数及去重。
                group = "sequence:" + json.dumps(payload)  # 同数列的多种指令不跨训练验证。
                prompt = {"ascending": "把数列按升序排列，只输出JSON数组：", "descending": "把数列按降序排列，只输出JSON数组：", "unique": "按首次出现顺序去重，只输出JSON数组：", "minmax": "找出最小值和最大值，依次输出JSON数组："}[family] + json.dumps(payload, ensure_ascii=False)  # 格式要求可程序核验。
                if family == "ascending":  # 从小到大，重复值仍保留。
                    value = sorted(payload)  # 标签为排序后的完整数列。
                elif family == "descending":  # 从大到小，仍保持元素重数。
                    value = sorted(payload, reverse=True)  # 仅改变顺序。
                elif family == "unique":  # 只保留每个值第一次出现的位置。
                    value = list(dict.fromkeys(payload))  # 字典插入顺序保留首现顺序。
                else:  # 极值任务只输出最小、最大两个数。
                    value = [min(payload), max(payload)]  # 即使两者相同，也保留两个位置。
                answer = json.dumps(value, ensure_ascii=False)  # 保留数字类型与数组结构。
            key = (family, group)  # 同族同输入不能重复生成。
            if key in seen:  # 随机重复时重新抽取。
                continue  # 不将重复行算成新任务覆盖。
            seen.add(key)  # 提交唯一输入。
            row = {"id": f"curriculum54:{family}:{made:03d}", "family": family, "payload": payload, "group_id": sha(group.encode("utf-8")), "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], "label_review": "authored template review plus full programmatic invariant verification; not external human annotation"}  # 保留可独立复核的原始输入。
            verify_task(row)  # 所有800条都检查，不只抽查标签。
            encode_dialogue(row["messages"], tokenizer)  # 符合真实SFT角色和长度要求。
            rows.append(row)  # 数据准备不意味着本课已SFT。
            made += 1  # 本类成功新增一条。
    groups = sorted({row["group_id"] for row in rows})  # 同底层输入整组划分。
    random.Random(754).shuffle(groups)  # 固定组级随机顺序。
    mapping = {g: "train" if i < len(groups) * 8 // 10 else "validation" for i, g in enumerate(groups)}  # 验证为未见输入，不是未见模板。
    stats, files = {}, {}  # 记录真正的独立组与任务种类。
    for split in ("train", "validation"):  # 两边保留十类任务覆盖。
        selected = [row for row in rows if mapping[row["group_id"]] == split]  # 不按模型表现移动样本。
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected).encode("utf-8")  # 输出完整问答及校验依据。
        persist(f"curriculum/{split}.jsonl", payload)  # 不覆盖第53课语料。
        files[split + ".jsonl"] = sha(payload)  # 固定数据内容。
        stats[split] = {"rows": len(selected), "groups": len({r["group_id"] for r in selected}), "families": dict(Counter(r["family"] for r in selected))}  # 变种或共有输入不算新组。
        assert len(stats[split]["families"]) == 10, "某集合缺少任务族"  # 不伪装成覆盖十类。
    manifest = {"format": "verified_curriculum54_v1", "authorship": "assistant-authored Chinese templates and deterministic generators", "review_scope": "all ten templates reviewed; every label verified by invariants; no claim of 800 independent manual reviews", "stats": stats, "files": files, "seed_generation": 654, "seed_group_split": 754, "same_templates_in_both_splits": True, "validation_scope": "new payloads within known task templates; not general chat test", "used_for_sft_in_lesson54": False}  # 先扩覆盖，再另设SFT预算。
    persist("curriculum/manifest.json", dump(manifest))  # 新任务课程可重建。
    return manifest  # 不触发模型更新。


def verify_task(row):  # 从原始输入独立核对可验证标签。
    family, payload, answer = row["family"], row["payload"], row["messages"][-1]["content"]  # 不用生成时的中间value。
    if family in ("add", "subtract", "multiply", "gcd"):  # 算术通过逆运算或约数定义检查。
        a, b = payload  # 使用原操作数。
        value = int(answer)  # 标签必须是整数。
        if family == "add":  # 和减去一个加数应恢复另一个加数。
            assert value - a == b, "加法标签错误"  # 用逆运算核对。
        elif family == "subtract":  # 差加回减数应恢复被减数。
            assert value + b == a, "减法标签错误"  # 不混淆a-b和b-a。
        elif family == "multiply":  # 正整数乘积除以一个因数应恢复另一个。
            assert value % a == 0 and value // a == b, "乘法标签错误"  # 同时检查整除，不能只向下取整。
        else:  # 从约数定义核验最大公约数。
            divisors = [d for d in range(1, min(a, b) + 1) if a % d == 0 and b % d == 0]  # 所有公共正约数。
            assert value == max(divisors), "最大公约数标签错误"  # 验证时不复用生成器的math.gcd。
    elif family == "reverse":  # 反转两次应恢复原文。
        assert answer[::-1] == payload, "倒序标签错误"  # 不能只看输出长度。
    elif family == "count":  # 明确按汉字码点计数。
        assert int(answer) == sum(1 for char in payload if "\u4e00" <= char <= "\u9fff"), "计数标签错误"  # 不把token数当字数。
    else:  # 数列任务用元素守恒、单调性及集合约束核验。
        value = json.loads(answer)  # 严格要求JSON数组。
        if family in ("ascending", "descending"):  # 排序必须保留全部元素重数。
            assert Counter(value) == Counter(payload) and all(a <= b if family == "ascending" else a >= b for a, b in zip(value, value[1:])), "排序标签错误"  # 不照抄生成器sorted作为唯一核验。
        elif family == "unique":  # 去重后集合不变且首现顺序保持。
            assert len(value) == len(set(value)) and set(value) == set(payload) and all(payload.index(a) < payload.index(b) for a, b in zip(value, value[1:])), "去重标签错误"  # 不允许重排或漏值。
        else:  # 最小最大必须来自输入并包围所有值。
            assert len(value) == 2 and all(x in payload for x in value) and all(value[0] <= x <= value[1] for x in payload), "极值标签错误"  # 不依赖生成器min/max调用。


def build_catalog():  # 将旧一般问答和新任务登记为可追溯的分开组件，不启动SFT。
    components, prompts = {}, {split: set() for split in ("train", "validation")}  # 两种数据不按行数盲目合并采样。
    for split in ("train", "validation"):  # 保留所有已有划分。
        entries = []  # 每种组件有独立来源与标签审核方式。
        for kind, relative in (("general", f"data/chat53/frozen_v1/{split}.jsonl"), ("verified_tasks", f"data/chat54/curriculum/{split}.jsonl")):  # 不碰旧测试集。
            path = ROOT / relative  # 已有冻结原文件。
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]  # 不改原消息。
            entries.append({"kind": kind, "path": relative, "sha256": sha(path.read_bytes()), "rows": len(rows), "groups": len({row["group_id"] for row in rows})})  # 组数不同于独立知识数。
            prompts[split].update(normalize(next(m["content"] for m in row["messages"] if m["role"] == "user")) for row in rows)  # 只核对完全相同的主问题。
        components[split] = entries  # 下一阶段可明确声明抽样配比。
    assert not prompts["train"] & prompts["validation"], "目录组件出现相同首问跨集合"  # 这不是完整语义等价检测。
    manifest = {"format": "expanded_sft54_catalog_v1", "components": components, "suggested_future_sampling": {"general": 0.5, "verified_tasks": 0.5, "within_verified_tasks": "balance ten families rather than uniform-over-all-rows dominating general dialogue"}, "used_for_sft_this_stage": False, "overlap_check": "normalized exact first user prompt only; preserves all parent splits and groups; semantic/commutative-equivalent cross-source overlap not guaranteed absent", "review_scope": "legacy source53 explicit assistant reviews plus new ten authored templates and every algorithmic label checked; not 800 independent human reviews", "validation_scope": "known templates new ordered payloads; not a final test of general chat", "additional_source": None}  # 不将目录准备说成已完成新微调。
    persist("sft_catalog.json", dump(manifest))  # 重跑应重建同一目录。
    return manifest  # 不加载或更新模型。


if __name__ == "__main__":  # 导入核验函数不生成或训练。
    print("语言混合：", build_language()["stats"], flush=True)  # 显示新增来源与实际保留规模。
    print("可验证中文任务：", build_curriculum()["stats"], flush=True)  # 不把数据准备说成已经SFT。
    build_catalog()  # 明确登记扩展组件与后续平衡采样建议。
