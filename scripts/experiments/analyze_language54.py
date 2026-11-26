import json  # 保存实际导出文件的全部固定续写。
from train_language54 import load_data, OUT, DIRECTORY, ROOT  # 使用同一语料和结构。
from chat_model import ChatLM, torch  # 只读评估，不更新参数。
from train_chat_pilot import generate  # 与阶段记录相同固定解码规则。
from prepare_chat_corpus import dump, sha  # 结果以实际文件摘要为证。


def main():  # 观察FP16导出后的输出，不挑好句子。
    torch.set_num_threads(4)  # 限制CPU线程。
    arrays, tokenizer, manifest, tasks = load_data()  # 确认所有来源和验证版本。
    state = torch.load(OUT / "base_fp16.pt", map_location="cpu", weights_only=True)  # 实际交付的最佳文件。
    model = ChatLM(**state["model_config"]).cuda()  # 不载入或更新优化器。
    model.load_state_dict(state["model_state"])  # 复算舍入后的真实参数。
    samples = [generate(model, tokenizer, prompt, greedy=greedy) for prompt in ("清晨，小明走进教室，", "学习新知识时，", "春风") for greedy in (True, False)]  # 保留全部六个固定输出。
    for item in samples:  # 原文包括重复、语义跳跃和解码替换符。
        item["replacement_characters"] = item["text"].count("\ufffd")  # 记录生成或截断解码问题，不归咎于未查证的训练文件乱码。
        print(item["mode"], repr(item["text"]), flush=True)  # 不手工润色输出。
    course = [json.loads(line) for split in ("train", "validation") for line in (DIRECTORY / "curriculum" / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()]  # 统计实际边界样例覆盖。
    edges = {"negative_subtractions": sum(r["family"] == "subtract" and r["payload"][0] < r["payload"][1] for r in course), "gcd_greater_than_one": sum(r["family"] == "gcd" and int(r["messages"][-1]["content"]) > 1 for r in course), "sequence_tasks_with_repeats": sum(r["family"] in ("ascending", "descending", "unique", "minmax") and len(r["payload"]) != len(set(r["payload"])) for r in course)}  # 类别数不等于边界覆盖。
    result = {"export_sha256": sha((OUT / "base_fp16.pt").read_bytes()), "samples": samples, "curriculum_edge_coverage": edges, "assessment": "fixed continuations still repeat and jump semantically; lower language validation loss is not usable chat; task set prepared but not SFT-trained; replacement character cause not fully diagnosed", "causal_limit": "mixed corpus, lower learning rate and microbatch changed together; no equal-budget old-only control, so improvements cannot be attributed solely to data expansion", "formal_additional_updates": 0}  # 不把模型选择后观察当独立测试。
    (ROOT / "docs/training_runs/lesson54_analysis.json").write_bytes(dump(result))  # 不修改正式模型或训练计数。


if __name__ == "__main__":  # 导入不加载GPU。
    main()  # 显式运行仅做生成与数据统计。
