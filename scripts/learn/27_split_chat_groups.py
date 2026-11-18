import argparse  # 明确输入文件、输出版本目录和随机种子。
import hashlib  # 保存输入与输出的内容指纹。
import json  # 读取对话并保存分组映射。
from pathlib import Path  # 相对脚本定位项目根目录。
import random  # 使用固定种子的独立随机数生成器。
ROOT = Path(__file__).resolve().parents[2]  # 当前脚本位于scripts/learn。
SPLITS = ("train", "validation", "test")  # 同一组只能属于其中一个集合。


def make_split(samples, seed=42):  # 根据已给出的group_id整组划分，不自动判断文本相似性。
    assert samples, "输入样本不能为空"  # 没有数据不能建立划分。
    for row in samples:  # 先检查划分所必需的字段。
        assert isinstance(row["id"], str) and row["id"] and isinstance(row["group_id"], str) and row["group_id"], "标识必须是非空字符串"  # 稳定标识不能用临时行号冒充。
        assert row["messages"] and all(isinstance(message["content"], str) for message in row["messages"]), "消息字段无效"  # 本课假设角色与内容质量已在上游审核。
    assert len({row["id"] for row in samples}) == len(samples), "样本id重复"  # 重复标识会破坏覆盖检查。
    groups = sorted({row["group_id"] for row in samples})  # 先排序，消除集合遍历顺序的不确定性。
    assert len(groups) >= 10, "当前80/10/10演示至少需要10组"  # 避免小样本取整使验证或测试为空。
    random.Random(seed).shuffle(groups)  # 不影响其他模块的全局随机状态。
    train_end, validation_end = len(groups) * 8 // 10, len(groups) * 9 // 10  # 比例按组数计算，整数除法向下取整。
    mapping = {group: "train" if i < train_end else "validation" if i < validation_end else "test" for i, group in enumerate(groups)}  # 每个组只分配一次集合。
    partitions = {split: [] for split in SPLITS}  # 分别收集三个集合。
    for row in sorted(samples, key=lambda item: item["id"]):  # 保存前按样本id排序，让输入行顺序变化不影响结果。
        partitions[mapping[row["group_id"]]].append(row)  # 所有同组样本使用同一个映射。
    group_sets = [{row["group_id"] for row in partitions[split]} for split in SPLITS]  # 检查的是组归属，不只是样本编号。
    assert not any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)), "不同集合存在组交集"  # 三个集合两两互斥。
    assert {row["id"] for split in SPLITS for row in partitions[split]} == {row["id"] for row in samples}, "样本覆盖不完整"  # 不遗漏或额外制造样本。
    return mapping, partitions  # 保留映射供未来新增样本查询已有组。


def encode_json(value):  # 统一JSON字节格式避免Windows/Linux换行差异。
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")  # 后续用write_bytes写入，不转换成CRLF。


def save_version(samples, source_bytes, output, seed=42):  # 保存或复现一个固定的数据划分版本。
    mapping, partitions = make_split(samples, seed)  # 先检查输入和所有分组约束。
    source_hash = hashlib.sha256(source_bytes).hexdigest()  # 输入内容或序列化变化都形成新的来源版本。
    manifest_path = output / "manifest.json"  # 该版本的来源和配置说明。
    if manifest_path.exists():  # 已存在的版本不能被新数据静默覆盖。
        old = json.loads(manifest_path.read_text(encoding="utf-8"))  # 读取已冻结的版本信息。
        assert old["source_sha256"] == source_hash and old["seed"] == seed, "数据或种子改变，请使用新的输出版本目录"  # 本课不自动重新分配旧版本。
    payloads = {"group_to_split.json": encode_json(mapping)}  # 映射与划分文件一起留存。
    for split in SPLITS:  # 一条完整对话写成一行JSON。
        payloads[f"{split}.jsonl"] = ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in partitions[split])).encode("utf-8")  # 每份内容可独立计算指纹。
    stats = {split: {"groups": len({row["group_id"] for row in partitions[split]}), "samples": len(partitions[split]), "characters": sum(len(message["content"]) for row in partitions[split] for message in row["messages"])} for split in SPLITS}  # 字符数包含用户和助手正文，不是分词后的token数。
    manifest = {"format": "chat_group_split_v1", "purpose": "format_demo_not_real_training", "source_sha256": source_hash, "seed": seed, "unit": "group_id", "ratios_by_group": [0.8, 0.1, 0.1], "stats": stats, "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}}  # 记录实际比例与字节证据。
    if manifest_path.exists():  # 同一版本重跑时必须得到完全相同的内容。
        assert old == manifest, "同版本重跑结果改变，请检查方法或环境"  # 不把复现失败悄悄写成新结果。
        assert all((output / name).read_bytes() == payload for name, payload in payloads.items()), "已有产物被改动，停止覆盖"  # 先报错保留现场。
    else:  # 第一次创建时写入新目录。
        output.mkdir(parents=True, exist_ok=True)  # 输出只位于明确指定的版本目录。
        assert all(not (output / name).exists() for name in payloads), "目录已有未登记文件，请换新版本目录"  # 不覆盖没有清单的残留文件。
        for name, payload in payloads.items():  # 所有JSON字节使用LF固定格式。
            (output / name).write_bytes(payload)  # 保证保存后指纹与清单一致。
        manifest_path.write_bytes(encode_json(manifest))  # 最后写入版本清单。
    return manifest  # 返回统计供学习者观察组数与条数的区别。


def main():  # 默认只处理独立教学示例，不接触正式语料。
    parser = argparse.ArgumentParser()  # 学习者可为新数据指定新版本目录。
    parser.add_argument("--input", type=Path, default=ROOT / "data/group_demo/samples.jsonl")  # 示例原始会话。
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/group_demo/splits_v1")  # 首个冻结划分版本。
    parser.add_argument("--seed", type=int, default=42)  # 数据划分种子与模型初始化种子各自管理。
    args = parser.parse_args()  # 开始前固定本次设置。
    source = args.input.read_bytes()  # 保留输入的实际字节版本。
    samples = [json.loads(line) for line in source.decode("utf-8-sig").splitlines() if line.strip()]  # 一条JSON是一段完整会话。
    manifest = save_version(samples, source, args.output_dir, args.seed)  # 保存映射、划分和指纹，或只读核验已冻结版本。
    for split in SPLITS:  # 三个集合都打印，不能只展示训练集。
        item = manifest["stats"][split]  # 当前集合真实统计。
        print(split, "组数：", item["groups"], "样本数：", item["samples"], "正文字符数：", item["characters"])  # 组数比例不等于样本数比例。
    print("组交集、样本覆盖及版本复现检查：通过")  # 本脚本没有训练模型。
    print("输出目录：", args.output_dir)  # 新版本应显式使用另一个目录。


if __name__ == "__main__":  # 导入测试工具时不写文件。
    main()  # 运行本课实操。
