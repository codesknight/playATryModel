import codecs  # 在下载字节截断处安全解码UTF-8。
import csv  # 读取带引号和换行的故事CSV。
import hashlib  # 记录样本文件和来源分组指纹。
import io  # 将有上限的下载内容交给CSV解析器。
import json  # 保存原始索引、分组及来源记录。
from pathlib import Path  # 样本单独存放，不混入四书训练划分。
import urllib.request  # 只下载固定版本的公开文件。

ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于scripts/experiments。
REVISION = "ad789c9d622bfbdeecdc90df2041b3ba458e0368"  # 固定本次核验的数据集提交。
BASE = "https://huggingface.co/datasets/joe-chiu/TinyChineseStories"  # 原作者的数据集页面。
MAX_BYTES, MAX_ROWS = 1048576, 50  # 只采集格式样本，不下载大规模语料。


def main():  # 采集未来数据实验候选，当前模型不使用这些文本。
    url = f"{BASE}/resolve/{REVISION}/stories.csv"  # 固定版本避免main变化导致不可复现。
    with urllib.request.urlopen(url, timeout=20) as response:  # 设置网络等待上限。
        payload = response.read(MAX_BYTES)  # 最多读取一MiB后关闭连接。
    text = codecs.getincrementaldecoder("utf-8-sig")().decode(payload, final=False)  # 忽略截断处未完成的最后一个字符。
    reader = csv.DictReader(io.StringIO(text))  # 保留CSV字段的完整引用关系。
    rows = []  # 固定保留源文件最前五十条，不能视为随机代表性样本。
    for index, source in enumerate(reader):  # 原始行号用于追溯。
        source = {key.strip(): value for key, value in source.items() if key is not None}  # 原文件部分表头有前导空格。
        story = source.get("story", "")  # 不合并到原训练文本。
        assert story and isinstance(story, str), "故事字段缺失"  # 无有效正文不能作为采集样本。
        grouping = {key: source.get(key, "") for key in ("names", "genre", "theme")}  # 同生成条件的故事未来需要成组划分。
        group_hash = hashlib.sha256(json.dumps(grouping, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()  # 防止相近生成条件泄漏到不同集合。
        rows.append({"source_index": index, "group_sha256": group_hash, "text": story, "generation_conditions": grouping})  # 保留来源条件，不编造人工标签。
        if len(rows) == MAX_ROWS:  # 严格限制采集数量。
            break  # 不为了凑训练量继续下载。
    assert len(rows) == MAX_ROWS, "下载上限内不足五十条完整记录"  # 格式问题应显式暴露。
    directory = ROOT / "data/external_samples/tiny_chinese_stories"  # 使用独立候选目录。
    directory.mkdir(parents=True, exist_ok=True)  # 确保输出位置存在。
    output = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")  # 原文只做JSON封装，不清洗或扩词表。
    (directory / "sample.jsonl").write_bytes(output)  # 原始样本单独保留。
    manifest = {"dataset": "joe-chiu/TinyChineseStories", "revision": REVISION, "source_url": url, "card_url": f"{BASE}/blob/{REVISION}/README.md", "credit": "joe-chiu, TinyChineseStories", "license_declared_in_readme": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "synthetic_source_declared": "GPT-3.5", "rows": len(rows), "download_limit_bytes": MAX_BYTES, "downloaded_prefix_bytes": len(payload), "sample_sha256": hashlib.sha256(output).hexdigest(), "sampling": "first_50_source_rows", "used_in_lesson42_training": False}  # 来源说明按作者数据卡记录，不冒充额外许可核验。
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存可追溯元数据。
    print(json.dumps({"rows": len(rows), "sample_bytes": len(output), "groups": len({row['group_sha256'] for row in rows}), "used_in_training": False}), flush=True)  # 明确候选样本未参与当前对照。


if __name__ == "__main__":  # 直接运行才执行小规模采集。
    main()  # 不是下载或训练整个外部数据集。
