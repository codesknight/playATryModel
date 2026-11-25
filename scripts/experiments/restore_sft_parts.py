import argparse  # 选择需要恢复的完整SFT检查点。
import hashlib  # 流式核对重组后的完整摘要。
import json  # 读取已发布的分片清单。
from pathlib import Path  # 使用明确的下载缓存和输出目录。
import urllib.request  # 仅显式要求时下载公开附件。
from prepare_chat_corpus import ROOT, sha  # 项目路径与单片摘要。


def reconstruct(name, directory, output, download=False):  # 不加载模型、不执行训练。
    assets = json.loads((ROOT / "docs/training_runs/lesson53_assets.json").read_text(encoding="utf-8"))  # 已验证远端摘要。
    artifact = assets["files"][name]  # 仅两个明确完整状态之一。
    assert not output.exists(), "输出已存在，拒绝覆盖完整检查点"  # 避免破坏正式最后进度。
    directory.mkdir(parents=True, exist_ok=True)  # 明确缓存目录。
    digest, size = hashlib.sha256(), 0  # 完整字节串摘要必须与报告一致。
    for part in artifact["parts"]:  # 按清单顺序先核验全部分片。
        path = directory / part["name"]  # 手工下载或先前缓存的分片。
        if not path.exists() and download:  # 只有显式参数才连接网络。
            assert not assets["draft"] and part["url"].startswith("https://github.com/codesknight/playATryModel/releases/download/lesson53-sft-pilot/"), "不是本课公开附件"  # 不向未知来源下载。
            with urllib.request.urlopen(part["url"], timeout=120) as response:  # 单片最多32MiB。
                data = response.read(part["bytes"] + 1)  # 不无限下载。
            assert len(data) == part["bytes"] and sha(data) == part["sha256"], "下载分片摘要错误"  # 核验后才缓存。
            path.write_bytes(data)  # 保留可复用片。
        assert path.is_file() and path.stat().st_size == part["bytes"] and sha(path.read_bytes()) == part["sha256"], f"缺失或损坏分片：{part['name']}"  # 不生成看似完整的坏文件。
    output.parent.mkdir(parents=True, exist_ok=True)  # 不移动或删除任何旧状态。
    with output.open("xb") as handle:  # 明确拒绝覆盖现有文件。
        for part in artifact["parts"]:  # 二进制顺序拼回原字节。
            data = (directory / part["name"]).read_bytes()  # 单片读取上限32MiB。
            handle.write(data)  # 不做文本换行转换。
            digest.update(data)  # 完整文件再次校验。
            size += len(data)  # 记录总大小。
    assert size == artifact["bytes"] and digest.hexdigest() == artifact["sha256"], "完整重组摘要错误"  # 保证与原360MB文件逐字节一致。
    return {"bytes": size, "sha256": digest.hexdigest()}  # 只返回校验结果，不启动恢复训练。


if __name__ == "__main__":  # 导入可供本地离线重组测试。
    parser = argparse.ArgumentParser()  # 指定缓存、目标文件及是否下载。
    parser.add_argument("--checkpoint", choices=("best_train.pt", "last_train.pt"), default="best_train.pt")  # 默认取最佳100步。
    parser.add_argument("--directory", type=Path, default=ROOT / "model/lesson53/parts")  # 缓存可重复使用。
    parser.add_argument("--output", type=Path, required=True)  # 必须显式决定新文件保存位置。
    parser.add_argument("--download", action="store_true")  # 默认离线，不偷偷下载数百MB。
    args = parser.parse_args()  # 明确本次仅重组文件。
    print(reconstruct(args.checkpoint, args.directory, args.output, args.download))  # 完整核验后才报告成功。
