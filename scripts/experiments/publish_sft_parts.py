import argparse  # 发布与上传分开执行。
import json  # 保存可重组的完整断点分片清单。
import os  # 禁止凭据交互。
import subprocess  # 仅在内存读取既有Git授权。
import urllib.parse  # 安全编码附件名。
import urllib.request  # 与GitHub官方API通信。
from prepare_chat_corpus import ROOT, sha, dump  # 固定文件摘要及JSON格式。
REPO, TAG = "codesknight/playATryModel", "lesson53-sft-pilot"  # 只操作已授权的本课仓库版本。


def main(publish=False):  # 大状态分32MiB片，单片失败不重传所有成功片。
    os.environ["GIT_TERMINAL_PROMPT"], os.environ["GCM_INTERACTIVE"] = "0", "Never"  # 不弹凭据窗口。
    result = subprocess.run(["git", "-c", "credential.interactive=false", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", text=True, capture_output=True, check=True, timeout=30)  # 复用本仓库已有授权。
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)  # 不打印或保存凭据。
    headers = {"Authorization": "Bearer " + fields["password"], "Accept": "application/vnd.github+json", "User-Agent": "playATryModel-sft53"}  # 凭据只发往GitHub。
    def api(path, method="GET", payload=None):  # 限定本仓库API。
        request = urllib.request.Request(f"https://api.github.com/repos/{REPO}/{path}", data=None if payload is None else json.dumps(payload).encode(), headers={**headers, "Content-Type": "application/json"}, method=method)  # 使用结构化参数。
        with urllib.request.urlopen(request, timeout=40) as response:  # 网络失败明确报错。
            data = response.read()  # 小JSON回复。
        return json.loads(data) if data else None  # DELETE可能没有正文。
    releases = api("releases?per_page=100")  # 草稿也可见，支持断点重跑上传。
    release = next((r for r in releases if r["tag_name"] == TAG), None)  # 不操作旧课Release。
    if release is None:  # 首次创建草稿。
        release = api("releases", "POST", {"tag_name": TAG, "target_commitish": "main", "name": "Lesson53 bounded Chinese SFT full state", "draft": True, "body": "Educational noncommercial research SFT pilot, not usable chat quality. Full best and last FP32/AdamW/Scaler/RNG/cursor states split into 32MiB parts. See lesson53_assets.json and restore_sft_parts.py for verified reconstruction. Source restrictions and hashes remain in course notes."})  # 不夸大模型能力。
    report = json.loads((ROOT / "docs/training_runs/lesson53.json").read_text(encoding="utf-8"))  # 只上传已完成正式训练文件。
    assert report["status"] == "completed", "正式SFT未完成"  # 不上传仍在变化的文件。
    existing = {item["name"]: item for item in api(f"releases/{release['id']}/assets")}  # 已成功分片可复用。
    for name in ("last_train.pt", "best_train.pt"):  # 清理本课失败的整文件上传占位，不删除成功附件。
        if name in existing and existing[name].get("state") != "uploaded":  # 网络重置可能留下starter。
            api(f"releases/assets/{existing[name]['id']}", "DELETE")  # 仅清理本课两个明确文件的失败对象。
    directory = ROOT / "model/lesson53/parts"  # 分片仅本地缓存，不放Git。
    directory.mkdir(exist_ok=True)  # 原完整状态保留。
    files = {}  # 记录每个逻辑断点的全文件摘要和有序分片。
    for name in ("last_train.pt", "best_train.pt"):  # 两者不同，不能复用错误的别名。
        path, artifact = directory.parent / name, report["artifacts"][name]  # 只接受本课正式文件。
        assert path.stat().st_size == artifact["bytes"] and sha(path.read_bytes()) == artifact["sha256"], "本地完整文件改变"  # 上传前核对全文件。
        parts = []  # 顺序即重组顺序。
        with path.open("rb") as handle:  # 流式拆分不复制整个文件到内存。
            while chunk := handle.read(32 * 1024 * 1024):  # 单个附件最多32MiB。
                part_name = f"{name}.part{len(parts) + 1:02d}"  # 稳定连续编号。
                part_hash = sha(chunk)  # 每片独立摘要。
                part_path = directory / part_name  # 便于离线验证重组。
                if part_path.exists():  # 已有分片不静默覆盖。
                    assert sha(part_path.read_bytes()) == part_hash, "本地分片不同"  # 与原文件必须一致。
                else:  # 初次拆分才写入。
                    part_path.write_bytes(chunk)  # 二进制原字节。
                current = existing.get(part_name)  # 已上传同片可跳过。
                if current and current.get("state") != "uploaded":  # 失败片只重试该片。
                    api(f"releases/assets/{current['id']}", "DELETE")  # 不删除其他成功分片。
                    current = None  # 后续重新上传。
                if current is None:  # 仅上传缺失分片。
                    url = release["upload_url"].split("{")[0] + "?name=" + urllib.parse.quote(part_name)  # 使用API返回地址。
                    assert urllib.parse.urlsplit(url).hostname == "uploads.github.com", "上传域名异常"  # 不向其他域名发送授权。
                    request = urllib.request.Request(url, data=chunk, headers={**headers, "Content-Type": "application/octet-stream"}, method="POST")  # 单片有界读取和发送。
                    with urllib.request.urlopen(request, timeout=180) as response:  # 失败退出，重跑可跳过成功片。
                        current = json.load(response)  # 服务器确认上传成功。
                assert current["size"] == len(chunk) and current.get("digest") == "sha256:" + part_hash, "远端分片摘要错误"  # 不只根据本地发送进度判完成。
                parts.append({"name": part_name, "bytes": len(chunk), "sha256": part_hash, "asset_id": current["id"], "url": current["browser_download_url"], "remote_digest": current["digest"]})  # 不保存凭据。
                print("已核验分片：", part_name, flush=True)  # 有限且明确的上传进度。
        files[name] = {**artifact, "parts": parts}  # 全文件摘要防止拼错顺序。
    if publish and release["draft"]:  # 代码推送后才公开发布完整附件。
        release = api(f"releases/{release['id']}", "PATCH", {"draft": False})  # 用户已授权每课交付。
    remote = {item["id"]: item for item in api(f"releases/{release['id']}/assets")}  # 发布后刷新永久URL。
    for artifact in files.values():  # 不保留草稿临时下载地址。
        for part in artifact["parts"]:  # 所有完整状态都刷新。
            part["url"] = remote[part["asset_id"]]["browser_download_url"]  # 公开标签地址。
    record = {"format": "chat_sft53_release_parts_v1", "tag": TAG, "url": f"https://github.com/{REPO}/releases/tag/{TAG}", "draft": release["draft"], "files": files}  # 完整重组所需清单。
    (ROOT / "docs/training_runs/lesson53_assets.json").write_bytes(dump(record))  # 保存全部远端摘要。
    print("完整状态分片就绪；draft：", release["draft"], flush=True)  # 未公开时不假称可下载。


if __name__ == "__main__":  # 导入不会访问凭据。
    parser = argparse.ArgumentParser()  # 显式发布选项。
    parser.add_argument("--publish", action="store_true")  # 默认只上传草稿。
    main(parser.parse_args().publish)  # 网络失败可重跑，不重传成功分片。
