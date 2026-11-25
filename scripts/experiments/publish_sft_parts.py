import argparse  # 发布与上传分开执行。
import json  # 保存可重组的完整断点分片清单。
import os  # 禁止凭据交互。
import subprocess  # 仅在内存读取既有Git授权。
import urllib.parse  # 安全编码附件名。
import urllib.request  # 与GitHub官方API通信。
from concurrent.futures import ThreadPoolExecutor, as_completed  # 独立分片最多三条连接并发，出错及时取消排队任务。
from prepare_chat_corpus import ROOT, sha, dump  # 固定文件摘要及JSON格式。
REPO, TAG = "codesknight/playATryModel", "lesson53-sft-pilot"  # 只操作已授权的本课仓库版本。


def main(publish=False):  # 大状态分8MiB片，单片失败不重传所有成功片。
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
        release = api("releases", "POST", {"tag_name": TAG, "target_commitish": "main", "name": "Lesson53 bounded Chinese SFT full state", "draft": True, "body": "Educational noncommercial research SFT pilot, not usable chat quality. Full best and last FP32/AdamW/Scaler/RNG/cursor states split into 8MiB parts. See lesson53_assets.json and restore_sft_parts.py for verified reconstruction. Source restrictions and hashes remain in course notes."})  # 不夸大模型能力。
    report = json.loads((ROOT / "docs/training_runs/lesson53.json").read_text(encoding="utf-8"))  # 只上传已完成正式训练文件。
    assert report["status"] == "completed", "正式SFT未完成"  # 不上传仍在变化的文件。
    def all_assets():  # 分片超过API默认30项时必须分页。
        collected, page = [], 1  # 不漏掉后面的成功片。
        while True:  # 按GitHub返回结果逐页读取。
            batch = api(f"releases/{release['id']}/assets?per_page=100&page={page}")  # 最大100项一页。
            collected.extend(batch)  # 合并完整附件列表。
            if len(batch) < 100:  # 最后一页已经读完。
                return collected  # 返回可复用的完整附件集合。
            page += 1  # 再读下一页。
    existing = {item["name"]: item for item in all_assets()}  # 已成功分片可复用。
    for name in ("last_train.pt", "best_train.pt"):  # 清理本课失败的整文件上传占位，不删除成功附件。
        if name in existing and existing[name].get("state") != "uploaded":  # 网络重置可能留下starter。
            api(f"releases/assets/{existing[name]['id']}", "DELETE")  # 仅清理本课两个明确文件的失败对象。
    directory = ROOT / "model/lesson53/parts"  # 分片仅本地缓存，不放Git。
    directory.mkdir(exist_ok=True)  # 原完整状态保留。
    def upload_part(part_name, part_path, part_hash, size):  # 每片独立，可并发且可恢复。
        current = existing.get(part_name)  # 成功片直接核验并复用。
        if current and current.get("state") != "uploaded":  # 仅删除本片失败占位。
            api(f"releases/assets/{current['id']}", "DELETE")  # 不触碰其他片。
            current = None  # 重新发送当前片。
        if current is None:  # 最多三片同时各自读取8MiB。
            url = release["upload_url"].split("{")[0] + "?name=" + urllib.parse.quote(part_name)  # 固定官方上传域名。
            assert urllib.parse.urlsplit(url).hostname == "uploads.github.com", "上传域名异常"  # 凭据不外传。
            request = urllib.request.Request(url, data=part_path.read_bytes(), headers={**headers, "Content-Type": "application/octet-stream"}, method="POST")  # 每片独立请求。
            with urllib.request.urlopen(request, timeout=180) as response:  # 单片失败明确退出，重跑复用成功片。
                current = json.load(response)  # 服务器回执。
        assert current["size"] == size and current.get("digest") == "sha256:" + part_hash, "远端分片摘要错误"  # 核对真实远端摘要。
        print("已核验分片：", part_name, flush=True)  # 并发完成顺序不代表重组顺序。
        return {"name": part_name, "bytes": size, "sha256": part_hash, "asset_id": current["id"], "url": current["browser_download_url"], "remote_digest": current["digest"]}  # 最后按编号顺序归档。
    pool = ThreadPoolExecutor(max_workers=3)  # 限制并发与内存，不无限启动线程。
    files = {}  # 记录每个逻辑断点的全文件摘要和有序分片。
    for name in ("last_train.pt", "best_train.pt"):  # 两者不同，不能复用错误的别名。
        path, artifact = directory.parent / name, report["artifacts"][name]  # 只接受本课正式文件。
        assert path.stat().st_size == artifact["bytes"] and sha(path.read_bytes()) == artifact["sha256"], "本地完整文件改变"  # 上传前核对全文件。
        parts = []  # 顺序即重组顺序。
        with path.open("rb") as handle:  # 流式拆分不复制整个文件到内存。
            while chunk := handle.read(8 * 1024 * 1024):  # 单个附件最多8MiB。
                part_name = f"{name}.part8m{len(parts) + 1:03d}"  # 与失败的旧大分片分开命名，稳定连续编号。
                part_hash = sha(chunk)  # 每片独立摘要。
                part_path = directory / part_name  # 便于离线验证重组。
                if part_path.exists():  # 已有分片不静默覆盖。
                    assert sha(part_path.read_bytes()) == part_hash, "本地分片不同"  # 与原文件必须一致。
                else:  # 初次拆分才写入。
                    part_path.write_bytes(chunk)  # 二进制原字节。
                parts.append(pool.submit(upload_part, part_name, part_path, part_hash, len(chunk)))  # 请求独立，重组顺序仍由列表固定。
        files[name] = {**artifact, "parts": parts}  # 全文件摘要防止拼错顺序。
    try:  # 网络出错时不继续等待全部排队上传逐个超时。
        for future in as_completed([task for artifact in files.values() for task in artifact["parts"]]):  # 实际完成顺序可不同。
            future.result()  # 立刻发现任一失败。
    except BaseException:  # 失败或用户中断均收尾线程。
        pool.shutdown(wait=True, cancel_futures=True)  # 取消未启动任务，成功片留给重跑复用。
        raise  # 不把不完整附件发布为完成。
    pool.shutdown(wait=True)  # 全部成功后关闭线程，不留后台任务。
    for artifact in files.values():  # 将有序任务转成有序远端回执。
        artifact["parts"] = [future.result() for future in artifact["parts"]]  # 任一失败不发布不完整版本。
    if publish and release["draft"]:  # 代码推送后才公开发布完整附件。
        release = api(f"releases/{release['id']}", "PATCH", {"draft": False})  # 用户已授权每课交付。
    remote = {item["id"]: item for item in all_assets()}  # 发布后分页刷新全部永久URL。
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
