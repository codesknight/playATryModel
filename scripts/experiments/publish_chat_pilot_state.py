import argparse  # 默认只上传草稿附件，推送代码后再显式发布。
import http.client  # 用文件流上传，避免再把360MB状态读入内存。
import json  # 读写GitHub标准API数据。
import os  # 关闭凭据交互，不输出用户凭据。
import subprocess  # 使用已授权的Git凭据管理器。
from time import perf_counter  # 为上传设置单次时长上限。
import urllib.parse  # 处理附件文件名和GitHub上传域名。
import urllib.request  # 创建或更新本仓库的版本附件。
from prepare_chat_corpus import ROOT, dump  # 统一项目路径与JSON字节格式。
REPO, TAG = "codesknight/playATryModel", "lesson48-pretrain-pilot"  # 仅用于本课已经授权上传的仓库。


def main():  # 大完整状态放Release附件，避免GitHub单文件100MB限制。
    parser = argparse.ArgumentParser()  # 发布动作与上传动作可分开恢复。
    parser.add_argument("--publish", action="store_true")  # 代码推送后将草稿转成公开发布。
    args = parser.parse_args()  # 默认不提前发布空版本。
    os.environ["GIT_TERMINAL_PROMPT"], os.environ["GCM_INTERACTIVE"] = "0", "Never"  # 凭据异常时退出，不弹窗口。
    response = subprocess.run(["git", "-c", "credential.interactive=false", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", text=True, capture_output=True, check=True, timeout=30)  # 复用现有仓库授权。
    fields = dict(line.split("=", 1) for line in response.stdout.splitlines() if "=" in line)  # 只在内存中读取凭据。
    token = fields["password"]  # 不打印、不写文件、不提交凭据。
    headers = {"Authorization": "Bearer " + token, "User-Agent": "playATryModel-lesson48", "Accept": "application/vnd.github+json"}  # 只发送到GitHub域名。

    def api(path, method="GET", payload=None):  # 限定本仓库API，避免凭据流向其他来源。
        request = urllib.request.Request("https://api.github.com/repos/" + REPO + "/" + path, data=None if payload is None else json.dumps(payload).encode(), headers={**headers, "Content-Type": "application/json"}, method=method)  # JSON结构化参数。
        with urllib.request.urlopen(request, timeout=40) as result:  # 网络错误直接报告，不假装上传完成。
            data = result.read()  # API回复很小。
        return json.loads(data) if data else None  # DELETE可能没有正文。

    releases = api("releases?per_page=100")  # 草稿对当前仓库授权也可见。
    release = next((item for item in releases if item["tag_name"] == TAG), None)  # 只恢复本课的固定标签。
    if release is None:  # 首次上传先创建草稿。
        release = api("releases", "POST", {"tag_name": TAG, "target_commitish": "main", "name": "Lesson 48: 30M pretraining pilot full state", "draft": True, "body": "Bounded educational pretraining pilot, not an SFT chat model. Full FP32 model, AdamW, GradScaler, RNG and data cursor for resuming. Source corpus terms and hashes are documented in docs/lesson48_data_tokenizer_pilot.md. Inference weights and tokenizer remain in the repository."})  # 不夸大模型能力。
    report = json.loads((ROOT / "docs/training_runs/lesson48.json").read_text(encoding="utf-8"))  # 只有完成后的训练产物才上传。
    assert report["status"] == "completed", "训练尚未完成"  # 不发布正在被覆盖的文件。
    assets, seen = [], {}  # 内容相同的best和last仅上传一份。
    for name in ("last_train.pt", "best_train.pt"):  # 两个完整训练状态，推理文件已在Git管理。
        artifact = report["artifacts"][name]  # 本地实测SHA和大小。
        if artifact["sha256"] in seen:  # 最佳恰好在最后时避免重复360MB。
            assets.append({"local_name": name, "same_as": seen[artifact["sha256"]], **artifact})  # 仍保存两个逻辑文件的对应关系。
            continue  # 不重复传输相同内容。
        path = ROOT / "model/lesson48" / name  # 只允许本课目录的明确文件名。
        assert path.stat().st_size == artifact["bytes"], "本地文件大小变化"  # 上传前不能与报告脱节。
        current = api(f"releases/{release['id']}/assets")  # 可恢复已经成功完成的附件。
        existing = next((item for item in current if item["name"] == name), None)  # 不触碰其他名称的附件。
        if existing and existing.get("state") != "uploaded":  # 失败传输留下的本课未完成附件可以重试。
            api(f"releases/assets/{existing['id']}", "DELETE")  # 仅清理本课同名未完成附件。
            existing = None  # 后续重新上传。
        if existing is None:  # 文件流上传，避免Git二进制单文件限制。
            upload = release["upload_url"].split("{")[0] + "?name=" + urllib.parse.quote(name)  # 使用GitHub返回的上传端点。
            parsed = urllib.parse.urlsplit(upload)  # 检查接收凭据的域名。
            assert parsed.hostname == "uploads.github.com", "异常上传域名"  # 不向不可信域名发送凭据。
            connection = http.client.HTTPSConnection(parsed.hostname, timeout=120, blocksize=1024 * 1024)  # 按MiB流式发送。
            started, sent, printed = perf_counter(), 0, 0  # 单个文件最多上传15分钟。
            class Reader:  # http.client会调用read，不把整个文件载入RAM。
                def read(self, size):  # 提供上传所需的下一段。
                    nonlocal sent, printed  # 更新有限进度计数。
                    assert perf_counter() - started < 900, "附件上传超过15分钟，请稍后重试"  # 不无限占用网络。
                    data = handle.read(size)  # handle仅在下方with作用域使用。
                    sent += len(data)  # 本地已交给HTTP连接的字节量。
                    if sent - printed >= 32 * 1024 * 1024:  # 长上传定期显示进度，不打印密钥或上传URL。
                        print(name, "已发送MiB：", sent // (1024 * 1024), flush=True)  # 不把尚未响应的进度当作完成。
                        printed = sent  # 下一次进度起点。
                    return data  # 空字节表示文件结束。
            with path.open("rb") as handle:  # 上传正式状态的只读文件流。
                connection.request("POST", parsed.path + "?" + parsed.query, body=Reader(), headers={**headers, "Content-Type": "application/octet-stream", "Content-Length": str(artifact["bytes"])})  # 不使用shell拼接二进制。
                result = connection.getresponse()  # 等待服务器确认上传完成。
                payload = result.read()  # 接收小JSON回复。
                assert result.status == 201, f"上传失败HTTP {result.status}"  # 不把失败记录为已同步。
            connection.close()  # 释放网络连接。
            existing = json.loads(payload)  # GitHub实际附件元数据。
        assert existing["size"] == artifact["bytes"], "远端附件大小不符"  # 内容长度必须匹配。
        if existing.get("digest"):  # 支持服务器端SHA-256时一并验证。
            assert existing["digest"] == "sha256:" + artifact["sha256"], "远端SHA-256不同"  # 不接受同名不同文件。
        seen[artifact["sha256"]] = name  # 后续可以复用相同内容。
        assets.append({"local_name": name, **artifact, "asset_id": existing["id"], "url": existing["browser_download_url"], "remote_digest": existing.get("digest")})  # 仅记录公开下载信息。
    if args.publish and release["draft"]:  # 等代码推送后才对外发布完整版本。
        release = api(f"releases/{release['id']}", "PATCH", {"draft": False, "target_commitish": "main", "make_latest": "false"})  # 不将教学附件冒充项目最新稳定版。
    final_assets = {item["id"]: item for item in api(f"releases/{release['id']}/assets")}  # 发布后草稿临时下载URL会改变。
    for item in assets:  # 更新为服务器当前返回的永久下载信息。
        if "asset_id" in item:  # 相同内容的逻辑别名不重复创建附件。
            item["url"] = final_assets[item["asset_id"]]["browser_download_url"]  # 不把untagged草稿地址留在正式记录中。
    record = {"tag": TAG, "release_id": release["id"], "url": "https://github.com/" + REPO + "/releases/tag/" + TAG, "draft": release["draft"], "assets": assets}  # 不包含凭据。
    (ROOT / "docs/training_runs/lesson48_assets.json").write_bytes(dump(record))  # 后续验收可检查远端附件。
    print(json.dumps(record, ensure_ascii=False, indent=2))  # 输出可审阅的元数据。


if __name__ == "__main__":  # CI导入不会访问凭据或发布。
    main()  # 显式运行才同步本课大型训练状态。
