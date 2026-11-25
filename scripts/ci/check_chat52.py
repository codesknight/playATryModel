import copy  # 仅在内存构造损坏审核，不改正式文件。
import json  # 读取版本化队列与决定。
import sys  # 导入冻结规则。
from pathlib import Path  # 定位仓库。
ROOT = Path(__file__).resolve().parents[2]  # 项目根目录。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 不导入训练器。
from freeze_sft_review import DIRECTORY, materialize, validate_reviews  # 调用正式冻结与审核校验函数。
from prepare_chat_corpus import sha  # 审核与队列以实际字节关联。
queue_path = DIRECTORY / "queue.json"  # 已冻结候选版本。
queue = json.loads(queue_path.read_text(encoding="utf-8"))  # 不打开测试文件。
review = json.loads((DIRECTORY / "reviews.json").read_text(encoding="utf-8"))  # 所有26条已读完整正文的逐条决定。
files, manifest = materialize()  # 重建技术统计、审核子集和门槛。
for name, data in files.items():  # 核对每个已发布文件的精确字节。
    assert (DIRECTORY / "frozen_v1" / name).read_bytes() == data, "冻结内容无法重建"  # CI不重写版本。
train, validation = manifest["statistics"]["train"], manifest["statistics"]["validation"]  # 训练与验证保持原身份。
assert train["accepted_conversations"] == 4 and validation["accepted_conversations"] == 2 and train["assistant_targets"] == 735 and validation["assistant_targets"] == 338, "冻结范围或BPE目标数改变"  # 逐条修改决定必须建立新版本。
assert not manifest["formal_sft_ready"] and len(manifest["blockers"]) == 4 and manifest["model_updates"] == 0 and manifest["test_content_opened"] is False, "将不足的子集误当正式训练就绪"  # 不用小型英文审核样本宣称中文SFT准备完成。
broken = []  # 收集三类实际风险的内存副本。
missing = copy.deepcopy(review)  # 漏掉最后一条审核。
missing["reviews"].pop()  # 未审样本必须失败，不能默认放行。
broken.append(missing)  # 测试漏审路径。
duplicate = copy.deepcopy(review)  # 模拟同条目出现两份决定。
duplicate["reviews"].append(copy.deepcopy(duplicate["reviews"][0]))  # 字典覆盖不能掩盖重复审核。
broken.append(duplicate)  # 测试重复路径。
stale = copy.deepcopy(review)  # 模拟审核所见正文与候选不一致。
stale["reviews"][0]["text_sha256"] = "0" * 64  # 只修改内存指纹。
broken.append(stale)  # 测试过期内容路径。
for case in broken:  # 所有损坏情况都应被正式入口拒绝。
    try:  # 使用真实校验函数，而非复制判断代码。
        validate_reviews(queue, case, sha(queue_path.read_bytes()))  # 不改审核文件或源语料。
    except AssertionError:  # 明确拒绝是预期结果。
        pass  # 继续测试下一种错误。
    else:  # 任一漏放都应使CI失败。
        raise AssertionError("不完整或过期审核被放行")  # 质量队列不能隐式接受未知记录。
print("审核漏项/重复/过期拒绝、原组隔离、冻结重建及正式SFT未就绪状态：通过")  # 本课不执行神经网络训练。
