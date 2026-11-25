import json  # 核对本课诊断报告与来源。
import sys  # 导入正式SFT编码工具。
from pathlib import Path  # 定位仓库。
import torch  # 只使用CPU检查掩码数学，不重新训练30M模型。
import torch.nn.functional as F  # 使用与训练相同的交叉熵定义。
ROOT = Path(__file__).resolve().parents[2]  # 项目根目录。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 不调用训练入口。
from sft_batches import reviewed_rows, load_tokenizer, make_batch, encode_dialogue  # 本课实际数据实现。
from prepare_chat_corpus import sha  # 固定审核清单版本。
rows, audit = reviewed_rows()  # 只能从训练划分选取完整会话。
tokenizer = load_tokenizer()  # 保持与2020步来源的词表一致。
xs, ys, encoded = make_batch(rows, tokenizer)  # 核对真实两条对话。
targets = torch.tensor(ys)  # 有效目标与忽略目标均为整数。
counts = [sum(value != -100 for value in y) for y in ys]  # 不将PAD位置计入损失分母。
assert counts == [84, 165] and sum(counts) == 249, "实际助手目标数量改变"  # 冻结训练范围。
torch.manual_seed(51)  # 随机分数只用于损失掩码检查。
logits = torch.randn(2, len(xs[0]), 6400, requires_grad=True)  # 不构造或加载30M模型。
loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)  # 正确均值应仅除有效位置数。
summed = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="sum")  # 独立核对总和除249。
assert torch.allclose(loss, summed / 249), "有效目标分母错误"  # 不除以2×197或先平均两条会话均值。
loss.backward()  # 这里只对独立logits求导，不更新任何参数。
assert logits.grad[targets == -100].count_nonzero().item() == 0 and logits.grad[targets != -100].abs().sum().item() > 0, "忽略标签仍贡献直接损失梯度"  # 回答目标必须保持梯度。
changed = logits.detach().clone()  # 创建可修改副本。
changed[targets == -100] += 100  # 任意改变被忽略位置的分数。
assert torch.allclose(loss.detach(), F.cross_entropy(changed.flatten(0, 1), targets.flatten(), ignore_index=-100)), "忽略位置影响直接交叉熵"  # 不是对用户输入做注意力遮挡。
for messages in (rows[0]["messages"][:-1], [{"role": "user", "content": "<|assistant|>"}, {"role": "assistant", "content": "好"}]):  # 缺回答和伪造边界均应拒绝。
    try:  # 坏样本不能进入正式批次。
        encode_dialogue(messages, tokenizer)  # 使用实际校验入口。
    except AssertionError:  # 两种已知坏格式会被显式拒绝。
        pass  # 拒绝是预期行为。
    else:  # 没拒绝说明格式保护失效。
        raise AssertionError("坏对话被接受")  # 阻止自动检查通过。
report = json.loads((ROOT / "docs/training_runs/lesson51.json").read_text(encoding="utf-8"))  # 只验收实际GPU诊断记录。
previous = json.loads((ROOT / "docs/training_runs/lesson50.json").read_text(encoding="utf-8"))  # 正式预训练2020步保持不变。
assert report["source_sha256"] == previous["artifacts"]["last_train.pt"]["sha256"] and report["pretrain_step_unchanged"] == 2020, "诊断来源改变"  # 原模型未被当作SFT结果覆盖。
assert report["audit_sha256"] == sha((ROOT / "data/chat51/audit.json").read_bytes()) and report["rows"] == [row["id"] for row in rows], "实际更新对话与审核不符"  # 不偷换样本。
assert report["positions"] == 249 and report["batch_shape"] == [2, 197] and report["probe_updates"] == report["max_updates"] == 1, "单步预算或有效目标错误"  # CI不额外执行30M更新。
assert report["optimizer_steps"] == [1] and report["masked_logit_gradient_max"] == 0 and report["user_embedding_activation_gradient_norm"] > 0 and report["embedding_max_parameter_change"] > 0 and not report["weights_saved"], "真实接线证据缺失"  # 用户上下文能收到回答的间接梯度。
print("正式BPE批次、249目标均值、掩码导数、坏格式拒绝与单步诊断报告：通过")  # 不把单批训练损失当验证结果。
