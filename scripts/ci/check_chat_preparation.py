from importlib import import_module  # 导入可执行教学模块但不启动演示。
from pathlib import Path  # 定位项目脚本。
import sys  # 设置模块查找路径。
ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于scripts/ci。
sys.path.insert(0, str(ROOT / "scripts/learn"))  # 导入本课的独立数据格式实现。
module = import_module("26_chat_sft_format")  # 模块名称有数字前缀，需要用import_module。
messages = [{"role": "user", "content": "甲"}, {"role": "assistant", "content": "乙"}, {"role": "user", "content": "丙"}, {"role": "assistant", "content": "丁"}]  # 用不同字符区分用户与助手目标。
tokens = module.SPECIAL + list("甲乙丙丁")  # 演示词表覆盖全部字符。
x, y, ids, flags = module.encode_dialogue(messages, tokens)  # 编码完整多轮，不跨会话拼接。
assert [tokens[target] for target in y if target != -100] == ["乙", "<|end|>", "丁", "<|end|>"], "非回答目标参与了损失"  # 手工列出两个回答及其停止标记。
assert ids[1:] == [target if target != -100 else original for target, original in zip(y, ids[1:])], "目标错位错误"  # 未被忽略的目标确实是下一符号。
for position, token in enumerate(x):  # 角色头位置应预测回答的首字符。
    if token == tokens.index("<|assistant|>"):  # 两个助手回答都要包含首字监督。
        assert y[position] in (tokens.index("乙"), tokens.index("丁")), "回答首字被漏掉"  # 防止loss mask错位一格。
for invalid in ([{"role": "system", "content": "甲"}], [{"role": "assistant", "content": "乙"}], [{"role": "user", "content": "甲"}]):  # 拒绝空回答或角色顺序错误的样本。
    rejected = False  # 明确检查坏样本是否被拒绝。
    try:  # 数据检查应失败而不是悄悄产生全忽略目标。
        module.encode_dialogue(invalid, tokens)  # 只测试格式，不训练模型。
    except AssertionError:  # 预期的数据验证错误。
        rejected = True  # 确认错误被正确识别。
    assert rejected, "无效会话未被拒绝"  # 所有监督位置被屏蔽的样本不能蒙混过关。
print("SFT回答掩码、首字对齐、多轮边界与坏样本拒绝检查：通过")  # 只读检查，不创建任何权重。
