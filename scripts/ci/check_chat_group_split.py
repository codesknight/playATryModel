from importlib import import_module  # 导入数字前缀的教学脚本。
import json  # 读取原始示例。
from pathlib import Path  # 定位仓库和临时版本目录。
import sys  # 配置学习模块路径。
import tempfile  # 测试版本保护时不修改真实产物。
ROOT = Path(__file__).resolve().parents[2]  # 项目根目录。
sys.path.insert(0, str(ROOT / "scripts/learn"))  # 查找本课划分函数。
module = import_module("27_split_chat_groups")  # 导入不会执行main。
source = (ROOT / "data/group_demo/samples.jsonl").read_bytes()  # 使用实际教学示例。
samples = [json.loads(line) for line in source.decode("utf-8").splitlines()]  # 读取12条完整会话。
mapping, partitions = module.make_split(samples)  # 核对固定种子的分组结果。
assert mapping == module.make_split(list(reversed(samples)))[0], "输入行顺序影响了组划分"  # 打乱前排序保证映射稳定。
assert partitions == module.make_split(list(reversed(samples)))[1], "输入行顺序影响输出记录"  # 保存的样本也按id排序。
assert [len(partitions[name]) for name in module.SPLITS] == [8, 3, 1], "样本条数不符"  # 组数比例不必等于条数比例。
assert mapping["g00"] == "validation" and all(row["group_id"] == "g00" for row in partitions["validation"]), "苹果题被拆组"  # 三个改名样本留在同一个验证组。
module.save_version(samples, source, ROOT / "data/group_demo/splits_v1")  # 只读核对已提交版本，没有差异时不重写文件。
with tempfile.TemporaryDirectory() as name:  # 在临时目录核对重跑和拒绝覆盖行为。
    output = Path(name) / "v1"  # 不触碰正式数据。
    first = module.save_version(samples, source, output)  # 创建一个独立版本。
    before = {path.name: path.read_bytes() for path in output.iterdir()}  # 保存全部文件字节以确认没有覆盖。
    assert first == module.save_version(samples, source, output), "同输入重跑不能复现"  # 相同输入应得到同一版本。
    for changed_source, changed_seed in ((source + b"\n", 42), (source, 43)):  # 来源字节或种子变化都不允许静默覆盖。
        rejected = False  # 检查是否显式拒绝。
        try:  # 预期版本保护会触发。
            module.save_version(samples, changed_source, output, changed_seed)  # 不应创建或覆盖任何文件。
        except AssertionError:  # 捕获本教学脚本的数据校验异常。
            rejected = True  # 确认保护生效。
        assert rejected and before == {path.name: path.read_bytes() for path in output.iterdir()}, "旧版本被覆盖"  # 不仅检查抛错，也检查文件不变。
    (output / "train.jsonl").write_bytes(b"corrupted")  # 故意破坏临时产物测试检测能力。
    rejected = False  # 不允许脚本自动修复并掩盖版本污染。
    try:  # 同来源但已保存文件变动也应报错。
        module.save_version(samples, source, output)  # 此调用必须拒绝覆盖。
    except AssertionError:  # 预期的内容完整性异常。
        rejected = True  # 确认检查成功。
    assert rejected and (output / "train.jsonl").read_bytes() == b"corrupted", "损坏产物被静默重写"  # 保留现场便于调查。
print("分组互斥、顺序稳定、版本复现与拒绝覆盖检查：通过")  # 不训练模型，不更换真实语料版本。
