import sys  # 导入实际冻结规则。
from pathlib import Path  # 相对脚本定位仓库。
ROOT = Path(__file__).resolve().parents[2]  # 终端目录变化不影响定位。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 不导入训练入口。
from freeze_sft_review import materialize, DIRECTORY  # 核对队列、逐条决定及冻结字节。


def main():  # 学习者只读检查，不写文件、不加载30M模型。
    files, manifest = materialize()  # 从原训练/验证文件重新推导所有统计。
    for name, data in files.items():  # 逐项比对已保存的冻结文件。
        assert (DIRECTORY / "frozen_v1" / name).read_bytes() == data, "冻结版本不一致"  # 重跑应得到相同结果。
    for split in ("train", "validation"):  # 不打开测试正文。
        inventory, stats = manifest["inventory"][split], manifest["statistics"][split]  # 自动结构筛查与逐条内容决定分别报告。
        print(split, "原记录：", inventory["total"], "完整超长：", inventory.get("over_context", 0), "可送审：", inventory["eligible_for_review"])  # 可送审不等于可训练。
        print("本次审核：", stats["reviewed"], "决定：", stats["decisions"])  # 待核实的内容也不能直接使用。
        print("冻结接纳会话：", stats["accepted_conversations"], "助手目标：", stats["assistant_targets"], "中文覆盖会话：", stats["cjk_conversations"])  # 对照中文聊天目标检查分布。
    print("允许启动正式SFT：", manifest["formal_sft_ready"])  # False可能是筛查正确生效，而不是程序出错。
    for reason in manifest["blockers"]:  # 展示具体阻止原因。
        print("未满足：", reason)  # 不因为训练按钮可运行就宣布准备完成。
    print("完整审核、原组隔离、冻结SHA及重建检查：通过；没有训练模型，未打开测试正文。")  # 只读操作不增加任何训练步数。


if __name__ == "__main__":  # 导入不会自动运行。
    main()  # 学习者显式运行才显示检查结果。
