import argparse  # 可选保存本次实验结果，默认只打印。
import copy  # 三个分支必须从完全相同的参数开始。
import json  # 保存可核对的梯度和更新误差。
from pathlib import Path  # 定位本课报告输出位置。
import torch  # 本实验只用CPU，不占用训练显卡。
from torch import nn  # 用很小的线性输出层隔离梯度累积规则。
from torch.nn import functional as F  # 与真实训练同样使用交叉熵。


def maximum_difference(left, right, gradients=False):  # 比较两份模型的全部参数或梯度。
    pairs = zip(left.parameters(), right.parameters())  # 对应同一结构的参数。
    return max(float(((a.grad - b.grad) if gradients else (a - b)).abs().max().item()) for a, b in pairs)  # 返回最大绝对误差。


def experiment():  # 三个分支各更新一次，不是给30M模型追加训练。
    torch.set_num_threads(2)  # 控制CPU线程。
    torch.manual_seed(49)  # 固定输入、目标和模型初始化。
    x = torch.randn(8, 3, 3, dtype=torch.float64)  # 八个窗口、每窗三位置、每位置三维模拟特征。
    y = torch.randint(0, 2, (8, 3))  # 每个位置的真实类别编号是0或1。
    y[-1, 1:] = -100  # 最后微批只有一个有效目标，另两位作补齐。
    valid_counts = (y != -100).sum(dim=1).tolist()  # 得到[3,3,3,3,3,3,3,1]。
    total_valid = sum(valid_counts)  # 正确的组内平均分母是22。
    original = nn.Linear(3, 2).double()  # 模拟输出层，不引入Dropout或跨样本统计。
    large, accumulated, naive = (copy.deepcopy(original) for _ in range(3))  # 三份相同初始参数。
    large_optimizer = torch.optim.SGD(large.parameters(), lr=0.1)  # 用无动量SGD单独验证梯度影响。
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)  # 对照分支学习率完全一致。
    naive_optimizer = torch.optim.SGD(naive.parameters(), lr=0.1)  # 错误加权分支也只更新一次。
    large_optimizer.zero_grad(set_to_none=True)  # 大批分支清除旧梯度。
    large_loss = F.cross_entropy(large(x).flatten(0, 1), y.flatten())  # 一次计算全部22个有效目标的平均损失。
    large_loss.backward()  # 大批只反向传播一次。
    accumulated_optimizer.zero_grad(set_to_none=True)  # 累积组开始清一次，不在微批之间清除。
    weighted_loss = 0.0  # 累加每个微批对总损失的贡献。
    for index in range(8):  # 每次只送一个微批，参数暂不改变。
        logits = accumulated(x[index:index + 1])  # 当前微批前向。
        loss_sum = F.cross_entropy(logits.flatten(0, 1), y[index:index + 1].flatten(), reduction="sum")  # 只把有效位置求和。
        contribution = loss_sum / total_valid  # 每个微批都除以整组22个目标。
        contribution.backward()  # 梯度累加到同一份.grad。
        weighted_loss += float(contribution.item())  # 应与大批平均损失一致。
    naive_optimizer.zero_grad(set_to_none=True)  # 错误示范只改变加权方式，不改变其他条件。
    for index in range(8):  # 不能把这段写法直接照搬到真实尾批。
        logits = naive(x[index:index + 1])  # 同一条输入和同一初始参数。
        micro_mean = F.cross_entropy(logits.flatten(0, 1), y[index:index + 1].flatten())  # 当前微批各自求平均。
        (micro_mean / 8).backward()  # 错误：各微批均值等权，使只有一个目标的尾批权重过大。
    correct_gradient_error = maximum_difference(large, accumulated, gradients=True)  # 检查正确累积与大批的梯度。
    naive_gradient_error = maximum_difference(large, naive, gradients=True)  # 证明不同有效长度时不能无条件除8。
    assert abs(large_loss.item() - weighted_loss) < 1e-12 and correct_gradient_error < 1e-12, "正确加权不等价"  # FP64只允许小浮点误差。
    assert naive_gradient_error > 1e-6, "错误示范未产生可观察差异"  # 防止无效示例蒙混通过。
    large_optimizer.step()  # 大批只执行一次参数更新。
    accumulated_optimizer.step()  # 八次backward之后只执行一次更新。
    naive_optimizer.step()  # 错误梯度导致不同的参数更新。
    parameter_error = maximum_difference(large, accumulated)  # 核对实际更新结果，不只比较损失。
    naive_parameter_error = maximum_difference(large, naive)  # 显示错误加权会改变训练轨迹。
    assert parameter_error < 1e-12 and naive_parameter_error > 1e-7, "更新结果检查失败"  # 相同学习率下应体现梯度差异。
    return {"scope": "tiny CPU FP64 linear-head demo; no extra 30M training", "valid_counts": valid_counts, "total_valid": total_valid, "large_batch_loss": float(large_loss.item()), "weighted_micro_loss": weighted_loss, "correct_gradient_max_error": correct_gradient_error, "naive_gradient_max_error": naive_gradient_error, "correct_parameter_max_error": parameter_error, "naive_parameter_max_error": naive_parameter_error, "backward_calls_large": 1, "backward_calls_accumulated": 8, "optimizer_updates_per_branch": 1}  # 真实30M记录不变。


def main():  # 学习者运行时默认不写任何模型文件。
    parser = argparse.ArgumentParser()  # 可选写出独立教学报告。
    parser.add_argument("--report", type=Path)  # CI不传此选项，不覆盖报告。
    args = parser.parse_args()  # 读取本次输出设置。
    result = experiment()  # 实际执行三份小模型各一次更新。
    print("每个微批有效目标：", result["valid_counts"])  # 先看尾批与普通批的区别。
    print("整组有效目标：", result["total_valid"])  # 总分母22，而不是填充后的24。
    print("大批平均损失：", result["large_batch_loss"])  # 作为等价对照。
    print("正确累积平均损失：", result["weighted_micro_loss"])  # 应与上一行一致。
    print("正确累积梯度最大误差：", result["correct_gradient_max_error"])  # 接近浮点舍入误差。
    print("错误平均梯度最大误差：", result["naive_gradient_max_error"])  # 直接除8会明显不同。
    print("正确累积参数更新最大误差：", result["correct_parameter_max_error"])  # 验证真实step后的结果。
    print("错误平均参数更新最大误差：", result["naive_parameter_max_error"])  # 错误梯度不是无关紧要的显示差异。
    print("大批backward次数：1；累积backward次数：8；每分支step次数：1")  # 区分微批反向与参数更新。
    print("有效目标加权、梯度与单步更新等价检查：通过；没有追加30M训练。")  # 不混淆小实验与正式训练。
    if args.report is not None:  # 只在显式要求时保存报告。
        args.report.parent.mkdir(parents=True, exist_ok=True)  # 支持独立报告目录。
        args.report.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))  # 固定LF，不写权重。


if __name__ == "__main__":  # 导入工具时不训练或打印。
    main()  # 执行本课的短CPU实操。
