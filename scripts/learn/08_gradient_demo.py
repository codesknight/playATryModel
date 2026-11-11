import numpy as np  # 导入 NumPy，进行稳定的对数概率和数值计算。


def loss_at(weight):  # 给定一个可调参数，计算当前预测的损失。
    logits = np.array([weight, 0.0, 0.0], dtype=np.float64)  # 三个候选中只调节第 0 项的分数。
    shifted = logits - logits.max()  # 减去最大分数，保持指数计算稳定。
    log_probs = shifted - np.log(np.exp(shifted).sum())  # 直接得到三个候选的对数概率。
    return float(-log_probs[0])  # 正确标签固定为 0，返回它的负对数概率。

def numerical_gradient(weight, epsilon=1e-5):  # 用中心差分估计当前参数处的导数。
    assert epsilon > 0, "试探距离必须大于零"  # 分母不能为零。
    loss_plus = loss_at(weight + epsilon)  # 从原参数位置向正方向试探一步。
    loss_minus = loss_at(weight - epsilon)  # 从同一个原位置向负方向试探一步。
    return (loss_plus - loss_minus) / (2 * epsilon)  # 损失差除以两个试探位置之间的距离。

def main():  # 定义演示入口，观察一次参数更新。
    weight = 0.0  # 从三个候选分数相等的位置开始。
    learning_rate = 0.5  # 设置实际更新所用的正学习率。
    epsilon = 1e-5  # 设置计算数值梯度时的试探距离。
    before = loss_at(weight)  # 记录更新前的损失。
    gradient = numerical_gradient(weight, epsilon)  # 估计损失相对这个参数的变化速率。
    updated_weight = weight - learning_rate * gradient  # 沿负梯度方向更新参数。
    after = loss_at(updated_weight)  # 重新计算更新后的损失。
    opposite_loss = loss_at(weight + learning_rate * gradient)  # 对照计算反方向步进后的损失。
    assert np.isfinite(gradient) and after < before < opposite_loss, "本例更新方向或损失不正确"  # 检查这个具体例子的更新结果。
    print("向正方向试探后的损失：", loss_at(weight + epsilon))  # 应略低于初始损失。
    print("向负方向试探后的损失：", loss_at(weight - epsilon))  # 应略高于初始损失。
    print("更新前参数：", weight)  # 初始为 0。
    print("数值梯度：", gradient)  # 当前约为 -0.666667。
    print("更新后参数：", updated_weight)  # 当前约为 0.333333。
    print("更新前损失：", before)  # 当前约为 1.098612。
    print("更新后损失：", after)  # 应小于初始损失。
    print("更新前正确答案概率：", np.exp(-before))  # 根据 loss=-ln(p) 还原正确答案概率。
    print("更新后正确答案概率：", np.exp(-after))  # 应比 1/3 大。
    print("数值梯度与单步更新方向检查：通过")  # 断言通过后显示。


if __name__ == "__main__":  # 直接运行文件时才执行演示。
    main()  # 调用入口，不启动真正的 RNN 训练。
