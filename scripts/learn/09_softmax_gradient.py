import numpy as np  # 导入 NumPy，用 float64 检查微小梯度差异。


def loss_and_grad(logits, target):  # 计算一个位置的损失、概率和对原始分数的梯度。
    logits = np.asarray(logits, dtype=np.float64)  # 本演示使用双精度，不改已有 RNN 的参数类型。
    assert logits.ndim == 1 and logits.size > 0 and np.isfinite(logits).all(), "需要有限的一维分数"  # 每次只检查一个位置。
    assert isinstance(target, (int, np.integer)) and 0 <= target < logits.size, "正确标签越界或类型错误"  # 标签是候选编号。
    shifted = logits - logits.max()  # 平移分数，保持计算稳定。
    log_probs = shifted - np.log(np.exp(shifted).sum())  # 计算所有候选的对数概率。
    probabilities = np.exp(log_probs)  # 恢复概率，供梯度公式使用。
    gradient = probabilities.copy()  # 梯度从 p 开始，复制以免改写概率。
    gradient[target] -= 1.0  # 正确类别减去 1，其他类别不变，实现 p-q。
    return float(-log_probs[target]), probabilities, gradient  # 返回损失、概率向量和分数梯度。

def numerical_gradient(logits, target, epsilon=1e-5):  # 用中心差分核对每个分数的偏导。
    assert epsilon > 0, "试探距离必须大于零"  # 分母不能为零。
    logits = np.asarray(logits, dtype=np.float64)  # 使用双精度进行差分。
    numerical = np.zeros_like(logits)  # 准备与分数向量同形状的数值梯度。
    for index in range(logits.size):  # 依次检查各个候选分数。
        plus = logits.copy()  # 正向试探从原分数开始。
        minus = logits.copy()  # 反向试探也从原分数开始。
        plus[index] += epsilon  # 只增加当前分数，其他分数固定。
        minus[index] -= epsilon  # 只减少当前分数，其他分数固定。
        plus_loss, _, _ = loss_and_grad(plus, target)  # 只使用损失，忽略额外返回结果。
        minus_loss, _, _ = loss_and_grad(minus, target)  # 计算反向试探损失。
        numerical[index] = (plus_loss - minus_loss) / (2 * epsilon)  # 估计当前分数的偏导。
    return numerical  # 返回全部分数的数值梯度。

def main():  # 定义检查入口。
    logits = np.array([0.15, -0.1, -0.3], dtype=np.float64)  # 设置天、风、雨的分数。
    target = 0  # 正确答案是第 0 项天。
    before, probabilities, gradient = loss_and_grad(logits, target)  # 用 p-q 计算解析梯度。
    numerical = numerical_gradient(logits, target)  # 用两侧损失计算数值梯度。
    max_error = np.max(np.abs(gradient - numerical))  # 统计两种方法的最大绝对差。
    assert np.allclose(gradient, numerical, rtol=1e-6, atol=1e-8), "解析与数值梯度不一致"  # 在双精度容差内核对。
    assert np.isclose(gradient.sum(), 0.0, atol=1e-12), "梯度和不为零"  # 同时平移所有分数不改变损失，因此偏导之和应为零。
    updated_logits = logits - 0.5 * gradient  # 仅为观察方向，对分数走一次负梯度小步。
    after, updated_probabilities, _ = loss_and_grad(updated_logits, target)  # 重新计算损失与概率。
    assert after < before and updated_probabilities[target] > probabilities[target], "本例下降方向不正确"  # 核对当前例子的效果。
    print("模型概率：", probabilities)  # 显示三个候选的概率。
    print("解析梯度：", gradient)  # 正确项为负，其他项为正。
    print("数值梯度：", numerical)  # 应与解析结果接近。
    print("最大绝对误差：", max_error)  # 应很小，尾数可能随平台变化。
    print("更新前损失：", before)  # 显示更新前的误差。
    print("更新后损失：", after)  # 本例应降低。
    print("解析梯度与数值差分检查：通过")  # 所有断言通过后显示。


if __name__ == "__main__":  # 直接运行文件时启动演示。
    main()  # 调用入口，不改变现有 RNN。