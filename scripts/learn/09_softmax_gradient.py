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

def output_loss(hidden, weight, bias, target):  # 计算单个位置经过输出层后的交叉熵。
    logits = hidden @ weight + bias  # 从 (1,H) 状态计算 (1,V) 分数。
    assert logits.shape[0] == 1, "本演示损失只处理一个位置"  # 复用已有单位置损失函数。
    loss, _, _ = loss_and_grad(logits[0], target)  # 取出这一行分数，计算标量损失。
    return loss  # 返回数值差分需要的损失。


def output_backward(hidden, weight, grad_logits):  # 接收分数梯度，计算输出层的三组梯度。
    assert hidden.ndim == weight.ndim == grad_logits.ndim == 2, "需要二维状态、权重和分数梯度"  # 统一使用矩阵。
    assert hidden.shape[0] == grad_logits.shape[0] and hidden.shape[1] == weight.shape[0] and grad_logits.shape[1] == weight.shape[1], "反向传播形状不匹配"  # 核对 N、H、V。
    grad_weight = hidden.T @ grad_logits  # 汇总各位置对共享权重的贡献，得到 (H,V)。
    grad_bias = grad_logits.sum(axis=0)  # 汇总各位置对共享偏移的贡献，得到 (V,)。
    grad_hidden = grad_logits @ weight.T  # 将梯度传回隐藏状态，得到 (N,H)。
    return grad_weight, grad_bias, grad_hidden  # 参数梯度用于更新，状态梯度继续传给前层。

def numerical_array_gradient(value, evaluate, epsilon=1e-5):  # 根据给定损失函数，逐元素检查数组梯度。
    assert epsilon > 0, "试探距离必须大于零"  # 防止无效差分距离。
    value = np.asarray(value, dtype=np.float64)  # 使用双精度进行微小扰动。
    gradient = np.zeros_like(value)  # 梯度形状与被检查数组相同。
    for index in np.ndindex(value.shape):  # 依次取得 (行,列) 等下标，兼容向量和矩阵。
        plus = value.copy()  # 正向试探使用独立副本。
        minus = value.copy()  # 反向试探也使用独立副本。
        plus[index] += epsilon  # 每次只增加一个元素。
        minus[index] -= epsilon  # 每次只减少同一个元素。
        gradient[index] = (evaluate(plus) - evaluate(minus)) / (2 * epsilon)  # 计算两侧损失的差分。
    return gradient  # 返回近似梯度，不修改原数组。

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

    hidden = np.array([[0.2, -0.1]], dtype=np.float64)  # 一个位置、两个隐藏分量，形状为 (1,2)。
    weight = np.array([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]], dtype=np.float64)  # 输出权重形状为 (2,3)。
    bias = np.array([0.0, 0.1, 0.0], dtype=np.float64)  # 每个候选一个偏移。
    target = 0  # 正确答案仍为第 0 项。
    scores = hidden @ weight + bias  # 得到 [0.15,-0.1,-0.3]，保留 (1,3) 形状。
    current_loss, _, score_gradient = loss_and_grad(scores[0], target)  # 先取得对分数的梯度。
    grad_weight, grad_bias, grad_hidden = output_backward(hidden, weight, score_gradient.reshape(1, -1))  # 恢复行维度并继续反传。

    numeric_weight = numerical_array_gradient(weight, lambda candidate: output_loss(hidden, candidate, bias, target))  # 只试探权重。
    numeric_bias = numerical_array_gradient(bias, lambda candidate: output_loss(hidden, weight, candidate, target))  # 只试探偏移。
    numeric_hidden = numerical_array_gradient(hidden, lambda candidate: output_loss(candidate, weight, bias, target))  # 只试探隐藏状态。

    assert np.allclose(grad_weight, numeric_weight, rtol=1e-6, atol=1e-8), "输出权重梯度不正确"  # 核对每个权重。
    assert np.allclose(grad_bias, numeric_bias, rtol=1e-6, atol=1e-8), "输出偏移梯度不正确"  # 核对每个偏移。
    assert np.allclose(grad_hidden, numeric_hidden, rtol=1e-6, atol=1e-8), "隐藏状态梯度不正确"  # 核对传给前层的梯度。

    updated_weight = weight - 0.1 * grad_weight  # 沿负梯度更新演示输出权重。
    updated_bias = bias - 0.1 * grad_bias  # 沿负梯度更新演示输出偏移。
    after_output_update = output_loss(hidden, updated_weight, updated_bias, target)  # 固定隐藏状态，重新前向计算。
    assert after_output_update < current_loss, "本例输出层更新没有降低损失"  # 验证这个具体小步。

    print("输出权重梯度：", grad_weight)  # 应为 (2,3)。
    print("输出偏移梯度：", grad_bias)  # 应为 (3,)。
    print("隐藏状态梯度：", grad_hidden)  # 应为 (1,2)。
    print("三组梯度最大误差：", np.max(np.abs(grad_weight - numeric_weight)), np.max(np.abs(grad_bias - numeric_bias)), np.max(np.abs(grad_hidden - numeric_hidden)))  # 显示数值检查误差。
    print("输出层更新前后损失：", current_loss, after_output_update)  # 这次更新的是权重和偏移，不直接改分数。
    print("输出层反向传播与数值检查：通过")  # 全部断言通过后显示。

if __name__ == "__main__":  # 直接运行文件时启动演示。
    main()  # 调用入口，不改变现有 RNN。