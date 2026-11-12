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

def recurrent_step(current, previous, weight, bias):  # 计算一个时间步，不处理整段序列。
    return np.tanh(current + previous @ weight + bias)  # 结合当前向量、上一状态和偏移，再激活。


def recurrent_step_loss(current, previous, recurrent_weight, hidden_bias, output_weight, output_bias, target):  # 将单步循环接到已有输出层损失。
    hidden = recurrent_step(current, previous, recurrent_weight, hidden_bias)  # 计算当前状态。
    return output_loss(hidden, output_weight, output_bias, target)  # 复用上一课的单位置损失。


def recurrent_backward(previous, weight, hidden, grad_hidden):  # 将上游梯度继续传过一个循环步。
    assert previous.ndim == 2 and previous.shape == hidden.shape == grad_hidden.shape, "状态与上游梯度形状不一致"  # 每行每个分量都要对应。
    assert weight.shape == (hidden.shape[1], hidden.shape[1]), "循环权重形状错误"  # 本模型使用 H×H 循环矩阵。
    grad_activation = grad_hidden * (1.0 - hidden ** 2)  # 逐元素乘 tanh 的局部导数。
    grad_current = grad_activation.copy()  # 当前字符向量直接相加，收到同样数值的梯度。
    grad_previous = grad_activation @ weight.T  # 将梯度传回上一时刻状态。
    grad_weight = previous.T @ grad_activation  # 累加各行对循环权重的贡献。
    grad_bias = grad_activation.sum(axis=0)  # 累加各行对共享隐藏偏移的贡献。
    return grad_current, grad_previous, grad_weight, grad_bias  # 返回梯度，暂不更新任何数组。


def two_step_loss(first, second, initial, recurrent_weight, hidden_bias, output_weight, output_bias):  # 重新计算两步平均损失，供数值差分检查。
    first_hidden = recurrent_step(first, initial, recurrent_weight, hidden_bias)  # 从初始状态计算第一步状态。
    second_hidden = recurrent_step(second, first_hidden, recurrent_weight, hidden_bias)  # 第一状态参与第二步计算。
    first_loss = output_loss(first_hidden, output_weight, output_bias, 0)  # 第一位置的正确编号固定为 0。
    second_loss = output_loss(second_hidden, output_weight, output_bias, 1)  # 第二位置的正确编号固定为 1。
    return (first_loss + second_loss) / 2.0  # 两个预测位置的损失取平均。

def sequence_hidden(vectors, initial, weight, bias):  # 按时间正向计算并保存全部状态。
    assert vectors.ndim == 3 and vectors.shape[1] > 0, "输入需要非空时间维"  # 输入形状为 (B,T,H)。
    assert initial.shape == (vectors.shape[0], vectors.shape[2]), "初始状态形状不对"  # 每条序列各有一个初始状态。
    states = np.zeros_like(vectors)  # 每个输入位置对应一个同维度状态。
    hidden = initial.copy()  # 从给定初始状态开始，不修改原数组。
    for time in range(vectors.shape[1]):  # 按 0、1、2 等顺序向前计算。
        hidden = recurrent_step(vectors[:, time, :], hidden, weight, bias)  # 复用已经验证的单步计算。
        states[:, time, :] = hidden  # 保存本步状态，反向传播时还要用到。
    return states  # 返回 (B,T,H) 的状态数组。

def sequence_backward(states, initial, weight, direct):  # 将各位置输出层的梯度沿时间反向传播。
    assert states.ndim == 3 and states.shape == direct.shape and states.shape[1] > 0, "状态与直接梯度形状不对"  # 每个状态必须有对应的直接梯度。
    assert initial.shape == (states.shape[0], states.shape[2]), "初始状态形状不对"  # 初始状态形状为 (B,H)。
    grad_vectors = np.zeros_like(states)  # 保存对每个输入向量的梯度。
    grad_weight = np.zeros_like(weight)  # 共享循环权重的总梯度从零开始。
    grad_bias = np.zeros(states.shape[2], dtype=states.dtype)  # 共享隐藏偏移的总梯度从零开始。
    future = np.zeros_like(initial)  # 最后位置没有更晚的步骤，未来贡献从零开始。
    for time in reversed(range(states.shape[1])):  # 按 T-1、T-2 到 0 的顺序反传。
        previous = initial if time == 0 else states[:, time - 1, :]  # 第零步的前态必须取 initial，不能取下标 -1。
        total = direct[:, time, :] + future  # 合并当前输出路径与后续步骤传回的梯度。
        grad_current, future, step_weight, step_bias = recurrent_backward(previous, weight, states[:, time, :], total)  # 反传本步，并取得传向更早一步的梯度。
        grad_vectors[:, time, :] = grad_current  # 不同位置的输入梯度分别保存。
        grad_weight += step_weight  # 同一个循环权重的各次使用贡献要累加。
        grad_bias += step_bias  # 同一个隐藏偏移的各次使用贡献也要累加。
    return grad_vectors, future, grad_weight, grad_bias  # 循环结束时 future 就是初始状态的梯度。

def sequence_loss(vectors, initial, recurrent_weight, hidden_bias, output_weight, output_bias, targets):  # 重算整段序列的平均损失，供数值核对。
    assert vectors.shape[0] == 1 and targets.shape == (vectors.shape[1],), "本例损失只演示一条序列"  # 单位置损失函数暂时逐步调用。
    states = sequence_hidden(vectors, initial, recurrent_weight, hidden_bias)  # 每次试探都重新计算整段状态。
    losses = [output_loss(states[:, time, :], output_weight, output_bias, int(targets[time])) for time in range(states.shape[1])]  # 计算各位置的损失。
    return float(np.mean(losses))  # 对全部时间位置取平均。

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

    current_vector = np.array([[0.1, 0.2]], dtype=np.float64)  # 当前字符的两维向量。
    previous_hidden = np.array([[0.2, -0.1]], dtype=np.float64)  # 上一步的两维状态。
    recurrent_weight = np.array([[0.5, 0.2], [-0.3, 0.4]], dtype=np.float64)  # 两维循环权重。
    hidden_bias = np.zeros(2, dtype=np.float64)  # 隐藏偏移，与输出层 bias 区分。

    step_hidden = recurrent_step(current_vector, previous_hidden, recurrent_weight, hidden_bias)  # 得到 tanh([0.23,0.2])。
    step_scores = step_hidden @ weight + bias  # 使用已有输出层计算候选分数。
    step_loss, _, step_score_gradient = loss_and_grad(step_scores[0], target)  # 计算损失和分数梯度。
    _, _, step_hidden_gradient = output_backward(step_hidden, weight, step_score_gradient.reshape(1, -1))  # 先通过输出层，取得状态梯度。
    grad_current, grad_previous, grad_recurrent, grad_hidden_bias = recurrent_backward(previous_hidden, recurrent_weight, step_hidden, step_hidden_gradient)  # 再通过 tanh 和循环连接。

    numeric_current = numerical_array_gradient(current_vector, lambda candidate: recurrent_step_loss(candidate, previous_hidden, recurrent_weight, hidden_bias, weight, bias, target))  # 只试探当前字符向量。
    numeric_previous = numerical_array_gradient(previous_hidden, lambda candidate: recurrent_step_loss(current_vector, candidate, recurrent_weight, hidden_bias, weight, bias, target))  # 只试探上一步状态。
    numeric_recurrent = numerical_array_gradient(recurrent_weight, lambda candidate: recurrent_step_loss(current_vector, previous_hidden, candidate, hidden_bias, weight, bias, target))  # 只试探循环权重。
    numeric_hidden_bias = numerical_array_gradient(hidden_bias, lambda candidate: recurrent_step_loss(current_vector, previous_hidden, recurrent_weight, candidate, weight, bias, target))  # 只试探隐藏偏移。

    checks = [("当前向量", grad_current, numeric_current), ("上一步状态", grad_previous, numeric_previous), ("循环权重", grad_recurrent, numeric_recurrent), ("隐藏偏移", grad_hidden_bias, numeric_hidden_bias)]  # 组织四组检查。
    for name, analytic, numeric in checks:  # 逐组核对解析与数值梯度。
        assert analytic.shape == numeric.shape and np.allclose(analytic, numeric, rtol=1e-6, atol=1e-8), name + "梯度不正确"  # 对照完整数组。
        print(name + "梯度：", analytic)  # 显示当前对象的解析梯度。
        print(name + "最大绝对误差：", np.max(np.abs(analytic - numeric)))  # 显示差分检查误差。

    print("当前状态：", step_hidden)  # 应约为 [[0.22602835,0.19737532]]。
    print("本例单步损失：", step_loss)  # 固定例子的核对值，不代表模型测评结果。
    print("tanh 与循环单步反向检查：通过")  # 四组检查通过后显示。

    second_vector = np.array([[-0.2, 0.3]], dtype=np.float64)  # 新增第二步输入，第一步沿用 current_vector。
    first_hidden = recurrent_step(current_vector, previous_hidden, recurrent_weight, hidden_bias)  # 从给定初始状态计算第一步。
    second_hidden = recurrent_step(second_vector, first_hidden, recurrent_weight, hidden_bias)  # 使用第一状态计算第二步。
    first_loss, _, first_score_grad = loss_and_grad((first_hidden @ weight + bias)[0], 0)  # 第一位置的正确编号为 0。
    second_loss, _, second_score_grad = loss_and_grad((second_hidden @ weight + bias)[0], 1)  # 第二位置的正确编号为 1。
    mean_loss = (first_loss + second_loss) / 2.0  # 本次目标是两个位置的平均损失。
    first_out_weight_grad, first_out_bias_grad, first_direct = output_backward(first_hidden, weight, first_score_grad.reshape(1, -1) / 2.0)  # 第一处输出梯度只除以位置数一次。
    second_out_weight_grad, second_out_bias_grad, second_direct = output_backward(second_hidden, weight, second_score_grad.reshape(1, -1) / 2.0)  # 第二处输出梯度也只除一次。

    second_input_grad, from_second, second_recurrent_grad, second_hidden_bias_grad = recurrent_backward(first_hidden, recurrent_weight, second_hidden, second_direct)  # 先反传第二步，取得传给第一状态的梯度。
    first_total = first_direct + from_second  # 第一状态收到两路贡献，逐元素相加。
    first_input_grad, initial_grad, first_recurrent_grad, first_hidden_bias_grad = recurrent_backward(previous_hidden, recurrent_weight, first_hidden, first_total)  # 合并后再反传第一步。
    total_recurrent_grad = first_recurrent_grad + second_recurrent_grad  # 同一个循环权重在两步使用，贡献相加。
    total_hidden_bias_grad = first_hidden_bias_grad + second_hidden_bias_grad  # 隐藏偏移也是两步共享的参数。
    total_output_weight_grad = first_out_weight_grad + second_out_weight_grad  # 汇总两处预测对输出权重的贡献。
    total_output_bias_grad = first_out_bias_grad + second_out_bias_grad  # 汇总两处预测对输出偏移的贡献。

    numeric_first = numerical_array_gradient(current_vector, lambda candidate: two_step_loss(candidate, second_vector, previous_hidden, recurrent_weight, hidden_bias, weight, bias))  # 改变第一输入后重算两步，检查跨步路径。
    numeric_recurrent = numerical_array_gradient(recurrent_weight, lambda candidate: two_step_loss(current_vector, second_vector, previous_hidden, candidate, hidden_bias, weight, bias))  # 两步都使用同一个候选循环权重。
    assert np.allclose(first_input_grad, numeric_first, rtol=1e-6, atol=1e-8), "第一输入梯度不正确"  # 漏掉第二步传回的梯度会使本检查失败。
    assert np.allclose(total_recurrent_grad, numeric_recurrent, rtol=1e-6, atol=1e-8), "共享循环权重梯度不正确"  # 检查是否正确累加两步贡献。
    print("两步平均损失：", mean_loss)  # 显示两个位置的平均损失。
    print("第一状态的直接梯度：", first_direct)  # 来自第一个位置的输出层。
    print("第二步传回的梯度：", from_second)  # 经第二步循环连接传回。
    print("第一状态的总梯度：", first_total)  # 第一处 tanh 真正接收的上游梯度。
    print("第一输入梯度：", first_input_grad)  # 显示继续传给第一个输入向量的梯度。
    print("共享循环权重总梯度：", total_recurrent_grad)  # 显示两个时间步的累计贡献。
    print("第一输入最大绝对误差：", np.max(np.abs(first_input_grad - numeric_first)))  # 核对第一输入的梯度。
    print("共享权重最大绝对误差：", np.max(np.abs(total_recurrent_grad - numeric_recurrent)))  # 核对共享权重的梯度。
    print("两步反向传播与梯度累加检查：通过")  # 两项断言通过后显示。

    vectors = np.array([[[0.1, 0.2], [-0.2, 0.3], [0.05, -0.1]]], dtype=np.float64)  # 一条序列、三个位置、每个向量两维。
    targets = np.array([0, 1, 2], dtype=np.int64)  # 三个位置各自的正确候选编号。
    states = sequence_hidden(vectors, previous_hidden, recurrent_weight, hidden_bias)  # 得到形状 (1,3,2) 的全部状态。
    direct = np.zeros_like(states)  # 保存各输出层直接传给对应状态的梯度。
    for time in range(states.shape[1]):  # 分别处理三个位置的输出层。
        scores = states[:, time, :] @ weight + bias  # 将本步状态转换为候选分数。
        _, _, score_grad = loss_and_grad(scores[0], int(targets[time]))  # 计算当前位置的分数梯度。
        _, _, hidden_grad = output_backward(states[:, time, :], weight, score_grad.reshape(1, -1) / states.shape[1])  # 平均损失的缩放只在这里处理一次。
        direct[:, time, :] = hidden_grad  # 这里只保存直接路径，未来路径由反向循环合并。
    grad_vectors, grad_initial, grad_recurrent, grad_hidden_bias = sequence_backward(states, previous_hidden, recurrent_weight, direct)  # 一次反向循环处理全部位置。
    numeric_vectors = numerical_array_gradient(vectors, lambda candidate: sequence_loss(candidate, previous_hidden, recurrent_weight, hidden_bias, weight, bias, targets))  # 核对全部输入位置的梯度。
    numeric_recurrent = numerical_array_gradient(recurrent_weight, lambda candidate: sequence_loss(vectors, previous_hidden, candidate, hidden_bias, weight, bias, targets))  # 核对所有时间步累计的循环权重梯度。
    assert np.allclose(grad_vectors, numeric_vectors, rtol=1e-6, atol=1e-8), "序列输入梯度不正确"  # 检查所有输入分量。
    assert np.allclose(grad_recurrent, numeric_recurrent, rtol=1e-6, atol=1e-8), "序列循环权重梯度不正确"  # 检查共享权重的全部分量。
    print("序列状态形状：", states.shape)  # 应为 (1,3,2)。
    print("序列平均损失：", sequence_loss(vectors, previous_hidden, recurrent_weight, hidden_bias, weight, bias, targets))  # 显示三处预测的平均损失。
    print("序列输入梯度：", grad_vectors)  # 每个位置都有自己的输入向量梯度。
    print("初始状态梯度：", grad_initial)  # 显示整段序列传回初始状态的梯度。
    print("序列循环权重梯度：", grad_recurrent)  # 显示三个时间步累计的参数梯度。
    print("序列输入最大绝对误差：", np.max(np.abs(grad_vectors - numeric_vectors)))  # 显示输入梯度核对误差。
    print("序列权重最大绝对误差：", np.max(np.abs(grad_recurrent - numeric_recurrent)))  # 显示循环权重核对误差。
    print("反向时间循环与数值检查：通过")  # 两项断言通过后显示。


if __name__ == "__main__":  # 直接运行文件时启动演示。
    main()  # 调用入口，不改变现有 RNN。
