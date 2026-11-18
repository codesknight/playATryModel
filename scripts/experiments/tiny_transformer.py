import os  # 在加载CUDA前配置可复现矩阵计算。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 同一设备上使用确定性矩阵运算。
import json  # 加载词表和数据清单。
import numpy as np  # 复现top-k采样。
import torch  # 使用PyTorch自动求导。
from torch import nn  # 实现每一层网络。
from torch.nn import functional as F  # 使用稳定交叉熵和因果注意力。
from prepare_transformer_data import ROOT, DIRECTORY, combined  # 复用可追溯的数据组合。


class Block(nn.Module):  # 一个先归一化的Transformer块。
    def __init__(self, width, heads):  # 隐藏维度必须可以平分为多个头。
        super().__init__()  # 注册参数。
        assert width % heads == 0, "隐藏维度不能平分注意力头"  # 每头维度为width除以heads。
        self.heads = heads  # 保存并行注意力头数量。
        self.norm_attention = nn.LayerNorm(width)  # 注意力前的层归一化。
        self.qkv = nn.Linear(width, 3 * width)  # 一次线性变换产生查询、键和值。
        self.projection = nn.Linear(width, width)  # 合并多头后的线性投影。
        self.norm_mlp = nn.LayerNorm(width)  # 前馈网络前的层归一化。
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))  # 逐位置的前馈网络。

    def forward(self, x):  # 输入和输出形状都是批大小、长度、隐藏维度。
        batch, length, width = x.shape  # 明确各维度的含义。
        parts = self.qkv(self.norm_attention(x)).chunk(3, dim=-1)  # 三个张量各为B、T、D。
        q, k, v = [part.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2) for part in parts]  # 调整为B、头数、T、每头维度。
        attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)  # 下三角遮罩保证不能读取未来位置。
        merged = attention.transpose(1, 2).contiguous().reshape(batch, length, width)  # 拼回多头输出。
        x = x + self.projection(merged)  # 第一条残差路径保留原表示。
        return x + self.mlp(self.norm_mlp(x))  # 第二条残差路径加入逐位置非线性变换。


class TinyTransformer(nn.Module):  # 只有因果自注意力的自回归语言模型。
    def __init__(self, vocab_size, width=128, heads=4, layers=4, context=128):  # 默认约一百多万参数。
        super().__init__()  # 初始化模块容器。
        self.spec = dict(vocab_size=vocab_size, width=width, heads=heads, layers=layers, context=context)  # 文件保存重建所需结构。
        self.embedding = nn.Embedding(vocab_size, width)  # 字符编号转向量。
        self.position = nn.Embedding(context, width)  # 学习每个上下文位置的向量。
        self.blocks = nn.Sequential(*(Block(width, heads) for _ in range(layers)))  # 堆叠独立参数的注意力块。
        self.norm = nn.LayerNorm(width)  # 输出前归一化。
        self.apply(self.initialize)  # 明确初始化尺度，不加载预训练大模型。

    @staticmethod  # 初始化不依赖实例中的其他字段。
    def initialize(module):  # 线性层与表示表使用小尺度正态分布。
        if isinstance(module, (nn.Linear, nn.Embedding)):  # LayerNorm保持默认缩放1偏移0。
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 固定分布而非使用任意大小的初始权重。
            if isinstance(module, nn.Linear) and module.bias is not None:  # 表示表没有偏移参数。
                nn.init.zeros_(module.bias)  # 线性偏移从零开始。

    def forward(self, ids):  # 返回每个位置的下一字符分数。
        assert 0 < ids.shape[1] <= self.spec["context"], "上下文长度超限"  # 不静默截断训练序列。
        positions = torch.arange(ids.shape[1], device=ids.device)  # 位置编号从0开始。
        x = self.embedding(ids) + self.position(positions)  # 将内容与顺序信息相加。
        return F.linear(self.norm(self.blocks(x)), self.embedding.weight)  # 输出层与输入表示表共享参数，节省VD个参数。


def load_windows(vocab, split, context=128):  # 按完整文档分别编码，不跨章节或故事。
    mapping = {char: index for index, char in enumerate(vocab["tokens"])}  # 新词表只来自混合训练集。
    xs, ys, sources = [], [], []  # 同时保留窗口的语料来源。
    for row in combined(split):  # 训练和验证各自组合，测试不在此训练入口读取。
        ids = [vocab["bos_id"]] + [mapping.get(char, vocab["unk_id"]) for char in row["text"]] + [vocab["eos_id"]]  # 显式标记文档开始和结束。
        for start in range(0, len(ids) - 1, context):  # 每个目标只出现一次，保留不足128的尾段。
            inputs, targets = ids[start:start + context], ids[start + 1:start + context + 1]  # 左右错位一格形成teacher forcing。
            inputs = inputs[:len(targets)]  # 文档最后一段不把EOS当成额外输入目标。
            xs.append(inputs + [0] * (context - len(inputs)))  # 右侧补零只为组批，真实前文不会看见右侧补齐。
            ys.append(targets + [-100] * (context - len(targets)))  # 交叉熵忽略补齐位置。
            sources.append(row["source"])  # 可分别报告四书和新故事验证结果。
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long), sources  # 保留CPU数据，在每批时送入GPU。


@torch.inference_mode()  # 验证不计算参数梯度。
def evaluate(model, data, batch_size=32):  # 完整验证并按实际目标数加权。
    x, y, sources = data  # 验证窗口及来源标签。
    device = next(model.parameters()).device  # 自动匹配模型设备。
    was_training = model.training  # 记录调用之前的模式。
    model.eval()  # 当前实现无dropout，但仍显式区分模式。
    totals = {source: {"loss_sum": 0.0, "correct": 0, "positions": 0} for source in sorted(set(sources))}  # 分来源统计防止混合平均掩盖差异。
    for start in range(0, len(x), batch_size):  # 尾批不丢弃。
        target = y[start:start + batch_size].to(device)  # 只搬运当前批次。
        logits = model(x[start:start + batch_size].to(device))  # 只前向，不能更新参数。
        losses = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="none").reshape_as(target)  # 补齐位置损失为零。
        correct = (logits.argmax(-1) == target) & (target != -100)  # 补齐不能计入准确率分母。
        for offset, source in enumerate(sources[start:start + batch_size]):  # 将实际位置加回对应语料。
            totals[source]["loss_sum"] += float(losses[offset].sum().item())  # 不按窗口平均掩盖尾段长度。
            totals[source]["correct"] += int(correct[offset].sum().item())  # 真实字符预测命中数。
            totals[source]["positions"] += int((target[offset] != -100).sum().item())  # 包含EOS，不包含BOS和补齐。
    model.train(was_training)  # 不改变后续训练行为。
    all_positions = sum(item["positions"] for item in totals.values())  # 两来源真实目标总数。
    return {"loss": sum(item["loss_sum"] for item in totals.values()) / all_positions, "positions": all_positions, "accuracy": sum(item["correct"] for item in totals.values()) / all_positions, "by_source": {source: {"loss": item["loss_sum"] / item["positions"], "positions": item["positions"], "accuracy": item["correct"] / item["positions"]} for source, item in totals.items()}}  # 保留可解释的来源指标。


@torch.inference_mode()  # 独立生成也不求梯度。
def generate(model, vocab, prompt, length=60, seed=42, greedy=False):  # 使用固定的温度和top-k，避免持续调采样掩盖问题。
    model.eval()  # 固定推断行为。
    device = next(model.parameters()).device  # 输入匹配模型设备。
    mapping = {char: index for index, char in enumerate(vocab["tokens"])}  # 编号含义来自检查点。
    history = [vocab["bos_id"]] + [mapping.get(char, 0) for char in prompt]  # 生成时提供完整前缀和文档起点。
    generated, rng = [], np.random.default_rng(seed)  # 每段样例独立可复现。
    for _ in range(length):  # 最多生成指定数量的新编号。
        logits = model(torch.tensor([history[-model.spec["context"]:]], device=device))[0, -1].cpu().numpy().astype(np.float64)  # 每次只读取最后位置的下一字分数。
        logits[:2] = -np.inf  # 不向用户输出未知符或文档开始符，EOS仍允许结束。
        candidates = np.argsort(-logits, kind="stable")[:40]  # 固定top-k40候选。
        weights = np.exp((logits[candidates] - logits[candidates].max()) / 0.7)  # 固定温度0.7。
        index = int(logits.argmax()) if greedy else int(rng.choice(candidates, p=weights / weights.sum()))  # 分别观察贪心和采样。
        generated.append(index)  # EOS也记录为实际生成编号。
        if index == vocab["eos_id"]:  # 文档结束后不再硬续写。
            break  # 不把EOS标签输出为普通文字。
        history.append(index)  # 新字回到下一次上下文。
    text = prompt + "".join(vocab["tokens"][index] for index in generated if index != vocab["eos_id"])  # 解码正文并保留原始编号。
    return {"prompt": prompt, "mode": "greedy" if greedy else "topk40_t07", "seed": seed, "text": text, "generated_ids": generated}  # 用于固定对照与人工阅读。


def restore(path, device):  # 恢复结构、参数与AdamW状态所需的文件内容。
    state = torch.load(path, map_location="cpu", weights_only=True)  # 不加载任意自定义Python对象。
    assert state["format"] == "playatry_transformer_v1", "文件格式错误"  # 与RNN文件明确区分。
    model = TinyTransformer(**state["model_config"]).to(device)  # 按原结构重建小模型。
    model.load_state_dict(state["model_state"])  # 载入共享表示表与所有注意力层。
    optimizer = torch.optim.AdamW(model.parameters(), lr=state["learning_rate"], betas=(0.9, 0.95), weight_decay=0.01)  # 与训练入口保持一致。
    optimizer.load_state_dict(state["optimizer_state"])  # 恢复m、v、t及原参数组。
    return model, optimizer, state  # 后续续训还应使用保存的数据游标与随机状态。
