import os  # 在导入CUDA前设置确定性矩阵计算环境。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 保持同设备复现实验设置。
import math  # 计算SwiGLU宽度与初始化尺度。
import torch  # 原生PyTorch自动求导，不加载预训练权重。
from torch import nn  # 逐块实现模型组件。
from torch.nn import functional as F  # 稳定交叉熵和因果注意力。


class RMSNorm(nn.Module):  # 不减均值，用均方根归一化向量。
    def __init__(self, width, eps=1e-6):  # 每个维度有独立缩放参数。
        super().__init__()  # 注册子模块。
        self.weight = nn.Parameter(torch.ones(width))  # 不使用偏移，参数只有width个。
        self.eps = eps  # 防止除零。

    def forward(self, x):  # 输入最后一维是要归一化的特征。
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)  # 用FP32计算均方根。
        return (normalized * self.weight).to(x.dtype)  # 恢复当前计算精度。


class Attention(nn.Module):  # 分组查询注意力，查询8头、键值4头。
    def __init__(self, width, heads, kv_heads, context):  # 头数与宽度必须匹配。
        super().__init__()  # 初始化模块。
        assert width % heads == 0 and heads % kv_heads == 0, "头数不匹配"  # 每组查询共享一个KV头。
        self.heads, self.kv_heads, self.dim = heads, kv_heads, width // heads  # 例如8、4、64。
        self.q = nn.Linear(width, heads * self.dim, bias=False)  # 查询投影。
        self.k = nn.Linear(width, kv_heads * self.dim, bias=False)  # 键的投影宽度只有查询的一半。
        self.v = nn.Linear(width, kv_heads * self.dim, bias=False)  # 值与键共享头数而不共享参数。
        self.output = nn.Linear(width, width, bias=False)  # 拼接各头后的输出投影。
        self.q_norm, self.k_norm = RMSNorm(self.dim), RMSNorm(self.dim)  # 每头的Q/K归一化参数也计入总量。
        frequencies = 1.0 / (1e6 ** (torch.arange(0, self.dim, 2).float() / self.dim))  # RoPE每对维度的旋转频率。
        angles = torch.outer(torch.arange(context).float(), frequencies)  # 所有上下文位置的相位。
        angles = torch.cat((angles, angles), dim=-1)  # 对应前后半向量的旋转布局。
        self.register_buffer("cos", angles.cos(), persistent=False)  # 可重建的缓存不是可学习参数。
        self.register_buffer("sin", angles.sin(), persistent=False)  # 保存时不把缓存算作模型权重。

    def rotary(self, x):  # 在Q与K上加入相对位置信息。
        left, right = x.chunk(2, dim=-1)  # 一半表示每对旋转的一个分量。
        rotated = torch.cat((-right, left), dim=-1)  # 构造90度旋转的对应分量。
        return x * self.cos[:x.shape[-2]].to(x.dtype) + rotated * self.sin[:x.shape[-2]].to(x.dtype)  # 按当前位置施加旋转。

    def forward(self, x):  # 输入B、T、D，输出形状不变。
        batch, length, width = x.shape  # 当前批大小、序列长度与宽度。
        q = self.q(x).reshape(batch, length, self.heads, self.dim).transpose(1, 2)  # B、8、T、64。
        k = self.k(x).reshape(batch, length, self.kv_heads, self.dim).transpose(1, 2)  # B、4、T、64。
        v = self.v(x).reshape(batch, length, self.kv_heads, self.dim).transpose(1, 2)  # 与K相同头数。
        q, k = self.rotary(self.q_norm(q)), self.rotary(self.k_norm(k))  # 分别归一化并旋转，不处理V。
        repeat = self.heads // self.kv_heads  # 两个查询头使用同一KV头。
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)  # 显式展开便于教学及CPU检查。
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)  # 当前位置不能读取未来输入。
        return self.output(attended.transpose(1, 2).contiguous().reshape(batch, length, width))  # 合并各头并投影。


class Block(nn.Module):  # 两条残差分支，先归一化后计算。
    def __init__(self, width, heads, kv_heads, context):  # 同结构不同层各有独立参数。
        super().__init__()  # 初始化模块。
        intermediate = math.ceil(width * math.pi / 64) * 64  # 宽512对应SwiGLU宽1664。
        self.norm_attention, self.norm_mlp = RMSNorm(width), RMSNorm(width)  # 每条残差分支前各一层。
        self.attention = Attention(width, heads, kv_heads, context)  # 本层分组因果注意力。
        self.gate = nn.Linear(width, intermediate, bias=False)  # 非线性门控分支。
        self.up = nn.Linear(width, intermediate, bias=False)  # 值的扩展分支。
        self.down = nn.Linear(intermediate, width, bias=False)  # 压回原维度以便残差相加。

    def forward(self, x):  # 形状在整个块中保持B、T、D。
        x = x + self.attention(self.norm_attention(x))  # 汇总前文后保留原路径。
        normalized = self.norm_mlp(x)  # 前馈输入归一化。
        return x + self.down(F.silu(self.gate(normalized)) * self.up(normalized))  # SwiGLU逐位置门控和第二次残差。


class ChatLM(nn.Module):  # 模型结构可用于预训练和SFT，能力由训练阶段决定。
    def __init__(self, vocab_size=6400, width=512, layers=8, heads=8, kv_heads=4, context=256):  # 默认实际约30M参数。
        super().__init__()  # 初始化模块容器。
        self.spec = dict(vocab_size=vocab_size, width=width, layers=layers, heads=heads, kv_heads=kv_heads, context=context)  # 保存精确重建配置。
        self.embedding = nn.Embedding(vocab_size, width)  # 输入表示表与输出权重共享。
        self.blocks = nn.ModuleList(Block(width, heads, kv_heads, context) for _ in range(layers))  # 堆叠八个独立块。
        self.norm = RMSNorm(width)  # 最终输出前归一化。
        self.apply(self.initialize)  # 明确从随机权重开始。
        for block in self.blocks:  # 对残差输出投影使用较小初始尺度。
            nn.init.normal_(block.attention.output.weight, std=0.02 / math.sqrt(2 * layers))  # 控制多层残差累计幅度。
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * layers))  # 不增加任何额外参数。

    @staticmethod  # 初始化函数不依赖实例状态。
    def initialize(module):  # 表示表与线性矩阵使用正态初始化。
        if isinstance(module, (nn.Linear, nn.Embedding)):  # RMSNorm保持缩放1。
            nn.init.normal_(module.weight, std=0.02)  # 所有模型参数由固定随机种子产生。

    def forward(self, ids):  # 每个输入位置预测下一token。
        assert 0 < ids.shape[1] <= self.spec["context"], "上下文超限"  # 不静默截断训练样本。
        x = self.embedding(ids)  # B、T、512。
        for block in self.blocks:  # 逐层更新表示。
            x = block(x)  # 注意力与门控前馈仍保留相同形状。
        return F.linear(self.norm(x), self.embedding.weight)  # B、T、6400，共享输出层不另计VD参数。
