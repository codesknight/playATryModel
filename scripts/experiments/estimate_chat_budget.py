import json  # 输出可核对的模型体积估算。
import math  # 按参考结构将前馈维度向64的倍数取整。
from pathlib import Path  # 定位已保存的预算方案。
ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于实验目录。


def estimate(width, layers, vocab_size=6400, heads=8, kv_heads=4):  # 估算MiniMind固定版本的稠密结构，不创建或训练模型。
    assert width % heads == 0 and heads % kv_heads == 0, "头数配置不匹配"  # GQA的查询头必须可被KV头整除。
    head_dim = width // heads  # 每个注意力头的维度。
    intermediate = math.ceil(width * math.pi / 64) * 64  # 参考版本默认SwiGLU中间维度。
    attention = 2 * width * heads * head_dim + 2 * width * kv_heads * head_dim  # 无偏移的Q、K、V、输出四个矩阵。
    feedforward = 3 * width * intermediate  # SwiGLU包含gate、up、down三个矩阵。
    norms = 2 * width + 2 * head_dim  # 每块两个RMSNorm及Q/K的逐头RMSNorm。
    parameters = vocab_size * width + layers * (attention + feedforward + norms) + width  # 共享输入输出表示表，末层另有RMSNorm。
    return {"width": width, "layers": layers, "vocab_size": vocab_size, "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim, "intermediate_size": intermediate, "parameters": parameters, "weights_fp32_bytes": parameters * 4, "weights_fp16_bytes": parameters * 2, "approx_fp32_weights_and_two_adam_moments_bytes": parameters * 12, "approx_fp32_weights_gradients_and_two_adam_moments_bytes": parameters * 16}  # 不把参数状态当成全部训练显存。


def main():  # 预算核对不分配GPU，也不下载模型或语料。
    plan = json.loads((ROOT / "docs/plans/chat_model_budget.json").read_text(encoding="utf-8"))  # 读取候选配置，实际进展以训练报告为准。
    for candidate in plan["candidates"]:  # 分别核对约30M和64M方案。
        result = estimate(candidate["width"], candidate["layers"])  # 通过公式独立复算预算。
        assert result == candidate, "预算文件与公式不一致"  # 改结构后必须同步预算。
        assert result["weights_fp32_bytes"] < plan["weight_limit_bytes"], "推理权重超过1GB"  # 限制适用于实际导出文件，公式只是预估。
        print(json.dumps(result, ensure_ascii=False, indent=2))  # 清楚区分FP32与FP16权重大小。
    print("候选结构与1GB权重预算核对通过；实际30M预训练进展见docs/training_runs/lesson48.json，64M仍为候选。")  # 不冒充真实显存测量或训练结果。


if __name__ == "__main__":  # 导入公式时不执行预算检查。
    main()  # 学习者可在base环境直接运行。
