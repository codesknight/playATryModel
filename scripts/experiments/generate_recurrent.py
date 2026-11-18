import argparse  # 从命令行接收模型、开头和采样设置。
from pathlib import Path  # 定位默认最佳模型。
import numpy as np  # 使用局部随机生成器复现采样。
import torch  # 只进行前向生成。
from bounded_recurrent import ROOT, restore_checkpoint  # 复用经过检查的模型加载。


@torch.inference_mode()  # 生成过程不构建反向计算图。
def generate(model, vocab, prompt, length, temperature, top_k, seed, greedy=False):  # 从固定开头连续生成指定数量编号。
    tokens = vocab["tokens"]  # 使用模型保存的词表。
    assert prompt and length >= 0 and temperature > 0 and 1 <= top_k <= len(tokens), "生成参数无效"  # 不接受空输入或无效候选设置。
    char_to_id = {char: index for index, char in enumerate(tokens)}  # 恢复字符到编号的映射。
    history = [char_to_id.get(char, vocab["unk_id"]) for char in prompt]  # 不扩充固定词表。
    rng, generated = np.random.default_rng(seed), []  # 整段持续使用同一个随机生成器。
    device = next(model.parameters()).device  # 输入与模型在同一设备。
    model.eval()  # 不训练，不更新参数。
    for _ in range(length):  # 每步产生一个新编号。
        scores = model(torch.tensor([history[-32:]], device=device))[0, -1].cpu().numpy()  # 沿用最近三十二个输入的上下文。
        if greedy:  # 可显式观察贪心的重复情况。
            index = int(scores.argmax())  # 直接选择最高分。
        else:  # 默认使用固定温度的top-k采样。
            candidates = np.argsort(-scores, kind="stable")[:top_k]  # 只保留规定数量的高分候选。
            weights = np.exp((scores[candidates].astype(np.float64) - scores[candidates].max()) / temperature)  # 稳定计算抽样权重。
            index = int(rng.choice(candidates, p=weights / weights.sum()))  # 在候选内重新归一化并抽取编号。
        history.append(index)  # 新字加入下一步上下文。
        generated.append(index)  # 只记录续写部分。
    return prompt + "".join(tokens[index] for index in generated), generated  # 保留原开头并返回可核验编号。


def main():  # 独立加载检查点续写，不调用训练入口。
    parser = argparse.ArgumentParser()  # 设置可复现生成参数。
    parser.add_argument("--model", type=Path, default=ROOT / "model/lesson42/lesson42_rnn_best.pt")  # 默认使用本次实际选中的模型。
    parser.add_argument("--prompt", default="唐僧道：")  # 可替换成自己的开头。
    parser.add_argument("--length", type=int, default=60)  # 控制生成编号数。
    parser.add_argument("--temperature", type=float, default=0.7)  # 控制候选概率集中程度。
    parser.add_argument("--top-k", type=int, default=40)  # 限制候选数量。
    parser.add_argument("--seed", type=int, default=42)  # 固定当前环境中的抽样轨迹。
    parser.add_argument("--greedy", action="store_true")  # 指定时直接选最高分。
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")  # 没有GPU时仍可加载生成。
    args = parser.parse_args()  # 读取用户设置。
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)  # 选择可用设备。
    model, _, state = restore_checkpoint(args.model, device)  # 加载对应参数和词表，不启动更新。
    text, _ = generate(model, state["vocab"], args.prompt, args.length, args.temperature, args.top_k, args.seed, args.greedy)  # 只进行前向续写。
    print(text)  # 显示原始生成结果，不做人工润色。


if __name__ == "__main__":  # 直接运行才生成。
    main()  # 模型质量限制与训练报告中说明相同。
