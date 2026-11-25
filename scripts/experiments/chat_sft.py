import argparse  # 独立聊天入口，不启动训练。
from pathlib import Path  # 读取指定推理文件。
from train_chat_sft import ChatLM, torch, reply, ROOT  # 复用正式训练结构与停止规则。
from sft_batches import load_tokenizer  # 复用6400词BPE。
from prepare_chat_corpus import sha  # 检查模型和词表对应关系。


def main():  # 支持单问、追问和交互多轮。
    parser = argparse.ArgumentParser()  # 参数只控制生成。
    parser.add_argument("--model", type=Path, default=ROOT / "model/lesson53/chat_fp16.pt")  # 默认加载正式SFT最佳导出。
    parser.add_argument("--prompt")  # 指定则运行一次问题后退出。
    parser.add_argument("--followup")  # 可追加一轮真实生成历史上的追问。
    parser.add_argument("--sample", action="store_true")  # 默认贪心，可选固定top-k采样。
    parser.add_argument("--max-new-tokens", type=int, default=48)  # 控制回答长度与上下文预算。
    args = parser.parse_args()  # 读取生成设置。
    assert 0 < args.max_new_tokens <= 64, "本入口回答长度需在1到64之间"  # 不越过已验证短上下文场景。
    torch.set_num_threads(4)  # CPU模式限制线程。
    state = torch.load(args.model, map_location="cpu", weights_only=True)  # 不载入优化器或训练循环。
    assert state["format"] == "chat_sft53_inference_v1", "请指定正式SFT推理文件"  # 不混用预训练续写入口。
    assert state["tokenizer_sha256"] == sha((ROOT / "data/chat48/tokenizer/tokenizer.json").read_bytes()), "词表不匹配"  # 编号必须对应相同文本。
    tokenizer, device = load_tokenizer(), "cuda" if torch.cuda.is_available() else "cpu"  # 没有显卡也能加载约60MB文件。
    model = ChatLM(**state["model_config"]).to(device)  # 主存储使用FP32，导出文件是FP16。
    model.load_state_dict(state["model_state"])  # 载入最佳SFT权重。
    messages = [{"role": "system", "content": state["system"]}]  # 保留训练时相同系统提示。
    def answer(prompt):  # 使用本模型之前实际生成的回答作为历史。
        proposed = messages + [{"role": "user", "content": prompt}]  # 失败前不修改正式历史。
        try:  # 长历史不静默截断。
            result = reply(model, tokenizer, proposed, args.max_new_tokens, args.sample)  # 结束符停止且不训练。
        except AssertionError as error:  # 将上下文上限转成可理解提示。
            print(str(error), "；输入 /clear 清空对话。")  # 不凭空声称记得已经丢弃的历史。
            return  # 保留之前有效历史。
        print("助手：", result["text"] or "（空回答）")  # 空回答也如实展示。
        print("停止原因：", result["stop_reason"])  # 区分正常结束与长度截断。
        messages.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": result["text"]}])  # 下一轮保留真实模型输出。
    if args.prompt:  # 自动检查或单次体验入口。
        answer(args.prompt)  # 不把用户问题追加为助手答案。
        if args.followup:  # 多轮命令行测试。
            answer(args.followup)  # 读取第一轮真实回答。
        return  # 完成后退出，不启动交互等待。
    print("输入 /clear 清空，/exit 退出。")  # 交互入口保持简洁。
    while True:  # 逐轮对话，无后台训练。
        try:  # 允许Ctrl+C或输入结束退出。
            prompt = input("你：").strip()  # 读取用户文字。
        except (EOFError, KeyboardInterrupt):  # 正常结束会话。
            break  # 不保存或发送用户聊天内容。
        if prompt == "/exit":  # 显式退出命令。
            break  # 不再生成。
        if prompt == "/clear":  # 显式重置短上下文。
            messages[:] = messages[:1]  # 只保留系统提示。
        elif prompt:  # 空输入不触发生成。
            answer(prompt)  # 参数保持不变。


if __name__ == "__main__":  # 导入不加载大模型或等待输入。
    main()  # 运行只进行推理。
