from collections import Counter  # 独立统计固定样例重复比例。
import hashlib  # 核对文件及训练数据顺序指纹。
from importlib import import_module  # 只导入训练工具，不执行训练入口。
import json  # 读取实际训练报告。
from pathlib import Path  # 定位仓库与检查点。
import sys  # 设置实验模块的导入路径。
import tempfile  # 用临时文件测试中断恢复。
import numpy as np  # 重建窗口排列及比较浮点指标。
import torch  # 在CPU上只读验证GPU训练产物。

ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于scripts/ci。
sys.path.insert(0, str(ROOT / "scripts/experiments"))  # 允许导入实际训练脚本。


def check_resume(module, kind):  # 用微型模型核对真实不中断和保存恢复的下一步。
    torch.manual_seed(7)  # 固定这个独立测试的随机状态。
    model = module.RecurrentLM(9, kind, width=8, layers=1)  # 小维度避免在CI重训真实模型。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 新建对应优化器。
    x = torch.randint(0, 9, (5, 32))  # 构造少量合法输入。
    y = x.roll(-1, dims=1)  # 此处仅测试恢复，不声称是实际语料。
    loss = torch.nn.functional.cross_entropy(model(x[:2]).flatten(0, 1), y[:2].flatten())  # 第一批平均损失。
    loss.backward()  # 创建非零梯度历史。
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 使用实际训练相同裁剪。
    optimizer.step()  # 先进行一次真实更新。
    state = {"format": "playatry_torch_recurrent_v1", "model_config": model.spec, "model_state": module.cpu_tree(model.state_dict()), "optimizer_state": module.cpu_tree(optimizer.state_dict()), "learning_rate": 0.001, "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": [], "cursor": {"next_epoch": 0, "next_start": 2}, "order_epoch": 0, "epoch_order": torch.arange(5), "data_seed": 47, "batch_size": 2}  # 保存与实际格式相同的关键状态。
    with tempfile.TemporaryDirectory(prefix="playatry_resume_") as directory:  # 临时目录只用于这个测试。
        path = Path(directory) / "state.pt"  # 临时状态文件不进入仓库。
        torch.save(state, path)  # 保存当前参数与动量。
        assert module.verify_resume(model, optimizer, path, (x, y), torch.device("cpu")) <= 1e-6, "恢复后下一步不一致"  # 比较两个副本的参数和Adam历史。


def main():  # 只读复核真实产物，不重复两轮训练。
    torch.set_num_threads(2)  # 限制CI的CPU线程争用。
    module = import_module("bounded_recurrent")  # 导入不会启动实验。
    generator = import_module("generate_recurrent")  # 单独生成入口不应启动训练。
    report_path = ROOT / "docs/training_runs/lesson42.json"  # 正式报告包含全部候选。
    saved = report_path.read_bytes()  # 留下原报告，核对检查没有覆盖它。
    report = json.loads(saved)  # 读取预算、配置及实测指标。
    config = report["config"]  # 预算以执行前保存的值为准。
    assert report["status"] == "completed" and config["max_steps"] == 2394 and config["max_epochs"] == 2 and config["max_seconds"] == 360, "实验预算或状态错误"  # 不把扩大预算的结果冒充原实验。
    assert config["width"] == 128 and config["layers"] == 2 and config["batch_size"] == 64 and config["learning_rate"] == 0.001, "对照配置错误"  # 两组维度及Adam设置相同。
    vocab, train, validation = module.load_data(torch.device("cpu"))  # 重建旧四书的同一窗口集合。
    assert report["train_sha256"] == vocab["train_sha256"] and report["validation_positions"] == 316960, "数据来源或验证范围错误"  # 外部故事样本没有混入。
    assert [row["kind"] for row in report["models"]] == ["rnn", "gru"] and report["matched_exposure"], "对照模型或数据预算未匹配"  # 当前提交的实际实验确已匹配。
    assert report["models"][0]["public_initial_sha256"] == report["models"][1]["public_initial_sha256"], "公共初始化不一致"  # 公共表示表和输出层相同。
    exposure = hashlib.sha256()  # 独立计算两整轮的窗口顺序。
    for epoch in range(2):  # 当前实验完成了两整轮。
        order = np.random.default_rng(config["data_seed"] + epoch).permutation(len(train[0])).astype(np.int64)  # 与模型初始化随机数分离。
        for start in range(0, len(order), 64):  # 最后一批按实际四十行计入。
            exposure.update(order[start:start + 64].tobytes())  # 核对真实训练数据前缀。
    directory = ROOT / "model/lesson42"  # 不依赖报告中的本机绝对路径。
    for name, artifact in report["artifacts"].items():  # 所有保存的状态文件都可追溯。
        path = directory / name  # 定位对应检查点。
        assert path.stat().st_size == artifact["bytes"] < 1_000_000_000 and hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"], "检查点大小或哈希不一致"  # 完整状态也小于一GB。
    for row in report["models"]:  # 各结构分别核验最佳模型及数据预算。
        assert row["completed_steps"] == 2394 and row["seen_positions"] == 2 * train[1].numel() == 4901376 and row["exposure_sha256"] == exposure.hexdigest(), "实际训练预算或数据顺序错误"  # 两组不是只在配置上看起来相同。
        assert row["restore_next_batch_max_abs_diff"] <= 1e-6 and row["reload_verified"], "恢复核验未通过"  # 原GPU运行实际核对了下一批。
        check_resume(module, row["kind"])  # 在CPU独立核对同类恢复逻辑。
        model, optimizer, state = module.restore_checkpoint(directory / row["best_checkpoint"], torch.device("cpu"))  # 使用受限加载模式恢复。
        assert state["vocab"] == vocab and state["step"] == row["best_step"] and sum(value.numel() for value in model.parameters()) == row["parameter_count"], "模型元数据错误"  # 参数数及词表均与报告一致。
        assert all(int(value["step"].item()) == row["best_step"] for value in optimizer.state.values()), "Adam历史位置错误"  # 优化器历史与最佳参数来自同一时刻。
        metrics = module.evaluate(model, validation)  # 在CPU复算GPU模型的完整验证损失。
        assert metrics["positions"] == 316960 and np.isclose(metrics["loss"], row["best_loss"], rtol=1e-4, atol=1e-4), "完整验证损失不一致"  # CPU和CUDA允许小浮点差。
        first = generator.generate(model, vocab, "唐僧道：", 12, 0.7, 40, 42)  # 独立生成固定种子的短续写。
        repeated = generator.generate(model, vocab, "唐僧道：", 12, 0.7, 40, 42)  # 在当前CPU环境中重复同一设置。
        assert first == repeated and len(first[1]) == 12, "生成入口无法复现"  # 不与GPU完整字符串作黄金值比较。
        greedy = generator.generate(model, vocab, "春风", 12, 0.7, 1, 42, True)  # 取得直接最高分的轨迹。
        top_one = generator.generate(model, vocab, "春风", 12, 0.7, 1, 43, False)  # 只有一个候选时抽样应与贪心一致。
        assert greedy == top_one, "生成入口的候选映射不正确"  # 核对top-k退化行为。
        best = min(row["history"], key=lambda item: item["loss"])  # 独立按全部已评估候选选优。
        assert best["step"] == row["best_step"] and np.isclose(best["loss"], row["best_loss"]), "最佳候选选择错误"  # 不默认最后一步最好。
        for candidate in row["history"]:  # 每个候选保留六段固定原始文字。
            assert len(candidate["samples"]) == 6, "固定样例不完整"  # 不只展示单句好结果。
            for sample in candidate["samples"]:  # 核对文本解码和重复指标。
                ids = sample["generated_ids"]  # 续写编号不包含开头。
                assert len(ids) == 60 and sample["text"] == sample["prompt"] + "".join(vocab["tokens"][value] for value in ids), "样例文字与编号不一致"  # 不比较跨平台整段随机字符串。
                counts = Counter(tuple(ids[index:index + 4]) for index in range(57))  # 独立计算重复出现的片段。
                assert np.isclose(sample["repeat4"], sum(value - 1 for value in counts.values()) / 57), "重复统计错误"  # 重复率不等于语言质量。
        print(f"lesson42 {row['kind']} CPU validation={metrics['loss']:.6f} resume passed", flush=True)  # 显示跨设备核对结果。
    winner = min(report["models"], key=lambda item: item["best_loss"])  # 记录同预算两组的赢家。
    assert report["paired_selected_kind"] == winner["kind"] and np.isclose(report["selected_loss"], min(winner["best_loss"], report["legacy_reference_loss"])), "最终候选选择错误"  # 旧模型仍是参考候选。
    assert report_path.read_bytes() == saved, "检查覆盖了正式报告"  # 所有核验仅只读已有产物。
    manifest = json.loads((ROOT / "data/external_samples/tiny_chinese_stories/manifest.json").read_text(encoding="utf-8"))  # 外部候选样本单独追溯。
    sample_path = ROOT / "data/external_samples/tiny_chinese_stories/sample.jsonl"  # 不读取它参与模型评估。
    assert manifest["rows"] == 50 and manifest["used_in_lesson42_training"] is False and hashlib.sha256(sample_path.read_bytes()).hexdigest() == manifest["sample_sha256"], "候选语料记录不一致"  # 采集与训练分开管理。
    print("lesson42 matched budget, checkpoints and separate data sample passed", flush=True)  # 不重新执行两轮模型训练。


if __name__ == "__main__":  # 直接运行才开始核验。
    main()  # 原有NumPy课程仍可使用原依赖运行。
