from importlib import import_module  # 复用已有训练、验证和生成函数。
from itertools import islice  # 只取预先规定的两百个训练批次。
from pathlib import Path  # 定位数据、权重及报告。
import hashlib  # 检查起点权重文件没有被覆盖。
import json  # 读取已有配置并保存对照实验。
import random  # 固定本次训练章节的打乱顺序。
import numpy as np  # 核对参数并检查损失是否有效。

ROOT = Path(__file__).resolve().parents[2]  # 从 scripts/learn 定位项目根目录。
network = import_module("07_tiny_rnn")  # 使用相同模型及梯度裁剪。
training = import_module("11_train_steps")  # 复用加载、保存和固定参数评估。
epoch_tools = import_module("17_train_epochs")  # 复用完整验证评估函数。
decoding = import_module("12_generate_text")  # 保持相同生成设置观察文字。
STEPS = 200  # 每组只更新两百步，不把这次短实验称为完整一轮。
RATES = (("lr010", 0.1), ("lr003", 0.03))  # 在看结果之前固定两种学习率。

def run_updates(model, batches, learning_rate):  # 在指定批次上连续更新同一个模型。
    for local_step, batch in enumerate(batches, start=1):  # 两组都从相同列表的第一批开始。
        loss, raw_grads = model.loss_and_grad(batch["x"], batch["y"])  # 用当前参数计算损失和完整梯度。
        assert np.isfinite(loss), "训练损失无效"  # 非有限损失不能继续更新。
        grads, _, _ = network.clip_gradients(raw_grads, 1.0)  # 两组都使用同一全局裁剪阈值。
        for name in model.params:  # 完整反向结束后再更新五组参数。
            model.params[name] -= learning_rate * grads[name]  # 本实验唯一主动改变的训练设置是学习率。
        assert all(np.isfinite(value).all() for value in model.params.values()), "更新后参数无效"  # 及时发现数值异常。
        if local_step % 50 == 0:  # 每五十步显示一次进度。
            print(f"lr={learning_rate} step={local_step}/{STEPS} batch_loss={loss:.6f}", flush=True)  # 当前批次损失不能替代完整验证损失。


def main(output_dir=None):  # 从共同起点比较两条独立训练分支。
    data = import_module("06_make_windows")  # 复用训练词表、窗口与批次函数。
    source = json.loads((ROOT / "docs/training_runs/lesson33.json").read_text(encoding="utf-8"))  # 使用已经选出的三轮最佳模型。
    assert source["status"] == "completed" and source["train_sha256"] == data.vocab["train_sha256"], "起点或数据版本不正确"  # 保证实验来源明确。
    source_path = ROOT / "model" / source["best_checkpoint"]  # 旧模型只读，新增分支使用不同文件名。
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()  # 记录源文件完整哈希。
    initial, vocab, start_step = training.load_checkpoint(source_path, network)  # 加载共同起点用于核对。
    assert vocab == data.vocab, "模型词表不一致"  # 输入编号必须对应正确表示表。
    output_dir = ROOT / "model" if output_dir is None else Path(output_dir)  # 默认保存到 model，也允许老师预跑时使用临时目录。
    records = list(data.train_records)  # 从同一原始章节顺序复制。
    random.Random(45).shuffle(records)  # 两组使用同一预先固定的章节顺序。
    batches = list(islice(data.iter_batches(data.iter_windows(records, 32, 32), 16), STEPS))  # 缓存相同的两百批，避免两组使用不同数据。
    assert len(batches) == STEPS and sum(batch["y"].size for batch in batches) == 102400, "训练规模不正确"  # 每组两百批共十万两千四百个目标位置。
    for batch in batches:  # 将缓存的输入和目标都设为只读。
        batch["x"].setflags(write=False)  # 防止训练过程意外修改共享输入。
        batch["y"].setflags(write=False)  # 防止训练过程意外修改共享正确答案。
    chapter_to_book = {record["id"]: record["book"] for record in records}  # 准备统计这段训练实际覆盖哪些书。
    book_windows = {record["book"]: 0 for record in records}  # 没有覆盖到的书也保留零计数。
    for batch in batches:  # 遍历缓存的训练来源信息。
        for chapter_id in batch["chapter_ids"]:  # 每一行对应一个训练窗口。
            book_windows[chapter_to_book[chapter_id]] += 1  # 短实验不假定四本书覆盖均衡。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 所有候选都使用同一完整验证集。
    initial_loss, positions = epoch_tools.evaluate_full(initial, data, validation)  # 重新测量共同起点，防止继续训练反而变差却被忽略。
    assert positions == 316960 and np.isclose(initial_loss, source["best_validation_loss"], rtol=1e-6, atol=1e-6), "起点指标不一致"  # 核对完整验证范围及权重。
    history = [{"label": "starting_model", "checkpoint": source_path.name, "completed_steps": start_step, "validation_loss": initial_loss}]  # 原始模型也参与最佳选择。
    print("共同起点完整验证损失：", initial_loss, flush=True)  # 两组使用同一条起跑线。

    for label, learning_rate in RATES:  # 每次循环是一组独立实验，不是同一模型的连续阶段。
        model, current_vocab, loaded_step = training.load_checkpoint(source_path, network)  # 每组开始重新加载共同起点。
        assert current_vocab == vocab and loaded_step == start_step and all(np.array_equal(value, initial.params[name]) for name, value in model.params.items()), "分支起点不同"  # 不能让第二组接着第一组训练。
        run_updates(model, batches, learning_rate)  # 只在本组内部连续更新。
        loss, current_positions = epoch_tools.evaluate_full(model, data, validation)  # 使用完整验证损失评价本组最终参数。
        assert current_positions == positions, "验证范围不一致"  # 同一分母才能公平比较。
        path = output_dir / f"lesson36_{label}.npz"  # 为两组分别命名，保留原权重。
        training.save_checkpoint(path, model, vocab, start_step + STEPS)  # 每组都只在共同起点上新增两百步。
        greedy, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32)  # 保持原来的开头和生成长度。
        history.append({"label": label, "learning_rate": learning_rate, "checkpoint": path.name, "completed_steps": start_step + STEPS, "validation_loss": loss, "change_from_start": loss - initial_loss, "greedy": greedy})  # 同时保存分数、变化量和生成观察。
        print(f"lr={learning_rate} full_validation={loss:.6f} change={loss-initial_loss:+.6f}", flush=True)  # 负变化量表示优于起点。
        print("贪心：", repr(greedy), flush=True)  # 文字不通顺或重复也如实保留。
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash, "共同起点文件被修改"  # 不允许新实验覆盖旧最佳权重。
    assert all(np.array_equal(value, initial.params[name]) for name, value in training.load_checkpoint(source_path, network)[0].params.items()), "共同起点参数被改变"  # 保留的起点对象也不能被分支更新影响。
    selected = min(history, key=lambda row: row["validation_loss"])  # 在起点和两组最终检查点中选最低分，相等保留较早候选。
    report = {  # 记录固定预算的短实验，而不是宣称找到全局最佳学习率。
        "starting_checkpoint": source_path.name, "starting_sha256": source_hash, "train_sha256": vocab["train_sha256"],  # 保存模型与数据来源。
        "config": {"steps_per_branch": STEPS, "learning_rates": [rate for _, rate in RATES], "batch_size": 16, "seq_length": 32, "max_norm": 1.0, "chapter_shuffle_seed": 45},  # 其余训练条件固定。
        "training_positions_per_branch": 102400, "cached_xy_bytes": sum(batch["x"].nbytes + batch["y"].nbytes for batch in batches), "book_windows": book_windows,  # 记录数据规模、数组内存及实际书籍覆盖。
        "validation_positions": positions, "history": history, "selected_label": selected["label"], "selected_checkpoint": selected["checkpoint"],  # 被选中的也可能仍是原模型。
    }  # 不使用测试集，不修改上一课的最佳模型记录。
    print("本次候选中选中：", selected["label"], selected["validation_loss"], flush=True)  # 只在本实验观察过的三个候选之间选择。
    print("实际训练窗口覆盖：", book_windows, flush=True)  # 明确短训练不等同于完整遍历。
    return report  # 返回结果供入口保存。

if __name__ == "__main__":  # 直接运行时才训练本课两个分支并保存报告。
    result = main()  # 默认把分支权重写入 model 目录。
    path = ROOT / "docs/training_runs/lesson36.json"  # 使用独立实验报告，不覆盖旧记录。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存可读 JSON。
    print("实验报告：", path)  # 输出结果位置。
