from importlib import import_module  # 复用数据、模型、检查点和生成诊断。
from itertools import islice  # 固定每组训练两百个批次。
from pathlib import Path  # 定位项目文件及实验输出。
import hashlib  # 检查共同起点权重没有变化。
import json  # 读取旧报告并保存新实验。
import random  # 固定训练章节顺序。
import numpy as np  # 保存优化器状态并执行逐元素计算。

ROOT = Path(__file__).resolve().parents[2]  # 当前脚本位于scripts/learn。
network = import_module("07_tiny_rnn")  # 模型结构保持不变。
training = import_module("11_train_steps")  # 复用权重加载和保存。
epoch_tools = import_module("17_train_epochs")  # 复用完整验证函数。
diag = import_module("23_generation_diagnostics")  # 保持上一课的生成和重复统计规则。
STEPS = 200  # 每组预先固定两百次更新。
CONFIGS = (("sgd", 0.03), ("adam", 0.001))  # 比较这两个指定配置，不宣称学习率已经分别调到最优。

class Adam:  # 为五组参数分别保存梯度历史。
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):  # 创建一次优化器，随后跨批次复用。
        assert np.isfinite(learning_rate) and learning_rate > 0 and np.isfinite(epsilon) and epsilon > 0, "学习率或稳定项无效"  # 两个数都必须为有限正数。
        assert 0 <= beta1 < 1 and 0 <= beta2 < 1, "历史保留系数无效"  # 系数控制指数移动平均。
        self.lr, self.beta1, self.beta2, self.epsilon = learning_rate, beta1, beta2, epsilon  # 保存优化器超参数。
        self.m = {name: np.zeros_like(value) for name, value in params.items()}  # m保存梯度的指数移动平均。
        self.v = {name: np.zeros_like(value) for name, value in params.items()}  # v保存梯度平方的指数移动平均。
        self.t = 0  # 新建Adam的历史从零开始，与模型累计训练步数分开。

    def step(self, params, grads):  # 用本批完整梯度更新所有参数。
        assert params.keys() == grads.keys() == self.m.keys(), "参数名称不一致"  # 每组参数必须有对应梯度和状态。
        self.t += 1  # 每个批次只增加一次，不能在参数循环里增加。
        for name in params:  # 每个状态数组与对应参数逐元素对齐。
            g = grads[name]  # 本课传入已经完成全局裁剪的梯度。
            assert g.shape == params[name].shape and np.isfinite(g).all(), "梯度形状或数值无效"  # 避免错误广播或无效更新。
            self.m[name] *= self.beta1  # 保留过去梯度平均的一部分。
            self.m[name] += (1 - self.beta1) * g  # 加入当前梯度。
            self.v[name] *= self.beta2  # 保留过去梯度平方平均的一部分。
            self.v[name] += (1 - self.beta2) * g * g  # 加入当前梯度的逐元素平方。
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)  # 修正从零初始化造成的偏小。
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)  # 对平方平均做同样的初始化偏差修正。
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)  # 用历史方向和平方尺度调整每个参数的更新。

def check_first_step():  # 用能手算的小例子验证公式和批次计数。
    params = {"weight": np.array([1.0]), "zero": np.array([2.0])}  # 两组双精度参数便于核对。
    optimizer = Adam(params)  # 默认学习率0.001，两组状态均从零开始。
    optimizer.step(params, {"weight": np.array([0.4]), "zero": np.array([0.0])})  # 一次更新同时处理两个参数组。
    expected = 1.0 - 0.001 * 0.4 / (0.4 + 1e-8)  # 第一步修正后m为0.4，v为0.16。
    assert optimizer.t == 1 and np.isclose(params["weight"][0], expected, rtol=0, atol=1e-12) and params["zero"][0] == 2.0, "Adam首步公式或计数错误"  # t只增一次，零历史零梯度不移动。
    print("Adam手算首步参数：", params["weight"][0], "优化器步数：", optimizer.t, flush=True)  # 预期约0.999和一步。

def measure(model, vocab, label, data, validation):  # 对各候选使用同一验证和生成规则。
    loss, positions = epoch_tools.evaluate_full(model, data, validation)  # 固定参数评估完整验证集。
    assert positions == 316960, "验证范围不一致"  # 不混用部分验证结果。
    print(f"{label} full_validation={loss:.6f}", flush=True)  # 完整验证损失用于候选选择。
    samples = []  # 额外保留固定开头的实际文字。
    for prompt in diag.PROMPTS:  # 沿用春风、唐僧道和宝玉听了三个开头。
        for mode, temperature, top_k in (("greedy", None, None), ("topk40_t07", 0.7, 40)):  # 对每个候选都使用同两种解码设置。
            text, ids = diag.generate_text(model, vocab, prompt, temperature, top_k, 42)  # 固定六十个续写编号和种子42。
            samples.append({"prompt": prompt, "mode": mode, "seed": 42, "text": text, "generated_ids": ids, "repeat4": diag.repeat_rate(ids)})  # 全部保留，避免挑选好看的样例。
            print(label, mode, repr(text), flush=True)  # 指标下降时也检查句子是否更容易读懂。
    return {"validation_loss": loss, "validation_positions": positions, "samples": samples}  # 返回当前候选的完整评估记录。

def main(output_dir=None):  # 同起点、同数据、同更新预算比较两种优化配置。
    check_first_step()  # 先确认Adam的基本公式无误。
    data = import_module("06_make_windows")  # 复用已划分的数据和固定词表。
    source = json.loads((ROOT / "docs/training_runs/lesson37.json").read_text(encoding="utf-8"))  # 从完整第四轮最佳模型开始。
    assert source["status"] == "completed" and source["train_sha256"] == data.vocab["train_sha256"], "模型来源或数据版本不一致"  # 不使用其他实验起点。
    source_path = ROOT / "model" / source["best_checkpoint"]  # 两组都加载同一最佳文件。
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()  # 保留共同起点校验值。
    initial, vocab, start_step = training.load_checkpoint(source_path, network)  # 用原模型作为共同起点和候选。
    assert vocab == data.vocab and start_step == 19148, "起点配置不正确"  # 本次从已核验的第四轮结束位置出发。
    output_dir = ROOT / "model" if output_dir is None else Path(output_dir)  # 老师预跑可把分支文件放入临时目录。
    records = list(data.train_records)  # 从原始章节列表复制。
    random.Random(46).shuffle(records)  # 预先固定本次短训练顺序。
    batches = list(islice(data.iter_batches(data.iter_windows(records, 32, 32), 16), STEPS))  # 两组精确复用相同的两百批。
    assert len(batches) == STEPS and sum(batch["y"].size for batch in batches) == 102400, "训练预算错误"  # 每组新增相同数量的目标预测。
    for batch in batches:  # 缓存输入目标不允许被训练代码修改。
        batch["x"].setflags(write=False)  # 保留输入数组。
        batch["y"].setflags(write=False)  # 保留正确目标数组。
    chapter_to_book = {record["id"]: record["book"] for record in records}  # 准备统计训练前缀实际覆盖。
    book_windows = {record["book"]: 0 for record in records}  # 没覆盖到的书保留零计数。
    for batch in batches:  # 每行对应一个章节内窗口。
        for chapter_id in batch["chapter_ids"]:  # 按窗口来源统计。
            book_windows[chapter_to_book[chapter_id]] += 1  # 不假定短前缀四书均衡。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 两组使用同一完整验证集。
    initial_row = measure(initial, vocab, "starting_model", data, validation)  # 起点也参加选优。
    assert np.isclose(initial_row["validation_loss"], source["best_validation_loss"], rtol=1e-6, atol=1e-6), "起点指标不一致"  # 确认未加载错模型。
    initial_row.update({"label": "starting_model", "checkpoint": source_path.name, "completed_steps": start_step})  # 加入来源信息。
    history = [initial_row]  # 保留全部候选，避免两个分支都变差时丢掉旧模型。

    for label, learning_rate in CONFIGS:  # 每组是独立训练分支。
        model, current_vocab, loaded_step = training.load_checkpoint(source_path, network)  # 每组重新加载共同起点。
        assert current_vocab == vocab and loaded_step == start_step and all(np.array_equal(value, initial.params[name]) for name, value in model.params.items()), "分支起点不同"  # 第二组不能接着第一组训练。
        optimizer = Adam(model.params, learning_rate) if label == "adam" else None  # Adam只创建一次，在组内所有批次间保留历史。
        for local_step, batch in enumerate(batches, start=1):  # 本组连续完成两百次更新。
            loss, raw_grads = model.loss_and_grad(batch["x"], batch["y"])  # 使用同一模型结构和反向公式。
            assert np.isfinite(loss), "训练损失无效"  # 无效数值不继续更新。
            grads, _, _ = network.clip_gradients(raw_grads, 1.0)  # 两组都先做同样的全局梯度裁剪。
            if optimizer is None:  # SGD使用固定学习率乘当前梯度。
                for name in model.params:  # 完整反向后统一更新。
                    model.params[name] -= learning_rate * grads[name]  # 不保存梯度历史。
            else:  # Adam使用过去和当前梯度形成的状态。
                optimizer.step(model.params, grads)  # 同一个优化器对象持续累积m、v和t。
            assert all(np.isfinite(value).all() for value in model.params.values()), "参数更新异常"  # 两组都检查数值稳定。
            if local_step % 50 == 0:  # 每五十步显示训练进度。
                print(f"{label} step={local_step}/{STEPS} batch_loss={loss:.6f}", flush=True)  # 当前批次损失不用于最终模型选择。
        assert optimizer is None or optimizer.t == STEPS, "Adam历史计数错误"  # Adam本次历史为200步，模型累计步数是19348。
        row = measure(model, vocab, label, data, validation)  # 训练结束后评估固定样例和完整验证。
        path = output_dir / f"lesson40_{label}.npz"  # 两组保存独立模型文件。
        training.save_checkpoint(path, model, vocab, start_step + STEPS)  # 本课只保存模型权重，不包含Adam状态。
        state_bytes = 0 if optimizer is None else sum(value.nbytes for group in (optimizer.m, optimizer.v) for value in group.values())  # 统计额外m和v数组大小。
        row.update({  # 补充本组训练条件和额外状态信息。
            "label": label, "learning_rate": learning_rate,  # 记录实际优化配置。
            "checkpoint": path.name, "completed_steps": start_step + STEPS,  # 保存文件及累计步数。
            "change_from_start": row["validation_loss"] - initial_row["validation_loss"],  # 负值才表示优于起点。
            "optimizer_steps": STEPS,  # 两组都完成两百次参数更新。
            "adam_state_steps": None if optimizer is None else optimizer.t,  # 只有Adam维护自己的历史计数。
            "optimizer_state_bytes": state_bytes,  # 记录m和v额外占用的数组字节数。
        })  # 不把额外优化器状态当作新增可学习参数。
        history.append(row)  # 无论结果好坏都保留。
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash and all(np.array_equal(value, initial.params[name]) for name, value in training.load_checkpoint(source_path, network)[0].params.items()), "共同起点被修改"  # 文件和初始模型对象都应保持不变。
    selected = min(history, key=lambda row: row["validation_loss"])  # 在起点和两组最终候选中选最低损失。
    print("实际训练窗口覆盖：", book_windows, flush=True)  # 明确本次只是相同短前缀。
    print("本次选中：", selected["label"], selected["validation_loss"], flush=True)  # 不预设Adam必须胜出。
    return {  # 保存两个明确配置的对照结果。
        "starting_checkpoint": source_path.name, "starting_sha256": source_hash,  # 保留共同起点来源。
        "train_sha256": vocab["train_sha256"],  # 记录训练数据版本。
        "config": {"steps_per_branch": STEPS, "sgd_lr": 0.03, "adam_lr": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "max_norm": 1.0, "batch_size": 16, "seq_length": 32, "chapter_shuffle_seed": 46},  # 完整记录预先规定的条件。
        "training_positions_per_branch": 102400, "book_windows": book_windows,  # 保留实际数据预算与书籍覆盖。
        "optimizer_state_saved": False,  # 当前权重文件没有保存Adam的m、v、t。
        "history": history,  # 保留起点与两个分支的全部评估结果。
        "selected_label": selected["label"], "selected_checkpoint": selected["checkpoint"],  # 最终仍可能选择旧模型。
    }  # 这里不声称支持精确恢复Adam训练。

if __name__ == "__main__":  # 直接运行才训练并保存正式报告。
    result = main()  # 默认把分支权重保存到model。
    path = ROOT / "docs/training_runs/lesson40.json"  # 使用独立的实验记录。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存配置、指标和所有固定样例。
    print("Adam对照报告：", path)  # 输出结果位置。
