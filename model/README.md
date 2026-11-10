# 模型权重

`tiny_rnn.npz` 是可以直接运行的微型示例权重，包含词表和模型参数，当前大小为 147,246 字节。

这个小文件直接纳入 Git，方便克隆后体验。其他训练产物默认由 `.gitignore` 忽略，避免反复提交二进制权重。

训练实验时指定另一个输出路径，可以保留基线：

```powershell
python scripts/train.py --output model/experiment_01.npz # 将实验权重写入另一个文件。
```

如需分享较大的模型，后续单独选择 GitHub Release、Git LFS 或模型托管服务，并在文档中记录配置、数据版本和测评结果。
