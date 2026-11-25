---
dataset_info:
  features:
  - name: instruction
    dtype: string
  - name: input
    dtype: string
  - name: output
    dtype: string
  splits:
  - name: train
    num_bytes: 32150579
    num_examples: 48818
  download_size: 35100559
  dataset_size: 32150579
license: cc-by-4.0
language:
- zh
pretty_name: Instruction Tuning with GPT-4
size_categories:
- 10K<n<100K
task_categories:
- text-generation
tags:
- gpt
- alpaca
- fine-tune
- instruct-tune
- instruction
---

# Dataset Description

- **Project Page:** https://instruction-tuning-with-gpt-4.github.io
- **Repo:** https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
- **Paper:** https://arxiv.org/abs/2304.03277

# Dataset Card for "alpaca-zh"

本数据集是参考Alpaca方法基于GPT4得到的self-instruct数据，约5万条。

Dataset from https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM 

It is the chinese dataset from https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json
# Usage and License Notices

The data is intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

train model with alpaca-zh dataset: https://github.com/shibing624/textgen

# English Dataset

[Found here](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data)

# Citation

```
@article{peng2023gpt4llm,
    title={Instruction Tuning with GPT-4},
    author={Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, Jianfeng Gao},
    journal={arXiv preprint arXiv:2304.03277},
    year={2023}
}
```