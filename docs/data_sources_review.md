# 外部语料候选审查

当前四书对照保持原数据不变，不能单凭生成不通顺就认定主因是数据量不足。

| 候选 | 来源声明 | 当前处理 |
|---|---|---|
| joe-chiu/TinyChineseStories | 作者卡说明GPT-3.5合成；README正文声明CC-BY-4.0 | 固定版本采集50条、最多1MiB，只审查格式，不训练 |
| RobinChen2001/TinyStories-Zh-2M | 卡标MIT，并注明翻译来源52AI/TinyStoriesZh和上游roneneldan/TinyStories | 下载规模约842MB，不在本次预算内；未下载，后续需沿来源分别核对说明 |
| 本地LightNovel5000_v1_txt | 用户已有目录及来源链接 | 本次不训练、不上传，尚未做格式、去重及来源审查 |

一手来源：[TinyChineseStories数据卡](https://huggingface.co/datasets/joe-chiu/TinyChineseStories/blob/ad789c9d622bfbdeecdc90df2041b3ba458e0368/README.md)、[中文2M整理版数据卡](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M/blob/main/README.md)。这里记录的是数据卡声明，不提供统一的额外再授权结论。任何新语料接入都与模型结构实验分开，避免同时改变多个变量。

## 第四十五课接入更新

学习者明确要求新增语料并实现小型Transformer后，已从同一固定提交下载完整TinyChineseStories原CSV（3045940字节、2277篇）。data/transformer45保存原文、署名许可、衍生划分和指纹。按主题而非姓名归组，260主题分为208/26/26组，故事1801/231/245篇；测试正文不用于建词表或评估选优。原四书划分保持，只有新训练故事加入新Transformer训练。此前50条候选样本仍保留为审查产物，不另重复计入训练。大体量轻小说压缩包未解压或接入。
