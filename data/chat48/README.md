# MiniMind来源的首次有界训练管线样本

来源：jingyaogong/minimind_dataset，提交312afb4f76391145c6902f765bb51691c09a12f5。
数据卡：https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/312afb4f76391145c6902f765bb51691c09a12f5/README.md 。原卡保存在raw/upstream_README.md，列出上游贡献者和来源。

数据卡声明Apache-2.0与CC-BY-NC-2.0混合条款，未将整个子集重新声明为纯Apache授权。本项目按非商业学习试验使用并保留署名、来源和原始说明；未作商用许可审查。协议链接：https://www.apache.org/licenses/LICENSE-2.0 ，https://creativecommons.org/licenses/by-nc/2.0/ 。

只下载每个文件八个分散位置的完整字节块：预训练16MiB、SFT8MiB；不是完整GB级数据集，不是全库均匀随机抽样。raw/*.bin保留收到的实际字节，块偏移和指纹见manifest.json。丢弃块边缘不完整JSON行；只统一正文首尾和换行，SFT保留可见messages字段，不保留reasoning_content；当前不收工具调用、空回答、不支持角色、来源身份与部分异常格式。

预训练和SFT在规范化完全去重后联合分组，首问/数字模板与采样字符SimHash只是启发式，不能证明消除全部语义泄漏，也不能证明文本事实正确。机械筛选不等于人工质量认证。两阶段共用同一组映射后分别留出train/validation/test，tokenizer只用两阶段train正文学习BPE，预训练只读取pretrain_train目标，不读SFT目标或测试指标。

本课为实际模型试跑而准备的数据，而非两条格式演示；它仍是有限采样的管线试验，不能作为已经完成高质量大规模语料建设的证据。预训练流以BOS/EOS连接文档，再按256目标打包；验证按文档独立并保留尾段。tokenizer中所有JSON与npy指纹固定，旧课程词表与权重不改动。
