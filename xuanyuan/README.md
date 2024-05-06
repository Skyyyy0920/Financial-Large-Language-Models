---
license: llama2
---

## 介绍

XuanYuan-6B系列模型是采用类LLaMA架构，从零开始进行预训练的金融大模型。我们构建了大规模、多样化、高质量的训练语料对模型进行了充分预训练，使模型具备各项能力。此外我们构建了丰富、高质量的问答数据和人类偏好数据，并通过指令微调和强化学习进一步对齐模型表现和人类偏好，显著提升了模型在对话场景中的表现。各项评估显示，XuanYuan-6B不仅具备较强的通用能力，更具备强大的金融能力。更多细节请参考我们的技术报告：[Report](https://github.com/Duxiaoman-DI/XuanYuan/blob/main/xuanyuan_6b_report.md)

XuanYuan-6B系列模型包含基座模型XuanYuan-6B，经指令微调和强化对齐的chat模型XuanYuan-6B-Chat，以及chat模型的量化版本XuanYuan-6B-Chat-4bit和XuanYuan-6B-Chat-8bit。各个模型的链接为：

| 基座模型                                                               | Chat模型                                                     | 8-bit量化Chat模型                                            | 4-bit量化Chat模型                                          |
| ------------------------------------------------------------          | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 🤗 [XuanYuan-6B](https://huggingface.co/Duxiaoman-DI/XuanYuan-6B) | 🤗 [XuanYuan-6B-Chat](https://huggingface.co/Duxiaoman-DI/XuanYuan-6B-Chat) | 🤗 [XuanYuan-6B-Chat-8bit](https://huggingface.co/Duxiaoman-DI/XuanYuan-6B-Chat-8bit ) | 🤗  [XuanYuan-6B-Chat-4bit](https://huggingface.co/Duxiaoman-DI/XuanYuan-6B-Chat-4bit) |


主要特点：

* 收集多个领域大量的训练预料，进行了多维度数据清洗和去重，保证数据的量级和质量
* 从零开始预训练，预训练中动态调整数据配比，模型基座能力较强
* 结合Self-QA方法构建高质量问答数据，采用混合训练方式进行监督微调
* 构建高质量人类偏好数据训练奖励模型并进行强化训练，对齐模型表现和人类偏好
* 模型尺寸小并包含量化版本，硬件要求低，适用性更强
* 在多个榜单和人工评估中均展现出良好的性能，具备领先的金融能力

## 模型细节

XuanYuan-6B具有4096个隐藏单元，由30层和32个注意⼒头组成。为了融⼊位置信息，我们采⽤了RoPE作为位置嵌⼊技术。模型中使⽤的激活函数是SwiGLU，并使⽤RMSNorm进⾏归⼀化处理。在训练过程中，我们将最⼤序列⻓度设置为2048个token。词表的⼤⼩为39438，与我们先前模型（XuanYuan-13B、XuanYuan-70B）使⽤的词表⼀致。

## 训练细节

训练前，我们从不同领域收集了大量训练语料，并对数据进行一系列处理来提升质量。

预训练中，我们不断评估模型在特定任务或基准上的性能，并根据评估结果动态调整不同来源的训练数据配⽐，不断优化模型训练过程，提升模型各项能力。

我们利用Self-QA的方法构建了高质量指令微调数据集，并结合无监督语言模型任务对预训练后的模型进行了混合微调。在增强模型chat场景下各项能力的同时，保证其泛化性。

最后，我们通过人工标注的方式构建了高质量的偏好数据，由此训练奖励模型并进行强化对齐训练，使其表现对齐人类偏好，以继续提升模型各项能力。

## 使用方法
XuanYuan-6B基座模型、chat模型及其量化模型的使用方法和[XuanYuan-70B](https://huggingface.co/Duxiaoman-DI/XuanYuan-70B)，[XuanYuan2-70B](https://huggingface.co/Duxiaoman-DI/XuanYuan2-70B)类似，但是tokenizer加载方式和在对话场景中使用的prompt格式不同（不包含system message）。下面以XuanYuan-6B-Chat模型为例，来展示XuanYuan-6B系列模型的使用方法。
```python
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

model_name_or_path = "your/model/path/"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
model.eval()

seps = [" ", "</s>"]
roles = ["Human", "Assistant"]

content = "介绍下你自己"
prompt = seps[0] + roles[0] + ": " + content + seps[0] + roles[1] + ":"
print(f"输入: {content}")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95)
outputs = tokenizer.decode(outputs.cpu()[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(f"输出: {outputs}")
```