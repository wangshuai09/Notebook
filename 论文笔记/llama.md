# 摘要
Llama 是一系列的基础语言模型，从7B到65B（B是什么概念），在万亿的token训练下得到
Llama-13B 在大部分的评测下都好于 GPT-3(175B)
Llama-65B 可以和 Chinchilla-70B/PaLM-540B 竞争最佳模型

# 简介
使用大量文本语料训练的LLM被证明在文字说明或者少量例子的情况下就可以在新任务中有好的表现，这种few-shot能力只有在模型尺寸比较大的情况下会有。
但是又有一种理论是，在算力一样的情况下，并非最大的模型最好，而是模型较小但数据量更大的效果最好。
Llama 只用公开可获取的数据来训练。

# 方法
### 预训练数据

- English CommonCrawl[67%]
  - 2017-2020
  - 使用 CCNet pipeline: 在 line level 去重，使用 fastText linear 分类器去除非英语 pages, 用 n-gram 语言模型过滤低质量内容
  - 训练一个linear model 来分类 pages 是否是 Wikipedia 的引用，丢弃那些分类器认为不是引用的 pages
- C4[15%]
  - 实验发现不同的 CommonCrawl 数据可以增加模型表现，所以增加了 C4 数据
  - 同样包括去重和语言区分步骤
  - 质量过滤区分与 ComonCrawl, 主要基于启发式的，比如标点符号，词数，网页中的句子等
- Github[4.5%]
  - 启发式过滤低质量文件，根据行长或者字母数字字符的比例，基于正则表达式删除了样板文件
- Wikipedia[4.5%]
  - June-Auguse 2022
  - 20 languages
  - 删除超连接、评论及其他格式样板
- Gutenberg and Books3[4.5%]
  - 两个书籍语料库
  - 在book level 进行去重，删除有90%以上内容重复的书籍
- ArXiv[2.5%]
  - arxiv latex files
  - 删除第一节之前的内容及参考书目
- Stack Exchange
  - 高质量的回答网站数据
  - 从 28 个最大的网站获得
  - 删除 HTML 标签，并把答案按分数进行排序

### Tokenizer
使用 Sentecepiece 实现的 bytepair encoding(BPE)算法，所有的数都切分成单独的数字
总体来看，大约包含1.4T tokens, 大部分的数据在训练过程中都使用了一次，Wikipedia和书籍使用了2次

### 结构
基于 Transformers 结构，同时结合了其他方面的提升：
- Pre-normalization[GPT-3]
- SwiGLU[PaLM]
- Rotary Embedding[GPTNeo]

### 优化器
使用 AdamW 优化器，
- β1 = 0.9
- β2 = 0.95
cosine 学习率采样，最终学习率是最大学习率的10%
weight decay: 0.1, gradient clip: 1
warmup: 2000
学习率和 batch size会根据模型大小而不同

### 有效实现
- 使用Xformers库的 causal multi-head attention
- 将反向中的compute expensive 的激活值保存下来
- 通过模型和序列并行减少内存占用
- 将激活值的计算和GPU间通信计算时间上重叠
训练一个65B模型，在2048台A100(80G)上可以达到 380 tokens/sec/Gpu，在1.4T tokens上需要 21 天

# 主要结果
### Common Sense Reasoning
### Closed-book Question Answering
### Reading Comprehension
### Mathematical reasoning
### Code generation
### Massive Multitask Language Understanding

# 指令微调
简单的指令微调可以快速提高模型在MMLU上的表现，虽然非微调版本已经可以遵循基础的指令，但是
很小量的微调数据就可以提高MMLU的表现

# 偏见、有害性、错误信息
###
###
