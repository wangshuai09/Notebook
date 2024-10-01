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
  - 删除