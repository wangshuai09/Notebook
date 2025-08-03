# 摘要
Low-Rank Adaptation, LoRA, 会冻结预训练模型权重，并在Transformer结构的每一层插入可训练的秩分解矩阵，
相比于GPT-3 175b with Adam，可以减少 10000 倍的训练参数，3倍的内存。
对语言模型自适应中的秩缺陷进行了研究，解释了LoRA的有效性。
代码：https://github.com/microsoft/LoRA

# 引言
大语言模型依赖于 adaption 来进行将预训练模型适配到下游任务，这个 adaption 通常叫 fine-tuning。但是这个方法的缺点
是需要全量参数更新，当前有1.只迁移少量参数 2. 添加额外的module，的方案，但这些方案会引入推理负载，模型精度方面也会
低于fine-tune baseline。
本文的方案灵感来于，可学习的过度参数的模型，实际上处于一个低维空间。训练稠密层的秩分解矩阵，冻结预训练矩阵，
<div align=center>
    <img src="https://raw.githubusercontent.com/wangshuai09/blog_img/main/images/20241014102835.png"/>
</div>

