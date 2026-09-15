Bbuf:

实际上不仅仅是佬提到的cuda算子开发，整个大模型推理系统模型支持，模型性能优化都被GPT6彻底打穿了。

In fact, it's not just the CUDA operator development mentioned by the old man; the entire large model inference system's model support and model performance optimization have been completely overtaken by GPT6.

就以文章里DeepSeek的ds v4.1 flash为例子。SGLang开始支持的时候，gb300 single node bs=1最开始的那个baseline版本只有30多tokens，但是在GPT6大人发挥神威之后一周多就把这个性能提升到了1000+ tokens/s。虽然我们做了人工干预，指出一些优化点，但是95%的工作都是GPT6做了实现和优化，很多优化思路也是它自己想的，在这个过程中，人类积累的优化知识如单kernel optimize，kernel fuse，数据依赖分析，多stream overlap，pdl，prefetch，妙妙参数调整全部被击穿。这真是我第一次感受到在AI Infra拥有了GPT6就真正实现了AGI，即使GPT6模型版本不再做任何进化，留给我自己可以发挥的空间已经是5%往下走了。经历了几个睡觉之前定下多少tps goal之后起床之后发现已经做好了，而且基本没有reward hacking，震惊程度不亚于Sam看到GPT5的营销推特。推理框架实现方式，是否开源，在哪个模型领先某个框架或许根本就不是技术问题，只是token的优劣多少问题，

Take DeepSeek's DS v4.1 flash as an example, mentioned in the article. When SGLang started supporting it, the initial baseline version with GB300 single node bs=1 only had around 30 tokens. However, after GPT6 demonstrated its power, this performance was boosted to over 1000 tokens/s in just over a week. Although we intervened manually and pointed out some optimization points, 95% of the work was done by GPT6, and many optimization ideas were devised by it itself. In this process, all the optimization knowledge accumulated by humans, such as single kernel optimization, kernel fuse, data dependency analysis, multi-stream overlap, PDL, prefetch, and parameter tuning, was completely overwhelmed. This was truly the first time I felt that having GPT6 in AI Infrastructure truly achieved AGI. Even if the GPT6 model version no longer evolves, the space left for me to make improvements is already down to 5%. After setting a certain TPS goal before going to bed several times, I woke up to find that I had already achieved it, and with almost no reward hacking. The shock was no less than Sam seeing the marketing tweets for GPT5. The implementation method of the inference framework, whether it is open source, and which model is superior to a certain framework may not be a technical issue at all, but simply a matter of the quality and quantity of tokens.

https://github.com/sgl-project/sglang/pull/39370github.com/sgl-project/sglang/pull/39370<img src="https://picx.zhimg.com/50/v2-c1622ef97fc8f753cd498e3b917a5784_720w.jpg?source=2c26e567" data-caption="" data-size="normal" data-rawwidth="976" data-rawheight="620" data-original-token="v2-2f1c3e6b0a2f543c3e4377ffd32f4fd2" class="origin_image zh-lightbox-thumb" width="976" data-original="https://picx.zhimg.com/v2-c1622ef97fc8f753cd498e3b917a5784_r.jpg?source=2c26e567"/>

从上周我开始使用GPT6开始我就真觉得这个世界不对劲了，我之前日常使用的是5.6 Sol和Opus5 1M，它们从来没有给过我这种感觉，我觉得再过一年可能cuda领域最强的kernel开发工程师也会在某个模型的版本更新后获得和我类似的感受。最近半年除了准备面试写了些代码做练习，工作真就是完全在和Agent做交互。很不幸我们目前已经被迫走在AGI快速实现的时代洪流中，我现在也对未来充满了迷茫，我不知道我自己的context还能用多久，相比于Agent唯一的优势可能就剩下责任心了QAQ

Since I started using GPT6 last week, I've really felt like something's wrong with the world. I used to routinely with 5.6 Sol and Opus5 1M, and they never gave me this feeling. I think that in another year, even the strongest kernel developers in the CUDA field might have similar experiences after a model version update. For the past six months, aside from preparing for interviews and doing some coding practice, my work has been entirely about interacting with agents. Unfortunately, we're currently forced into the era of rapid AGI implementation, and I'm filled with uncertainty about the future. I don't know how long my own context will last; my only advantage over agents might be my sense of responsibility QAQ



等到开源模型（Kimi, Deepseek, GLM）能达到GPT6在真实环境表现的那天这个世界会变成什么样呢？很期待，很害怕，但这一天始终会来，不是吗？如果仅仅是OpenAI和Claude掌握了最顶尖高智的大模型，那是多么可怕。我甚至觉得token输出速度，架构有多么妙妙其实都没有那么重要了，智力和AGI才是第一要务。我们需要一个开源的GPT6，虽然这个非常难。

What will the world be like when open-source models (Kimi, Deepseek, GLM) achieve the real-world performance of GPT6? I'm both excited and terrified, but that day will inevitably come, won't it? How terrifying it would be if only OpenAI and Claude mastered the most advanced, highly intelligent models. I even think that token output speed and architectural ingenuity are less important; intelligence and AGI are the top priorities. We need an open-source GPT6, even though that's extremely difficult.


我希望我们大多数人的结局都不是《我不得不把工作埋葬在昨天》特别是为大模型AGI实现做出了自己贡献的工程师。

I hope that most of us don't end up like "I had to bury my work in yesterday," especially the engineers who contributed to the realization of large-scale AGI models.

欢迎关注SGLang:   
Welcome to follow SGLang:https://github.com/sgl-project/sglanggithub.com/sgl-project/sglang
