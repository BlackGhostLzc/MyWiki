









## GRPO

下面是GRPO的核心公式：

![](./img/GRPO-1.jpg)



其中$\hat{A_{i,t}} = \frac{r_i - mean(r)}{std(r)}$，$r={r_1,r_2,...r_G}$。

公式一个核心的就是概率比$\frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{old}(o_{i,t} | q, o_{i,<t})}$，也就是新模型在