## GRPO前置知识

* state：prompt加上目前为止生成的响应
* action：生成下一个token

在这里的实现中，我们将专注于结果奖励，就是reward是整个完整响应的函数。

**什么是策略梯度？**

在GRPO算法中，我们省去了Value Model（价值函数），策略梯度核心思想就是直接优化策略本身，让这个策略更倾向于做出能带来高奖励的行为。

我们希望最大化期望奖励$E(R)$，这个期望可以写成一个积分（或求和），它考虑了所有可能的状态 $s$ 和行为 $a$：
$$
E[R] = \int p(s)\pi(a|s)R(s,a)
$$

* $p(s)$：出现状态 $s$ 的概率
* $\pi(a | s)$：在状态 $s$ 下，策略 $\pi$ 选择行为 $a$ 的概率
* $R(s,a)$：在状态 $s$ 下做出行为 $a$ 后获得的奖励

为了最大化 $E[R]$，我们使用梯度上升法，也就是计算期望奖励**关于策略参数**的梯度 $∇E[R]$，然后沿着梯度的方向更新参数。
$$
∇E[R]=∫p(s)∇π(a∣s)R(s,a)
$$

问题在于我们不知道 $∇π(a∣s)$ 该怎么计算，因为它依赖于策略内部复杂的参数。这里用一个很巧妙的数学技巧。我们知道 $(logf(x))'= \frac{f'(x)}{f(x)}$。

$$
∇π(a∣s)=π(a∣s)∇logπ(a∣s)
$$

代入：
$$
∇E[R]=\int p(s)π(a∣s)∇logπ(a∣s)R(s,a)
$$
这个积分形式正好是某个东西的期望值。
$$
∇E[R]=E[∇logπ(a∣s)R(s,a)]
$$
它告诉我们**可以用采样的方式来估计梯度**。朴素的策略梯度思想是：从$p(s)$中采样一个prompt，从$a$中采样一个response，然后就根据期望里面的内容来更新参数。本质和SFT差不多，但是参数的更新会根据奖励reward进行加权。

但是如果奖励是稀疏的（比如很难的问题，只有少数response回得到奖励，大部分的response得到奖励为0），那么梯度更新的策略就会被卡住。

比如说下面这个例子：

* s1: a1 → reward 11, a2 → reward 9

* s2: a1 → reward 0, a2 → reward 2

> 在一次采样中，对于s1采样动作a2得到了奖励 `9`。算法会认为这是个很棒的奖励！我必须**大力增加**在 `s1` 时选择 `a2` 的概率！**但这是个错误的结论**，因为在 `s1` 时选择 `a1` 可以得到更高的 `11` 分。`a2` 其实是次优选择。在状态 `s2`，采样了动作 `a2`，得到了奖励 `2`。这是一个较小的正数，算法会认为2分还行吧。就**小力增加**在 `s2` 时选择 `a2` 的概率。**这同样是低效的**。在 `s2` 中，`a2` (奖励2) 是明显优于 `a1` (奖励0) 的选择，它应该得到更强的鼓励。算法无法分辨出哪个动作是“相对更好”的。这就是**高方差**问题：学习信号的波动性太大，导致训练不稳定且效率低下。

策略梯度方法的关键思想是，有一个叫做基线的模型。

我们的原始目标是最大化期望回报 $E[R]$。现在的新目标看起来是最大化带基线的回报 $E[R - b(s)]$。基线 $b(s)$ 只和状态 $s$ 有关，而与策略 $π$ 无关，这一项求梯度为0。$$∇E[R−b(s)]=∇E[R]−∇E[b(s)]=∇E[R]−0=∇E[R]$$，优化带基线的目标和优化原始目标的梯度是完全一样。

虽然从期望上看梯度是相同的，但我们每次更新时用的都是**单次采样**的估计值。原来的更新依赖于：$∇ log π(a | s) * R(s, a)$， 现在的更新依赖于$∇ log π(a | s) * (R(s, a) - b(s))$。这里的 $(R(s, a) - b(s))$ 就是我们之前讨论过的**优势函数 (Advantage Function)**。它虽然没有改变梯度的期望值（保持了**无偏性**），但通过将原始回报 $R$ 中心化，有效**降低了梯度的方差**，使得训练过程更稳定、更高效。



## GRPO极简实现

这个任务是这样的，给定一个prompt（数组），返回的response是排序好的结果。比如prompt = [1, 0, 2]，理想的response = [0, 1, 2]。然后我们有很多不同的方法定义response的reward，比如与response排好序数组的匹配数、相对位置等等。

模型结构如下：

1. 输入数据$X \in [vocab\_size]$
2. 经过embedding层，每个vocab编码成向量$[vocab\_size, dim]$
3. 每个单独的编码向量通过矩阵运算再相加得到中间表示$[dim]$
4. 再把这个单独的表示乘以多个不同的矩阵得到解码向量$[vocab\_size,vocab\_size]$，表示每一个token选择某个pos的logits得分

后面会对这个logits得分进行softmax操作。

有一个函数叫做`generate_responses`，用于生成`N`个response（随机性取样），它的返回结果是$Response \in [N, vocab\_size]$，$Response[i,j]$ 表示第`i`个response中第`j`个token选择的位置。

生成response后，需要对其进行打分$[N]$，然后计算$\delta$（比如说归一化），这个是梯度更新的幅度。

如何计算responses的`log_probs` 对数概率呢？

> 我们把数据prompt输入到model进行一次forward得到logits，然后进行`log_probs = F.log_softmax(logits, dim=-1)`得到对数概率，`log_probs[i,j]`表示第对于当前prompt，第`i`个token选择第`j`个位置的对数概率。
>
> responses 张量 `[N, vocab_size]`，把`log_probs`进行复制与responses进行匹配得到`[N,vocab_size,vocab_size]`，我们只关心实际出现在 responses 中的那个特定词元的概率。`log_probs = log_probs.gather(dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)`进行查找得到`[N, vocab_size]`，这就是最终的`log_porbs`。

如何计算loss呢？

上面我们可以计算ref model和model的对数概率，将其一起输入到`compute_loss`中。可以算出`ratios`，`[N, vocab_size]`，然后将`ratios`与计算的`deltas`相乘（ `ratios`对应的每一项都乘了这个相应回答的`delta`），再把结果相加得到最终的loss。

```Python
def compute_loss(log_probs: torch.Tensor, deltas: torch.Tensor, mode: str, old_log_probs: torch.Tensor | None = None) -> torch.Tensor:
    if mode == "naive":
        return -einsum(log_probs, deltas, "batch trial pos, batch trial -> batch trial pos").mean()
    if mode == "unclipped":
        ratios = log_probs / old_log_probs  
        return -einsum(ratios, deltas, "batch trial pos, batch trial -> batch trial pos").mean()
    if mode == "clipped":
        epsilon = 0.01
        unclipped_ratios = log_probs / old_log_probs  
        unclipped = einsum(unclipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")
        clipped_ratios = torch.clamp(unclipped_ratios, min=1 - epsilon, max=1 + epsilon)
        clipped = einsum(clipped_ratios, deltas, "batch trial pos, batch trial -> batch trial pos")
        return -torch.minimum(unclipped, clipped).mean()
```

训练代码：

```Python
    for epoch in tqdm(range(num_epochs), desc="epoch"):
        # If using KL penalty, need to get the reference model (freeze it every few epochs)
        if kl_penalty != 0:
            if epoch % compute_ref_model_period == 0:
                ref_model = model.clone()
        # Sample responses and evaluate their rewards
        responses = generate_responses(prompts=prompts, model=model, num_responses=num_responses)  # [batch trial pos]
        rewards = compute_reward(prompts=prompts, responses=responses, reward_fn=reward_fn)  # [batch trial]
        deltas = compute_deltas(rewards=rewards, mode=deltas_mode)  # [batch trial]
        if kl_penalty != 0:  # Compute under the reference model
            with torch.no_grad():
                ref_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=ref_model)  # [batch trial]
        if loss_mode != "naive":  # Compute under the current model (but freeze while we do the inner steps)
            with torch.no_grad():
                old_log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial]
        # Take a number of steps given the responses
        for step in range(num_steps_per_epoch):
            log_probs = compute_log_probs(prompts=prompts, responses=responses, model=model)  # [batch trial]
            loss = compute_loss(log_probs=log_probs, deltas=deltas, mode=loss_mode, old_log_probs=old_log_probs)  # @inspect loss
            if kl_penalty != 0:
                loss += kl_penalty * compute_kl_penalty(log_probs=log_probs, ref_log_probs=ref_log_probs)
            # Print information
            print_information(epoch=epoch, step=step, loss=loss, prompts=prompts, rewards=rewards, responses=responses, log_probs=log_probs, deltas=deltas, out=out)
            global_step = epoch * num_steps_per_epoch + step
            records.append({"epoch": epoch, "step": global_step, "loss": loss.item(), "mean_reward": rewards.mean().item()})
            # Backprop and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```















