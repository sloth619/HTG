# HTG：老师模型内部的“状态变化”能不能帮助学生学得更好？

## 一、研究目的

标准 on-policy distillation（OPD）主要利用老师的输出概率监督学生。我们想进一步问：

> **除了“下一个 token 应该是什么”，老师模型内部“状态如何随推理过程变化”的信息，能不能给学生提供额外帮助？**

研究对象：

- 老师：Qwen3-8B
- 学生：Qwen3-1.7B-Base

我们最后重点研究 **HTG（Hidden Transition Geometry）**。

对于相邻两个 token 的 hidden state：

\[
h_t,\quad h_{t+1}
\]

先计算内部状态变化：

\[
\Delta h_t = h_{t+1}-h_t
\]

然后做特征中心化和归一化：

\[
\widetilde{\Delta h}_t
=
\Delta h_t-\operatorname{mean}(\Delta h_t)
\]

\[
d_t=
\frac{\widetilde{\Delta h}_t}
{\|\widetilde{\Delta h}_t\|_2+\epsilon}
\]

把多个位置的变化方向堆起来：

\[
D=[d_1,d_2,\ldots,d_P]^\top
\]

再计算不同位置之间的关系：

\[
G=DD^\top
\]

其中：

\[
G_{ij}=\cos(d_i,d_j)
\]

表示第 \(i\) 个位置和第 \(j\) 个位置的内部变化方向有多相似。

HTG 的目标是让学生和老师的关系矩阵接近：

\[
L_{\mathrm{HTG}}
=
\frac{1}{P^2}
\|G_S-G_T\|_F^2
\]

它不要求：

\[
h_t^S \approx h_t^T
\]

所以即使老师和学生的 hidden size 不一样，也可以比较。

---

## 二、做过的实验

| 实验 | 想回答的问题 |
|---|---|
| 比较 Attention、MLP、Hidden | 模型内部哪一部分最值得蒸馏？ |
| 比较 static 和 transition | 直接看当前状态，还是看相邻 token 的状态变化更有信息？ |
| 正确位置 vs 打乱位置 | 这些内部关系是不是真的依赖正确位置？ |
| OPD 前后对比 | 标准 OPD 自己会不会已经学到这些内部关系？ |
| HTG 辅助训练 | 把这些关系加入训练后，数学能力会不会提高？ |
| 几何干预 | 直接把学生内部关系改得更像老师，后续输出会不会改善？ |
| 行为签名干预 | 让学生内部方向对输出概率的影响更像老师，会不会更有效？ |
| 扰动传播 | 中间层的修改能不能真正传到最后输出？ |

当前主要 HTG 设置：

\[
\text{Student layer }11
\leftarrow
\text{Teacher layer }14
\]

即 S11/T14，每条较长回答采样：

\[
P=128
\]

个连续 hidden transition。

---

## 三、实验结果

### 1. 为什么最后选 Hidden Transition

为了判断某种内部关系是不是依赖正确位置，我们比较：

- 正确位置下的损失 \(L_{\mathrm{true}}\)
- 把老师位置随机打乱后的损失 \(L_{\mathrm{perm}}\)

定义：

\[
R=
\frac{L_{\mathrm{perm}}-L_{\mathrm{true}}}
{L_{\mathrm{true}}}
\]

这个数越大，说明：

> **老师和学生按原位置比较时，明显比把老师位置打乱后更接近。**

结果：

| 看哪一部分 | 层 | Static \(R\) | Transition \(R\) | 结果 |
|---|---|---:|---:|---|
| Attention | S16/T21 | 0.692 | 0.433 | 看变化后反而变弱 |
| MLP | S16/T21 | 0.920 | 0.949 | 基本没变化 |
| Hidden | S16/T21 | 0.195 | **1.419** | 看变化后明显增强 |
| Hidden | S11/T14 | 约 0.48 | **1.743** | 最后选这一组 |
| Hidden | S22/T29 | — | **1.728** | 后层也能看到类似现象 |

最明显的现象是：

\[
\boxed{
\text{Transition 对 Hidden 特别有效，
但对 Attention 并不有效}
}
\]

所以最后重点研究 Hidden Transition。

### 2. 为什么 Hidden 做 transition 后更清楚

不同 token 位置的 raw hidden 本身有很强的共同成分。

平均位置间 cosine：

| 表征 | Student | Teacher | 做 transition 后 |
|---|---:|---:|---:|
| Hidden | +0.425 | +0.549 | 约 −0.005 |
| Attention | +0.375 | +0.283 | 约 −0.004 |
| MLP | +0.113 | +0.085 | 约 −0.005 |

可以粗略写成：

\[
h_t \approx c+r_t
\]

其中 \(c\) 是很多位置共享的部分。

做差后：

\[
h_{t+1}-h_t
=
(c+r_{t+1})-(c+r_t)
=
r_{t+1}-r_t
\]

共享部分 \(c\) 被抵消，因此更容易看到“这一步内部状态到底怎么变”。

这能解释 Hidden 的 static relation 为什么弱，而 transition relation 为什么明显增强。

但 Attention 为什么做 transition 后反而变弱，目前还没有完全解释清楚。

### 3. 标准 OPD 不会自动把这个规律完全学到

我们在同一批回答上比较：

- 初始学生
- 做过标准 OPD 的学生

结果发现：

- 一些 static hidden relation 会随着 OPD 自然接近老师；
- 但 S11/T14 的 hidden transition geometry 变化很小。

所以：

\[
\boxed{
\text{HTG 不是标准 OPD 本来就会自动学到的东西}
}
\]

这也是它值得单独测试的原因。

### 4. 历史训练没有看到稳定提升

MATH500：

| 配置 | Avg@8 |
|---|---:|
| 纯 OPD | 61.55% |
| HTG 历史 run 1 | 60.67% |
| HTG 历史 run 2 | 61.38% |
| HTG 历史 run 3 | 61.30% |
| 打乱老师位置的 HTG | 61.85% |

目前只能说：

> **历史实验没有看到稳定的 HTG 提升。**

---

## 四、机制验证

训练没有稳定提升以后，我们把问题拆成三步：

\[
\text{内部能不能改}
\rightarrow
\text{改动能不能传到输出}
\rightarrow
\text{输出会不会真的变好}
\]

### 1. 内部目标确实能改

直接修改 hidden 后，学生的几何关系可以明显更接近老师。

我们还定义了一个更接近输出的“行为签名”。

沿某个内部方向 \(d\) 对 hidden 做正负小扰动：

\[
h^+=h+\epsilon d
\]

\[
h^-=h-\epsilon d
\]

观察 token log-prob 的变化：

\[
s
\approx
\frac{
\log p(y|h+\epsilon d)
-
\log p(y|h-\epsilon d)
}{
2\epsilon
}
\]

这个向量描述：

> 沿这个内部方向移动，会让哪些 token 更可能、哪些更不可能。

然后比较学生和老师的 \(s\)。

代表性实验中，学生和老师的 signature cosine 平均提高：

\[
\Delta\cos \approx +0.251
\]

16/16 条 rollout 都提高。

所以问题不是“这个内部目标改不动”。

### 2. 改动确实能传到输出

中间层干预之后：

\[
\|\Delta h\|
\]

在后续层没有消失，最终输出概率也明显变化。


### 3. 但内部更像老师，没有稳定让输出更像老师

我们用：

\[
D_{\mathrm{KL}}
\left(
p_{\mathrm{student}}
\|
p_{\mathrm{teacher}}
\right)
\]

衡量学生输出和老师输出的差距。

这个值下降，表示学生更接近老师。

代表性结果：

| 干预方法 | 内部变化 | 后续 reverse KL |
|---|---|---:|
| 让 signature 更像老师 | 明显改善 | +0.00052，区间包含 0 |
| 直接沿 KL 下降方向改 hidden | 不以 signature 为目标 | −0.01285，明显改善 |

所以：

\[
\boxed{
\text{内部更像老师}
\not\Rightarrow
\text{输出更像老师}
}
\]


---

## 五、下一步方向


> **如果我们真的控制住 HTG 的更新力度，它到底有没有用？**

定义：

\[
r_t=
\frac{
\|\lambda g_{\mathrm{HTG},t}\|
}{
\|g_{\mathrm{OPD},t}\|
}
\]

这个数表示：

> **HTG 这部分参数更新，相当于 OPD 主更新的多少。**

先在真实 trainer 路径上把这个量测准，然后跑三组：

| 实验 | 设置 | 要回答的问题 |
|---|---|---|
| **T5** | 正确 HTG，step 0 的 \(r_0\approx5\%\) | 少量 HTG 有没有帮助？ |
| **T10** | 正确 HTG，step 0 的 \(r_0\approx10\%\) | 更强 HTG 后效果怎么变？ |
| **P10** | 打乱老师位置，step 0 的 \(r_0\approx10\%\) | 相同更新力度下，正确位置关系有没有额外价值？ |

训练开始后 λ 固定，不会每一步重新强行调回 5% 或 10%。

继续记录：

\[
r_t
\]

观察训练过程中真实强度怎么变化。

之后根据结果决定：

- 如果 \(T5>T10\)：HTG 可能加多了会伤，再研究“只在前期用”或“达到一定程度就停”；
- 如果 \(T10>T5\)：继续测试更高强度；
- 如果正确 HTG 和纯 OPD 差不多，但明显好于打乱版本：说明这个内部规律本身有意义，但还没有变成最终能力提升；

