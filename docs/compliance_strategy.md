# 当前 compliance 策略简述

## 背景

当前 policy 的训练数据主要来自纯拖动状态。进入 BRE/compliance 后，机器人-人系统的动力学和观测分布会偏离训练分布，policy 直接规划容易出现反向、横向或大角度掉头。当前实现采用 zero-shot shield 思路：policy 只在 guide 分布下提供轨迹，compliance 状态由外层控制逻辑保守接管。

## 在线策略

1. **guide 状态缓存轨迹**
   - 在 guide 状态，每次 policy inference 后缓存当前 action sequence。
   - 从 guide 切到 BRE/compliance 前，额外缓存当前 cached plan 的剩余部分。

2. **compliance 状态不执行 OOD policy**
   - compliance 状态下仍可运行 policy 用于日志分析。
   - 实际执行使用上一次 guide 缓存轨迹的前半段。
   - 前半段用完后停止，不再 fallback 到 compliance 状态下的新 policy 输出。

3. **compliance control 只减速**
   - forward 被限制为非负小速度，避免倒退。
   - heading 保留缓存 guide 轨迹中的方向，不使用 compliance 阶段重新生成的大幅转向。

4. **safe filter 只允许缩放或停止**
   - 对 compliance 段，QP 只作为可行性检查器。
   - 依次测试原 nominal delta、`0.75`、`0.5`、`0.25`、`0.0` 倍缩放。
   - 不直接采用 QP 输出的二维改向 delta，避免安全层引入 90/180 度转向。

5. **名义预览使用非 BRE 动力学**
   - stashed guide action 的 nominal preview 使用 `bre_override=False`。
   - 真实执行仍保持当前 BRE 状态，只是 planner/safety 的参考方向不被 BRE 物理扰动污染。

## 日志检查字段

重点检查 inference 日志中的字段：

- `using_stashed_compliance_plan`
- `stashed_compliance_source`
- `stashed_compliance_cursor`
- `stashed_compliance_front_half_len`
- `policy_raw_first3`
- `safe_delta_seq`
- `robot_qp_debug_seq[*].forward_only`
- `robot_qp_debug_seq[*].resolution_stage`

如果 BRE 后 `using_stashed_compliance_plan=true`，且 `resolution_stage` 主要是 `forward_only` / `forward_only_backoff` / `forward_only_stop`，说明当前执行路径走的是缓存 guide 轨迹加缩放安全过滤，而不是 compliance 状态下的 OOD policy。

## Tether-Mode Compliance Planning

在得到交互状态后，控制器需要根据 guide/tether 运动学选择不同的交互策略。传统 compliance control 在所有时刻都根据交互力修正控制输入：

$$
u_t^{\mathrm{trad}} =
u_t^\star + K_c(f_t - f_0),
$$ {#eq-traditional-compliance}

其中，$f_t$ 为当前交互力，$f_0$ 为期望交互力，$K_c$ 为顺应增益。持续采用这一策略虽然能够提高柔顺性，但会使机器人在 guide 状态下不必要地减速或偏离路径。为适应变化的人机关系，本文仅在识别到 tether 状态时生成顺应候选动作：

$$
\tilde{u}_t =
u_t^\star +
\mathbf{1}[z_t^\star=\mathrm{tether}]
K_c(f_t - f_0).
$$ {#eq-compliance-candidate}

当 $z_t^\star=\mathrm{guide}$ 时，指示函数为零，控制器直接保留 human-safe diffusion 动作 $u_t^\star$；当 $z_t^\star=\mathrm{tether}$ 时，机器人根据人体输入进行顺应。由于顺应修正可能使动作重新接近障碍物，本文将候选动作在 tether 运动学约束下再次投影到安全控制集合：

$$
\begin{aligned}
u_t =
\arg\min_{u\in\mathcal{U}}\quad
& \frac{1}{2}\lVert u-\tilde{u}_t\rVert^2\\
\mathrm{s.t.}\quad
& x_{t+1}=F_{\mathrm{tether}}(x_t,u,f_t,\hat a_t^h),\\
& h_r(x_t,u;\mathcal{O})\ge 0,\qquad h_h(x_t,u;\mathcal{O})\ge 0.
\end{aligned}
$$ {#eq-compliance}

其中，$F_{\mathrm{tether}}$ 表示 tether 模式下的人机联合状态转移，$\hat a_t^h$ 表示由交互观测或力响应估计的人体主动输入；若没有显式估计器，也可以将其视为来自 $\mathcal{P}_h$ 的有界扰动。$h_r$ 和 $h_h$ 与 @eq-pointcloud-h 或 @eq-vectormap-h 中的定义一致，分别约束 tether 运动学预测得到的机器人和人体位置到障碍物的安全距离。由此，guide 状态保持扩散策略的导航效率，tether 状态顺应人体主动输入，而两种状态下的最终控制输入均满足机器人和人体安全约束。

policy 输出的 forward-heading action 序列为

$$
A_t^\pi = \{a_{t:t+H-1}^\pi\},\qquad
a_i^\pi = [\Delta s_i,\Delta\psi_i].
$$
机器人名义运动为

$$
\begin{aligned}
\theta_{n+1}^r &= \theta_n^r + u_\omega\omega_{\max}\Delta t,\\
\bar v_{n+1}^r &= \beta_r v_n^r + (1-\beta_r)v_{\max}u_v
\begin{bmatrix}
\cos\theta_{n+1}^r\\
\sin\theta_{n+1}^r
\end{bmatrix},\\
\bar p_{n+1}^r &= p_n^r + \bar v_{n+1}^r\Delta t .
\end{aligned}
$$

人体 tether 响应由机器人-人体连线决定。令

$$
d_n=\bar p_n^h-\bar p_n^r,\qquad
\ell_n=\lVert d_n\rVert,\qquad
e_n=d_n/\ell_n .
$$

当 $\ell_n>L$ 时，人体受到沿绳方向的拖拽项：

$$
\bar v_{n+1}^h =
\lambda_h
\left(
v_n^h - k_L(\ell_n-L)e_n\Delta t
\right),
\qquad
\bar p_{n+1}^h = p_n^h+\bar v_{n+1}^h\Delta t .
$$

在 BRE/tether 触发后，若机器人和人体距离仍超过绳长，代码使用一阶位置投影把二者拉回绳长约束附近：

$$
\begin{aligned}
p_{n+1}^r &=
\kappa \bar p_{n+1}^r
+(1-\kappa)(\bar p_{n+1}^h-Le_n),\\
p_{n+1}^h &=
\kappa(\bar p_{n+1}^r+Le_n)
+(1-\kappa)\bar p_{n+1}^h,
\end{aligned}
$$

并用相对速度中的切向分量和远离分量做阻尼：

$$
\begin{aligned}
v_{\mathrm{rel}} &= \bar v_{n+1}^h-\bar v_{n+1}^r,\\
v_{\mathrm{rad}} &= (v_{\mathrm{rel}}^\top e_n)e_n,\\
v_{\mathrm{tan}} &= v_{\mathrm{rel}}-v_{\mathrm{rad}},\\
c_v &= c_{\mathrm{tan}}v_{\mathrm{tan}}
+\max(0,v_{\mathrm{rel}}^\top e_n)e_n,\\
v_{n+1}^r &= \bar v_{n+1}^r+\frac{1}{2}c_v,\\
v_{n+1}^h &= \bar v_{n+1}^h-\frac{1}{2}c_v .
\end{aligned}
$$

这组运动学给出了 @eq-compliance 中 $F_{\mathrm{tether}}$ 的具体实现形式。安全层在固定参考方向上做缩放搜索：

$$
\alpha^\star =
\max_{\alpha\in\{1.0,0.75,0.5,0.25,0.0\}}
\alpha
\quad
\mathrm{s.t.}\quad
\alpha\delta_t^0 \in \mathcal{C}_{\mathrm{safe}}.
$$

最终执行位移为

$$
\delta_t = \alpha^\star \delta_t^0 .
$$

需要注意，policy 只在纯拖动数据上训练，tether 状态对 policy 是 out-of-distribution，因此沿用上一帧stash的action。
