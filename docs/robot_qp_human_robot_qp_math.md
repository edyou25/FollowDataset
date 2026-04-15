# `robot_qp` 与 `human_robot_qp` 的数学表达式整理

本文档按当前代码实现整理，目标是把 `planning.py` 和 `src/safety_filter.py` 中的在线安全过滤写成统一的数学形式，便于分析、汇报和后续改造。

对应代码入口:

- `planning.py` 中的 `_safety_filter_delta()`
- `src/safety_filter.py` 中的 `QPSafetyFilter.project_delta()`
- `src/safety_filter.py` 中的 `_project_halfspace_qp()`

## 1. 记号

设:

- 机器人当前位姿为 $(p_r, \theta_r)$，其中 $p_r \in \mathbb{R}^2$
- 人当前位姿为 $p_h \in \mathbb{R}^2$
- 机器人半径为 $r_r$
- 人半径为 $r_h$
- 第 $j$ 个圆形障碍的圆心和半径为 $(c_j, R_j)$
- 第 $k$ 个线段障碍为 $[a_k, b_k]$
- 安全 margin 为 $m$
- 约束缩放系数为 $\alpha$
- 最大约束数为 $K$
- 影响距离为 $d_{\mathrm{inf}}$

在线安全层处理的变量不是原始网络 action，而是单步二维位移:

$$
\delta_{\mathrm{ref}} \in \mathbb{R}^2
$$

它表示当前时刻 nominal planner 希望机器人在一个 data step 内走出的平面位移。

安全过滤后的输出为:

$$
\delta^\star \in \mathbb{R}^2
$$

## 2. 从策略 action 到 nominal 位移

在 `forward_heading` 模式下，策略输出:

$$
a = \begin{bmatrix}\Delta s \\ \Delta \psi \end{bmatrix}
$$

其中:

- $\Delta s$ 是前进位移意图
- $\Delta \psi$ 是 heading 增量意图

代码不会直接把这个二元组当成 QP 变量，而是先通过一次动力学 preview 得到 nominal 位移:

$$
\delta_{\mathrm{ref}} = p_r^{\mathrm{preview}} - p_r
$$

其中 $p_r^{\mathrm{preview}}$ 来自按当前控制映射和仿真步长 rollout 一次后的机器人位置。

在实现上，控制量近似满足:

$$
u_{\mathrm{turn}} = \mathrm{clip}\left(
\frac{\mathrm{turn\_gain} \cdot \Delta \psi}{\omega_{\max}\Delta t},
-1, 1
\right)
$$

$$
u_{\mathrm{fwd}} = \mathrm{clip}\left(
\frac{\Delta s}{v_{\max}\Delta t},
-1, 1
\right)
$$

若开启 `curvature_slowdown`，则还会乘一个速度缩放:

$$
\rho = \min\left(1, \frac{|\mathrm{turn\_gain}\cdot \Delta \psi|}{\omega_{\max}\Delta t}\right)
$$

$$
\gamma = \max\left(\gamma_{\min}, 1 - c_{\kappa}\rho \right)
$$

$$
u_{\mathrm{fwd}} \leftarrow \gamma u_{\mathrm{fwd}}
$$

其中:

- $\omega_{\max}$ 对应 `turn_speed`
- $v_{\max}$ 对应 `robot_speed`
- $\gamma_{\min}$ 对应 `min_speed_scale`
- $c_{\kappa}$ 对应 `curvature_scale`

对 `delta`/`velocity` 模式，也会先转成世界坐标系下的 $\delta_{\mathrm{ref}}$，后续安全优化形式相同。

## 3. 统一的 QP 风格投影问题

安全层的核心目标是: 在尽量贴近 nominal 位移的前提下，让位移满足一组线性化安全约束。

统一写成:

$$
\delta^\star
=
\arg\min_{\delta \in \mathbb{R}^2}
\frac{1}{2}\|\delta - \delta_{\mathrm{ref}}\|_2^2
$$

subject to

$$
g_i^\top \delta \le h_i,\quad i = 1,\dots,N
$$

这里:

- $g_i \in \mathbb{R}^2$
- $h_i \in \mathbb{R}$
- 每个约束对应一个障碍物对某个被保护实体的局部线性化安全边界

`robot_qp` 和 `human_robot_qp` 的区别不在目标函数，而在约束集合由哪些“被保护实体”生成。

## 4. 被保护实体集合

### 4.1 `robot_qp`

`robot_qp` 模式保护两组实体:

$$
\mathcal{E}_{\mathrm{robot}} =
\left\{
(\text{robot}, p_r, r_r),
(\text{robot\_future}, p_r^+, r_r)
\right\}
$$

其中 $p_r^+$ 是按 nominal 位移 preview 后的机器人位置。

也就是说，当前实现不仅保护“当前机器人”，还保护“nominal rollout 后的未来机器人”。

### 4.2 `human_robot_qp`

`human_robot_qp` 在上式基础上额外保护两组 human 实体:

$$
\mathcal{E}_{\mathrm{human\_robot}} =
\left\{
(\text{robot}, p_r, r_r),
(\text{robot\_future}, p_r^+, r_r),
(\text{human}, p_h, r_h),
(\text{human\_future}, p_h^+, r_h)
\right\}
$$

其中 $p_h^+$ 是 nominal preview 后的人位置。

所以从数学上看:

- `robot_qp` 只要求机器人相关实体安全
- `human_robot_qp` 同时要求机器人和人都对障碍安全

两者使用同一组障碍物、同一个优化目标、同一个投影求解器。

## 5. 圆形障碍约束

对任意被保护实体 $e = (p_e, r_e)$ 和圆形障碍 $(c_j, R_j)$，代码先构造相对向量:

$$
\mathrm{rel}_{e,j} = p_e - c_j
$$

距离为:

$$
d^{\mathrm{dist}}_{e,j} = \|\mathrm{rel}_{e,j}\|_2
$$

带半径与 margin 的 clearance 为:

$$
\mathrm{clr}_{e,j} =
\|\mathrm{rel}_{e,j}\|_2 - (R_j + r_e) - m
$$

单位法向取为:

$$
n_{e,j} =
\mathrm{normalize}(\mathrm{rel}_{e,j})
$$

若 $\mathrm{rel}_{e,j}$ 退化为零向量，则代码使用 $-\delta_{\mathrm{ref}}$ 作为 fallback 方向。

对 clearance 做一阶线性化，位移后的近似 clearance 为:

$$
\widehat{\mathrm{clr}}_{e,j}(\delta)
\approx
\mathrm{clr}_{e,j} + n_{e,j}^\top \delta
$$

若要求线性化 clearance 非负，并引入代码中的 $\alpha$ 缩放，则约束写成:

$$
n_{e,j}^\top \delta \ge -\alpha \,\mathrm{clr}_{e,j}
$$

等价地，可以写成标准半空间形式:

$$
(-n_{e,j})^\top \delta \le \alpha \,\mathrm{clr}_{e,j}
$$

因此有:

$$
g_{e,j} = -n_{e,j}, \qquad
h_{e,j} = \alpha \,\mathrm{clr}_{e,j}
$$

代码还会计算 nominal 位移对应的预测 clearance:

$$
\mathrm{pclr}_{e,j}
=
\mathrm{clr}_{e,j} + n_{e,j}^\top \delta_{\mathrm{ref}}
$$

它不直接进入目标函数，而是用于约束筛选和排序。

## 6. 线段障碍约束

对任意实体 $e=(p_e,r_e)$ 和线段障碍 $[a_k,b_k]$，先求点到线段的最近点:

$$
z_{e,k} = \Pi_{[a_k,b_k]}(p_e)
$$

相对向量为:

$$
\mathrm{rel}_{e,k} = p_e - z_{e,k}
$$

clearance 为:

$$
\mathrm{clr}_{e,k}
=
\|\mathrm{rel}_{e,k}\|_2 - r_e - m
$$

单位法向为:

$$
n_{e,k} = \mathrm{normalize}(\mathrm{rel}_{e,k})
$$

如果 $\mathrm{rel}_{e,k}$ 退化，则代码优先使用线段法向，若线段法向也退化，再回退到 $-\delta_{\mathrm{ref}}$。

同样做一阶线性化，可得约束:

$$
n_{e,k}^\top \delta \ge -\alpha\,\mathrm{clr}_{e,k}
$$

等价写成:

$$
g_{e,k}^\top \delta \le h_{e,k}
$$

其中

$$
g_{e,k} = -n_{e,k}, \qquad
h_{e,k} = \alpha\,\mathrm{clr}_{e,k}
$$

对应的 nominal 预测 clearance 为:

$$
\mathrm{pclr}_{e,k}
=
\mathrm{clr}_{e,k} + n_{e,k}^\top \delta_{\mathrm{ref}}
$$

## 7. 约束筛选与截断

当前实现不会把所有障碍都放入 QP，而是先做影响距离筛选。

某条约束会被纳入候选集合，当且仅当:

$$
\mathrm{clr}_i \le d_{\mathrm{inf}}
\quad \text{or} \quad
\mathrm{pclr}_i \le d_{\mathrm{inf}}
$$

得到所有候选约束后，按如下关键字升序排序:

$$
(\mathrm{pclr}_i,\ \mathrm{clr}_i)
$$

然后只保留前 $K$ 条:

$$
\mathcal{I}_{\mathrm{sel}} = \mathrm{TopK}\left(
\mathrm{sort}_{(\mathrm{pclr},\mathrm{clr})}
(\mathcal{I})
\right)
$$

因此最终真正进入投影问题的约束个数为:

$$
N = \min(|\mathcal{I}|, K)
$$

## 8. 当前实现中的“QP 求解器”

虽然名字叫 QP style，但当前代码没有调用通用二次规划库，而是在二维空间里直接枚举候选解。

### 8.1 原始参考解

若 $\delta_{\mathrm{ref}}$ 已满足全部约束，则它本身是候选解:

$$
g_i^\top \delta_{\mathrm{ref}} \le h_i,\quad \forall i
$$

### 8.2 单约束投影

对每个约束 $i$，若原解违反该约束，则投影到其边界:

$$
v_i = g_i^\top \delta_{\mathrm{ref}} - h_i
$$

$$
\delta_i
=
\delta_{\mathrm{ref}}
- \frac{\max(0,v_i)}{\|g_i\|_2^2}g_i
$$

若 $\delta_i$ 满足全部约束，则纳入候选集合。

### 8.3 双约束交点

对任意一对约束 $(i,j)$，若

$$
\det
\begin{bmatrix}
g_i^\top \\
g_j^\top
\end{bmatrix}
\neq 0
$$

则求解:

$$
\begin{bmatrix}
g_i^\top \\
g_j^\top
\end{bmatrix}
\delta_{ij}
=
\begin{bmatrix}
h_i \\
h_j
\end{bmatrix}
$$

如果 $\delta_{ij}$ 满足全部约束，则也纳入候选集合。

### 8.4 选解准则

最终从所有可行候选中选取与 nominal 位移最接近的那个:

$$
\delta_{\mathrm{qp}}
=
\arg\min_{\delta \in \mathcal{C}_{\mathrm{feasible}}}
\|\delta - \delta_{\mathrm{ref}}\|_2^2
$$

若候选集合为空，则直接返回:

$$
\delta_{\mathrm{qp}} = 0
$$

## 9. QP 后的精确仿真校验与 backoff

代码不会无条件执行 $\delta_{\mathrm{qp}}$，而是再做一次精确仿真校验。

记仿真碰撞判定算子为:

$$
\mathrm{Collide}(\delta; \mathrm{protect\_robot}, \mathrm{protect\_human})
\in \{0,1\}
$$

若

$$
\mathrm{Collide}(\delta_{\mathrm{qp}}; 1, \mathbf{1}_{\mathrm{human\_robot\_qp}})=0
$$

则最终输出:

$$
\delta^\star = \delta_{\mathrm{qp}}
$$

否则代码按固定缩放序列尝试 backoff:

$$
\lambda \in \{0.75, 0.5, 0.25, 0\}
$$

依次测试:

$$
\delta = \lambda \delta_{\mathrm{qp}}
$$

选择第一个不碰撞的解作为最终输出。也就是说:

$$
\delta^\star
=
\lambda^\star \delta_{\mathrm{qp}}
$$

其中

$$
\lambda^\star
=
\min_{\lambda \in \{0.75,0.5,0.25,0\}}
\left\{
\lambda \mid
\mathrm{Collide}(\lambda \delta_{\mathrm{qp}}; 1, \mathbf{1}_{\mathrm{human\_robot\_qp}})=0
\right\}
$$

这里的“最小”是按代码尝试顺序意义上的第一个成功值，而不是数值最小值。

当前代码中曾经存在的基于 `min_clearance` 的强制 stop gate 已被注释掉，因此当前有效流程是:

1. 线性化约束投影
2. 精确仿真碰撞检查
3. 固定比例 backoff

而不是 “QP 后再按 clearance 阈值直接停车”。

## 10. `robot_qp` 与 `human_robot_qp` 的最终数学区别

把前面的内容合并后，两种模式可以写成完全相同的优化框架:

$$
\delta^\star
=
\mathrm{Backoff}\left(
\arg\min_{\delta \in \mathbb{R}^2}
\frac{1}{2}\|\delta-\delta_{\mathrm{ref}}\|_2^2
\quad
\text{s.t. }
g_i^\top \delta \le h_i,\ i \in \mathcal{I}(\mathcal{E})
\right)
$$

其中差别只在实体集合 $\mathcal{E}$:

### `robot_qp`

$$
\mathcal{E} = \mathcal{E}_{\mathrm{robot}}
=
\{(\text{robot}, p_r, r_r), (\text{robot\_future}, p_r^+, r_r)\}
$$

### `human_robot_qp`

$$
\mathcal{E} = \mathcal{E}_{\mathrm{human\_robot}}
=
\{(\text{robot}, p_r, r_r), (\text{robot\_future}, p_r^+, r_r),
(\text{human}, p_h, r_h), (\text{human\_future}, p_h^+, r_h)\}
$$

因此:

- `robot_qp` 约束更少，可行域通常更大
- `human_robot_qp` 约束更多，可行域通常更小，也更保守

## 11. 可以直接用于论文或汇报的简化表述

如果只需要一个简洁版本，可以直接写成:

> We project the nominal 2D displacement $\delta_{\mathrm{ref}}$ predicted by the diffusion policy onto a set of linearized safety halfspaces:
>
> $$
> \delta^\star
> =
> \arg\min_{\delta}
> \frac{1}{2}\|\delta-\delta_{\mathrm{ref}}\|_2^2
> \quad
> \text{s.t. }
> g_i^\top \delta \le h_i
> $$
>
> where each halfspace is derived from either a circular obstacle or a segment obstacle around the protected entities. In `robot_qp`, the protected entities are the current robot state and its nominal future state. In `human_robot_qp`, the protected set additionally includes the current human state and the nominal future human state. The projected solution is then validated by exact rollout simulation and, if necessary, scaled down by a fixed backoff schedule.

## 12. 备注

- 本文档描述的是当前代码实现，不是理想化的连续时间最优控制形式。
- 当前求解器是二维候选枚举式 QP 风格投影，不是通用数值 QP solver。
- `human_robot_qp` 本质上是 `robot_qp` 在约束实体集合上的扩展，而不是完全不同的优化器。
