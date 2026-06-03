# QP Safety Filter Optimization Model

本文档解释 `FollowDataset/src/safety_filter.py` 中
`QPSafetyFilter.project_delta()` 的数学模型。该函数的目标是把策略网络输出的
nominal 2D 位移修正为一个更安全的位移，同时尽量少偏离原始动作。

对应代码:

- `FollowDataset/src/safety_filter.py::QPSafetyFilter.project_delta`
- `FollowDataset/src/safety_filter.py::_circle_constraint`
- `FollowDataset/src/safety_filter.py::_segment_constraint`
- `FollowDataset/src/safety_filter.py::_project_halfspace_qp`

## 1. 优化变量

安全过滤器处理的变量是单步二维位移:

$$
\delta \in \mathbb{R}^2
$$

策略网络或上层 planner 给出的原始位移记为:

$$
\delta_{\mathrm{ref}} \in \mathbb{R}^2
$$

函数最终输出:

$$
\delta^\star
$$

其中 $\delta^\star$ 是经过安全投影后的位移。

## 2. 总体优化问题

`project_delta()` 实现的是一个二维 QP 风格投影问题:

$$
\delta^\star =
\arg\min_{\delta \in \mathbb{R}^2}
\frac{1}{2}\|\delta - \delta_{\mathrm{ref}}\|_2^2
$$

subject to

$$
g_i^\top \delta \le h_i,\quad i=1,\dots,N
$$

含义是:

- 目标函数要求新的位移尽量接近原始位移
- 约束要求新的位移不要违反线性化后的安全边界
- 每个约束 $g_i^\top \delta \le h_i$ 来自一个被保护实体和一个障碍物

代码中没有调用外部 QP solver，而是在二维空间中枚举可行候选点，选出距离
$\delta_{\mathrm{ref}}$ 最近的一个。

## 3. 被保护实体

函数先构造被保护实体集合 $\mathcal{E}$。

机器人一定在集合中:

$$
(\mathrm{robot}, p_r, r_r)
$$

其中 $p_r$ 是机器人位置，$r_r$ 是机器人半径。

如果 `include_human=True`，并且传入了 human 状态，则加入:

$$
(\mathrm{human}, p_h, r_h)
$$

如果调用者传入 `extra_entities`，例如 `robot_future` 或 `human_future`，这些实体也会加入约束构造。

因此统一写作:

$$
\mathcal{E} =
\{(name_e, p_e, r_e)\}
$$

其中 $p_e \in \mathbb{R}^2$，$r_e$ 是实体半径。

## 4. 圆形障碍约束

设圆形障碍为:

$$
o_j = (c_j, R_j)
$$

其中 $c_j \in \mathbb{R}^2$ 是圆心，$R_j$ 是障碍半径。

对某个被保护实体 $e=(p_e,r_e)$，先计算实体到障碍圆心的相对向量:

$$
q_{e,j} = p_e - c_j
$$

单位外法向为:

$$
n_{e,j} = \frac{q_{e,j}}{\|q_{e,j}\|_2}
$$

如果 $q_{e,j}$ 太小，代码会用 $-\delta_{\mathrm{ref}}$ 作为 fallback 方向。

考虑实体半径、障碍半径和安全 margin $m$ 后，当前 clearance 为:

$$
c_{e,j}
= \|p_e - c_j\|_2 - (r_e + R_j) - m
$$

如果实体沿位移 $\delta$ 移动，一阶近似后的 clearance 为:

$$
\widehat{c}_{e,j}(\delta)
\approx c_{e,j} + n_{e,j}^\top \delta
$$

安全要求是不要让该近似 clearance 过度下降。代码使用参数 $\alpha$ 构造约束:

$$
n_{e,j}^\top \delta \ge -\alpha c_{e,j}
$$

写成标准半空间形式:

$$
g_{e,j}^\top \delta \le h_{e,j}
$$

其中:

$$
g_{e,j} = -n_{e,j}
$$

$$
h_{e,j} = \alpha c_{e,j}
$$

这对应代码中的:

```python
g = -grad
h = alpha * clearance
```

## 5. 线段障碍约束

设线段障碍为端点:

$$
s_k = [a_k, b_k]
$$

先求实体位置到线段的最近点:

$$
z_{e,k} = \Pi_{[a_k,b_k]}(p_e)
$$

相对向量:

$$
q_{e,k} = p_e - z_{e,k}
$$

单位外法向:

$$
n_{e,k} = \frac{q_{e,k}}{\|q_{e,k}\|_2}
$$

如果 $q_{e,k}$ 退化，代码会用线段法向或 $-\delta_{\mathrm{ref}}$ 作为 fallback。

当前 clearance 为:

$$
c_{e,k} = \|p_e - z_{e,k}\|_2 - r_e - m
$$

线性化后:

$$
\widehat{c}_{e,k}(\delta)
\approx c_{e,k} + n_{e,k}^\top \delta
$$

对应半空间约束:

$$
(-n_{e,k})^\top \delta \le \alpha c_{e,k}
$$

即:

$$
g_{e,k} = -n_{e,k}, \qquad h_{e,k} = \alpha c_{e,k}
$$

## 6. 约束筛选

不是所有障碍都会进入 QP。代码会计算原始动作下的预测 clearance:

$$
c^{\mathrm{pred}}_i =
c_i + n_i^\top \delta_{\mathrm{ref}}
$$

只有满足以下任一条件的约束才会保留:

$$
c_i \le d_{\mathrm{inf}}
$$

或

$$
c^{\mathrm{pred}}_i \le d_{\mathrm{inf}}
$$

其中 $d_{\mathrm{inf}}$ 对应代码中的 `influence_distance`。

保留下来的约束会按:

$$
(c^{\mathrm{pred}}_i, c_i)
$$

从小到大排序，也就是优先处理预测最危险、当前也更接近的障碍。

然后只取前 $K$ 个:

$$
K = \texttt{max\_constraints}
$$

得到最终参与投影的约束集合:

$$
\mathcal{C}_K = \{(g_i,h_i)\}_{i=1}^{K'}
$$

其中 $K' \le K$。

## 7. 二维投影求解

最终问题为:

$$
\min_\delta \frac{1}{2}\|\delta-\delta_{\mathrm{ref}}\|_2^2
$$

subject to

$$
g_i^\top \delta \le h_i,\quad (g_i,h_i)\in\mathcal{C}_K
$$

`_project_halfspace_qp()` 用二维枚举方式求解:

1. 检查 $\delta_{\mathrm{ref}}$ 是否已经满足所有约束。如果满足，它就是最优解。
2. 对每个单独约束，把 $\delta_{\mathrm{ref}}$ 正交投影到该约束边界:

   $$
   \delta_i =
   \delta_{\mathrm{ref}}
   -
   \frac{\max(0, g_i^\top \delta_{\mathrm{ref}} - h_i)}
        {\|g_i\|_2^2}
   g_i
   $$

   如果 $\delta_i$ 满足所有约束，就作为候选。

3. 对每对约束，求两个边界的交点:

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

   如果该交点满足所有约束，就作为候选。

4. 在所有可行候选中选择离 $\delta_{\mathrm{ref}}$ 最近的:

   $$
   \delta^\star =
   \arg\min_{\delta \in \mathcal{D}}
   \|\delta-\delta_{\mathrm{ref}}\|_2^2
   $$

5. 如果没有任何可行候选，代码返回零位移:

   $$
   \delta^\star = [0,0]^\top
   $$

## 8. 参数含义

`margin`:

额外安全距离。margin 越大，clearance 越小，约束越保守。

`alpha`:

约束松紧系数。代码使用 $h=\alpha c$。当 $\alpha$ 较大时，允许的位移范围通常更宽；当 $\alpha$ 较小时，限制更保守。

`influence_distance`:

约束激活距离。只有当前或预测 clearance 进入该距离内，障碍才参与优化。

`max_constraints`:

最多保留多少个最危险约束。该值控制计算量，也影响是否忽略较远的障碍。

## 9. 输出解释

函数返回 `SafetyProjectionResult`。

核心字段:

- `delta`: 投影后的安全位移 $\delta^\star$
- `modified`: 是否修改了原始 $\delta_{\mathrm{ref}}$
- `constraint_count`: 实际参与投影的约束数
- `total_constraint_count`: 筛选后、截断前的约束总数
- `min_clearance`: 参与投影约束里的最小当前 clearance
- `ref_feasible`: 原始 $\delta_{\mathrm{ref}}$ 是否已经满足所有选中约束
- `candidate_count`: 求解器枚举出的可行候选数量
- `best_candidate_kind`: 最优候选类型，可能是 `ref`、`single`、`pair` 或 `zero_fallback`
- `selected_constraints`: 调试用的约束明细，包括 $g$、$h$、clearance 和 violation

## 10. 直观理解

该函数做的事情可以概括为:

1. 把机器人、人、未来位置都看作需要保护的圆。
2. 把圆形障碍和线段障碍转换成局部线性安全边界。
3. 在这些半空间约束内，找一个最接近 policy 原始动作的二维位移。
4. 如果原动作安全，就完全不改；如果原动作危险，就做最小幅度修正。

因此它不是重新规划完整路径，而是一个单步在线 safety projection。
