# BIFrost 分析图表完整说明

本文档解释 BIFrost 生成的所有 SwanLab 图表的含义和诊断方法。

图表按 SwanLab 目录结构组织，分为两大阶段：

- **采样阶段**（`4_1_bif/`）：由 `bif_runner.py` 在 SGLD 采样时实时记录
- **分析阶段**（`1_diag/`、`2_scores/`、`3_influence/`、`4_2_influence/`）：由 `bif_analyzer.py` 在采样完成后分析时生成

---

## 一、采样阶段图表（`4_1_bif/`）

这些图表在 `run-bif` 运行时实时产生，用于监控 SGLD 采样的健康状况。

### 4_1_bif/chain{N}/pool_loss_mean

- **类型**：时序折线图（SwanLab native）
- **x轴**：draw_idx（采样步，含burnin）
- **y轴**：该链上 pool 数据集的平均序列 loss
- **含义**：SGLD 采样过程中，模型在 pool 数据上的 loss 随采样步的变化
- **怎么看**：
  - burnin 期间（is_burnin=1）loss 应该先快速变化后趋于稳定
  - post-burnin 期间 loss 应该围绕某个值平稳波动
  - 如果 loss 持续上升或出现 NaN，说明 SGLD 发散（学习率太大或 nβ 太小）
  - 如果 loss 完全不动，说明 SGLD 冻结（学习率太小或噪声太小）

### 4_1_bif/chain{N}/query_loss_mean

- **类型**：时序折线图
- **x轴**：draw_idx（采样步，含burnin）
- **y轴**：该链上 query 数据集的平均序列 loss
- **含义**：SGLD 采样过程中，模型在 query 数据上的 loss 变化
- **怎么看**：与 pool_loss_mean 类似，但反映 query 方向的响应

### 4_1_bif/chain{N}/param_dist_from_anchor

- **类型**：时序折线图（仅 post-burnin）
- **x轴**：draw_idx
- **y轴**：当前参数与锚点（初始模型）的 L2 距离
- **含义**：SGLD 链在参数空间中走了多远
- **怎么看**：
  - 距离持续增长 → 模型在漂移（drift），不是在平衡分布附近采样
  - 距离平稳波动 → 采样健康
  - 距离为0或极小 → SGLD 没有动（步长太小）

### 4_1_bif/chain{N}/is_burnin

- **类型**：时序折线图
- **值**：1 = burnin 阶段，0 = post-burnin 阶段
- **含义**：标记哪些步是 burnin，哪些是正式采样

### 4_1_bif/chain{N}/sgld_step_loss

- **类型**：时序折线图
- **x轴**：原始 SGLD step（每一步都记录）
- **y轴**：该步的 training loss
- **含义**：每个 SGLD 步的即时 loss（比 draw 级别更细粒度）
- **怎么看**：用于诊断 SGLD 每步是否稳定

### 4_1_bif/chain{N}/sgld_grad_norm

- **类型**：时序折线图
- **x轴**：原始 SGLD step
- **y轴**：该步梯度的 L2 范数
- **含义**：SGLD 每步的梯度大小
- **怎么看**：与 noise_norm 对比，判断信号噪声比

### 4_1_bif/chain{N}/sgld_noise_norm

- **类型**：时序折线图
- **x轴**：原始 SGLD step
- **y轴**：该步注入噪声的 L2 范数
- **含义**：SGLD 每步的随机噪声大小
- **怎么看**：
  - 如果 noise_norm >> grad_norm → 噪声主导，BIF 中的梯度项被淹没
  - 如果 noise_norm << grad_norm → 梯度主导，采样可能不够探索
  - 理想情况：两者量级可比（SNR 在 0.1~1 之间）

### 4_1_bif/chain{N}/sgld_signal_noise_ratio

- **类型**：时序折线图
- **x轴**：原始 SGLD step
- **y轴**：grad_norm / noise_norm
- **含义**：每步的信号噪声比（SNR）
- **怎么看**：
  - SNR >> 1：梯度主导，采样不够随机，可能陷入局部
  - SNR << 1：噪声主导，梯度信息被淹没，BIF 分数不可靠
  - SNR ≈ 0.1~1：较健康的采样

### 4_1_bif/chain{N}/draw/grad_norm_mean, grad_norm_max

- **类型**：时序折线图
- **x轴**：draw_idx（含burnin）
- **y轴**：该 draw 周期内所有 SGLD 步的梯度范数均值/最大值
- **含义**：draw 级别的梯度统计（比 step 级别更粗粒度）

### 4_1_bif/chain{N}/draw/noise_norm_mean, noise_norm_max

- **类型**：时序折线图
- **x轴**：draw_idx
- **y轴**：该 draw 周期内噪声范数的均值/最大值

### 4_1_bif/chain{N}/draw/snr_mean, snr_min

- **类型**：时序折线图
- **x轴**：draw_idx
- **y轴**：该 draw 周期内 SNR 的均值/最小值
- **怎么看**：重点看 snr_min，如果长时间 snr_min << 0.01，说明采样质量差

### 4_1_bif/chain{N}/draw/step_loss_mean

- **类型**：时序折线图
- **x轴**：draw_idx
- **y轴**：该 draw 周期内所有 SGLD 步的 loss 均值

### 4_1_bif/chain{N}/draw/num_steps

- **类型**：时序折线图
- **x轴**：draw_idx
- **y轴**：该 draw 周期包含的 SGLD 步数
- **含义**：等于 `num_steps_bw_draws`（通常为 1）

### 4_1_bif/chain{N}/draw/actual_sgld_step

- **类型**：时序折线图
- **x轴**：draw_idx
- **y轴**：该 draw 对应的实际 SGLD 全局步号
- **含义**：用于核对 draw 与 step 的对应关系

### 4_1_bif/pool_loss_all_chains/chain_{N}

- **类型**：多链叠加时序图（SwanLab native overlay）
- **x轴**：draw_idx（从 0 开始，包含 burnin）
- **y轴**：各链的 pool loss 均值
- **含义**：所有链的 pool loss 叠加在一张图上，用于直观对比链间行为
- **怎么看**：
  - 各链应该收敛到相近的 loss 水平 → 链间一致
  - 如果某条链明显偏离 → 该链可能有问题
  - burnin 阶段（早期）各链可能差异较大，post-burnin 应趋同

### 4_1_bif/query_loss_all_chains/chain_{N}

- **类型**：多链叠加时序图
- **含义**：同上，但是 query loss
- **怎么看**：同上

---

## 二、分析阶段图表

### 1_diag/ — 诊断图表

#### 1_diag/{ckpt}/pool_loss/chain_{N}

- **类型**：时序折线图（SwanLab native，post-burnin only）
- **x轴**：draw_idx（从 burnin 结束后开始）
- **y轴**：该链在 pool 数据上的 loss 均值
- **含义**：正式采样阶段各链的 loss 轨迹（不含 burnin）
- **怎么看**：应围绕某值平稳波动，无趋势

#### 1_diag/{ckpt}/query_loss/chain_{N}

- **类型**：时序折线图
- **含义**：同上，query 方向

#### 1_diag/{ckpt}/pool_loss(chains_mean), query_loss(chains_mean)

- **类型**：时序折线图
- **y轴**：所有链的均值
- **含义**：链平均 loss 轨迹

#### 1_diag/{ckpt}/convergence/bif_mean_avg, bif_mean_std

- **类型**：时序折线图
- **x轴**：n_draws（使用的 draw 数量）
- **y轴**：用前 n_draws 计算的 BIF 均值的平均 / 标准差
- **含义**：BIF 分数是否随 draw 数量增加而收敛
- **怎么看**：
  - bif_mean_avg 应该随 n_draws 增加趋于稳定
  - bif_mean_std 应该随 n_draws 增加而下降
  - 如果持续剧烈波动 → draw 不够，需要更多采样

#### 1_diag/rhat/{ckpt}

- **类型**：柱状图
- **4个柱**：mean（R-hat 均值）、max（R-hat 最大值）、frac<1.1（R-hat<1.1 的比例）、frac<1.2（R-hat<1.2 的比例）
- **含义**：Gelman-Rubin R-hat 诊断，衡量多链是否收敛到同一分布
- **怎么看**：
  - R-hat < 1.1 → 链已收敛（好）
  - R-hat > 1.2 → 链未收敛（需要更多 burnin 或 draw）
  - frac<1.1 越接近 1.0 越好
  - max R-hat 是最差样本的收敛情况，应关注

#### 1_diag/chain_vs_rest/chain_{N}/{ckpt}

- **类型**：散点图
- **x轴**：chain_N 单独计算的每个 pool 样本的 BIF 分数
- **y轴**：其余链计算的同一 pool 样本 BIF 分数的均值
- **含义**：链间一致性检验——不同链对"谁重要"的判断是否一致
- **怎么看**：
  - 点分布在对角线附近 → 链间一致，BIF 可信
  - 点散乱 → 链间不一致，采样不足或 SGLD 参数有问题
  - 如果有系统性偏离对角线（如某链系统性偏高），说明该链有问题

---

### 2_scores/ — BIF 分数分析

#### 2_scores/distribution/{ckpt}

- **类型**：直方图（柱状图）
- **x轴**：BIF 分数区间
- **y轴**：落入该区间的 pool 样本数量
- **含义**：所有 pool 样本 BIF 分数的分布
- **怎么看**：
  - 如果分布很窄且集中在 0 附近 → BIF 没区分度（所有样本影响力差不多）
  - 如果有明显的长尾 → 有少量高影响力样本（好）
  - 如果呈双峰 → 可能有两个不同影响力的子群体

#### 2_scores/cross_cov_vs_selfvar/{ckpt}

- **类型**：散点图
- **x轴**：pool_self_variance（pool 样本在 SGLD 采样过程中的 loss 自方差）
- **y轴**：cross_cov_avg_over_queries（该 pool 样本对所有 query 的平均交叉协方差，即 BIF 中的"影响力"项）
- **含义**：区分"真实影响力"和"自身不稳定"。BIF = cross_cov - self_var/nβ，如果 self_var 很大而 cross_cov 不大，说明 BIF 低是因为该样本自身不稳定而非没有影响力。
- **怎么看**：
  - 点在对角线附近 → cross_cov 和 self_var 强相关，BIF 分数被自方差主导（不好，说明 BIF 没区分出真实影响力）
  - 有明显偏离对角线的点（y 大 x 小）→ 这些样本有真实影响力且自身稳定（好）
  - y 整体偏低 → nβ 太大，梯度项被过度惩罚
  - x 整体偏大 → SGLD 采样不稳定，loss 方差大

#### 2_scores/eigenvalue_spectrum/{ckpt}

- **类型**：柱状图
- **x轴**：特征值编号（ev0, ev1, ...）
- **y轴**：双系列——特征值大小 + 累积方差占比（cumulative_frac）
- **含义**：BIF 矩阵（对称化，对角线置零）的特征值谱。反映影响力的"有效维度"。
- **怎么看**：
  - 前几个特征值远大于后续（cumulative_frac 快速趋 1）→ 影响力集中，少数方向主导
  - 特征值缓慢衰减 → 影响力分散在多个方向
  - 有大的负特征值 → BIF 矩阵不是正半定的，可能采样质量差或 nβ 不合适

---

### 3_influence/ — 影响力分析

#### 3_influence/pool_corr_distribution/{ckpt}

- **类型**：直方图
- **x轴**：pool-pool 相关系数区间
- **y轴**：数量
- **含义**：BIF 矩阵上三角（pool 样本两两之间）的相关系数分布
- **怎么看**：
  - 分布集中在 0 附近 → pool 样本间相互独立（好）
  - 有大量高相关 → pool 样本间有冗余影响力，top-K 选择可能不太稳定
  - 有大量负相关 → 某些 pool 样本的影响方向相反

#### 3_influence/cross_corr_distribution/{ckpt}

- **类型**：直方图
- **x轴**：pool-query 交叉相关系数区间
- **y轴**：数量
- **含义**：每个 pool 样本对所有 query 的平均交叉相关系数的分布。这是 BIF 的核心——pool 样本对 query loss 变化的协方差。
- **怎么看**：
  - 分布宽 → 不同 pool 样本对 query 的影响力差异大（好，有区分度）
  - 分布窄且集中在 0 → BIF 没有区分出有意义的影响力
  - 正偏 → 多数 pool 样本对 query 有正向影响

#### 3_influence/source_x_query_heatmap/{ckpt}（有 source 信息时）

- **类型**：热力图
- **x轴**：query 编号（q0, q1, ...）
- **y轴**：数据来源（source），按平均影响力排序
- **颜色**：该 source 对该 query 的平均交叉相关系数
- **含义**：哪些 source 对哪些 query 的影响力大
- **怎么看**：
  - 某行整体颜色深 → 该 source 影响力大
  - 某列整体颜色深 → 该 query 容易被影响
  - 特定行列交叉深 → 该 source 对该 query 有特殊影响

#### 3_influence/pool_x_query_heatmap/{ckpt}（无 source 信息时）

- **类型**：热力图
- **含义**：同上，但每个 pool 样本单独一行（不按 source 聚合）

#### 3_influence/bif_topK_heatmap/{ckpt}

- **类型**：热力图
- **x轴/y轴**：top-K BIF 样本（按排名，标注 rank 和 source）
- **颜色**：BIF 相关系数
- **含义**：影响力最大的 K 个 pool 样本之间的相互关系
- **怎么看**：
  - 对角线为 0（已置零）
  - 高亮的块结构 → 同 source 的样本间有强相关
  - 整体偏暗 → top-K 样本间相互独立

#### 3_influence/bif_source_blocks/{ckpt}

- **类型**：热力图
- **x轴/y轴**：数据来源（按 BIF 均分排序）
- **颜色**：source 间的平均 BIF 相关系数（对角线 = source 内部非对角线均值）
- **含义**：source 粒度的影响力块结构
- **怎么看**：
  - 对角线亮 + 非对角线暗 → 同 source 的样本影响力相似，跨 source 不同
  - 某行/列整体亮 → 该 source 整体影响力大

#### 3_influence/source_distribution/{ckpt}

- **类型**：分组柱状图
- **x轴**：数据来源
- **y轴**：三系列——top_k（top-K 中该 source 占比）、bottom_k（bottom-K 中该 source 占比）、pool（pool 中该 source 占比）
- **含义**：各 source 在 top-K / bottom-K / 全体 pool 中的占比
- **怎么看**：
  - top_k >> pool → 该 source 被富集（影响力大）
  - top_k << pool → 该 source 影响力小
  - top_k ≈ pool → 该 source 影响力与随机相当

#### 3_influence/score_by_source/{ckpt}

- **类型**：箱线图
- **x轴**：数据来源
- **y轴**：BIF 分数分布（中位数、Q1、Q3、须）
- **含义**：各 source 内部 BIF 分数的分布
- **怎么看**：
  - 箱体位置高 → 该 source 整体影响力大
  - 箱体窄 → 该 source 内部影响力均匀
  - 有离群点 → 某些特定样本影响力异常

#### 3_influence/enrichment/{ckpt}

- **类型**：柱状图
- **x轴**：数据来源
- **y轴**：enrichment_ratio = top_k_fraction / pool_fraction
- **含义**：各 source 在 top-K 中的富集程度（>1 表示富集，<1 表示缺失）
- **怎么看**：
  - enrichment > 1 → 该 source 在 top-K 中被富集（影响力大）
  - enrichment ≈ 1 → 无富集（随机水平）
  - enrichment < 1 → 该 source 在 top-K 中缺失（影响力小）

---

### 4_2_influence/ — 全局影响力分析（跨 checkpoint）

以下图表只在有多个 checkpoint（如 finetune 过程中保存的多个模型）时才有意义。如果只有 `final_model` 一个 checkpoint，部分图表不会生成或意义有限。

#### 4_2_influence/samples/top, samples/bottom

- **类型**：表格
- **列**：rank、source、各 checkpoint 分数、traj_mean（轨迹均分）、emergence（最后一个 checkpoint 减第一个的分数差）、text（样本全文）
- **含义**：按 traj_mean 排序的 top/bottom 样本详情
- **怎么看**：
  - top 样本 = 全程高影响力的样本
  - bottom 样本 = 全程低影响力的样本
  - emergence > 0 → 该样本影响力在训练过程中逐渐增大（"涌现"）

#### 4_2_influence/samples/top/{ckpt}, samples/bottom/{ckpt}

- **类型**：表格
- **列**：rank、sample_id、source、score、cross_corr、mean_loss、text
- **含义**：每个 checkpoint 独立排名的 top/bottom 样本

#### 4_2_influence/trajectory/topk_mean, topk_std

- **类型**：时序折线图
- **x轴**：checkpoint 编号
- **y轴**：top-K 样本在各 checkpoint 的 BIF 均值 / 标准差
- **含义**：高影响力样本的影响力随训练的变化

#### 4_2_influence/trajectory/top_by_mean

- **类型**：折线图
- **x轴**：checkpoint 名称
- **y轴**：多条线，每条 = 一个 top 样本的 BIF 分数轨迹
- **含义**：影响力最高的 N 个样本的分数随训练的变化
- **怎么看**：
  - 上升 → 训练使该样本影响力增大
  - 下降 → 训练使该样本影响力减小
  - 平稳 → 训练对该样本影响力无影响

#### 4_2_influence/trajectory/top_emergent

- **类型**：折线图
- **含义**：按 emergence（最后减第一）排序的 top N 样本的分数轨迹
- **怎么看**：这些是训练过程中影响力变化最大的样本

#### 4_2_influence/source/score_vs_checkpoint

- **类型**：热力图（多 checkpoint 时才有）
- **x轴**：checkpoint
- **y轴**：source
- **颜色**：该 source 在该 checkpoint 的平均 BIF 分数
- **含义**：各 source 的影响力如何随训练变化

#### 4_2_influence/source/topk_count_vs_checkpoint

- **类型**：热力图（多 checkpoint 时才有）
- **x轴**：checkpoint
- **y轴**：source
- **颜色**：该 source 在 top-K 中的数量
- **含义**：各 source 在高影响力样本中的数量变化

#### 4_2_influence/source/shift_topk

- **类型**：堆叠柱状图（多 checkpoint 时才有）
- **x轴**：checkpoint
- **y轴**：各 source 在 top-K 中的占比（堆叠）
- **含义**：top-K 的 source 组成如何随训练变化
- **怎么看**：
  - 堆叠比例变化 → 训练改变了哪些 source 的样本更有影响力
  - 比例稳定 → 训练没有改变影响力来源结构

#### 4_2_influence/rank_stability_spearman

- **类型**：热力图（多 checkpoint 时才有）
- **x轴/y轴**：checkpoint 名称
- **颜色**：Spearman 相关系数
- **含义**：不同 checkpoint 之间 BIF 排名的 Spearman 相关
- **怎么看**：
  - 相关系数高（>0.9）→ 排名稳定，BIF 结果可信
  - 相关系数低 → 排名不稳定，BIF 结果对训练步数敏感

#### 4_2_influence/topK_overlap

- **类型**：热力图（多 checkpoint 时才有）
- **x轴/y轴**：checkpoint 名称
- **颜色**：top-K 集合的 Jaccard 重叠度
- **含义**：不同 checkpoint 选出的 top-K 样本有多大的交集
- **怎么看**：
  - 重叠度高 → 选择的样本稳定
  - 重叠度低 → 不同训练阶段选出的高影响力样本不同

---

## 三、关键诊断流程

按以下顺序检查图表，可以快速判断 BIF 结果是否可靠：

1. **SGLD 采样健康**：
   - 看 `4_1_bif/chain{N}/pool_loss_mean`：post-burnin 是否平稳波动
   - 看 `4_1_bif/chain{N}/sgld_signal_noise_ratio`：SNR 是否在 0.01~1 范围
   - 看 `4_1_bif/pool_loss_all_chains`：多链是否收敛到同一水平

2. **链间一致性**：
   - 看 `1_diag/rhat`：frac<1.1 是否 >0.8
   - 看 `1_diag/chain_vs_rest`：点是否在对角线附近

3. **BIF 区分度**：
   - 看 `2_scores/distribution`：是否有长尾
   - 看 `2_scores/cross_cov_vs_selfvar`：cross_cov 是否独立于 self_var
   - 看 `3_influence/cross_corr_distribution`：分布是否宽

4. **影响力解释**：
   - 看 `3_influence/source_distribution` 和 `enrichment`：哪些 source 影响力大
   - 看 `3_influence/bif_topK_heatmap`：top 样本间的关系
   - 看 `4_2_influence/samples/top`：具体哪些文本影响力最大
