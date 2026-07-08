# 论文图像 Nature 风格重绘规范（数据已逐项核验）

> 硬约束：**所有数据点必须来自下列真实数据源，不得为美观平滑/修改/臆造**。柱状图数值已从论文表格逐项核验，直接采用本文件给定的精确值。时间序列从 metrics.csv 原样读取（仅允许标注的滑动平均）。每张图须在脚本注释中标注数据来源文件。
> 输出目录：`docs/论文相关/5.29需要修订论文/thesis/thesis-latex/figures/`，**覆盖同名 PNG**（LaTeX 已用这些文件名，保持文件名不变以避免交叉引用错位）。
> **去除图内一切"图4.X"标题与中文大标题**（编号与标题由 LaTeX caption 提供）；仅保留坐标轴标签、图例、必要的数值标注与注记。
> 字体：`WenQuanYi Micro Hei`（matplotlib 可用的简体中文字体）；`axes.unicode_minus=False`；导出 `dpi=300`，`bbox_inches='tight'`，白底。

## 统一 Nature 风格基线（建议 rcParams）
- 配色（色盲友好，全图统一）：基线/对照=蓝 `#4C72B0`，本文 GCGL/重点=红 `#C44E52`，第三系列=绿 `#55A868`，第四系列=橙 `#CC8963`/紫 `#8172B3`。
- 字号：坐标轴标签 10–11，刻度 9，图例 9，数值标注 8–8.5。
- 线宽 1.3–1.6；去顶/右边框（`spines top/right off`）；网格 `alpha 0.3, ls '--'` 仅 y 轴或浅网格。
- 图尺寸：单图约 (7,4.2)；双面板 (8,6)；多面板按内容。误差棒用真实 ±std（见下）。

---

## A. 训练时间序列（数据源：`checkpoints/thesis/<exp>/metrics.csv`，仅风格重绘，不加会与表格冲突的数值标注）
实验目录：Baseline-MSE=`exp01_baseline_mse`，GCGL=`exp21_grad_clip_goal`，GradClip-Only=`exp22_grad_clip_only`，GoalLoss-Only=`exp17_goal_reaching`。横轴=`step`(训练迭代次数)。建议滑动平均窗口 100（原始值浅色 alpha≈0.15 叠加）。

- **image5.png**（图4.2 训练过程加权到达指数(AR)曲线对比）：列 `ar`，画 4 条（Baseline/GCGL/GradClip/GoalLoss）。可在基线峰值后区域用浅色背景或箭头标注"性能退化区"，**不要标注与正文冲突的具体数值**。Y 轴=加权到达指数(AR)。
- **image6.png**（图4.3 基线实验损失尺度分析）：仅 `exp01_baseline_mse`。上面板对数 y 轴画 `loss_collide`（碰撞损失）与 `loss_v`（速度跟踪损失）；下面板画比值 `loss_collide/loss_v`，标注 top15% 高占比区间。
- **image9.png**（图4.4 训练过程速度与无碰撞率对比）：Baseline vs GCGL。上面板 `avg_speed`（平均速度 m/s），下面板 `collision_free_rate`（无碰撞率 CFR）。
- **image7.png**（图4.5 训练过程目标距离对比）：Baseline vs GCGL。`goal_distance_final`（终端距离，实线）与 `goal_distance_best`（最优距离，虚线）。Y 轴=目标距离(m)。

## B. 柱状图（数值已核验，直接采用；SR/CFR 一律为**离线**评估）

- **image11.png**（图4.6 五维度实验稳定AR对比）：5 子面板（损失函数/传感器/架构/训练策略/CMA-ES），各画稳定AR。数据=各实验 metrics.csv 末10步 `ar` 均值，等于下列表4.5 值：
  - 损失函数：Baseline-MSE 0.78，VelDecomp 0.67，VelAdaptive 0.70
  - 传感器：Depth(基线) 0.78，LiDAR 0.60，Fusion 0.58
  - 架构：CNN-GRU(基线) 0.78，CBAM 0.67，Lightweight 0.44
  - 训练策略：Baseline 0.78，GoalLoss 0.70，GradClip 0.83，GCGL 0.91
  - CMA-ES：Baseline 0.78，Decay 0.67，Guide 0.63，Meta 0.30
  （每面板基线柱用统一灰/蓝；GCGL 柱高亮红色描边。）

- **image13.png**（图4.7 消融实验对比）：4 方法 × 3 指标分组柱：稳定AR / **离线SR(%)** / 保留率(%)。SR 加 ±std 误差棒。GCGL 高亮。
  | 方法 | 稳定AR | 离线SR | 保留率 |
  |---|---|---|---|
  | Baseline-MSE | 0.78 | 74.4±4.6 | 60.0 |
  | GoalLoss-Only | 0.70 | 77.5±4.1 | 55.1 |
  | GradClip-Only | 0.83 | 76.9±5.2 | 63.7 |
  | **GCGL** | **0.91** | **83.1±4.7** | **69.1** |
  （为统一 y 轴 0–1：SR/保留率以百分比/100 绘制，柱顶标注原始 % 文本。）

- **image15.png**（图4.9 梯度裁剪阈值灵敏度实验）：4 阈值 × 3 指标分组柱：稳定AR / **离线SR(%)** / **离线CFR(%)**。SR、CFR 加 ±std。c=1.0 高亮。
  | c | 稳定AR | 离线SR | 离线CFR |
  |---|---|---|---|
  | 0.5 | 0.90 | 81.9±3.4 | 90.0±4.1 |
  | **1.0(默认)** | **0.91** | **83.1±4.7** | 88.1±3.4 |
  | 2.0 | 0.88 | 79.4±3.6 | 88.1±1.4 |
  | 5.0 | 0.75 | 75.6±4.6 | 86.9±2.6 |

- **image17.png**（图4.10 消融实验失败模式分布）：4 方法 × 4 类别（成功/碰撞/停滞/超时）分组柱，单位 %。N=160/方法。
  | 方法 | 成功 | 碰撞 | 停滞 | 超时 |
  |---|---|---|---|---|
  | Baseline-MSE | 74.4 | 11.9 | 5.6 | 8.1 |
  | GoalLoss-Only | 77.5 | 13.8 | 3.8 | 5.0 |
  | GradClip-Only | 76.9 | 11.2 | 4.4 | 7.5 |
  | **GCGL** | **83.1** | 11.9 | **2.5** | **2.5** |

## C. 分布图（保留小提琴+箱线形态，仅风格重绘）
- **image18.png**（图4.11 失败episode最佳进度分布图）：数据源=`viz_results/thesis_eval/<exp>/seed<N>/episode_*_log.csv`，逐 episode 取"最佳进度 = (初始距离-全程最近距离)/初始距离"。参考脚本 `testscript/analyze_progress_distribution.py` 的口径。
  - 左：全部失败 episode（N=141）最佳进度的小提琴+箱线图，标注停滞阈值线 20%、`<20%: 27 (19.1%)`、`>90%: 76 (53.9%)`。
  - 右：4 方案（Baseline n=41 / GoalLoss n=36 / GCGL n=27 / GradClip n=37）分组箱线对比。
  - 注：必须复现 N=141、19.1%、53.9% 这三项（若重算不符，停止报告，不得强改）。

## D. 已有新图（风格协调）
- **grad_norm.png**（图4.8 梯度范数监控对比）：已存在（脚本 `testscript/plot_grad_norm_figure.py`，数据 `exp_gradnorm_{baseline,gcgl}/metrics.csv`）。按上面统一配色/字号重新导出一版，保持其数据与统计不变（峰度24.7→15.1、最大34.5→26.8、>20:10→3）。

## 不重绘
- `image1.png`/`image2.png`（校徽/封面 logo）、`image3.png`（CNN 架构示意图，非数据图）：保持不动。
