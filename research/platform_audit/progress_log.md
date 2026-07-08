# 自主执行进展日志（夜间运行）

> 用户授权「到早晨 8 点前自主决策」，Codex（codex-plugin-cc）作下级实现、我做规划+GPU 验证。
> 起点：审计 v1 完成（[capability_audit.md](capability_audit.md)）。本日志记录每个工作包的动作与验证证据。

## 关键运行约束（实测得出）
- **Codex 沙箱无 CUDA**（`torch.cuda.is_available()==False`，nvidia-smi 不可达）→ Codex 只能做代码编辑 + CPU 逻辑/纯 python 验证；**所有 GPU 依赖代码的验证由我（有 GPU）跑**。
- env `pytorch`；GPU cuda:0=3060(6G)/cuda:1=3080(10G)；测试须 `PYTHONPATH=repo_root`（run_all.py 已自动处理，默认 `--gpu 1`）。
- 不提交（commit 留给用户决定）；不删用户文件。

---

## 2026-07-01 夜

### ✅ WP0.1 train.py 可复现 seed（Codex 实现 / 我 GPU 验证）
- train.py 加 `--seed`，main() 中 seed random/numpy/torch/cuda；**关键正确性**：`SceneGenerator` 自带私有 `torch.Generator`，故 seed 须线程进 `SceneGenerator(seed=...)`（Codex 发现并处理，全局 manual_seed 不够）。
- 新增 `testscript/test_repro_seed.py`：同 seed 场景/出生点 bitwise 一致、异 seed 发散。GPU 验证 **PASS**。
- 诚实边界：PyTorch3D 光栅器 GPU 路径不保证完整 bitwise determinism（已写进 --help）。

### ✅ WP0.3 一键 test runner + 修 test-rot（Codex / 我 GPU 验证）
- 新增 `testscript/run_all.py`：subprocess 跑全部 `test_*.py`（PYTHONPATH 注入、`--gpu` 自动路由、SKIP 识别、CI 退出码）。ponytail：subprocess 而非 pytest（这些是带自计数器/device 参数的脚本式测试，pytest 收集不干净；完整 pytest 化是 WP4）。
- 修 2 个 test-rot：`test_audit_fixes` 的 `train_adaptive.py` 失效断言→现行入口；`test_ground_ratio_fix` 缺 ckpt→SKIP。

### ✅ P1-C 多机碰撞（我三角定位+修，read-only 调查 + GPU 验证）
- **非数值 bug**（详见 audit §0/§3）：组件齐、`train.py:1025` 未接线、coef 默认 0、2 测试 stale。
- 修 `test_inter_drone`（打真 API `inter_drone_distances()`+`n_drones_per_group=2`，期望值据代码 = 0.5−0.3−0.3=**−0.1**，旧测试错减一个 margin 写 0.2）；修 `test_inter_drone_collision_loss`（断 `loss_drone_collide` 真 key，非 `loss_collide`）；`test_visualize_all_features` 缺 ckpt→SKIP。
- **遗留决策项（后置，非 bug）**：`n_drones_per_group>1` 时把 `inter_drone_dist_history` 接进 train.py 使机间碰撞可训——属功能补全、改训练行为，留待专门一轮 + 多机训练 smoke。

### 套件状态（GPU，run_all.py）
`13 PASS, 2 SKIP, 0 FAIL (15)` —— 一键全绿（2 SKIP 均为缺 ckpt 的可视化/eval 测试）。

### ✅ WP1 dynamics 梯度一致性（Codex 起草 / 我补齐+GPU 验证）
- `testscript/test_gradients.py`（7 检查，float64 CPU）：`simulate_position_step` gradcheck 全过（Verlet+线性/二次/诱导阻力+airmode+全 6 输入）；`solve_attitude` gradcheck + 输出 R 正交(det=1)；`g_decay` 前向恒等/反向缩放（标量&逐样本广播）+ 正确接进步（p/v grad 被 `grad_decay**dt` 缩放、动作 grad 不受影响）；Verlet 前向闭式。→ **7 PASS/0 FAIL**。
- Codex 交付了完整实现（不是桩），质量高（含逐输入隔离诊断）；我补了前向闭式 + R 正交 + 诱导/二次阻力 gradcheck 两配置。
- 坑：`g_decay` 故意非保守（前向恒等、反向×param），不能直接 gradcheck → 物理 gradcheck 须 `grad_decay=1.0`（令 GDecay 成梯度恒等），再单独验缩放。

### ✅ WP0.2 评估去 stdout-regex（Codex 实现 / 我 review+GPU 验证）
- `visualize_eval.py` 已写 `{output_dir}/metrics.json`（16 key，你此前工作树改动）；Codex 给两聚合器加 `load_metrics()`：JSON 优先、regex 回退（兼容旧 DONE run），替换全部调用点。
- `testscript/test_eval_aggregator.py` 自检（我补，Codex stall 前未写）：JSON 优先 + regex 回退双分支 **4 PASS**。
- 真实短 eval（exp21 ckpt, 2 ep）实测产出 metrics.json 全 16 key → 全链闭合；**顺带 smoke 了 eval 管线**（审计原记"尚未实跑"）。
- Codex 本包 stall（stream watchdog）在编辑落盘之后，代码实为完成。

### ✅ WP1b 渲染梯度路径（我，GPU）
- `testscript/test_render_gradients.py`（3 检查 @ cuda:1）：深度→`p_ros`/`T_view` 梯度连通/有限/非零；平滑区方向 FD 与解析**同号**、rel-err 13.7%。
- **诚实边界**：渲染器硬光栅（`drone_renderer.py:203` `blur_radius=0`）→ 深度在轮廓不连续 → 严格逐像素 gradcheck 对硬光栅本就不适用（13.7% 差额即来自轮廓像素）。故做连通性 + 平滑区方向 FD，而非强 gradcheck。
- 探针实测 pose `(-6,-6,1)` ROS 见 100% 几何（深度 0.31–31m）作测试锚点。

### 套件状态（GPU，run_all.py，含 3 个新测试文件）
**`16 PASS, 2 SKIP, 0 FAIL (18)`** —— 一键全绿。**P0-A 梯度正确性缺口完全闭合**（dynamics+rendering）。

### ✅ P0-B canonical 结果表 + README 头条复现（我，GPU）
- 用修好的 `multi_seed_eval.py`（读 metrics.json）跑 3 seed×16 ep，落盘 `viz_results/thesis_eval/summary_aggregated.{csv,json}`——**首个 committed 机器可读 canonical 结果表**（此前头条只在 PDF/论文）。
- **复现 README 头条**：exp21_grad_clip_goal SR **85.4±1.8%** ≈ README 83.13%（种子/回合差异内一致）；也验证了 WP0.2 的 load_metrics 在真实规模（cache+fresh 混合）读数正确。

### ✅ P1-B 架构横向对比表（我，GPU；可对比支柱首个交付）
项目自夸的 10 架构里 7 个的同场景 SR/RR/CFR 对照（3 seed×16 ep，按 SR 排）：

| 实验 | SR% | RR% | CFR% | prog% |
|---|---|---|---|---|
| exp21_grad_clip_goal | **85.4±1.8** | 94.8 | 89.6 | 96.6 |
| exp22_grad_clip_only | 76.0±6.5 | 86.5 | 89.6 | 91.1 |
| exp01_baseline_mse | 75.0±6.2 | 83.3 | 89.6 | 88.8 |
| exp09_sensor_fusion | 75.0±8.3 | 79.2 | **91.7** | 90.4 |
| exp10_model_attention | 72.9±9.6 | 84.4 | 84.4 | 93.4 |
| exp08_sensor_lidar | 71.9±8.3 | 82.3 | 85.4 | 92.6 |
| exp11_model_lightweight | 60.4±4.8 | 68.8 | 77.1 | 90.1 |

结论：grad_clip+goal(exp21) 最优；lightweight 用 ~25pt SR 换轻量；fusion CFR 最高(最避碰)但 SR 非最优。**注意**：`multi_seed_eval` 每次按 `--experiments` 覆盖写 summary_aggregated（不累积）——须一次传全部实验才得完整表。

**已扩到全 19 exp**（57 评测点，`summary_aggregated.{csv,json}` 现为完整消融表）。关键读出：
- **clip 敏感性**：clip@0.5 82.3% > @2.0 80.2% > @5.0 75.0%——越紧越好（单调）。
- exp05_cmaes_guide：RR **99%** 但 CFR 78%——到达强、避碰弱（激进）。
- ⚠️ **exp07_cmaes_lossnet：SR 1.0% / RR 3.1%**——**已查明根因**：权重健康(无 NaN、量级正常)、`model_type=bigger`/`sensor=depth` 与 eval 默认**一致**(非评测错配)；但 `best_ar.pth` 停在 **iteration 253、ar_ema 仅 0.044**（~4% 抵达）→ 该学习损失网络 CMA-ES 变体**训练就没学会导航**（真实失败/欠训，非 checkpoint 坏也非评测 artifact）。合法负结果。
- exp02_loss_decomposed CFR **93.8%**（最避碰）但 SR 中游。

### 🔎 train.py 决定性实测（我，GPU）
- 同 seed(777) 两次短训（6 iter）对比 `metrics.csv`：**step-1 loss 完全一致**、场景/init 一致（test_repro_seed 已证 bitwise）；随迭代 GPU 渲染/反传的非确定归约累积出 **~1e-6 相对漂移**（step-6 loss 第 6 位后不同、grad_norm ~1e-13 起）。
- 结论：可复现级别 = **seed 控制 + 近似确定**，非 GPU bitwise（正合 `--seed --help` 声明）。诚实边界，非 bug。

### ✅ WP4-核心 run_all.py --cpu-only（我）
- `run_all.py --cpu-only`：隐藏 CUDA + 把「No CUDA GPUs」类失败**自动记为 SKIP**（无硬编码 allowlist，免维护）→ 12 PASS/6 SKIP/**0 FAIL** 退出 0，供无 GPU 环境/CI/Codex 沙箱自验 CPU 子集。（GitHub Actions YAML 从简未加：你本地跑为主，flag 是可复用原语。）

### ✅ 动力学各轴隔离测试（我）
- `test_dynamics_axes.py`(5)：R=I 闭式钉死 执行器延迟（一阶低通+delay 越大越快）/线性阻力(a=act−k1·v，反向)/二次阻力(−k2·v|v|)/风扰(作用于 v−v_wind，v=v_wind 零阻力)/airmode(仅推力变化诱导、沿推力方向)。闭合审计「各轴无隔离验证」🟡→✅。
- 🔎 **发现**：airmode 的 `acos` 输入 clamp 到 1−1e-6 → 零角速仍有 ~0.01 m/s² 小地板（数值边界小瑕疵，非 bug，已在测试注释记录）。

### 套件 & 里程碑
- 套件（run_all.py，19 测试文件，5 个本夜新增）：**17 PASS / 2 SKIP / 0 FAIL**；`--cpu-only`：12 PASS/6 SKIP/0 FAIL。
- **P0-A（梯度正确性）完全闭合**（dynamics gradcheck + rendering 连通/平滑区 FD）。
- **P0-B（可复现头条）闭合**（seed + metrics.json + 全 19-exp canonical 表复现 README）。
- **P1-B（架构/消融对照）交付**（19-exp 完整表 + drag/delay/wind/airmode 隔离）。

### 下一步（建议优先级）
1. **WP3 真 baseline**（可对比支柱补全）：已把接口摸清——策略 `forward(x_depth, v_state[dim_obs=10], hx)→(act[6], img_feat, hx)`，rollout 层 `policy.infer→(act_cmd[3], v_pred[3], target_v, h)`，`env.step(act_cmd=…)`。实现"走向目标+深度排斥"经典控制器作 drop-in（加进 visualize_eval 的 `_model_map`、baseline 免 checkpoint），同 harness 出 SR/RR/CFR。**需 GPU 验证导航合理性**（故我做，非 Codex）。风险=obs/act 帧与量纲，须先小场景 smoke 校准。
2. **WP4** CI 骨架：`run_all.py --cpu-only`（CPU 安全子集）+ GitHub Actions 冒烟（GPU 测试本地留）。
3. **WP2** DiffPhysDrone parity（需参考实现，本机未装→先记协议）。
4. **P1-C 决策项**：多机时把 `inter_drone_dist_history` 接进 `train.py:1025` 使可训。
5. 扩 canonical 表到全 19 exp（cache 多、快）。
