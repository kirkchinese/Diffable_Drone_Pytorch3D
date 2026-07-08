# DiffDrone-PyTorch3D 能力审计 v1

> 上级研究员审计 · 2026-07-01 · 分支 `chore/track-quadruped-go2-research`
> 目标：判定该可微无人机仿真/训练平台距「可商用级」的真实差距。**结论以代码/测试/结果文件为准，不以 README 自述为准。**

## 0. 证据基准与环境

- **何谓证据**：`✅` = 有 assert 测试**或**已落地实验结果文件支撑；`🟡` = 代码实现且可配置，但仅 smoke/print 或无隔离验证；`🟠` = 缺失或仅雏形。
- **环境**：conda env `pytorch`（base 当前激活）；torch 2.4.1 / pytorch3d 0.7.8；GPU `cuda:0`=RTX 3060 Laptop **6 GB**、`cuda:1`=RTX 3080 **10 GB**（均空闲）。注意 nvidia-smi 索引 0=3060/1=3080。
- **测试现状**：`testscript/` 14 个 `test_*.py`，约 161 处 assert；**无 pytest 套件/conftest/CI**；其中 4 个为 0-assert 的可视/print「测试」（`test_coordinate_system.py`、`test_ground_affinity_fix.py`、`test_ground_ratio_fix.py`、`test_visualize_all_features.py`）。
- **依赖缺口**：`requirements.txt` **无任何 RL 库**（无 stable-baselines3 / gym / cleanrl）→ 目标要求的 SAC 对照当前**被依赖阻塞**；亦无经典规划库（A*/RRT/MPC）。
- **实测验证（2026-07-01，env `pytorch`，`PYTHONPATH=repo_root`，13/14 test 实跑，xbox 跳过）**：**9 PASS / 4 FAIL**。
  - PASS(9)：`test_lidar_sensor`、`test_navigation_utils`(21)、`test_integration`(44, renderer+6 动态障碍模式)、`test_coordinate_system`、`test_coordinate_system_consistency`、`test_drone_collision_refactor`(单机碰撞)、`test_drone_decimate`、`test_ground_affinity_fix`(13/13)、`test_random_heading`。
  - FAIL — 🧹 **test-rot（非代码 bug，但使套件不可一键绿/无法 CI）**：
    - `test_audit_fixes` 9/10——子测 `train_entrypoints_unified` 找不到 `train_adaptive.py`（入口已改名/移除，测试未更）。
    - `test_ground_ratio_fix`——依赖缺失 ckpt `checkpoints/single_run_20260322/checkpoint_final.pth`（属 eval-on-ckpt，非单测）。
  - FAIL — 🟠 **多机碰撞：组件已实现但训练未接通 + 测试 stale（非数值 bug，已三角定位 2026-07-01）**：
    - `test_inter_drone`：调 `env._knn_query()` 期望机间距 0.2，但 `_knn_query`(drone_env.py:953) **按设计只算静态障碍**（knn vs obstacle_pcd），机间距在**专用方法** `inter_drone_distances()`(drone_env.py:1081)/`inter_drone_vec_subdivided()`(:1146)，且 `n_drones_per_group>1` 才生效；测试既调错方法又没设 `n_drones_per_group=2` → **API 漂移后未更新的 stale 测试**。
    - `test_inter_drone_collision_loss`：断言 `metrics['loss_collide']`（静态碰撞，两次相同），但机间惩罚写在 `metrics['loss_drone_collide']`(loss.py:326)；**断错 key 的 stale 测试**，功能本身有算。
    - **真问题**：`train.py` 唯一调用点(line 1025)硬编码 `inter_drone_dist_history=None`、`coef_drone_collide` 默认 0 → **机间碰撞惩罚训练中从不激活**（能力在 env+loss 有件、训练未接线，不可训）。单机碰撞 PASS。
  - **test 调用契约不一致**：部分自举 `sys.path`，部分假设 root 在 path；无统一 runner/conftest → 直接从根 `python testscript/test_x.py` 半数 import 失败。归 P2，WP0 顺手修。
  - 训练/评估全量**尚未实跑**；下方非 ✅-实测 的判断仍需后续确认。

---

## 1. 能力覆盖矩阵（7 语义链）

### 链1 · 场景与随机化域
| 子能力 | 代码位置 | 证据 | 桶 |
|---|---|---|---|
| 随机障碍/簇生/接地分布 | `scene_generator.py`(52K), cfg `random_scene/ground_ratio` | `test_drone_decimate`(3), `test_audit_fixes::scene_generator_generate_smoke` | 🟡 |
| 安全出生点 | cfg `safe_spawn/safe_clearance/force_cross_map/spawn_z_max` | `test_random_heading`(8) 间接 | 🟡 |
| 动态障碍物 | `drone_renderer_dynamic.py`, cfg `enable_dynamic_obstacles` | `test_audit_fixes`(dynamic_obstacle_accepts_list / randomize_torch_only) **仅 smoke** | 🟡 |
| 相机外参/噪声/FOV | cfg `hfov/cam_angle/cam_rand_rpy/cam_rand_xy/depth_max` | 配置面齐全，无隔离验证 | 🟡 |
| 无人机尺寸/安全半径随机化 | cfg `margin_min/max`（碰撞 margin） | 尺寸随机化是否存在**待确认**（margin≠body size） | 🟠 |

### 链2 · 动力学
| 子能力 | 代码位置 | 证据 | 桶 |
|---|---|---|---|
| Verlet 积分 | `drone_dynamics.py` | `test_gradients.py`：前向闭式 + gradcheck 通过 | ✅ |
| 推力/姿态解算 | `drone_dynamics.py` | `test_gradients.py`：gradcheck 通过 + 输出 R 正交(det=1) | ✅ |
| GDecay 梯度衰减 | 5 文件, cfg `grad_decay 0.4` | 实验 `exp_gradnorm_gcgl` / `viz_results/GDCAY` 支撑 | 🟡→✅ |
| 空气阻力 / 控制延迟 / 风扰 / Airmode | `drone_dynamics.py`,`drone_env.py` | **2026-07-01** `test_dynamics_axes.py`(5)：各轴闭式隔离验证（延迟一阶低通/线性+二次阻力/风扰相对速度/airmode 沿推力）；顺带发现 airmode acos-clamp 有 ~0.01 小地板(已记) | ✅ |
| 多机交互与碰撞 | env `inter_drone_distances`/`_vec_subdivided`(1081/1146)、loss `loss_drone_collide`(326) | 组件实现且结构合理，但 **train.py 硬编码 `inter_drone_dist_history=None`(1025)+coef 默认0→训练never激活**；2 个测试均 stale(调错方法/断错key)；视觉互见(eye_mask)已接 render | 🟠 |

### 链3 · 渲染与传感器
| 子能力 | 代码位置 | 证据 | 桶 |
|---|---|---|---|
| PyTorch3D 深度/RGB 渲染 | `drone_renderer.py`(35K) | `test_integration`(44, 含 renderer) + 每-episode depth/rgb mp4 | ✅ |
| 相机坐标系 | `drone_renderer.py` | `test_coordinate_system_consistency`(4)；`test_coordinate_system`(**0 assert**) | 🟡 |
| 动态网格合成 | `drone_renderer_dynamic.py` | 仅 smoke | 🟡 |
| LiDAR | `lidar_sensor.py` | `test_lidar_sensor`(22) | ✅ |
| 深度+LiDAR 融合 | `model.py` fusion 头 | 架构存在；`exp09_sensor_fusion` ckpt；无融合专项 eval | 🟡 |
| **渲染梯度路径** | renderer 全链 | `test_render_gradients.py`：p_ros/T_view 梯度连通有限非零 + 平滑区方向FD同号(rel 13.7%)；硬光栅(blur=0)故严格逐像素gradcheck不适用(已记) | ✅ |

### 链4 · 策略网络
| 子能力 | 代码位置 | 证据 | 桶 |
|---|---|---|---|
| 10 种架构(cnn+gru/bigger/attention/multiscale/residual/lightweight/lidar/fusion…) | `model.py`(39K) | **2026-07-01**：7 架构同场景对照表 `viz_results/thesis_eval/`(3seed)：bigger+clip+goal 85.4% > fusion/attention/lidar ~72–75% > lightweight 60.4%（P1-B 首块）；仍仅 lidar 有单测、multiscale/residual 未 eval | 🟡→部分✅ |
| 统一推理接口 | `model.py` | `test_audit_fixes::train_entrypoints_unified` 间接 | 🟡 |

### 链5 · 梯度反传
| 子能力 | 代码位置 | 证据 | 桶 |
|---|---|---|---|
| BPTT / 截断 | `train.py`(detach) | 无单测 | 🟡 |
| 梯度衰减 GDecay | 见链2 | 实验支撑 | 🟡→✅ |
| 梯度裁剪 | `train.py` `clip_grad` | **专项消融** `exp21/22`+`exp_clip_0p5/2p0/5p0` | ✅ |
| **有限差分/解析梯度一致性** | `testscript/test_gradients.py`(7)+`test_render_gradients.py`(3) | ✅ **2026-07-01 补齐**：dynamics float64 gradcheck 全过(Verlet+drag+airmode+姿态)；渲染硬光栅故做连通+平滑区方向FD(同号,rel-err 13.7%) | ✅ |
| 长视野梯度稳定性 | `testscript/analyze_grad_norm.py` | `exp_gradnorm_baseline/gcgl` 分析，无形式化判据 | 🟡 |

### 链6 · 损失函数
| 子能力 | 代码位置 | 证据 | 桶 |
|---|---|---|---|
| 速度跟踪/预测速度/碰撞/障碍回避/d_acc/jerk/横向/接地高度 (8 项) | `loss.py`(17K), cfg `coef_*` 全套 | `test_drone_collision_refactor`(9)+`test_inter_drone_collision_loss`(2)；`diagnose_loss_*.py` 分析；`test_ground_affinity_fix`(**0 assert**) | 🟡 |
| 多机碰撞损失 | `loss.py` | 轻量测试 | 🟡 |

### 链7 · 实验评价
| 子能力 | 代码位置 | 证据 | 桶 |
|---|---|---|---|
| SR/RR/CFR/速度/距离/进度/碰撞率 | `visualize_eval.py`→`metrics.json`；聚合器 `load_metrics`(JSON 优先) | ✅ **2026-07-01**：去 stdout-regex（WP0.2）；canonical 表 `viz_results/thesis_eval/summary_aggregated.{csv,json}` 落盘=exp21 SR **85.4±1.8%**(3seed×16ep, ≈README 83.13%)>exp22 76.0>exp01 75.0，复现头条 | ✅ |
| 消融实验 | `checkpoints/thesis/exp01–22` | 损失分解/传感器/模型/裁剪/gradnorm 多组 ckpt | ✅ |
| 多种子统计 | `multi_seed_eval.py` | 脚本在；是否全 exp 跑齐**存疑**（log 空） | 🟡 |
| 显存/速度 profile | `profile_training.py`/`profile_spawn.py` | 脚本在，无基准表 | 🟡 |
| 训练稳定性 | `training_monitor.py`, `logs/*/curves` | 曲线在，无判据 | 🟡 |

---

## 2. 与基线的差距

- **vs DiffPhysDrone（本源）**：本项目是其纯 PyTorch3D 复现+扩展，但**无 head-to-head 对齐实验**（同场景→同 SR 的 parity 证据未落盘）。README 称对齐，证据缺。
- **vs 传统模拟器**（Flightmare / AirSim / gym-pybullet-drones / Isaac）：**零 baseline adapter，零同场景对照**。
- **vs 经典规划/控制**（A*/RRT*/MPC）：无对照实现。
- **vs RL（SAC，目标点名）**：**被依赖阻塞**——无 RL 库、无 gym 环境包装。须先实现 `DroneEnv→gym.Env` 适配 + 引入 SB3/cleanrl，再谈同场景对照。
- **vs 商业平台**（Isaac Sim/Gazebo）：无对参考引擎的物理校核、无 sim2real、无传感器噪声模型标定。

---

## 3. 关键缺口（按优先级）

- **P0-A 梯度正确性**：平台卖点是「端到端可微」，却**无渲染/动力学梯度的有限差分 vs 解析一致性测试**。这是可商用的第一块短板（Go2 分支已有范式可移植）。
- **P0-B 头条数字可复现**：SR 83.13% 等无 committed 的 canonical 结果表（仅存于 `final_comparison.pdf`/论文）；聚合靠 stdout-regex（脆弱）；**`train.py` 无 seed** → 训练不可 bit 复现。
- **P1-A 横向对照**：至少 1 个传统/规划/RL baseline 的同场景对照（SAC 需先解依赖）。
- **P1-B 能力隔离验证**：drag/delay/wind/airmode 各轴、9/10 个未单独 eval 的策略架构。
- **P1-C 多机碰撞未接通（已 triage）**：非数值 bug——env+loss 有件但 `train.py:1025` 硬编码 `inter_drone_dist_history=None`、coef 默认 0，训练中惩罚从不激活；2 个测试 stale（调错方法/断错 key）。修复 = ①改两测试打真 API（`n_drones_per_group=2`+`inter_drone_distances()`+断 `loss_drone_collide`）；②（决策项，非纯修 bug）当 `n_drones_per_group>1` 时把 `inter_drone_dist_history` 接进 train.py 使其可训。
- **P2 工程化**：pytest 化 + 统一 runner/conftest（修 test 调用契约 + 2 个 test-rot：`train_adaptive.py` 引用、缺失 ckpt 依赖）、CI、实验注册表、结果表标准路径。

---

## 4. 「可商用级」验收标准（可勾选 DoD）

**可复现**：① `train.py` 接受 `--seed` 并 `manual_seed` 全 RNG；② 每实验 = 配置文件 + seed + ckpt + 一条评估命令；③ 头条指标由**单条命令**重生成到固定路径的 CSV+JSON（非 stdout 抓取）；④ 关键实验 ≥3 seed 报均值±std。

**可验证**：① pytest 套件，关键模块行覆盖有基线；② 坐标系测试有 assert（补齐 `test_coordinate_system.py`）；③ **梯度一致性测试**（renderer+dynamics，finite-diff vs autograd，相对误差阈值）；④ 渲染非空/值域测试；⑤ 性能基准表（it/s、显存 @batch）。

**可对比**：① `BaselineAdapter` 抽象 + ≥1 落地实现（先经典规划或 SAC，依本机可行性）；② 同场景或等价场景下与本平台同表对照（SR/RR/CFR/速度）；③ 若 SAC 因依赖不可装→写明阻塞原因，先交付 gym 适配接口 + 实验协议。

**可维护**：① 清晰 API + 配置管理（现 `@configs/*.args` 可用，补 schema/校验）；② 实验注册表（exp→config+seed+ckpt+结果映射）；③ 失败诊断（现 `diagnose_*`/`analyze_*` 收编为统一 CLI）；④ 最小文档。

**可扩展**：场景/动力学/渲染/传感器/损失/策略/baseline 均为可插拔模块（策略已有统一接口，baseline adapter 待建）。

---

## 5. 建议工作包顺序（交 Codex 执行，逐包先 smoke 后全量）

1. **WP0 复现地基**：`train.py --seed` + 评估指标函数化（去 stdout-regex）+ canonical 结果表路径。*（小改，解锁一切对照）*
2. **WP1 梯度一致性测试**：移植 Go2 分支 gradcheck 范式到 renderer/dynamics。*（验证核心卖点）*
3. **WP2 DiffPhysDrone parity**：同场景对齐实验，落盘 parity 表。
4. **WP3 baseline adapter + 首个对照**：先经典规划（无新依赖）或 gym+SAC（需解依赖）。
5. **WP4 能力隔离 + pytest 化 + CI**。

> 约束：尊重现有风格、不删用户结果/ckpt/论文/未追踪文件；大实验前确认显存（3080 仅 10 GB）；成本高先 smoke。
