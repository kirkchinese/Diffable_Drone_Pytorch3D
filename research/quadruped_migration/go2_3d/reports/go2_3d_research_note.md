# 3D 迁移阶段研究笔记：Unitree Go2 可微数字孪生

> 版本：3D 阶段 · E3D-0 + E3D-1（2026-06-09）· 配套代码：[`models/`](../models/)、[`dynamics/`](../dynamics/)、[`scripts/`](../scripts/)、[`parameters/`](../parameters/)、[`figures/`](../figures/)
> 复现：`conda activate pytorch`；E3D-0 `python scripts/{convert_dae_to_obj,emit_model_summary,render_standing_pose}.py`；E3D-1 `python scripts/e3d1_floating_base_checks.py`
> 承接 2D 阶段结论见 [`../research_note.md`](../research_note.md)。证据等级沿用：〔代码〕〔论文〕〔推导〕〔实验〕〔假设/猜想〕。

本阶段把 2D 平面四足建模迁移到 **3D Unitree Go2 数字机体**。与之前不同，本阶段的核心要求是
**整条链路（运动学 + 渲染）全部可微**，并基于 PyTorch / PyTorch3D **自建可微数字孪生**，
而不是接入一个黑箱仿真器。本笔记记录第一步 **E3D-0：模型读取 + 可微数字孪生骨架**。

---

## 0. 速览（TL;DR）

**一句话**：用宇树官方 URDF（`go2_description`）构建了一个**端到端可微**的 Go2 数字孪生——
URDF→运动学树→刚体前向运动学（FK）→逐连杆网格刚性蒙皮（rigid skinning）→PyTorch3D 可微光栅化。
所有几何与物理参数**逐项可追溯到官方 URDF 的固定 commit**；FK 的足端位置与零位解析值逐点吻合、
梯度物理正确；渲染像素对关节角 `q` 的梯度有限且 12 维全非零——**"可微数字孪生"在 3D Go2 上成立**〔实验〕。

| 验收问题（§12-Q1） | 速答 |
|---|---|
| Unitree Go2 数字模型是否已正确加载与参数化？ | **是**。官方 URDF @ `8bd6717` 解析为 29 连杆 / 28 关节 / 12 驱动关节，总质量 **15.019 kg**，参数表见 [`parameters/`](../parameters/)，逐项标注来源〔代码/实验〕 |

---

## 1. E3D-0 目标与交付

| 目标 | 交付物 | 状态 |
|---|---|---|
| 确认官方数字模型可被正确加载、参数可信、坐标系/命名清楚 | URDF 解析器 + 参数摘要 | ✅ |
| 把"下载 obj + PyTorch3D 骨骼绑定"落地为可微孪生 | DAE→OBJ 转换 + FK + 蒙皮 + 渲染 | ✅ |
| 端到端可微性验证 | 像素→`q` 梯度检验 | ✅ |

模块划分（遵循"多小文件"）：
- [`models/go2_urdf.py`](../models/go2_urdf.py)：纯标准库 URDF 解析器（无 urdfpy 依赖）。
- [`models/go2_kinematics.py`](../models/go2_kinematics.py)：批量可微 FK（4×4 齐次变换）。
- [`models/go2_render.py`](../models/go2_render.py)：刚体蒙皮 + PyTorch3D 渲染（`Go2Twin`）。
- [`scripts/convert_dae_to_obj.py`](../scripts/convert_dae_to_obj.py)、[`emit_model_summary.py`](../scripts/emit_model_summary.py)、[`render_standing_pose.py`](../scripts/render_standing_pose.py)。

---

## 2. 模型来源与可追溯性〔代码〕

- 来源：`https://github.com/Unitree-Go2-Robot/go2_description`，分支 `humble`，**固定 commit `8bd6717ff0c7b5ca388c0e10e426dd9ad873ceaf`**。
- 物理参数（质量、惯量、关节限位、连杆偏置）**全部取自官方 URDF**；网格仅用于渲染。
- 关键工程事实：官方可视化网格是 **COLLADA `.dae`**，而 PyTorch3D 仅读 `.obj/.ply`。
  故新增 DAE→OBJ 转换步骤（`trimesh`+`pycollada`，烘焙场景图变换、单网格化）。
  转换后单位经核对为**米**（base 包围盒 0.46×0.19×0.19 m、thigh 0.278 m），缩放 1.0〔实验〕。
- 来源记录：[`assets/go2_description/PROVENANCE.md`](../assets/go2_description/PROVENANCE.md)，并写入参数摘要 JSON 的 `provenance` 字段。

---

## 3. 参数摘要（关键数字）〔代码〕

完整表见 [`parameters/go2_model_summary.md`](../parameters/go2_model_summary.md) / `.json`。

- **base link**：`base_link`（浮动基），质量 6.921 kg，惯量对角 `(0.02448, 0.098077, 0.107)`。
- **总质量 15.019 kg** = base 6.921 + 4×(hip 0.678 + thigh 1.152 + calf 0.154 + foot 0.04) + 2×Head 0.001。
- **腿几何**：hip 安装点 `(±0.1934, ±0.0465, 0)`；hip→thigh 侧向偏置 `0.0955`；**thigh 长 = calf 长 = 0.213 m**。
- **12 驱动关节**（canonical 顺序 `LEGS=(FL,FR,RL,RR) × (hip,thigh,calf)`）：
  - hip 外摆，轴 `[1,0,0]`，限位 ±1.0472 rad，力矩 23.7 N·m；
  - thigh 屈伸，轴 `[0,1,0]`，前腿 `[-1.5708, 3.4907]` / 后腿 `[-0.5236, 4.5379]`；
  - calf 膝，轴 `[0,1,0]`，限位 `[-2.7227, -0.83776]`，力矩 45.43 N·m。
- **足端帧**（SRBD 接触点）：`{FL,FR,RL,RR}_foot`，零位（腿伸直）相对 base 偏置 `(±0.1934, ±0.142, -0.426)`。
- **坐标系**：URDF/ROS 机体系 `+X 前、+Y 左、+Z 上`，重力 `-Z`——与无人机项目的 ENU 约定一致，便于复用。
- ⚠ **顺序陷阱**〔假设/待核对〕：Unitree 底层 SDK 电机序为 `FR,FL,RR,RL`，与 URDF 的 `FL,FR,RL,RR` **不同**；
  后续与 Unitree 工具对齐（E3D-5）必须重映射。

---

## 4. 可微数字孪生构建〔实验〕

### 4.1 设计：刚体骨骼绑定（rigid skinning）〔推导/代码〕

对**刚性**机器人，"骨骼绑定"= 一个连杆即一根刚性骨；用 FK 求每个连杆的世界变换，
再把该连杆**整块网格**刚性变换后用 `join_meshes_as_scene` 合成单一场景 Mesh。
**无需**带顶点混合权重的线性混合蒙皮（LBS，那是给可形变角色的）。

> 关键复用：合成所用的 `join_meshes_as_scene` 正是无人机渲染器
> [`drone_renderer.py`](../../../drone_renderer.py) 合成"静态场景 + 动态机体"的同一 PyTorch3D 习语；
> 相机用 `PerspectiveCameras(in_ndc=False)` + `MeshRasterizer` + `SoftPhongShader`，与原项目一致。
> **PyTorch3D 本身没有内建关节/骨骼动画系统**——FK 由我们自建，PyTorch3D 只提供可微的网格/变换/光栅化原语。

整条 `q → 连杆世界变换 → 顶点 → 像素` 由 torch 可微算子组成，梯度可回传到 `q` 与浮动基位姿。

### 4.2 FK 数值与梯度验证〔实验〕

- **零位一致性**：`foot_positions(q=0)` 给出 `FL=(0.1934, 0.142, -0.426)` 等，与参数摘要的零位偏置**逐点吻合**。
- **梯度物理正确**：`d(FL_foot_z)/d[hip,thigh,calf] |_{q=0} = [0.0955, 0, 0]`。
  - `0.0955` = hip 轴到足端的侧向力臂（绕 X 抬腿，一阶灵敏度 = 侧向偏置）；
  - thigh/calf 为 0，因零位足端正悬于这两个 Y 轴正下方，绕 Y 一阶不改变高度——**与解析预期一致**。

### 4.3 渲染与端到端可微性〔实验〕

![twin](../figures/go2_digital_twin_poses.png)

- 装配正确：灰躯干 / 蓝 hip / 橙 thigh / 绿 calf / 黑 foot，左右腿镜像（FR/RR 用 `*_mirror` 网格 + 翻转 rpy）正确，**机身直立（Z-up 经 DAE→OBJ 保持）**。
- **站立高度核对**：nominal `q=(0, 0.9, -1.8)` 下四足同高 `z=-0.265`，base 距足 **0.265 m**（Go2 站高 ~0.28–0.32 m 量级合理）〔假设：nominal 角为 legged_gym 典型值，未拟合真实站姿〕。
- **端到端可微**：`d(mean pixel)/dq` 范数 `6.5e-3`，有限、**12 维全非零**——确认"可微数字孪生"成立。

---

## 5. 与 2D / 无人机框架的对照

| 维度 | 无人机 | 2D 四足（已验证） | 3D Go2（E3D-0 现状） |
|---|---|---|---|
| 状态 | 质点 + 代数姿态 | 平面 SRBM `[p_x,p_z,θ]` | 浮动基 `[p,q,v,ω]`（待 E3D-1/3 接入动力学） |
| 可微渲染 | 单 obj 场景 | 无（纯动力学实验） | **URDF 多连杆刚性蒙皮 + PyTorch3D**（本阶段新增） |
| 几何来源 | 手写无人机 mesh | 手写简化几何 | **官方 URDF，逐项可追溯** |
| 可微性 | C∞ 动力学 | 平滑接触下良好 | **运动学 + 渲染已确认可微**；接触动力学留待 E3D-2/3 |
| GDecay/坐标系 | ENU, `R` 列向量 | 同 | **沿用同一约定**，迁移成本低 |

**迁移可行性的第一块拼图已就位**：Go2 的几何/惯量/关节结构能被干净加载，且"自建可微孪生"在 3D 多连杆形态下确实可微——
这是后续把 2D 接触/姿态/残差结论搬到 3D 的**前提条件**，现已满足。

---

## 6. 结论分级

**已验证结论〔实验〕**
1. 官方 Go2 URDF 可被正确解析为 29 连杆 / 28 关节 / 12 驱动，总质量 15.019 kg，参数逐项可追溯。
2. 自建 FK 数值正确（零位吻合）、梯度物理正确。
3. DAE→OBJ + 刚体蒙皮 + PyTorch3D 渲染装配正确、单位为米、机身直立。
4. 整条 `q→像素` 链路端到端可微（梯度有限、满秩非零）。

**从 2D 继承〔实验/论文〕**
- 接触平滑性是可训练性分水岭；GDecay 压幅不压偏；残差须放在误差发生处（见 2D 笔记 §9/§10）。本阶段尚未触及接触。

**假设 / 待核对〔假设〕**
- nominal 站姿关节角为典型值，未拟合真实站高；
- SDK 电机序 `FR,FL,RR,RL` 与 URDF 序不同，E3D-5 对齐时须重映射；
- 渲染世界系直接用 URDF 系（相机 up=+Z），与无人机 `transform_pos_ros2pt3d` 的 ROS↔PT3D 映射尚未统一，后续若要复用无人机感知管线需对齐。

**3D 新发现 / 工程经验〔实验〕**
- 官方 `.dae` 与 PyTorch3D 的 `.obj` 不兼容，需 `pycollada` 转换；
- 全分辨率孪生约 **40 万面**，单姿态渲染在 6 GB 显存可行，但批量训练前需减面（decimation）。

**环境校正〔代码〕**
- 实际机器为**单块 RTX 3060 Laptop（6 GB）**，与研究提示词所述 3080+3060 双卡桌面**不符**；`torch.cuda.device_count()==1`，全部用 `cuda:0`。后续涉及显存/并行规划须以此为准。

---

## 7. E3D-1：floating-base 单刚体动力学（坐标系/四元数/积分/可微钉死）〔实验〕

> 配套：[`dynamics/floating_base_srbd.py`](../dynamics/floating_base_srbd.py)、[`models/go2_inertia.py`](../models/go2_inertia.py)、[`scripts/e3d1_floating_base_checks.py`](../scripts/e3d1_floating_base_checks.py)、[`notebooks/go2_01_3d_rigid_body.ipynb`](../notebooks/go2_01_3d_rigid_body.ipynb)。
> 指导原则（用户）：**先不追求复杂，先把坐标系/惯量/四元数积分/可微性钉死**——这一层错了，E3D-2 的接触与摩擦锥会放大错误且难定位。

### 7.1 全局约定（一次定清，全局统一）〔推导/代码〕

状态 `x=(p,q,v,w)`：

| 量 | frame | 说明 |
|---|---|---|
| `p` 位置 | **WORLD** | |
| `q` 姿态 | Body→World | **单位四元数 (w,x,y,z) 标量在前**；`R=quaternion_to_matrix(q)`，`v_world=R·v_body`（与 FK / 无人机项目列向量约定一致，已数值核对）|
| `v` 线速度 | **WORLD** | |
| `w` 角速度 | **BODY** | `omega_body`，使 `I_body` 在体系常量 |
| `f_world` 外力 | **WORLD** | 不含重力 |
| `tau_body` 外力矩 | **BODY** | 关于 COM；接触换算 `f_world+=f_i, tau_body+=r_i×(Rᵀf_i)` 由调用方在调用点显式完成 |

运动方程：`m dv/dt = m g + f_world`（世界系平动）、`I dw/dt = tau_body − w×(I w)`（体系 Euler 方程）、姿态用**体系旋转向量 `w·dt` 的指数映射**积分 `q_{t+1}=normalize(q_t ⊗ exp_quat(w·dt))`（右乘，因 `w` 是体系率）。**绝不用欧拉角积分**。积分器：半隐式（辛）Euler。
SRBD 惯量来自 URDF 复合刚体（parallel-axis，nominal 站姿）：mass 15.019 kg、`I diag≈(0.158, 0.469, 0.525)`、SPD（Ixx 最小=绕长轴）。

### 7.2 五项验证〔实验〕

![e3d1](../figures/e3d1_floating_base_checks.png)

| 检验 | 结果 | 含义 |
|---|---|---|
| **C1 自由落体 vs 解析** | pz 误差 dt 减半→误差减半（比值 2.00，一阶收敛）；速度精确到 7.99e-14 | 坐标系/重力符号正确 |
| **C2 四元数单位范数** | 8000 步 `max‖q‖-1 = 2.2e-16` | 姿态恒为合法单位四元数 |
| **C3 力矩自由守恒（最关键）** | **世界系角动量** L 相对漂移 1.6e-3、转动能 2.8e-3，均**有界** | frame/符号正确（`w×(Iw)` 符号或四元数乘序错则此处发散）|
| **C4 可微 vs 视野** | 梯度范数 50→1600 步 = 0.09→2.02，温和有界 | BPTT 梯度不爆炸 |
| **C5 exp-map vs 欧拉角** | 欧拉角在 pitch→90° 梯度爆到 ~10（**≈200× 于 exp-map**），exp-map 始终 ≤0.05 | **定量复现"欧拉角积分致梯度爆炸"的踩坑，确认 exp-map 是正确选择** |

### 7.3 结论〔实验〕
- floating-base 单刚体动力学的**坐标系、惯量、四元数表示与积分、外力/力矩 frame、可微性**已全部钉死并交叉验证。
- C3 是最有判别力的检验：世界系角动量/能量有界漂移直接证明 frame 与符号正确——这是接触层之前必须先拿到的"干净地基"。
- C5 用 Go2 复合惯量定量复现了用户预警的欧拉角梯度爆炸，并证明 exp-map 指数映射积分免疫该病理。
- 半隐式 Euler 的一阶漂移（C1 的 0.5·a·t·dt、C3 的 ~1e-3）是已知且有界的；若 E3D-3+ 长视野需要更紧守恒，可换 RK2/midpoint（接口不变）。

## 8. 下一步（E3D-2 起）

1. **E3D-2**：把 2D 的平滑接触/摩擦锥扩展到足端 3D 接触（法向平滑罚 + 平滑库仑/摩擦锥 + 接触门控），复刻 2D 的 E1"接触力-梯度"分析；外力/力矩按 §7.1 的 frame 换算接入 SRBD。
2. **E3D-3**：3D SRBD 原地站立（平滑接触 + 目标损失 + GDecay/短视野），对照 smooth/hard。
3. 减面版孪生（rendering LOD）以支持批量与短视野 BPTT。
