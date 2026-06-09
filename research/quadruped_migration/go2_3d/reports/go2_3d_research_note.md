# 3D 迁移阶段研究笔记：Unitree Go2 可微数字孪生

> 版本：3D 阶段 · E3D-0 … E3D-3（2026-06-09）· 配套代码：[`models/`](../models/)、[`dynamics/`](../dynamics/)、[`scripts/`](../scripts/)、[`parameters/`](../parameters/)、[`figures/`](../figures/)
> 复现：`conda activate pytorch`；E3D-0 `python scripts/{convert_dae_to_obj,emit_model_summary,render_standing_pose}.py`；E3D-1 `python scripts/e3d1_floating_base_checks.py`；E3D-2 `python scripts/e3d2_contact_checks.py --device cuda:0`；E3D-3 `python scripts/e3d3_static_audit.py` + `python scripts/e3d3_standing_train.py --device cuda:0`
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
| `p` 位置 | **WORLD** | **COM 位置**（非 base 原点）；base 原点 `= p − R·c_body`（平动/转动仅在 COM 处解耦）|
| `q` 姿态 | Body→World | **单位四元数 (w,x,y,z) 标量在前**；`R=quaternion_to_matrix(q)`，`v_world=R·v_body`（与 FK / 无人机项目列向量约定一致，已数值核对）|
| `v` 线速度 | **WORLD** | |
| `w` 角速度 | **BODY** | `omega_body`，使 `I_body` 在体系常量 |
| `f_world` 外力 | **WORLD** | 不含重力 |
| `tau_body` 外力矩 | **BODY** | 关于 COM |

**接触力矩换算（写死，不靠脑补——frame 混淆会悄悄翻转力矩）**〔推导/代码〕：世界系接触力 `f_i_world` 作用于世界系接触点 `p_i_world`，COM 在 `p_com_world = p`，`R=quaternion_to_matrix(q)`（Body→World）：
```
r_world  = p_i_world − p_com_world
f_world  += f_i_world
tau_body += Rᵀ (r_world × f_i_world)        # 关于 COM 的力矩，旋到体系
```
等价体系形式（恒等式 `R(a×b)=(Ra)×(Rb)`，R∈SO(3)）：`tau_body += (Rᵀr_world)×(Rᵀf_i_world) = r_body×f_body`。两种写法已**数值核对一致到 ~1e-7（CPU 与 3080 GPU）**、可微、GPU-clean。统一走 [`contact_to_body_wrench()`](../dynamics/floating_base_srbd.py) 以防误用；body-frame 优化版（`r_body=foot_body−c_body`）留待世界系版确认无误后再做。

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

## 8. E3D-2：3D 足端接触（平滑法向 + 摩擦锥，frame 钉死）〔实验〕

> 配套：[`dynamics/contact_3d.py`](../dynamics/contact_3d.py)、[`scripts/e3d2_contact_checks.py`](../scripts/e3d2_contact_checks.py)、[`notebooks/go2_02_smooth_contact_3d.ipynb`](../notebooks/go2_02_smooth_contact_3d.ipynb)。把 2D 的 E1/F1 接触梯度结论迁到 3D 平地足端。

### 8.1 设计（按项目踩坑清单的优先级）〔推导/代码〕
1. **frame 全部 WORLD**：足端位置/速度入、接触力出，地面法向 `n=+Z`。**绝不**混用 body 足速与 world 法向（会翻转摩擦方向）。world↔body 只在 [`contact_to_body_wrench`](../dynamics/floating_base_srbd.py) 一处发生。
2. **可微近似不硬投影**：`pen=ε·softplus(−gap/ε)`；`f_n=k_n·pen+k_d·gate·srelu(−vn)`（`gate=sigmoid(−gap/ε)`，`srelu(x)=x·sigmoid(x/v_d)` 在 `vn=0` 恰为 0，**无静止伪力**）；`v_t=v_foot−vn·n`；`f_t=−μf_n·tanh(‖v_t‖/v_ε)·v_t/‖v_t‖`，故 **‖f_t‖≤μf_n 结构成立**、梯度处处有界（无 sign/硬 clamp）。
3. 摩擦**对抗**切向速度（耗散）。

### 8.2 六项验证（含 SRBD 落地积分）〔实验〕

![e3d2](../figures/e3d2_contact_checks.png)

| 检验 | 结果 | 对应 2D |
|---|---|---|
| **K1 法向力 vs 穿深** | 平滑梯度连续有界（~k_n=1e4）；硬接触起始处梯度不连续（阶跃）| E1 |
| **K2 摩擦 vs 切向速度** | 平滑库仑梯度**有界且有信息**（峰 544）；硬库仑梯度**恒 0**（无信息）| F1 |
| **K3 摩擦锥** | `max‖f_t‖/(μf_n)=1.0000`，违反率 **0%**（结构成立）| 新 |
| **K4 摩擦方向** | `f_t·v_t≤0` 恒成立，非耗散率 **0%** | frame 踩坑检查 |
| **K6 落地积分** | Go2 SRBD+4 足落地稳定 settle 到 COM z≈**0.243 m**，静止 **Σf_n=m·g（误差 0%）** | 新（接触→SRBD 回路）|

全程可微、3080 GPU 运行（CPU/GPU 数值一致）。

### 8.3 结论〔实验〕
- 2D 的"平滑接触给可用梯度、硬接触梯度无信息"在 3D 足端**复现**（K1/K2）：摩擦的平滑库仑切向梯度有界有信息，硬库仑恒 0——这是 3D 可训练性的同一分水岭。
- 摩擦锥用 **tanh 饱和**实现"可微近似"，`‖f_t‖≤μf_n` 结构成立、零违反，**避免了硬投影**（呼应用户踩坑 #3）。
- K4 是 frame 踩坑的判别检验：world 足速 × world 法向 → 摩擦严格耗散，证明方向没反（呼应踩坑 #2）。
- K6 把 contact_3d 经 §7.1 的 `contact_to_body_wrench` 接入 E3D-1 的 SRBD，落地稳定且静力平衡精确——**接触层与动力学层的接口闭合**。
- 待办：body-frame 优化版接触换算（确认无误后）；接触门控的 softplus 尾巴（离地微小残力）已被 gate 压制但非严格 0，必要时可加更陡门控；E3D-3 将引入策略闭环。

## 9. E3D-3：3D SRBD 原地站立闭环最小可微训练（2D-F3 的 3D 版）〔实验〕

> 配套：[`dynamics/srbd_standing.py`](../dynamics/srbd_standing.py)、[`scripts/e3d3_static_audit.py`](../scripts/e3d3_static_audit.py)、[`scripts/e3d3_standing_train.py`](../scripts/e3d3_standing_train.py)、[`notebooks/go2_03_srbd_standing.ipynb`](../notebooks/go2_03_srbd_standing.ipynb)、`results/e3d3_standing.json`。分两阶段：先静力审计，后闭环训练。

### 9.1 动作空间与任务（明确控制器输出）〔推导/代码〕
- **动作 `a∈R⁴`=四腿竖直伸长**（tanh 限幅 ±0.10 m），足端体系高度=nominal−a；**接触力由接触模型从穿深产生，不直接命令力**——保持接触梯度问题居中、策略贡献可量化（对应 2D-F3 的腿伸长动作）。
- 足端接触速度只取**刚体部分** `v_com+ω×r`（不做位置差分，规避 2D-F4 的 1/dt 摩擦梯度爆炸）。姿态损失用**无奇异** `1−R[2,2]`（体 z 对世界 z），**不用欧拉角**。
- 任务：从随机扰动（roll/pitch ±0.08 rad、高度 rest±0.03）regulate 到**目标高度 z\*=rest+0.04 m**（高于被动平衡，故**需策略**）+ 水平 + 速度收敛。

### 9.2 Stage A 静力审计（站立前必查，对应踩坑 #1/#2/#7）〔实验〕

![audit](../figures/e3d3_static_audit.png)

| 检验 | 结果 |
|---|---|
| 足端命名/符号 | FL(+x,+y)/FR(+x,−y)/RL(−x,+y)/RR(−x,−y) 全部正确 |
| 平衡（**不止竖直力**）| 净力 `[0,0,147.34]=[0,0,mg]`、**净力矩 `[0,0,0]`**、roll/pitch≈0 |
| 静止高度来自几何 | settle 0.2426 vs 解析 `−foot_z_rel_com−mg/(4k_n)`=0.2423（差 0.34mm）|
| 负载分配 | 前 73.0N/后 74.4N，比值 0.981=力臂比 0.1916/0.1952（恰好令净俯仰力矩=0）|
| 落足重分配（命名验证）| pitch 扰动→纯**前后**载荷差（+439N，左右=0）；roll→纯**左右**（−332N，前后=0）——错配则会串扰，这里不串 |

> 关键：`Σf_n=mg` 不够，**净力矩=0** 才是站立。审计同时确认了踩坑 #6——**被动接触能把对称四足站到水平**（故任务用高于被动平衡的目标高度逼出策略贡献）。

### 9.3 Stage B 闭环训练〔实验〕

![train](../figures/e3d3_standing_train.png)

| 配置 | 训练损失 | 梯度范数(中位,裁剪前) | 长 rollout(5×=1.5s) 终态 |
|---|---|---|---|
| **smooth+noGDecay** | 0.020→**0.004** | 5.5e-3 | z 差 **6.5mm**、tilt 0.20° ✓最佳 |
| smooth+GDecay | 0.020→0.006 | 2.4e-2 | z 差 30.5mm、tilt 0.20° |
| smooth+shortH | 0.021→0.011 | 1.2e-2 | z 差 34mm、**tilt 6.45°(漂)** |
| **hard+GDecay** | 0.028→**0.71(发散)** | **2.1e9** | z 差 162mm、**tilt 115°(倒)** |
| passive(无策略) | — | — | z 差 39.7mm、tilt 0.003° |

训练前梯度自检（踩坑 #4）：BPTT 梯度范数 smooth=**0.15** vs hard=**2.6×10¹⁰**。

### 9.4 结论〔实验〕
- **梯度分水岭 3D 复刻 2D-F3/E5**：硬接触浮动基 BPTT 梯度爆到 **10⁹–10¹⁰**（2D ">10⁶" 的放大版）；**smooth 训得动、hard 训不动**，同配方同裁剪——裁剪封幅不纠偏（2D-E4），故 hard 发散倒地。**失败是梯度性的（#9 非稻草人）**。
- **策略必要性（#6）**：被动站到水平但够不到目标高度（差 39.7mm）；smooth+noGDecay 闭到 6.5mm——策略贡献可量化。
- **GDecay 边际（诚实边界）**：平滑接触已梯度有界，GDecay 反略伤跟踪（30.5 vs 6.5mm）——价值域在刚性/长视野，与 2D-F3 附注一致。
- **短视野漂移（#8）**：smooth+shortH 训练损失尚可，5× 长 rollout 暴露姿态慢漂 6.45°（全视野 0.2°）——**只看训练视野内 loss 会被骗，必须长 rollout 验证**。
- **摩擦锥占用（#3）**：max cone=1.0 出现在起始 settle 暂态（刚性足随机身翻正而横扫滑动），稳态 cone→0；暂态饱和未阻碍 standing 收敛——锥饱和弱梯度对**持续蹬地的 locomotion（E3D-4）**更关键。建模注记：足端刚连机身、无落足锚定。

## 10. 下一步（E3D-4 起）

1. **E3D-4**：预定步态 3D 前向速度跟踪（3D 版 F4）——含落足锚定 / 抬腿，正面处理摩擦锥持续占用与切向梯度。
2. body-frame 优化版接触换算；E3D-5 与高保真仿真器状态对齐；E3D-6 残差动力学。
3. 减面版孪生（rendering LOD）以支持批量与短视野 BPTT。
