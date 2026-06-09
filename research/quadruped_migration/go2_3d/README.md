# go2_3d — Unitree Go2 可微数字孪生（3D 迁移阶段）

把 2D 平面四足建模迁移到 **3D Unitree Go2**。核心要求：运动学 + 渲染**全程可微**，
基于 PyTorch / PyTorch3D 自建数字孪生（**非黑箱仿真器**）。承接 2D 阶段见
[`../research_note.md`](../research_note.md)；本阶段结论见
[`reports/go2_3d_research_note.md`](reports/go2_3d_research_note.md)。

## 目录
```
go2_3d/
├── models/                    # 可复用核心模块
│   ├── go2_urdf.py            #   纯标准库 URDF 解析器（无 urdfpy 依赖）
│   ├── go2_kinematics.py      #   批量可微 FK（4×4 齐次变换）
│   └── go2_render.py          #   刚体骨骼绑定 + PyTorch3D 渲染（Go2Twin）
├── scripts/
│   ├── convert_dae_to_obj.py  #   官方 .dae → .obj（PyTorch3D 只读 obj/ply）
│   ├── emit_model_summary.py  #   生成 parameters/go2_model_summary.{json,md}
│   └── render_standing_pose.py#   E3D-0 渲染 + 站姿核对 + 端到端可微检验
├── notebooks/go2_00_model_inspection.ipynb   # 叙事/展示（已执行，含图）
├── assets/
│   ├── go2_description/        #   官方 URDF + dae（PROVENANCE.md 记录 commit）
│   └── obj/                    #   转换后的 obj（渲染用）
├── parameters/                # go2_model_summary.{json,md}
├── figures/  results/  reports/
```

## 复现（conda 环境 `pytorch`，Python 3.9 / torch 2.4.1 / pytorch3d 0.7.8）
```bash
conda activate pytorch
cd research/quadruped_migration/go2_3d
python scripts/convert_dae_to_obj.py            # 一次性：dae→obj
python scripts/emit_model_summary.py            # 参数摘要
python scripts/render_standing_pose.py --device cuda:0   # 渲染 + 可微检验
```
Notebook 执行需用 **env 的 ipykernel**（见下方"踩坑"）：
```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=go2py39 notebooks/go2_00_model_inspection.ipynb
```

## 关键结果（E3D-0）
- 官方 Go2 URDF @ `8bd6717`：**29 连杆 / 28 关节 / 12 驱动，总质量 15.019 kg**，thigh=calf=0.213 m，参数逐项可追溯。
- 可微 FK：零位足端 `(±0.1934, ±0.142, -0.426)` 与解析值吻合；`d(foot_z)/d(hip)=0.0955`（侧向力臂）物理正确。
- **可微数字孪生成立**：DAE→OBJ + 刚体蒙皮 + PyTorch3D 渲染装配正确、单位米、机身直立；
  像素→`q` 梯度有限、12 维全非零。

## 设计要点
- **刚体骨骼绑定（rigid skinning）**：一个连杆 = 一根刚性骨；FK 给世界变换，整块网格刚性变换后
  `join_meshes_as_scene` 合成——复用无人机渲染器同一习语。**PyTorch3D 无内建骨骼动画**，FK 自建；
  刚性机器人不需要顶点混合蒙皮（LBS）。
- **坐标系**：URDF/ROS 机体系 `+X 前、+Y 左、+Z 上`，与无人机 ENU 约定一致。渲染世界系暂直接用 URDF 系
  （相机 up=+Z）；若复用无人机感知管线需对齐 `transform_pos_ros2pt3d`。
- **canonical 顺序** `q`：`(FL,FR,RL,RR) × (hip,thigh,calf)`。⚠ Unitree SDK 电机序 `FR,FL,RR,RL` 不同，对齐须重映射。

## 踩坑记录
- 官方网格是 COLLADA `.dae`，PyTorch3D 不读 → 需 `pycollada`（已装）+ `trimesh` 转 obj。
- **实际硬件为单块 RTX 3060 Laptop（6 GB）**，与研究提示词的 3080+3060 双卡不符；`device_count==1`，统一 `cuda:0`。
- `jupyter nbconvert` 的默认 `python3` kernel 解析到 **base anaconda py3.13**（无 pytorch3d）。
  已注册 env kernel `go2py39`（`jupyter kernelspec remove go2py39` 可撤销），执行时显式指定它。
- 全分辨率孪生约 **40 万面**，单姿态渲染 OK；批量训练前需减面。
