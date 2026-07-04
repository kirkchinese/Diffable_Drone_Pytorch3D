# DGX Spark 迁移清单 — 2×2×3种子补充实验

目的：把 `testscript/experiments/multiseed_campaign.sh`（4 格 × 训练种子
{1001,1002,1003}，共 12 个 run）搬到公司 DGX Spark 上跑，替代本地 3080
（单 run ≈ 11.5h，串行 ≈ 5.7 天；DGX 显存充裕可多 run 并行）。

## 0. 硬件/环境注意（Grace-Blackwell 专有坑）

- DGX Spark 是 **ARM64 (aarch64) CPU + Blackwell GPU**，x86 wheel 全部不可用。
- PyTorch：用 NVIDIA 官方渠道装 aarch64 + CUDA 版（NGC PyTorch 容器最省事，
  或 `pip install torch --index-url https://download.pytorch.org/whl/cu128`，以
  当机 CUDA 版本为准）。
- **PyTorch3D 没有 aarch64 预编译包，必须源码编译**：
  ```bash
  pip install "git+https://github.com/facebookresearch/pytorch3d.git"
  # 需要 nvcc 与 torch 的 CUDA 版本一致; Blackwell 须 TORCH_CUDA_ARCH_LIST 含 12.0
  # 编译 ~20-40 分钟。失败时优先检查: gcc 版本 / CUB / TORCH_CUDA_ARCH_LIST
  ```
- 其余依赖见 `requirements.txt`（numpy/pandas/matplotlib/tqdm/tensorboard 均有 aarch64 wheel）。

## 1. 克隆与检查

```bash
git clone <repo> && cd Diffable_Drone_Pytorch3D
git checkout <本清单所在分支>
git log -1        # 核对 commit hash（checkpoint 里会记录 git_hash 用于溯源）
ls data/base_model/drone.obj data/sample/sample4.obj   # 网格已在库内
```

## 2. 冒烟测试（必须先过，再上全量）

```bash
python train.py @configs/thesis_base.args --loss_v_mode mse \
  --grad_clip_norm 1.0 --coef_goal_reaching 0.2 --seed 9999 --gpu 0 \
  --num_iters 2 --save_dir /tmp/smoke_ckpt --log_dir /tmp/smoke_log
```
通过标准：正常走完 2 个 iter、无 CUDA/渲染报错、`/tmp/smoke_ckpt/metrics.csv` 生成。
顺带记录单 iter 耗时（本地 3080 ≈ 8.3s/it，Blackwell 应显著更快）。

## 3. 启动战役

```bash
# 先按 DGX 的 GPU 情况改脚本头部的 GPU 变量（nvidia-smi 确认序号）
nohup bash testscript/experiments/multiseed_campaign.sh \
  > logs/multiseed_campaign_nohup.log 2>&1 &
```

- 断点续跑已修复并验证：崩溃/重启后**直接重跑同一脚本**即可——完成的 run 有
  DONE 标记自动跳过，半途的 run 从最新 checkpoint 恢复（optimizer/scheduler/RNG/
  迭代数全部连续，与不间断训练等价；resume 不要加 `--reset_lr`，那是 fine-tune 语义）。
- 多卡并行：把脚本按 seed 拆 3 份、GPU 变量各指一张卡即可（每 run 显存 ≈ 5.7GB）。

## 4. 监控

```bash
tail -f logs/thesis_multiseed/<run名>/train.log     # 单 run 进度
ls checkpoints/thesis_multiseed/*/DONE              # 完成计数 (共 12)
```

## 5. 训练完成后的评估（预注册准则，勿改）

- 选择准则：每个 run 的 **best_ar.pth**（训练期 AR 最优，与论文一致，先于评估固定）。
- 评估：与论文相同的 5 个场景种子 `0 42 123 456 789` × 32 episodes（默认值）。
  **必须指定 `--out_base`，防止覆盖 `viz_results/thesis_eval` 下的论文真值**：
  ```bash
  python testscript/multi_seed_eval.py \
    --ckpt_base checkpoints/thesis_multiseed \
    --out_base viz_results/multiseed_eval \
    --experiments baseline_s1001 goal_s1001 clip_s1001 gcgl_s1001 \
                  baseline_s1002 goal_s1002 clip_s1002 gcgl_s1002 \
                  baseline_s1003 goal_s1003 clip_s1003 gcgl_s1003
  ```
- 主不确定度按**训练种子**维度汇报（3 个/格），场景种子波动是次要维度
  （REVIEW_REPORT.md §4 的设计）。

## 6. 结果回传

12 个 run 只需回传：`checkpoints/thesis_multiseed/*/best_ar.pth`、`*/metrics.csv`
与评估输出目录（几百 MB 内），全量 checkpoint 不必搬。
