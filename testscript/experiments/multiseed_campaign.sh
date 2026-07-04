#!/usr/bin/env bash
# ================================================================
# multiseed_campaign.sh — 2×2 消融 × 3 训练种子 (REVIEW_REPORT.md §4)
# ================================================================
# 目的: 把"四个选中检查点的差异"升级为跨训练种子的可信方法比较。
# 设计: 4 格 (baseline / goal-only / clip-only / gcgl) × 种子 {1001,1002,1003}，
#       与原论文完全同配置 (@configs/thesis_base.args + --loss_v_mode mse)，
#       仅新增 --seed 与 --gpu 0。
# ⚠ GPU 序号陷阱: CUDA 默认按"最快优先"排序 → cuda:0 = RTX 3080 (10GB)，
#       cuda:1 = 3060 Laptop (6GB)；与 nvidia-smi 的 index 正好相反！
#       nvidia-smi 里 3080 显示为 index 1。务必用 --gpu 0。
# 选择准则(预注册): 训练期 AR 最优检查点 best_ar.pth，评估用与论文相同的
#       5 个场景种子 × 32 episodes (评估另行批量执行)。
# 断点: 每 run 完成后写 DONE 标记; 重跑脚本自动跳过已完成、续跑半途的。
# 顺序: 种子外层——先跑完 seed 1001 全部 4 格，尽早得到首个完整配对副本。
# 预计: ~8.3 s/it × 5000 it ≈ 11.5h/run, 共 12 runs ≈ 5.7 天。
# ================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."   # → 仓库根目录

GPU=0   # cuda:0 = RTX 3080 (CUDA fastest-first; nvidia-smi 里它是 index 1)
SEEDS=(1001 1002 1003)
CELLS=(baseline goal clip gcgl)

cell_args() {
    case "$1" in
        baseline) echo "" ;;
        goal)     echo "--coef_goal_reaching 0.2" ;;
        clip)     echo "--grad_clip_norm 1.0" ;;
        gcgl)     echo "--grad_clip_norm 1.0 --coef_goal_reaching 0.2" ;;
    esac
}

for seed in "${SEEDS[@]}"; do
  for cell in "${CELLS[@]}"; do
    name="${cell}_s${seed}"
    SAVE_DIR="checkpoints/thesis_multiseed/${name}"
    LOG_DIR="logs/thesis_multiseed/${name}"
    mkdir -p "$SAVE_DIR" "$LOG_DIR"

    if [ -f "$SAVE_DIR/DONE" ]; then
        echo "[skip] $name already DONE"
        continue
    fi

    RESUME_FLAG=""
    LATEST_CKPT=$(find "$SAVE_DIR" -name 'checkpoint_*.pth' ! -name 'best_*' 2>/dev/null | sort | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        # 不带 --reset_lr: 恢复 optimizer/scheduler/RNG 并从存档迭代继续，
        # 与不间断训练等价（--reset_lr 会重开 cosine 周期并从 0 重跑, 是 fine-tune 语义）
        RESUME_FLAG="--resume $LATEST_CKPT"
        echo "[resume] $name from $LATEST_CKPT"
    fi

    echo "[start] $name  $(date '+%F %T')"
    # shellcheck disable=SC2046
    python train.py \
        @configs/thesis_base.args \
        --loss_v_mode mse \
        $(cell_args "$cell") \
        --seed "$seed" \
        --gpu "$GPU" \
        --save_dir "$SAVE_DIR" \
        --log_dir "$LOG_DIR" \
        $RESUME_FLAG \
        >> "${LOG_DIR}/train.log" 2>&1
    status=$?

    if [ $status -eq 0 ]; then
        touch "$SAVE_DIR/DONE"
        echo "[done]  $name  $(date '+%F %T')"
    else
        echo "[FAIL]  $name exit=$status  $(date '+%F %T')  (继续下一个; 重跑本脚本可续)"
    fi
  done
done
echo "[campaign] all runs processed  $(date '+%F %T')"
