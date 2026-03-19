"""
训练监控模块

提供训练过程中的实时监控功能：
- CSV 日志：所有指标写入 CSV 文件，方便离线分析 (pandas/Excel)
- 损失曲线自动保存：定期生成 PNG 图片，无需 TensorBoard 即可查看趋势
- 增强进度条：tqdm 显示平滑损失 + 关键指标
- 控制台摘要：定期打印格式化的指标摘要表

用法：
    monitor = TrainingMonitor(log_dir='./logs/run_xxx')
    for i in range(num_iters):
        loss, metrics = ...
        monitor.step(i, loss, metrics)  # 一行搞定所有日志
    monitor.close()

作者: Kirk
"""

import os
import csv
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，不需要 GUI
import matplotlib.pyplot as plt
import numpy as np


class TrainingMonitor:
    """
    训练监控器，聚合所有日志/可视化功能。
    
    Args:
        log_dir: 日志输出目录（CSV、PNG 都写在这里）
        smoothing_window: 指标平滑窗口大小
        csv_flush_interval: 每 N 步刷新 CSV 文件到磁盘
        curve_save_interval: 每 N 步保存损失曲线 PNG
        console_summary_interval: 每 N 步打印控制台摘要
        key_metrics: 在 tqdm 和控制台摘要中显示的关键指标
    """
    
    def __init__(self,
                 log_dir,
                 smoothing_window=50,
                 csv_flush_interval=25,
                 curve_save_interval=500,
                 console_summary_interval=100,
                 key_metrics=None):
        
        self.log_dir = log_dir
        self.smoothing_window = smoothing_window
        self.csv_flush_interval = csv_flush_interval
        self.curve_save_interval = curve_save_interval
        self.console_summary_interval = console_summary_interval
        
        self.key_metrics = key_metrics or [
            'loss', 'loss_v', 'loss_collide', 'loss_obj_avoidance',
            'success_rate', 'reach_rate', 'collision_free_rate', 'goal_progress', 'avg_speed', 'ar', 'task_score'
        ]
        
        # 滑动窗口存储（用于实时平滑显示）
        self._windows = defaultdict(lambda: deque(maxlen=smoothing_window))
        
        # 全量历史（用于曲线绘制）
        self._history_steps = []
        self._history = defaultdict(list)
        
        # CSV 相关
        self._csv_path = os.path.join(log_dir, 'metrics.csv')
        self._csv_file = None
        self._csv_writer = None
        self._csv_columns = None
        self._csv_buffer = []
        
        # 曲线图目录
        self._curve_dir = os.path.join(log_dir, 'curves')
        os.makedirs(self._curve_dir, exist_ok=True)
        
        # 时间追踪
        self._start_time = time.time()
        self._last_step_time = time.time()
        self._step_times = deque(maxlen=100)  # 最近 100 步的耗时
        
        print(f"[TrainingMonitor] CSV 日志: {self._csv_path}")
        print(f"[TrainingMonitor] 损失曲线: {self._curve_dir}/")
    
    def step(self, iteration, loss, metrics, pbar=None):
        """
        记录一步训练结果。每次 train loop 迭代调用一次。
        
        Args:
            iteration: 当前迭代编号 (0-based)
            loss: 标量损失值
            metrics: dict，各项指标
            pbar: tqdm 进度条对象（可选，用于更新显示）
        """
        step = iteration + 1  # 1-based 用于显示
        now = time.time()
        step_time = now - self._last_step_time
        self._last_step_time = now
        self._step_times.append(step_time)
        
        # 合并所有指标
        all_metrics = {'loss': float(loss)}
        for k, v in metrics.items():
            try:
                all_metrics[k] = float(v)
            except (TypeError, ValueError):
                continue
        all_metrics['step_time'] = step_time
        all_metrics['lr'] = metrics.get('lr', 0.0)
        
        # 更新滑动窗口
        for k, v in all_metrics.items():
            self._windows[k].append(v)
        
        # 更新全量历史
        self._history_steps.append(step)
        for k, v in all_metrics.items():
            self._history[k].append(v)
        
        # 缓存 CSV 行
        self._csv_buffer.append((step, all_metrics))
        
        # 更新 tqdm
        if pbar is not None:
            pbar.set_postfix_str(self._format_tqdm(step), refresh=False)
        
        # 定期操作
        if step % self.csv_flush_interval == 0:
            self._flush_csv()
        
        if step % self.console_summary_interval == 0:
            self._print_summary(step)
        
        if step % self.curve_save_interval == 0:
            self._save_curves(step)
    
    def close(self):
        """训练结束时调用，刷新所有缓冲。"""
        self._flush_csv()
        if len(self._history_steps) > 0:
            self._save_curves(self._history_steps[-1])
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        print(f"[TrainingMonitor] 已保存最终曲线和 CSV")
    
    def get_smoothed(self, key):
        """获取某指标的滑动平均值。"""
        w = self._windows.get(key)
        if w is None or len(w) == 0:
            return 0.0
        return sum(w) / len(w)
    
    # ---- 内部方法 ----
    
    def _format_tqdm(self, step):
        """生成 tqdm 后缀字符串：平滑损失 + 关键指标。"""
        parts = []
        
        # 核心损失
        loss_sm = self.get_smoothed('loss')
        parts.append(f'L={loss_sm:.3f}')
        
        # 子项损失（只显示有值的）
        for sub in ['loss_v', 'loss_collide', 'loss_obj_avoidance']:
            val = self.get_smoothed(sub)
            if val > 0:
                short_name = sub.replace('loss_', '')
                parts.append(f'{short_name}={val:.3f}')
        
        # 成功率和速度
        sr = self.get_smoothed('success_rate')
        if sr > 0:
            parts.append(f'SR={sr:.0%}')
        
        ar = self.get_smoothed('ar')
        if ar > 0:
            parts.append(f'AR={ar:.2f}')
        
        # 速度 (it/s)
        if len(self._step_times) > 0:
            avg_time = sum(self._step_times) / len(self._step_times)
            parts.append(f'{avg_time:.2f}s/it')
        
        return ' | '.join(parts)
    
    def _flush_csv(self):
        """将缓冲区写入 CSV 文件。"""
        if not self._csv_buffer:
            return
        
        # 收集所有出现过的列名
        all_keys = set()
        for _, m in self._csv_buffer:
            all_keys.update(m.keys())
        
        # 初始化 CSV（首次写入时创建文件头）
        if self._csv_file is None:
            self._csv_columns = ['step'] + sorted(all_keys)
            self._csv_file = open(self._csv_path, 'w', newline='')
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_columns,
                                               extrasaction='ignore')
            self._csv_writer.writeheader()
        else:
            # 如果有新的列，需要重新打开（罕见）
            new_keys = all_keys - set(self._csv_columns[1:])
            if new_keys:
                self._csv_file.close()
                self._csv_columns = ['step'] + sorted(
                    set(self._csv_columns[1:]) | new_keys
                )
                # 重写整个文件（包含旧数据 + 新列）
                self._csv_file = open(self._csv_path, 'w', newline='')
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_columns,
                                                   extrasaction='ignore')
                self._csv_writer.writeheader()
                # 回写全量历史
                for idx, s in enumerate(self._history_steps[:-len(self._csv_buffer)]):
                    row = {'step': s}
                    for k in self._csv_columns[1:]:
                        if k in self._history and idx < len(self._history[k]):
                            row[k] = self._history[k][idx]
                    self._csv_writer.writerow(row)
        
        # 写入缓冲行
        for step, m in self._csv_buffer:
            row = {'step': step}
            row.update(m)
            self._csv_writer.writerow(row)
        
        self._csv_file.flush()
        self._csv_buffer.clear()
    
    def _print_summary(self, step):
        """打印控制台指标摘要表。"""
        elapsed = time.time() - self._start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        # ETA
        if len(self._step_times) > 0:
            avg_time = sum(self._step_times) / len(self._step_times)
        else:
            avg_time = 0
        
        print(f"\n{'─' * 65}")
        print(f"  # Iter {step:>6d}  |  已用时 {elapsed_str}  |  {avg_time:.2f} s/it")
        print(f"{'─' * 65}")
        
        # 按类别分组显示
        groups = {
            '# 损失': ['loss', 'loss_v', 'loss_v_pred', 'loss_collide',
                        'loss_obj_avoidance', 'loss_d_acc', 'loss_d_jerk',
                        'loss_speed', 'loss_ground_affinity', 'loss_bias'],
            '# 模型性能': ['success_rate', 'reach_rate', 'collision_free_rate',
                        'goal_progress', 'goal_distance_best', 'goal_distance_final',
                        'avg_speed', 'max_speed', 'ar', 'task_score'],
            '# 迭代速度': ['step_time', 'lr'],
        }
        
        for group_name, keys in groups.items():
            displayed = []
            for k in keys:
                w = self._windows.get(k)
                if w is None or len(w) == 0:
                    continue
                val = sum(w) / len(w)
                # 格式化
                if k in {'success_rate', 'reach_rate', 'collision_free_rate', 'goal_progress'}:
                    displayed.append(f"  {k:25s}  {val:>10.1%}")
                elif k == 'lr':
                    displayed.append(f"  {k:25s}  {val:>10.2e}")
                elif k == 'step_time':
                    displayed.append(f"  {k:25s}  {val:>10.3f} s")
                else:
                    displayed.append(f"  {k:25s}  {val:>10.4f}")
            
            if displayed:
                print(f"  {group_name}:")
                for line in displayed:
                    print(line)
        
        print(f"{'─' * 65}\n")
    
    def _save_curves(self, step):
        """保存损失曲线和性能曲线为 PNG。"""
        if len(self._history_steps) < 2:
            return
        
        steps = np.array(self._history_steps)
        
        # ---- 图 1: 总损失 + 子项损失 ----
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Curves (iter {step})', fontsize=14)
        
        # 1a: 总损失
        ax = axes[0, 0]
        if 'loss' in self._history:
            vals = np.array(self._history['loss'])
            ax.plot(steps, vals, alpha=0.3, color='C0', linewidth=0.5)
            if len(vals) >= 10:
                smoothed = self._moving_average(vals, min(50, len(vals) // 3))
                ax.plot(steps[len(steps)-len(smoothed):], smoothed, color='C0', linewidth=2, label='smoothed')
            ax.set_title('Total Loss')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 1b: 关键子项损失
        ax = axes[0, 1]
        sub_losses = ['loss_v', 'loss_collide', 'loss_obj_avoidance', 'loss_v_pred']
        for i, key in enumerate(sub_losses):
            if key in self._history and len(self._history[key]) > 0:
                vals = np.array(self._history[key])
                if len(vals) >= 10:
                    smoothed = self._moving_average(vals, min(50, len(vals) // 3))
                    ax.plot(steps[len(steps)-len(smoothed):], smoothed,
                            label=key.replace('loss_', ''), linewidth=1.5)
                else:
                    ax.plot(steps[:len(vals)], vals, label=key.replace('loss_', ''), linewidth=1)
        ax.set_title('Sub-Losses')
        ax.set_xlabel('Iteration')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 1c: 严格成功率/抵达率/任务分数
        ax = axes[1, 0]
        for key, label, color in [
            ('success_rate', 'Strict Success Rate', 'C2'),
            ('reach_rate', 'Reach Rate', 'C6'),
            ('task_score', 'Task Score', 'C3'),
        ]:
            if key in self._history and len(self._history[key]) > 0:
                vals = np.array(self._history[key])
                ax.plot(steps[:len(vals)], vals, alpha=0.2, color=color, linewidth=0.5)
                if len(vals) >= 10:
                    smoothed = self._moving_average(vals, min(50, len(vals) // 3))
                    ax.plot(steps[len(steps)-len(smoothed):], smoothed,
                            color=color, linewidth=2, label=label)
        ax.set_title('Strict Success / Reach / Task Score')
        ax.set_xlabel('Iteration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 1d: 平均速度 + 最大速度
        ax = axes[1, 1]
        for key, label, color in [('avg_speed', 'Avg Speed', 'C4'), ('max_speed', 'Max Speed', 'C5')]:
            if key in self._history and len(self._history[key]) > 0:
                vals = np.array(self._history[key])
                if len(vals) >= 10:
                    smoothed = self._moving_average(vals, min(50, len(vals) // 3))
                    ax.plot(steps[len(steps)-len(smoothed):], smoothed,
                            color=color, linewidth=2, label=label)
                else:
                    ax.plot(steps[:len(vals)], vals, color=color, label=label, linewidth=1)
        ax.set_title('Speed Metrics')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('m/s')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self._curve_dir, 'latest.png')
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        # 同时保存带步数的版本（方便对比不同阶段）
        milestone_path = os.path.join(self._curve_dir, f'curves_{step:06d}.png')
        fig.savefig(milestone_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def _moving_average(data, window):
        """一维滑动平均。"""
        if window <= 1 or len(data) < window:
            return data
        cumsum = np.cumsum(data)
        cumsum[window:] = cumsum[window:] - cumsum[:-window]
        return cumsum[window - 1:] / window
