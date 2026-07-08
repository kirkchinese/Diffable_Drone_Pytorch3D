"""Direct-BPTT trainer dedicated to perceptive Go2 locomotion."""

from __future__ import annotations

from dataclasses import fields
import math
from pathlib import Path
import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from .. import EnvBuildContext, make_env
from ..losses import LossBuildContext, make_loss
from ..losses.go2 import Go2LossContext
from .policy import Go2Policy, mirror_action, mirror_observation
from .types import Go2State


def _scale_state_gradient(state: Go2State, decay: float) -> Go2State:
    """Identity in the forward pass, configurable GDecay in the backward pass."""

    if decay == 1.0:
        return state
    return Go2State(
        **{
            field.name: getattr(state, field.name).detach()
            + decay * (getattr(state, field.name) - getattr(state, field.name).detach())
            for field in fields(Go2State)
        }
    )


class Go2Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.control_dt = args.go2_control_dt
        self.batch_size = args.go2_batch_size or 64
        self.sensor_mode = args.go2_sensor_mode
        args.batch_size = self.batch_size
        args.ctl_dt = self.control_dt

        self.env = make_env(
            "go2",
            EnvBuildContext(
                args=args,
                device=self.device,
                control_dt=self.control_dt,
                focal_length=32.0,
            ),
        )
        self.policy = Go2Policy(self.sensor_mode, hidden_size=args.go2_hidden_size).to(self.device)
        self.loss_fn = make_loss(
            "go2_locomotion", LossBuildContext(args=args, control_dt=self.control_dt)
        ).to(self.device)
        self.optimizer = AdamW(self.policy.parameters(), lr=args.lr)
        updates = math.ceil(args.go2_rollout_steps / args.go2_tbptt_steps)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            max(1, args.num_iters * updates),
            eta_min=args.lr * 0.05,
        )
        self.start_iteration = 0
        self.global_update = 0
        if args.resume:
            self._load(args.resume)
        self.save_dir = Path(args.save_dir) / "go2"
        self.log_dir = Path(args.log_dir) / "go2"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def _load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "optimizer_state_dict" in checkpoint and not self.args.reset_lr:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_iteration = int(checkpoint.get("iteration", 0))
        self.global_update = int(checkpoint.get("global_update", 0))

    def _checkpoint(self, iteration: int) -> dict:
        return {
            "iteration": iteration,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "sensor_mode": self.sensor_mode,
            "joint_order": self.env.model.joint_names,
            "train_seed": self.args.seed,
            "global_update": self.global_update,
        }

    def _optimize_segment(self, losses: list[torch.Tensor]) -> float:
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = clip_grad_norm_(self.policy.parameters(), self.args.go2_grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.writer.add_scalar("train/loss", loss.detach(), self.global_update)
        self.writer.add_scalar("train/gradient_norm", gradient_norm.detach(), self.global_update)
        self.writer.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], self.global_update)
        self.global_update += 1
        return float(loss.detach())

    def train(self) -> None:
        print(
            f"[Go2] device={self.device} B={self.batch_size} sensor={self.sensor_mode} "
            f"rollout={self.args.go2_rollout_steps} TBPTT={self.args.go2_tbptt_steps}"
        )
        observation = self.env.reset().observation
        hidden = self.policy.initial_hidden(self.batch_size, self.device)
        for iteration in range(self.start_iteration, self.args.num_iters):
            started = time.perf_counter()
            alive = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
            fallen = torch.zeros_like(alive)
            segment_losses: list[torch.Tensor] = []
            iteration_losses = []
            terminations = torch.zeros((), device=self.device)
            for step in range(self.args.go2_rollout_steps):
                hidden_input = hidden
                previous_action = self.env.previous_action
                action, hidden = self.policy(observation, hidden_input)
                reflected_action, _ = self.policy(mirror_observation(observation), hidden_input)
                symmetry = (reflected_action - mirror_action(action)).square().mean()
                output = self.env.step(action)
                result = self.loss_fn(
                    Go2LossContext(
                        state=output.state,
                        action=action,
                        previous_action=previous_action,
                        command=self.env.command,
                        torque=self.env.last_dynamics.torque,
                        contact=self.env.last_contact,
                        scene=self.env.scene,
                        model=self.env.model,
                        valid=alive,
                    )
                )
                segment_losses.append(
                    result.loss + self.args.go2_symmetry_weight * symmetry
                )
                terminated = output.terminated
                fallen |= terminated
                alive &= ~terminated
                terminations = terminations + terminated.float().mean()
                self.env.state = _scale_state_gradient(output.state, self.args.go2_gdecay)
                observation = self.env.observe()

                boundary = (step + 1) % self.args.go2_tbptt_steps == 0
                boundary |= step + 1 == self.args.go2_rollout_steps
                if boundary:
                    iteration_losses.append(self._optimize_segment(segment_losses))
                    segment_losses = []
                    hidden = hidden.detach()
                    self.env.state = self.env.state.detach()
                    self.env.previous_action = self.env.previous_action.detach()
                    observation = self.env.reset(mask=fallen).observation
                    fallen.zero_()
                    alive.fill_(True)

            elapsed = time.perf_counter() - started
            mean_loss = sum(iteration_losses) / len(iteration_losses)
            fall_rate = float(terminations.detach() / self.args.go2_rollout_steps)
            self.writer.add_scalar("train/fall_rate", fall_rate, iteration)
            throughput = self.batch_size * self.args.go2_rollout_steps / elapsed
            self.writer.add_scalar("train/control_steps_per_second", throughput, iteration)
            print(
                f"[Go2 {iteration + 1}/{self.args.num_iters}] loss={mean_loss:.5f} "
                f"fall={fall_rate:.3f} elapsed={elapsed:.2f}s"
            )
            if (iteration + 1) % self.args.save_freq == 0:
                torch.save(
                    self._checkpoint(iteration + 1),
                    self.save_dir / f"checkpoint_{iteration + 1:06d}.pth",
                )
        torch.save(self._checkpoint(self.args.num_iters), self.save_dir / "checkpoint_final.pth")
        self.writer.close()
