"""PD, standing load, friction, and short-trajectory MuJoCo reconciliation."""

from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffsim.go2.contact import contact_wrenches  # noqa: E402
from diffsim.go2.dynamics import actuator_torque, dynamics_step  # noqa: E402
from diffsim.go2.scene import flat_scene  # noqa: E402
from diffsim.go2.types import Go2EnvConfig  # noqa: E402
from diffsim.envs.go2 import Go2Env  # noqa: E402
from testscript.test_go2_mujoco_gate import _load_matching_urdf  # noqa: E402


def _mj_addresses(model):
    names = [
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    ]
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names
    ]
    return (
        np.array([model.jnt_qposadr[index] for index in joint_ids]),
        np.array([model.jnt_dofadr[index] for index in joint_ids]),
    )


def _copy_to_mujoco(state, data, qpos_address, dof_address):
    data.qpos[:3] = state.base_pos[0].detach().cpu().numpy()
    data.qpos[3:7] = state.base_quat[0].detach().cpu().numpy()
    data.qpos[qpos_address] = state.joint_pos[0].detach().cpu().numpy()
    data.qvel[:3] = state.base_vel[0].detach().cpu().numpy()
    data.qvel[3:6] = state.base_omega[0].detach().cpu().numpy()
    data.qvel[dof_address] = state.joint_vel[0].detach().cpu().numpy()


def _mj_torque(data, qpos_address, dof_address, target, action, model, config):
    desired = model.default_joint_pos.cpu().numpy() + config.action_scale * np.tanh(action)
    desired = np.clip(
        desired,
        model.joint_lower.cpu().numpy() + 0.02,
        model.joint_upper.cpu().numpy() - 0.02,
    )
    alpha = 1.0 - np.exp(-config.physics_dt / config.actuator_time_constant)
    target = target + alpha * (desired - target)
    raw = config.kp * (target - data.qpos[qpos_address]) - config.kd * data.qvel[dof_address]
    effort = model.joint_effort.cpu().numpy()
    return effort * np.tanh(raw / effort), target


def main():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA is required for the full self-contact reconciliation rollout")
        return
    device = torch.device("cuda:0")
    config = Go2EnvConfig(batch_size=1)
    env = Go2Env(config, device, randomize=False)
    model, state = env.model, env.state
    scene = flat_scene(config, device, torch.float32)

    mj_model = _load_matching_urdf(with_floor=True)
    mj_model.opt.timestep = config.physics_dt
    mj_model.geom_friction[:, 0] = config.friction
    mj_data = mujoco.MjData(mj_model)
    qpos_address, dof_address = _mj_addresses(mj_model)
    _copy_to_mujoco(state, mj_data, qpos_address, dof_address)
    mujoco.mj_forward(mj_model, mj_data)
    target_mj = mj_data.qpos[qpos_address].copy()
    action_zero = torch.zeros(1, 12, device=device)

    with torch.no_grad():
        # PD step: compare the controller's torque before either engine advances.
        torque_torch, _, _ = actuator_torque(model, state, action_zero, config)
        torque_mj, target_mj = _mj_torque(
            mj_data, qpos_address, dof_address, target_mj, np.zeros(12), model, config
        )
        pd_error = np.max(np.abs(torque_torch[0].cpu().numpy() - torque_mj))
        assert pd_error < 2e-5

        # Settle under standing load in both engines.
        for _ in range(250):
            contact = contact_wrenches(model, state, scene, config)
            state = dynamics_step(
                model, state, action_zero, config, external_wrench_body=contact.wrench_body
            ).state
            torque_mj, target_mj = _mj_torque(
                mj_data, qpos_address, dof_address, target_mj, np.zeros(12), model, config
            )
            mj_data.qfrc_applied[dof_address] = torque_mj
            mujoco.mj_step(mj_model, mj_data)
        standing_height_error = abs(float(state.base_pos[0, 2].cpu()) - mj_data.qpos[2])
        foot_penetration = float(
            torch.relu(-contact.gap[:, model.collisions.sample_is_foot]).amax().cpu()
        )
        assert foot_penetration <= 0.003
        assert standing_height_error < 0.035

        # Friction audit: both contacts must dissipate the same imposed slip.
        state.base_vel[:, 0] = 0.5
        mj_data.qvel[0] = 0.5
        speed_initial = 0.5
        for _ in range(150):
            contact = contact_wrenches(model, state, scene, config)
            state = dynamics_step(
                model, state, action_zero, config, external_wrench_body=contact.wrench_body
            ).state
            torque_mj, target_mj = _mj_torque(
                mj_data, qpos_address, dof_address, target_mj, np.zeros(12), model, config
            )
            mj_data.qfrc_applied[dof_address] = torque_mj
            mujoco.mj_step(mj_model, mj_data)
        speed_torch = abs(float(state.base_vel[0, 0].cpu()))
        speed_mj = abs(float(mj_data.qvel[0]))
        assert speed_torch < speed_initial and speed_mj < speed_initial

        # Same smooth target sequence: require bounded short-horizon divergence.
        for step in range(150):
            phase = 2.0 * np.pi * step / 150.0
            action_np = 0.12 * np.sin(phase + np.arange(12) * 0.31)
            action = torch.tensor(action_np, device=device)[None].float()
            contact = contact_wrenches(model, state, scene, config)
            state = dynamics_step(
                model, state, action, config, external_wrench_body=contact.wrench_body
            ).state
            torque_mj, target_mj = _mj_torque(
                mj_data, qpos_address, dof_address, target_mj, action_np, model, config
            )
            mj_data.qfrc_applied[dof_address] = torque_mj
            mujoco.mj_step(mj_model, mj_data)
        joint_rmse = float(
            np.sqrt(
                np.mean(
                    (state.joint_pos[0].cpu().numpy() - mj_data.qpos[qpos_address]) ** 2
                )
            )
        )
        assert joint_rmse < 0.20
        assert torch.isfinite(state.base_pos).all()

    print(f"PD torque max error: {pd_error:.3e} Nm")
    print(
        f"standing: height delta={standing_height_error * 1e3:.2f} mm "
        f"soft-contact penetration={foot_penetration * 1e3:.2f} mm"
    )
    print(f"friction slip: vx |torch|={speed_torch:.3f} |mujoco|={speed_mj:.3f} m/s")
    print(f"same-action short trajectory: joint RMSE={joint_rmse:.4f} rad")
    print("[PASS] MuJoCo PD/contact/friction/trajectory reconciliation")


if __name__ == "__main__":
    main()
