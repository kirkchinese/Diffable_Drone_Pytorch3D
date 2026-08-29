#!/usr/bin/env python3
"""ROS 1/MAVROS adapter for the trained depth-navigation policy."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from model import Model_bigger
from navigation_utils import DronePolicy


ROS_DEPTH_SCALE = 0.001
HARD_SPEED_LIMIT_MPS = 0.5
HARD_ACCEL_LIMIT_MPS2 = 1.0
DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 24.0
MODEL_HEIGHT = 48
MODEL_WIDTH = 64


def clamp_vector(value, limit: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("expected one finite xyz vector")
    if not np.isfinite(limit) or limit < 0.0:
        raise ValueError("limit must be finite and non-negative")
    norm = float(np.linalg.norm(value))
    return value if norm <= limit or norm == 0.0 else value * (limit / norm)


def speed_limit(requested: float) -> float:
    if not np.isfinite(requested) or requested < 0.0:
        raise ValueError("max_speed must be finite and non-negative")
    return min(float(requested), HARD_SPEED_LIMIT_MPS)


def quaternion_to_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(q).all() or norm < 1e-8:
        raise ValueError("invalid odometry quaternion")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _center_crop_4_3(depth: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    if width * 3 > height * 4:
        crop_width = height * 4 // 3
        left = (width - crop_width) // 2
        return depth[:, left : left + crop_width]
    crop_height = width * 3 // 4
    top = (height - crop_height) // 2
    return depth[top : top + crop_height]


def decode_depth_image(msg, depth_scale: float = ROS_DEPTH_SCALE) -> tuple[np.ndarray, float]:
    """Decode ROS Image, apply metric scale, center-crop, and resize to 48x64."""
    if msg.encoding == "16UC1":
        dtype = np.dtype(">u2" if msg.is_bigendian else "<u2")
        scale = depth_scale
    elif msg.encoding == "32FC1":
        dtype = np.dtype(">f4" if msg.is_bigendian else "<f4")
        scale = 1.0
    else:
        raise ValueError(f"unsupported depth encoding: {msg.encoding}")
    if not np.isfinite(scale) or scale <= 0.0 or msg.step % dtype.itemsize:
        raise ValueError("invalid depth scale or row step")
    row_items = msg.step // dtype.itemsize
    if row_items < msg.width or len(msg.data) < msg.step * msg.height:
        raise ValueError("truncated depth image")
    raw = np.frombuffer(msg.data, dtype=dtype, count=row_items * msg.height)
    depth = raw.reshape(msg.height, row_items)[:, : msg.width].astype(np.float32)
    depth *= scale
    invalid = ~np.isfinite(depth) | (depth <= 0.0) | (depth > DEPTH_MAX_M)
    depth[invalid] = -1.0
    depth = cv2.resize(
        _center_crop_4_3(depth),
        (MODEL_WIDTH, MODEL_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.ascontiguousarray(depth), float(np.mean(depth > 0.0))


def velocity_setpoint(
    velocity_world: np.ndarray,
    acceleration_world: np.ndarray,
    dt: float,
    requested_speed: float,
    requested_acceleration: float,
) -> np.ndarray:
    """Short-horizon acceleration adapter with a non-bypassable 3-D speed cap."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    acceleration_limit = min(float(requested_acceleration), HARD_ACCEL_LIMIT_MPS2)
    if not np.isfinite(acceleration_limit) or acceleration_limit < 0.0:
        raise ValueError("max_acceleration must be finite and non-negative")
    acceleration = clamp_vector(acceleration_world, acceleration_limit)
    return clamp_vector(
        np.asarray(velocity_world, dtype=np.float32) + acceleration * dt,
        speed_limit(requested_speed),
    )


class Ros1MavrosAdapter:
    def __init__(self) -> None:
        import message_filters
        import rospy
        from geometry_msgs.msg import PoseStamped
        from mavros_msgs.msg import PositionTarget
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image

        self.rospy = rospy
        self.PositionTarget = PositionTarget
        self.lock = threading.Lock()
        self.latest = None
        self.goal = None
        self.goal_frame = ""
        self.goal_version = 0
        self.hidden = None
        self.last_image_ns = None
        self.last_command_stamp = None

        root = Path(__file__).resolve().parent
        checkpoint = Path(
            rospy.get_param(
                "~checkpoint",
                str(root / "checkpoints/thesis/exp21_grad_clip_goal/best_ar.pth"),
            )
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.device = torch.device(rospy.get_param("~device", "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.set_num_threads(int(rospy.get_param("~torch_threads", 1)))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = Model_bigger(dim_obs=10, dim_action=6)
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.model.eval().to(self.device)
        self.policy = DronePolicy(
            self.model,
            torch.tensor([0.0, 0.0, -9.80665], device=self.device),
            DEPTH_MIN_M,
            DEPTH_MAX_M,
            no_odom=False,
            sensor_mode="depth",
        )

        self.rate = float(rospy.get_param("~inference_rate", 15.0))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.max_input_age = float(rospy.get_param("~max_input_age", 0.2))
        if self.rate <= 0.0 or self.sync_slop <= 0.0 or self.max_input_age <= 0.0:
            raise ValueError("rate, sync_slop, and max_input_age must be positive")

        depth_sub = message_filters.Subscriber(
            rospy.get_param("~depth_topic", "/camera/depth/image_rect_raw"), Image
        )
        odom_sub = message_filters.Subscriber(
            rospy.get_param("~odom_topic", "/mavros/local_position/odom"), Odometry
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, odom_sub], queue_size=10, slop=self.sync_slop
        )
        self.sync.registerCallback(self._sync_callback)
        self.goal_sub = rospy.Subscriber(
            rospy.get_param("~goal_topic", "/drone_policy/goal"),
            PoseStamped,
            self._goal_callback,
            queue_size=1,
        )
        self.publisher = rospy.Publisher(
            rospy.get_param("~setpoint_topic", "/mavros/setpoint_raw/local"),
            PositionTarget,
            queue_size=1,
        )
        self.timer = rospy.Timer(rospy.Duration.from_sec(1.0 / self.rate), self._tick)
        rospy.logwarn(
            "drone policy ready; publish_setpoints defaults false, hard speed limit %.2f m/s",
            HARD_SPEED_LIMIT_MPS,
        )

    def _sync_callback(self, depth, odom) -> None:
        with self.lock:
            self.latest = (depth, odom)

    def _goal_callback(self, msg) -> None:
        goal = np.asarray([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if not np.isfinite(goal).all():
            self.rospy.logerr("rejected non-finite goal")
            return
        with self.lock:
            changed = self.goal is None or self.goal_frame != msg.header.frame_id or not np.allclose(self.goal, goal)
            self.goal = goal.astype(np.float32)
            self.goal_frame = msg.header.frame_id
            if changed:
                self.goal_version += 1
                self.hidden = None

    def _reject(self, reason: str) -> None:
        self.hidden = None
        self.rospy.logerr_throttle(2.0, "no setpoint: %s", reason)

    def _tick(self, _event) -> None:
        with self.lock:
            pair = self.latest
            goal = None if self.goal is None else self.goal.copy()
            goal_frame = self.goal_frame
            goal_version = self.goal_version
            hidden = self.hidden
        if pair is None or goal is None:
            self.rospy.logwarn_throttle(5.0, "waiting for synchronized depth/odom and goal")
            return
        depth_msg, odom = pair
        image_ns = depth_msg.header.stamp.to_nsec()
        if image_ns == self.last_image_ns:
            return
        if self.last_image_ns is not None and image_ns < self.last_image_ns:
            self.last_image_ns = image_ns
            self.last_command_stamp = None
            self._reject("depth time moved backwards")
            return
        self.last_image_ns = image_ns
        try:
            now = self.rospy.Time.now()
            age = (now - depth_msg.header.stamp).to_sec()
            sync_error = abs((depth_msg.header.stamp - odom.header.stamp).to_sec())
            if age < -0.05 or age > self.max_input_age:
                raise ValueError(f"depth age {age:.3f}s exceeds limit")
            if sync_error > self.sync_slop:
                raise ValueError(f"depth/odom delta {sync_error:.3f}s exceeds limit")
            if goal_frame != odom.header.frame_id:
                raise ValueError(f"goal frame {goal_frame!r} != odom frame {odom.header.frame_id!r}")
            expected_body = self.rospy.get_param("~body_frame", "base_link")
            expected_depth = self.rospy.get_param("~depth_frame", "camera_depth_optical_frame")
            if odom.child_frame_id != expected_body:
                raise ValueError(f"odom child frame {odom.child_frame_id!r} != {expected_body!r}")
            if depth_msg.header.frame_id != expected_depth:
                raise ValueError(f"depth frame {depth_msg.header.frame_id!r} != {expected_depth!r}")

            if self.last_command_stamp is not None:
                gap = (depth_msg.header.stamp - self.last_command_stamp).to_sec()
                if gap <= 0.0:
                    raise ValueError("non-increasing depth time")
                if gap > self.max_input_age:
                    hidden = None

            depth, valid_fraction = decode_depth_image(
                depth_msg, float(self.rospy.get_param("~depth_scale", ROS_DEPTH_SCALE))
            )
            minimum_valid = float(self.rospy.get_param("~min_valid_depth_fraction", 0.5))
            if not 0.0 <= minimum_valid <= 1.0:
                raise ValueError("min_valid_depth_fraction must be in 0..1")
            if valid_fraction < minimum_valid:
                raise ValueError(f"valid depth fraction {valid_fraction:.3f} is too low")

            pose = odom.pose.pose
            rotation = quaternion_to_rotation(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
            body_velocity = np.asarray(
                [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z],
                dtype=np.float32,
            )
            velocity = rotation @ body_velocity
            position = np.asarray(
                [pose.position.x, pose.position.y, pose.position.z], dtype=np.float32
            )
            requested_speed = float(self.rospy.get_param("~max_speed", HARD_SPEED_LIMIT_MPS))
            applied_speed = speed_limit(requested_speed)
            margin = float(self.rospy.get_param("~margin", 0.6))
            if not 0.3 <= margin <= 0.8:
                raise ValueError("margin must stay in the trained 0.3..0.8 m range")

            tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                acceleration, _, _, next_hidden = self.policy.infer(
                    tensor(depth).unsqueeze(0),
                    tensor(rotation).unsqueeze(0),
                    tensor(velocity).unsqueeze(0),
                    tensor(goal - position).unsqueeze(0),
                    tensor([margin]),
                    tensor([[applied_speed]]),
                    tensor([[1.0]]),
                    hidden,
                    depth_noise_std=0.0,
                )
            acceleration = acceleration[0].cpu().numpy()
            if not np.isfinite(acceleration).all():
                raise ValueError("model produced NaN/Inf")
            dt = 1.0 / self.rate
            if self.last_command_stamp is not None:
                dt = min(max((depth_msg.header.stamp - self.last_command_stamp).to_sec(), 1e-3), 0.2)
            command = velocity_setpoint(
                velocity,
                acceleration,
                dt,
                requested_speed,
                float(self.rospy.get_param("~max_acceleration", HARD_ACCEL_LIMIT_MPS2)),
            )
            with self.lock:
                if goal_version != self.goal_version:
                    self.hidden = None
                    return
                self.hidden = next_hidden
            self.last_command_stamp = depth_msg.header.stamp

            if bool(self.rospy.get_param("~publish_setpoints", False)):
                msg = self.PositionTarget()
                msg.header.stamp = now
                msg.header.frame_id = odom.header.frame_id
                msg.coordinate_frame = self.PositionTarget.FRAME_LOCAL_NED
                msg.type_mask = (
                    self.PositionTarget.IGNORE_PX
                    | self.PositionTarget.IGNORE_PY
                    | self.PositionTarget.IGNORE_PZ
                    | self.PositionTarget.IGNORE_AFX
                    | self.PositionTarget.IGNORE_AFY
                    | self.PositionTarget.IGNORE_AFZ
                    | self.PositionTarget.IGNORE_YAW
                    | self.PositionTarget.IGNORE_YAW_RATE
                )
                msg.velocity.x, msg.velocity.y, msg.velocity.z = map(float, command)
                self.publisher.publish(msg)
            self.rospy.loginfo_throttle(
                5.0,
                "depth_valid=%.1f%% max_speed=%.2f command=[%.3f %.3f %.3f] publish=%s",
                100.0 * valid_fraction,
                applied_speed,
                *command,
                bool(self.rospy.get_param("~publish_setpoints", False)),
            )
        except Exception as exc:
            self._reject(str(exc))


def self_test() -> None:
    from types import SimpleNamespace

    raw = np.full((480, 848), 2000, dtype="<u2")
    raw[0, 104] = 0
    msg = SimpleNamespace(
        encoding="16UC1", is_bigendian=0, height=480, width=848, step=1696, data=raw.tobytes()
    )
    depth, valid = decode_depth_image(msg)
    assert depth.shape == (48, 64) and depth.dtype == np.float32
    assert depth[0, 0] == -1.0 and np.isclose(depth[20, 20], 2.0) and valid > 0.99
    rotation = quaternion_to_rotation(0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))
    np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-6)
    command = velocity_setpoint(np.array([0.49, 0.0, 0.0]), np.array([10.0, 0.0, 0.0]), 0.1, 2.0, 3.0)
    assert np.linalg.norm(command) <= HARD_SPEED_LIMIT_MPS + 1e-6
    print("ros_compat self-test passed")


def main() -> None:
    if "--self-test" in sys.argv:
        self_test()
        return
    import rospy

    rospy.init_node("drone_ros_compat")
    Ros1MavrosAdapter()
    rospy.spin()


if __name__ == "__main__":
    main()
