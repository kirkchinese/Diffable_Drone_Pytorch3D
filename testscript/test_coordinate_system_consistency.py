import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_coordinate_system_consistency():
    from drone_renderer import DroneRenderer, transform_pos_ros2pt3d, transform_rot_ros2pt3d

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pos_ros = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    pos_pt3d = transform_pos_ros2pt3d(pos_ros)
    expected_pos = torch.tensor([[-1.0, 3.0, 2.0]], device=device)
    assert torch.allclose(pos_pt3d, expected_pos, atol=1e-6), \
        f'ENU→PyTorch3D 坐标映射错误: {pos_pt3d.tolist()} != {expected_pos.tolist()}'
    print('[PASS] ENU → PyTorch3D 位置映射正确')

    R_ros = torch.eye(3, device=device).unsqueeze(0)
    R_pt3d = transform_rot_ros2pt3d(R_ros)
    expected_rot = torch.tensor([
        [[-1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, 1.0, 0.0]]
    ], device=device)
    assert torch.allclose(R_pt3d, expected_rot, atol=1e-6), \
        f'FLU/ENU → PyTorch3D 旋转映射错误: {R_pt3d.tolist()} != {expected_rot.tolist()}'
    print('[PASS] FLU/ENU → PyTorch3D 旋转映射正确')

    renderer = DroneRenderer(
        mesh_path='./data/sample/sample4.obj',
        device=device,
        image_size=(48, 64),
        focal_length=32.0,
        num_samples=1000,
        subdivide_times=0,
    )
    R_view, T_view = renderer.compute_view_matrix(
        p_ros=torch.zeros(1, 3, device=device),
        R_ros=R_ros,
        camera_pitch_deg=0.0,
        cam_offset_body=[0.1, 0.0, 0.0],
    )
    camera_center = -torch.bmm(R_view, T_view.unsqueeze(-1)).squeeze(-1)
    expected_center = torch.tensor([[-0.1, 0.0, 0.0]], device=device)
    assert torch.allclose(camera_center, expected_center, atol=1e-5), \
        f'相机安装位置错误: {camera_center.tolist()} != {expected_center.tolist()}'
    print('[PASS] 相机位置与文档一致: Body X=0.10, Y=0, Z=0')


if __name__ == '__main__':
    test_coordinate_system_consistency()
