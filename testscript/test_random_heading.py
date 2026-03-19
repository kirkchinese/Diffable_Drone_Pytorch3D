"""
测试随机初始朝向功能

验证：
1. reset() 后每架无人机有不同的 R 矩阵
2. R 矩阵是合法旋转矩阵 (正交 + det=1)
3. pitch 和 roll 为零 (R[2,2] == 1)
4. 多次 reset 产生不同的朝向
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

def test_random_heading():
    from drone_env import DroneSimulator
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    B = 16
    
    sim = DroneSimulator(
        batch_size=B,
        device=device,
        mesh_path='./data/sample/sample4.obj',
        image_size=(48, 64),
        focal_length=32.0,
        num_samples=1000,
        subdivide_times=0,
    )
    
    # Test 1: R 矩阵在 batch 内不同
    sim.reset()
    R = sim.R.clone()
    yaws = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    assert not torch.allclose(yaws, yaws[0].expand_as(yaws), atol=1e-3), \
        "所有无人机朝向相同，随机化未生效"
    print(f"[PASS] Batch内偏航角不同: {yaws.cpu().numpy()}")
    
    # Test 2: R 是合法旋转矩阵
    identity = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
    RRt = torch.bmm(R, R.transpose(1, 2))
    assert torch.allclose(RRt, identity, atol=1e-5), "R 不是正交矩阵"
    dets = torch.det(R)
    assert torch.allclose(dets, torch.ones(B, device=device), atol=1e-5), "det(R) != 1"
    print("[PASS] R 是合法旋转矩阵")
    
    # Test 3: pitch=roll=0 (R[2,2]=1, R[2,0]=R[2,1]=0)
    assert torch.allclose(R[:, 2, 2], torch.ones(B, device=device), atol=1e-5), "R[2,2] != 1"
    assert torch.allclose(R[:, 2, 0], torch.zeros(B, device=device), atol=1e-5), "R[2,0] != 0"
    assert torch.allclose(R[:, 2, 1], torch.zeros(B, device=device), atol=1e-5), "R[2,1] != 0"
    print("[PASS] pitch=roll=0, 仅 yaw 随机化")
    
    # Test 4: 多次 reset 产生不同结果
    yaws_1 = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    sim.reset()
    yaws_2 = torch.atan2(sim.R[:, 1, 0], sim.R[:, 0, 0])
    assert not torch.allclose(yaws_1, yaws_2, atol=1e-3), \
        "两次 reset 产生相同朝向"
    print("[PASS] 多次 reset 产生不同朝向")
    
    print("\n=== 所有随机朝向测试通过 ===")

if __name__ == '__main__':
    test_random_heading()
