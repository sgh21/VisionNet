import torch
import numpy as np
from extensions.chamfer_dist import ChamferDistanceL2, ChamferDistanceL1

def test_simple_2d_case():
    """
    测试二维点集的Chamfer距离计算
    使用简单的已知距离点集进行验证
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化2D点集A - 2个点，形成一条直线
    points_a = torch.tensor([
        [[0.0, 0.0],
         [1.0, 0.0]]
    ], device=device)  # 形状为 [1, 2, 2]
    
    # 初始化2D点集B - 2个点，也形成一条直线，但向上平移1个单位
    points_b = torch.tensor([
        [[0.0, 1.0],
         [1.0, 1.0]]
    ], device=device)  # 形状为 [1, 2, 2]
    
    # 将2D点转换为3D点（添加z=0）
    points_a_3d = torch.cat([points_a, torch.zeros_like(points_a[:,:,:1])], dim=2)
    points_b_3d = torch.cat([points_b, torch.zeros_like(points_b[:,:,:1])], dim=2)
    
    # 计算预期的Chamfer距离
    # 从A到B: 对于A中的每个点，到B中最近点的距离均为1.0
    # 从B到A: 对于B中的每个点，到A中最近点的距离均为1.0
    # L2距离: 1.0^2 = 1.0
    # L1距离: sqrt(1.0) = 1.0
    expected_l2 = 1.0  # L2距离的平均值
    expected_l1 = torch.sqrt(torch.tensor(expected_l2))  # L1距离的平均值
    
    # 初始化Chamfer距离计算模块
    chamfer_l2 = ChamferDistanceL2()
    chamfer_l1 = ChamferDistanceL1()
    
    # 计算Chamfer距离
    dist_l2 = chamfer_l2(points_a_3d, points_b_3d)
    dist_l1 = chamfer_l1(points_a_3d, points_b_3d)
    
    # 验证结果
    print(f"测试简单2D点集...")
    print(f"L2 Chamfer距离: 计算结果={dist_l2.item():.6f}, 预期结果={expected_l2:.6f}")
    print(f"L1 Chamfer距离: 计算结果={dist_l1.item():.6f}, 预期结果={expected_l1:.6f}")
    print(f"L2测试通过: {abs(dist_l2.item() - expected_l2) < 1e-5}")
    print(f"L1测试通过: {abs(dist_l1.item() - expected_l1) < 1e-5}")
    
    return abs(dist_l2.item() - expected_l2) < 1e-5 and abs(dist_l1.item() - expected_l1) < 1e-5

def test_complex_2d_case():
    """
    测试更复杂的二维点集情况
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化一个正方形点集
    points_a = torch.tensor([
        [[0.0, 0.0],
         [1.0, 0.0],
         [0.0, 1.0],
         [1.0, 1.0]]
    ], device=device)  # 形状为 [1, 4, 2]
    
    # 初始化一个平移后的正方形点集
    points_b = torch.tensor([
        [[2.0, 2.0],
         [3.0, 2.0],
         [2.0, 3.0],
         [3.0, 3.0]]
    ], device=device)  # 形状为 [1, 4, 2]
    
    # 将2D点转换为3D点（添加z=0）
    points_a_3d = torch.cat([points_a, torch.zeros_like(points_a[:,:,:1])], dim=2)
    points_b_3d = torch.cat([points_b, torch.zeros_like(points_b[:,:,:1])], dim=2)
    
    # 计算预期的Chamfer L2距离
    # 对于A中的点(0,0)，到B中最近点(2,2)的距离为sqrt((2-0)^2 + (2-0)^2) = sqrt(8) ≈ 2.83
    # 对于A中的点(1,0)，到B中最近点(2,2)的距离为sqrt((2-1)^2 + (2-0)^2) = sqrt(5) ≈ 2.24
    # 对于A中的点(0,1)，到B中最近点(2,2)的距离为sqrt((2-0)^2 + (2-1)^2) = sqrt(5) ≈ 2.24
    # 对于A中的点(1,1)，到B中最近点(2,2)的距离为sqrt((2-1)^2 + (2-1)^2) = sqrt(2) ≈ 1.41
    # L2距离为上述距离的平方，平均值为 (8 + 5 + 5 + 2) / 4 = 5
    
    # 同理计算B到A的距离，应该是相同的
    # 因此Chamfer L2距离的预期值为5
    
    chamfer_l2 = ChamferDistanceL2()
    chamfer_l1 = ChamferDistanceL1()
    
    dist_l2 = chamfer_l2(points_a_3d, points_b_3d)
    dist_l1 = chamfer_l1(points_a_3d, points_b_3d)
    
    # L1距离是L2距离的平方根
    expected_l2 = 5.0  # 平均平方距离
    expected_l1 = torch.sqrt(torch.tensor(expected_l2))  # 平均距离约为2
    
    print(f"\n测试复杂2D点集...")
    print(f"L2 Chamfer距离: 计算结果={dist_l2.item():.6f}, 预期结果={expected_l2:.6f}")
    print(f"L1 Chamfer距离: 计算结果={dist_l1.item():.6f}, 预期结果={expected_l1:.6f}")
    print(f"L2测试通过: {abs(dist_l2.item() - expected_l2) < 1e-5}")
    print(f"L1测试通过: {abs(dist_l1.item() - expected_l1) < 1e-4}")  # L1使用误差容忍度稍大一些
    
    return abs(dist_l2.item() - expected_l2) < 1e-5 and abs(dist_l1.item() - expected_l1) < 1e-4

def test_batch_2d_case():
    """
    测试批处理模式下的二维点集情况
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建批量数据 - 2个批次
    points_a = torch.tensor([
        # 批次1: 简单点集
        [[0.0, 0.0],
         [1.0, 0.0]],
        # 批次2: 简单点集的变种
        [[0.0, 0.0],
         [2.0, 0.0]]
    ], device=device)  # 形状为 [2, 2, 2]
    
    points_b = torch.tensor([
        # 批次1: 平移的点集
        [[0.0, 1.0],
         [1.0, 1.0]],
        # 批次2: 平移的点集
        [[0.0, 2.0],
         [2.0, 2.0]]
    ], device=device)  # 形状为 [2, 2, 2]
    
    # 将2D点转换为3D点（添加z=0）
    points_a_3d = torch.cat([points_a, torch.zeros_like(points_a[:,:,:1])], dim=2)
    points_b_3d = torch.cat([points_b, torch.zeros_like(points_b[:,:,:1])], dim=2)
    
    chamfer_l2 = ChamferDistanceL2()
    chamfer_l1 = ChamferDistanceL1()
    
    dist_l2 = chamfer_l2(points_a_3d, points_b_3d)
    dist_l1 = chamfer_l1(points_a_3d, points_b_3d)
    
    # 批次1和2的预期距离都是1和2的平均值
    expected_l2 = 2.5  # (1 + 4) / 2 (计算L2)
    expected_l1 = (1+2)/2  # (1 + 1.5) / 2 (计算L1)
    
    print(f"\n测试批处理2D点集...")
    print(f"L2 Chamfer距离: 计算结果={dist_l2.item():.6f}, 预期结果={expected_l2:.6f}")
    print(f"L1 Chamfer距离: 计算结果={dist_l1.item():.6f}, 预期结果={expected_l1:.6f}")
    print(f"L2测试通过: {abs(dist_l2.item() - expected_l2) < 1e-5}")
    print(f"L1测试通过: {abs(dist_l1.item() - expected_l1) < 1e-4}")
    
    return abs(dist_l2.item() - expected_l2) < 1e-5 and abs(dist_l1.item() - expected_l1) < 1e-4

if __name__ == '__main__':
    print("开始测试2D点集的Chamfer距离计算...")
    test_simple = test_simple_2d_case()
    test_complex = test_complex_2d_case()
    test_batch = test_batch_2d_case()
    
    if test_simple and test_complex and test_batch:
        print("\n所有测试都通过了！Chamfer距离在2D点集上计算正确。")
    else:
        print("\n有测试未通过，请检查实现或期望值计算。")