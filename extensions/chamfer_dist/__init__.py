# # -*- coding: utf-8 -*-
# # @Author: Thibault GROUEIX
# # @Date:   2019-08-07 20:54:24
# # @Last Modified by:   Haozhe Xie
# # @Last Modified time: 2019-12-18 15:06:25
# # @Email:  cshzxie@gmail.com

# import torch

# import chamfer


# class ChamferFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, xyz1, xyz2):
#         dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
#         ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

#         return dist1, dist2

#     @staticmethod
#     def backward(ctx, grad_dist1, grad_dist2):
#         xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
#         grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
#         return grad_xyz1, grad_xyz2


# class ChamferDistanceL2(torch.nn.Module):
#     f''' Chamder Distance L2
#     '''
#     def __init__(self, ignore_zeros=False):
#         super().__init__()
#         self.ignore_zeros = ignore_zeros

#     def forward(self, xyz1, xyz2):
#         batch_size = xyz1.size(0)
#         if batch_size == 1 and self.ignore_zeros:
#             non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
#             non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
#             xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
#             xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

#         dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
#         return (torch.mean(dist1) + torch.mean(dist2))/2.0

# class ChamferDistanceL2_split(torch.nn.Module):
#     f''' Chamder Distance L2
#     '''
#     def __init__(self, ignore_zeros=False):
#         super().__init__()
#         self.ignore_zeros = ignore_zeros

#     def forward(self, xyz1, xyz2):
#         batch_size = xyz1.size(0)
#         if batch_size == 1 and self.ignore_zeros:
#             non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
#             non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
#             xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
#             xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

#         dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
#         return torch.mean(dist1), torch.mean(dist2)

# class ChamferDistanceL1(torch.nn.Module):
#     f''' Chamder Distance L1
#     '''
#     def __init__(self, ignore_zeros=False):
#         super().__init__()
#         self.ignore_zeros = ignore_zeros

#     def forward(self, xyz1, xyz2):
#         batch_size = xyz1.size(0)
#         if batch_size == 1 and self.ignore_zeros:
#             non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
#             non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
#             xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
#             xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

#         dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
#         # import pdb
#         # pdb.set_trace()
#         dist1 = torch.sqrt(dist1)
#         dist2 = torch.sqrt(dist2)
#         return (torch.mean(dist1) + torch.mean(dist2))/2


# class ChamferDistanceUDF(torch.nn.Module):
#     f''' Chamder Distance L1
#     '''
#     def __init__(self, ignore_zeros=False):
#         super().__init__()
#         self.ignore_zeros = ignore_zeros

#     def forward(self, xyz1, xyz2):
#         '''
#         xyz1 to match xyz2
#         '''
#         batch_size = xyz1.size(0)
#         if batch_size == 1 and self.ignore_zeros:
#             non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
#             non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
#             xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
#             xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

#         dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
#         return torch.sqrt(dist1)

# class PatialChamferDistanceL1(torch.nn.Module):
#     f''' Chamder Distance L1
#     '''
#     def __init__(self, ignore_zeros=False):
#         super().__init__()
#         self.ignore_zeros = ignore_zeros

#     def forward(self, xyz1, xyz2):
#         '''
#         xyz1 to match xyz2
#         '''
#         batch_size = xyz1.size(0)
#         if batch_size == 1 and self.ignore_zeros:
#             non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
#             non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
#             xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
#             xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

#         dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
#         dist1 = torch.sqrt(dist1)
#         return torch.mean(dist1)

# class PatialChamferDistanceL2(torch.nn.Module):
#     f''' Chamder Distance L2
#     '''
#     def __init__(self, ignore_zeros=False):
#         super().__init__()
#         self.ignore_zeros = ignore_zeros

#     def forward(self, xyz1, xyz2):
#         '''
#         xyz1 to match xyz2
#         '''
#         batch_size = xyz1.size(0)
#         if batch_size == 1 and self.ignore_zeros:
#             non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
#             non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
#             xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
#             xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

#         dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
#         return torch.mean(dist1)


# # if __name__ == '__main__':
# #     # check ChamferDistanceUDF
# #     gt_udf = ChamferDistanceUDF()
# #     q = query_points.view(B, -1, 3)[0]
# #     k = key_points.view(B, -1, 3)[0]
# #     torch.norm(k - q[0], dim=-1).min() == gt_distance[0][0]

# -*- coding: utf-8 -*-
# @Author: Thibault GROUEIX
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-18 15:06:25
# @Email:  cshzxie@gmail.com

import torch

import chamfer


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, grad_idx1=None, grad_idx2=None):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2, weights1=None, weights2=None):
        """
        Computes weighted Chamfer Distance between two point clouds
        
        Args:
            xyz1, xyz2: Point clouds [B, N, 3]
            weights1, weights2: Point weights [B, N] or None (equal weights)
            
        Returns:
            Average Chamfer distance
        """
        print(f"xyz1: {xyz1.shape}, xyz2: {xyz2.shape}")
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
            
            # Also filter weights if provided
            if weights1 is not None:
                weights1 = weights1[non_zeros1].unsqueeze(dim=0)
            if weights2 is not None:
                weights2 = weights2[non_zeros2].unsqueeze(dim=0)

        # Get raw distances and indices
        dist1, dist2, _, _ = ChamferFunction.apply(xyz1, xyz2)
        
        # Apply weights if provided
        if weights1 is not None:
            # Ensure weights are properly shaped
            if weights1.dim() == 2:  # [B, N]
                dist1 = dist1 * weights1
                print(f"dist1: {dist1.shape}, weights1: {weights1.shape}")
            elif weights1.dim() == 3 and weights1.size(2) == 1:  # [B, N, 1]
                dist1 = dist1 * weights1.squeeze(-1)
                
        if weights2 is not None:
            # Ensure weights are properly shaped
            if weights2.dim() == 2:  # [B, N]
                dist2 = dist2 * weights2
            elif weights2.dim() == 3 and weights2.size(2) == 1:  # [B, N, 1]
                dist2 = dist2 * weights2.squeeze(-1)
        
        # Compute average
        if weights1 is not None:
            dist1_avg = torch.sum(dist1) / torch.sum(weights1)
        else:
            dist1_avg = torch.mean(dist1)
            
        if weights2 is not None:
            dist2_avg = torch.sum(dist2) / torch.sum(weights2)
        else:
            dist2_avg = torch.mean(dist2)
            
        return (dist1_avg + dist2_avg) / 2.0


class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2, weights1=None, weights2=None):
        """
        Computes weighted Chamfer Distance between two point clouds and returns
        separate distances from each point cloud to the other
        
        Args:
            xyz1, xyz2: Point clouds [B, N, 3]
            weights1, weights2: Point weights [B, N] or None (equal weights)
            
        Returns:
            Tuple of distances (dist1_avg, dist2_avg)
        """
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
            
            # Also filter weights if provided
            if weights1 is not None:
                weights1 = weights1[non_zeros1].unsqueeze(dim=0)
            if weights2 is not None:
                weights2 = weights2[non_zeros2].unsqueeze(dim=0)

        # Get raw distances and indices
        dist1, dist2, _, _ = ChamferFunction.apply(xyz1, xyz2)
        
        # Apply weights if provided
        if weights1 is not None:
            # Ensure weights are properly shaped
            if weights1.dim() == 2:  # [B, N]
                dist1 = dist1 * weights1
            elif weights1.dim() == 3 and weights1.size(2) == 1:  # [B, N, 1]
                dist1 = dist1 * weights1.squeeze(-1)
                
        if weights2 is not None:
            # Ensure weights are properly shaped
            if weights2.dim() == 2:  # [B, N]
                dist2 = dist2 * weights2
            elif weights2.dim() == 3 and weights2.size(2) == 1:  # [B, N, 1]
                dist2 = dist2 * weights2.squeeze(-1)
        
        # Compute average
        if weights1 is not None:
            dist1_avg = torch.sum(dist1) / torch.sum(weights1)
        else:
            dist1_avg = torch.mean(dist1)
            
        if weights2 is not None:
            dist2_avg = torch.sum(dist2) / torch.sum(weights2)
        else:
            dist2_avg = torch.mean(dist2)
            
        return dist1_avg, dist2_avg


class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2, weights1=None, weights2=None):
        """
        Computes weighted Chamfer Distance L1 between two point clouds
        
        Args:
            xyz1, xyz2: Point clouds [B, N, 3]
            weights1, weights2: Point weights [B, N] or None (equal weights)
            
        Returns:
            Average Chamfer distance
        """
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
            
            # Also filter weights if provided
            if weights1 is not None:
                weights1 = weights1[non_zeros1].unsqueeze(dim=0)
            if weights2 is not None:
                weights2 = weights2[non_zeros2].unsqueeze(dim=0)

        # Get raw distances
        dist1, dist2, _, _ = ChamferFunction.apply(xyz1, xyz2)
        
        # Apply square root for L1 norm
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)
        
        # Apply weights if provided
        if weights1 is not None:
            # Ensure weights are properly shaped
            if weights1.dim() == 2:  # [B, N]
                dist1 = dist1 * weights1
            elif weights1.dim() == 3 and weights1.size(2) == 1:  # [B, N, 1]
                dist1 = dist1 * weights1.squeeze(-1)
                
        if weights2 is not None:
            # Ensure weights are properly shaped
            if weights2.dim() == 2:  # [B, N]
                dist2 = dist2 * weights2
            elif weights2.dim() == 3 and weights2.size(2) == 1:  # [B, N, 1]
                dist2 = dist2 * weights2.squeeze(-1)
        
        # Compute average
        if weights1 is not None:
            dist1_avg = torch.sum(dist1) / torch.sum(weights1)
        else:
            dist1_avg = torch.mean(dist1)
            
        if weights2 is not None:
            dist2_avg = torch.sum(dist2) / torch.sum(weights2)
        else:
            dist2_avg = torch.mean(dist2)
            
        return (dist1_avg + dist2_avg) / 2


class PatialChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2, weights1=None):
        '''
        xyz1 to match xyz2 with optional weights
        '''
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
            
            # Also filter weights if provided
            if weights1 is not None:
                weights1 = weights1[non_zeros1].unsqueeze(dim=0)

        dist1, dist2, _, _ = ChamferFunction.apply(xyz1, xyz2)
        dist1 = torch.sqrt(dist1)
        
        # Apply weights if provided
        if weights1 is not None:
            # Ensure weights are properly shaped
            if weights1.dim() == 2:  # [B, N]
                dist1 = dist1 * weights1
            elif weights1.dim() == 3 and weights1.size(2) == 1:  # [B, N, 1]
                dist1 = dist1 * weights1.squeeze(-1)
            
            # Weighted average
            return torch.sum(dist1) / torch.sum(weights1)
        else:
            # Regular average
            return torch.mean(dist1)


class PatialChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2, weights1=None):
        '''
        xyz1 to match xyz2 with optional weights
        '''
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
            
            # Also filter weights if provided
            if weights1 is not None:
                weights1 = weights1[non_zeros1].unsqueeze(dim=0)

        dist1, dist2, _, _ = ChamferFunction.apply(xyz1, xyz2)
        
        # Apply weights if provided
        if weights1 is not None:
            # Ensure weights are properly shaped
            if weights1.dim() == 2:  # [B, N]
                dist1 = dist1 * weights1
            elif weights1.dim() == 3 and weights1.size(2) == 1:  # [B, N, 1]
                dist1 = dist1 * weights1.squeeze(-1)
            
            # Weighted average
            return torch.sum(dist1) / torch.sum(weights1)
        else:
            # Regular average
            return torch.mean(dist1)


# Example usage
def test_weighted_chamfer():
    """
    Test weighted Chamfer distance
    """
    # Create two point clouds
    pc1 = torch.randn(2, 1000, 3).cuda()
    pc2 = torch.randn(2, 1000, 3).cuda()
    
    # Create weights for each point
    weights1 = torch.ones(2, 1000).cuda()
    weights1[0, :500] = 0.1  # Lower weight for first 500 points in first batch
    weights1[1, :500] = 10.0  # Higher weight for first 500 points in second batch
    
    weights2 = torch.ones(2, 1000).cuda()
    weights2[0, 500:] = 5.0  # Higher weight for last 500 points in first batch
    
    # Create Chamfer distance modules
    chamfer_l2 = ChamferDistanceL2().cuda()
    chamfer_l1 = ChamferDistanceL1().cuda()
    chamfer_l2_split = ChamferDistanceL2_split().cuda()
    patial_chamfer_l1 = PatialChamferDistanceL1().cuda()
    patial_chamfer_l2 = PatialChamferDistanceL2().cuda()
    
    # Test without weights
    dist1 = chamfer_l2(pc1, pc2)
    print(f"Standard L2: {dist1.item()}")
    
    # Test with weights
    dist2 = chamfer_l2(pc1, pc2, weights1, weights2)
    print(f"Weighted L2: {dist2.item()}")
    
    # Test L2_split with weights
    dist3_1, dist3_2 = chamfer_l2_split(pc1, pc2, weights1, weights2)
    print(f"Weighted L2 split: {dist3_1.item()}, {dist3_2.item()}")
    
    # Test L1 with weights
    dist4 = chamfer_l1(pc1, pc2, weights1, weights2)
    print(f"Weighted L1: {dist4.item()}")
    
    # Test patial L1/L2 with weights
    dist5 = patial_chamfer_l1(pc1, pc2, weights1)
    print(f"Weighted Patial L1: {dist5.item()}")
    
    dist6 = patial_chamfer_l2(pc1, pc2, weights1)
    print(f"Weighted Patial L2: {dist6.item()}")


# Uncomment to test
if __name__ == '__main__':
    test_weighted_chamfer()