import torch

def apply_hard_bc(pred, bc_tensor):
    """
    强制修正：
    pred: [B, 5, D, H, W] (rho, u, v, w, T)
    bc_tensor: [B, 3, D, H, W] (y+, T_wall, Mask)
    Mask=1 表示壁面
    """
    # 假设 Mask 在通道 2
    mask = bc_tensor[:, 2:3, ...] > 0.5
    
    # 无滑移条件: u, v, w = 0 (indices 1, 2, 3)
    # 创建一个 mask，扩展到 u,v,w 通道
    uvw_mask = mask.expand(-1, 3, -1, -1, -1)
    pred[:, 1:4, ...].masked_fill_(uvw_mask, 0.0)
    
    # 简单的等温壁面修正 (假设 T_wall 在通道 1)
    # 实际操作中，T 的修正可能更复杂，这里做简化
    t_wall = bc_tensor[:, 1:2, ...]
    pred[:, 4:5, ...].masked_fill_(mask, 0.0)
    pred[:, 4:5, ...] = pred[:, 4:5, ...] + (t_wall * mask.float())
    
    return pred
