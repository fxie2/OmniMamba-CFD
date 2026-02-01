import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from torch.utils.checkpoint import checkpoint

class SS3DScan(nn.Module):
    """
    [VMamba 思想] SS3D: 6方向扫描 (X/Y/Z 的 Forward + Backward)
    [SegMamba 思想] Gating: 自适应融合不同方向的特征
    """
    def __init__(self, d_model, d_state=64, d_head=64, expand=2, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # 为了节省显存和参数，复用 Mamba 核心，或者定义三个独立核心
        # 这里定义三个独立核心处理不同维度，以捕捉不同维度的特定物理特征
        self.stream_mamba = Mamba2(d_model=d_model, d_state=d_state, d_head=d_head, expand=expand)
        self.normal_mamba = Mamba2(d_model=d_model, d_state=d_state, d_head=d_head, expand=expand)
        self.span_mamba   = Mamba2(d_model=d_model, d_state=d_state, d_head=d_head, expand=expand)
        
        # SegMamba 风格的门控权重生成器
        self.gate_fc = nn.Linear(d_model, 6) # 6个方向的权重

    def forward(self, x):
        # x input: [B, D, H, W, C] (注意：这里输入已经是 Channel Last)
        B, D, H, W, C = x.shape
        
        # --- 1. 数据重排与反转 (VMamba) ---
        # Streamwise (X) - D dimension
        x_s_fwd = x.reshape(B, -1, C) # [B, L, C]
        x_s_bwd = torch.flip(x_s_fwd, dims=[1])
        
        # Wall-normal (Y) - H dimension (permute H to first spatial dim)
        x_n = x.permute(0, 2, 1, 3, 4) # [B, H, D, W, C]
        x_n_fwd = x_n.reshape(B, -1, C)
        x_n_bwd = torch.flip(x_n_fwd, dims=[1])
        
        # Spanwise (Z) - W dimension
        x_z = x.permute(0, 3, 1, 2, 4) # [B, W, D, H, C]
        x_z_fwd = x_z.reshape(B, -1, C)
        x_z_bwd = torch.flip(x_z_fwd, dims=[1])
        
        # --- 2. Mamba 扫描 ---
        def run_mamba(layer, inp):
            return layer(inp)

        # Streamwise
        out_s_fwd = self.stream_mamba(x_s_fwd).view(B, D, H, W, C)
        out_s_bwd = self.stream_mamba(x_s_bwd)
        out_s_bwd = torch.flip(out_s_bwd, dims=[1]).view(B, D, H, W, C)
        
        # Wall-normal (还原维度)
        out_n_fwd = self.normal_mamba(x_n_fwd).view(B, H, D, W, C).permute(0, 2, 1, 3, 4)
        out_n_bwd = self.normal_mamba(x_n_bwd)
        out_n_bwd = torch.flip(out_n_bwd, dims=[1]).view(B, H, D, W, C).permute(0, 2, 1, 3, 4)
        
        # Spanwise (还原维度)
        out_z_fwd = self.span_mamba(x_z_fwd).view(B, W, D, H, C).permute(0, 2, 3, 1, 4)
        out_z_bwd = self.span_mamba(x_z_bwd)
        out_z_bwd = torch.flip(out_z_bwd, dims=[1]).view(B, W, D, H, C).permute(0, 2, 3, 1, 4)

        # --- 3. 门控融合 (SegMamba) ---
        # 堆叠结果: [B, 6, D, H, W, C]
        stack_out = torch.stack([out_s_fwd, out_s_bwd, out_n_fwd, out_n_bwd, out_z_fwd, out_z_bwd], dim=1)
        
        # 计算 Pixel-wise Gate
        # x: [B, D, H, W, C]
        gates = F.softmax(self.gate_fc(x), dim=-1) # [B, D, H, W, 6]
        gates = gates.unsqueeze(-1) # [B, D, H, W, 6, 1] -> 广播到 C
        gates = gates.permute(0, 4, 1, 2, 3, 5) # [B, 6, D, H, W, 1]
        
        # 加权求和
        out = (stack_out * gates).sum(dim=1) # [B, D, H, W, C]
        
        return out

class ChannelAttention3D(nn.Module):
    """
    [MambaIR 思想] Channel Attention: 强化物理量耦合
    """
    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(num_feat, num_feat // squeeze_factor, 1, padding=0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(num_feat // squeeze_factor, num_feat, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class OmniMambaBlock(nn.Module):
    def __init__(self, dim, d_state=64, use_checkpoint=False):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.use_checkpoint = use_checkpoint
        
        # 分支 1: 全局 SS3D
        self.ss3d_mamba = SS3DScan(dim, d_state=d_state, use_checkpoint=use_checkpoint)
        
        # 分支 2: 局部卷积 (Depthwise)
        self.local_conv = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv3d(dim, dim, 1),
            nn.SiLU()
        )
        
        # 后处理: 通道注意力
        self.ca = ChannelAttention3D(dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x input: [B, C, D, H, W]
        residual = x
        
        # 调整为 Channel Last 以适配 Mamba 和 LayerNorm
        x_perm = x.permute(0, 2, 3, 4, 1) # [B, D, H, W, C]
        x_norm = self.norm(x_perm)
        
        # 分支 1 (Mamba 需要 Channel Last)
        if self.use_checkpoint and x.requires_grad:
             global_feat = checkpoint(self.ss3d_mamba, x_norm)
        else:
             global_feat = self.ss3d_mamba(x_norm)
        
        global_feat = global_feat.permute(0, 4, 1, 2, 3) # Back to [B, C, D, H, W]
        
        # 分支 2 (Conv 需要 Channel First)
        # 这里我们需要先把 x_norm 变回 Channel First 传给 Conv
        x_norm_cf = x_norm.permute(0, 4, 1, 2, 3)
        local_feat = self.local_conv(x_norm_cf)
        
        # 融合
        fused = self.alpha * global_feat + (1 - self.alpha) * local_feat
        
        # 注意力修正
        out = self.ca(fused)
        
        return residual + out
