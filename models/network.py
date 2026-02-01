import torch
import torch.nn as nn
from .omni_mamba_block import OmniMambaBlock

class OmniMambaCFDSolver(nn.Module):
    def __init__(self, in_vars=5, bc_vars=3, embed_dim=96, num_blocks=6, d_state=64):
        super().__init__()
        
        # 1. 特征融合嵌入
        self.flow_embed = nn.Conv3d(in_vars, embed_dim, 3, 1, 1)
        self.bc_embed = nn.Sequential(
            nn.Conv3d(bc_vars, embed_dim // 2, 1),
            nn.SiLU(),
            nn.Conv3d(embed_dim // 2, embed_dim, 1)
        )
        self.fuse_layer = nn.Conv3d(embed_dim * 2, embed_dim, 1)
        
        # 2. Deep OmniMamba Encoder
        # use_checkpoint=True 可以显著降低显存占用，适合 3D 大数据
        layers = [OmniMambaBlock(embed_dim, d_state=d_state, use_checkpoint=True) for _ in range(num_blocks)]
        self.encoder = nn.Sequential(*layers)
        
        # 3. Decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim * 2, 1),
            nn.PixelShuffle(1), # 这里的 PixelShuffle 在 3D 中通常等价于 Channel 变换或自定义上采样
            # 简化起见，我们直接用 Conv 映射回物理量
            nn.SiLU(),
            nn.Conv3d(embed_dim * 2, in_vars, 3, 1, 1)
        )
        
    def forward(self, vles, bc):
        # vles: [B, 5, D, H, W]
        # bc:   [B, 3, D, H, W]
        
        f = self.flow_embed(vles)
        b = self.bc_embed(bc)
        
        # 融合
        x = torch.cat([f, b], dim=1)
        x = self.fuse_layer(x)
        
        # 核心处理
        feat = self.encoder(x)
        
        # 残差预测
        delta = self.decoder(feat)
        
        return vles + delta
