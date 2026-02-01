import torch
import torch.nn as nn

class PhysicsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def gradient_loss(self, pred, target):
        # 计算三个方向的梯度差异，强化小尺度涡结构
        dy_pred = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        dy_target = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
        
        dx_pred = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
        dx_target = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
        
        return self.l1(dy_pred, dy_target) + self.l1(dx_pred, dx_target)

    def forward(self, pred, target):
        # 基础 MSE
        loss_val = self.mse(pred, target)
        # 梯度损失 (权重 0.5)
        loss_grad = self.gradient_loss(pred, target)
        return loss_val + 0.5 * loss_grad
