import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from config import Config
from models.network import OmniMambaCFDSolver
from utils.boundary import apply_hard_bc

def predict(input_vles_path, input_bc_path, output_path=None):
    # 0. 准备环境
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(Config.CHECKPOINT_DIR, Config.MODEL_NAME)
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Run train.py first.")
        return

    print(f"Loading model from {model_path}...")

    # 1. 初始化并加载模型
    model = OmniMambaCFDSolver(
        in_vars=Config.IN_VARS, 
        bc_vars=Config.BC_VARS, 
        embed_dim=Config.EMBED_DIM, 
        num_blocks=Config.NUM_BLOCKS,
        d_state=Config.D_STATE
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. 加载输入数据 (支持单个样本或Batch)
    # 这里为了演示，我们从训练数据中取一个切片，实际使用时请替换为新的 .npy 文件路径
    print(f"Loading input data from {input_vles_path}...")
    vles_data = np.load(input_vles_path) # [B, 5, D, H, W]
    bc_data = np.load(input_bc_path)     # [B, 3, D, H, W]
    
    # 转换为 Tensor
    vles_tensor = torch.from_numpy(vles_data).to(device)
    bc_tensor = torch.from_numpy(bc_data).to(device)
    
    # 确保维度正确 (如果是单个样本 [5, D, H, W]，增加 Batch 维度)
    if vles_tensor.ndim == 4:
        vles_tensor = vles_tensor.unsqueeze(0)
        bc_tensor = bc_tensor.unsqueeze(0)

    # 3. 推理 (Inference)
    print("Running inference...")
    with torch.no_grad():
        # 模型预测
        raw_pred = model(vles_tensor, bc_tensor)
        
        # 4. 后处理：应用物理硬约束 (Hard Constraints)
        final_pred = apply_hard_bc(raw_pred, bc_tensor)

    # 5. 保存或可视化结果
    result = final_pred.cpu().numpy()
    
    if output_path:
        np.save(output_path, result)
        print(f"Prediction saved to {output_path}")
    
    # 简单验证输出
    print("\n--- Prediction Statistics ---")
    print(f"Input Shape: {vles_tensor.shape}")
    print(f"Output Shape: {result.shape}")
    
    # 验证壁面速度 (应该接近 0)
    # 假设 Mask=1 处为壁面，检查 u 速度分量
    wall_mask = bc_data[:, 2, ...] > 0.5
    u_velocity = result[:, 1, ...]
    max_wall_u = np.max(np.abs(u_velocity[wall_mask]))
    print(f"Validation: Max U-velocity at wall (BC enforced): {max_wall_u:.6f}")
    
    return result

if __name__ == "__main__":
    # 示例运行：使用训练数据中的第一帧作为测试输入
    # 在实际工程中，这里应该指向新的 .npy 文件
    vles_file = Config.TRAIN_VLES
    bc_file = Config.TRAIN_BC
    
    predict(vles_file, bc_file, output_path='./data/prediction_result.npy')
