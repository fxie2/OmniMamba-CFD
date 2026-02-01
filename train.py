import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from config import Config
from models.network import OmniMambaCFDSolver
from utils.physics_loss import PhysicsLoss

def train():
    # 0. 准备环境
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on {device}...")

    # 1. 加载数据
    if not os.path.exists(Config.TRAIN_VLES):
        print(f"Error: Data not found at {Config.TRAIN_VLES}. Run preprocess.py first.")
        return

    print("Loading datasets...")
    vles = torch.from_numpy(np.load(Config.TRAIN_VLES))
    dns = torch.from_numpy(np.load(Config.TRAIN_DNS))
    bc = torch.from_numpy(np.load(Config.TRAIN_BC))
    
    dataset = TensorDataset(vles, bc, dns)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 2. 初始化模型
    model = OmniMambaCFDSolver(
        in_vars=Config.IN_VARS, 
        bc_vars=Config.BC_VARS, 
        embed_dim=Config.EMBED_DIM, 
        num_blocks=Config.NUM_BLOCKS,
        d_state=Config.D_STATE
    ).to(device)
    
    criterion = PhysicsLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # 3. 训练循环
    min_loss = float('inf')
    save_path = os.path.join(Config.CHECKPOINT_DIR, Config.MODEL_NAME)

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_vles, batch_bc, batch_dns in loader:
            batch_vles = batch_vles.to(device)
            batch_bc = batch_bc.to(device)
            batch_dns = batch_dns.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            pred = model(batch_vles, batch_bc)
            
            # Loss Calculation
            loss = criterion(pred, batch_dns)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {avg_loss:.6f}")
        
        # 保存最佳模型
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model saved to {save_path}")

    print("Training complete.")

if __name__ == "__main__":
    train()
