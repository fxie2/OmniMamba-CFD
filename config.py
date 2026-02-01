# config.py

class Config:
    # 数据路径
    DATA_DIR = './data'
    TRAIN_VLES = f'{DATA_DIR}/train_vles.npy'
    TRAIN_DNS = f'{DATA_DIR}/train_dns.npy'
    TRAIN_BC = f'{DATA_DIR}/train_bc.npy'
    
    # 模型保存路径
    CHECKPOINT_DIR = './checkpoints'
    MODEL_NAME = 'omni_mamba_cfd_best.pth'
    
    # 模型超参数 (训练和预测必须一致)
    IN_VARS = 5       # rho, u, v, w, T
    BC_VARS = 3       # y+, T_wall, Mask
    EMBED_DIM = 64    # 嵌入维度
    NUM_BLOCKS = 4    # OmniMamba Block 层数
    D_STATE = 64      # SSM 状态维度
    
    # 训练参数
    BATCH_SIZE = 2
    EPOCHS = 50
    LEARNING_RATE = 5e-4
    DEVICE = 'cuda'   # 'cuda' or 'cpu'
