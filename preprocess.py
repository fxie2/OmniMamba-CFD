import numpy as np
import os

def generate_dummy_data():
    os.makedirs('data', exist_ok=True)
    print("Generating dummy data for testing...")
    
    # 模拟小块数据: Batch=4, Time/Stream=32, Normal=32, Span=32
    # 变量: 5 (rho, u, v, w, T)
    shape = (4, 5, 32, 32, 32)
    
    dns = np.random.randn(*shape).astype(np.float32)
    vles = dns + np.random.normal(0, 0.1, shape).astype(np.float32) # 加噪模拟 vLES
    
    # BC: 3 channels (y+, T_wall, Mask)
    bc = np.zeros((4, 3, 32, 32, 32), dtype=np.float32)
    # 模拟 y 轴两端是壁面
    bc[:, 2, :, 0, :] = 1.0
    bc[:, 2, :, -1, :] = 1.0
    bc[:, 1, :, 0, :] = 300.0 # Wall Temp
    
    np.save('data/train_vles.npy', vles)
    np.save('data/train_dns.npy', dns)
    np.save('data/train_bc.npy', bc)
    print("Data saved to ./data/")

if __name__ == "__main__":
    generate_dummy_data()
