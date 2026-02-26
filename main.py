import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pickle  # Python对象序列化（保存/加载数据）
import yaml  # 读取YAML配置文件
import shutil  # 文件操作（复制、移动等）
import torch  # PyTorch深度学习框架
import numpy as np

from timeit import default_timer
from einops import rearrange  # 张量重排（维度变换）的简洁语法
from torch.utils.data import TensorDataset, DataLoader
from libs.model import Model
from libs.utils import *

# 设置随机种子（确保实验可重复）
def set_random_seed(seed):
    np.random.seed(seed)  # NumPy随机种子
    torch.manual_seed(seed)  # PyTorch CPU随机种子
    torch.cuda.manual_seed(seed)  # PyTorch GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 所有GPU的随机种子
    
set_random_seed(1234)


# -------------------------- parameters --------------------------
# read the parameters file
with open(os.path.join('configs', 'IFactFormer.yml'), 'r') as f:
    config = yaml.safe_load(f)
config = dict2namespace(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create save_dirs
i = 0
save_dirs = os.path.join(config.log_dir, f'dim{config.model.dim}_depth{config.model.depth}_iter{config.model.n_layer}_T{config.model.in_time_window}_{i}')
while os.path.exists(save_dirs):
    i += 1
    save_dirs = os.path.join(config.log_dir, f'dim{config.model.dim}_depth{config.model.depth}_iter{config.model.n_layer}_T{config.model.in_time_window}_{i}')
os.makedirs(save_dirs)
print(save_dirs)

# copy the parameters file
shutil.copyfile('configs/IFactFormer.yml', f'{save_dirs}/IFactFormer.yml')


# -------------------------- data_generation --------------------------
# data
# vor_data: b,t,nx,ny,nz,c, vor_data size is (21,400,32,33,16,4)
data = np.load('../data/fDNS_data/Retau180_dim21x400x32x33x16x4_dT200.npy')
data = data[0:20, ..., 0:3]
data = torch.from_numpy(data).float()

input_list = []
output_list = []

b, t, nx, ny, nz, c = data.shape
in_time_window, out_time_window = config.model.in_time_window, 1
sample_num = t - (in_time_window + out_time_window)

for i in range(b):
    for j in range(sample_num):
        input_list.append(data[i, j: j + in_time_window, ...])
        output_list.append(data[i, j + in_time_window: j + in_time_window + out_time_window, ...])

input_data = torch.stack(input_list) # input_data: b t nx ny nz c
output_data = torch.stack(output_list) # output_data: b t nx ny nz c

output_data = rearrange(output_data, 'b 1 nx ny nz c -> b nx ny nz c')

ntrain = int(0.8 * input_data.shape[0])
ntest = input_data.shape[0] - ntrain
train_x = input_data[:ntrain]
train_y = output_data[:ntrain]
test_x = input_data[ntrain:]
test_y = output_data[ntrain:]

norm = {}
norm['x_mean'] = torch.mean(train_x, dim=(0,1,2,3,4)).to(device)
norm['x_std'] = torch.std(train_x, dim=(0,1,2,3,4)).to(device)
norm['y_mean'] = torch.mean(train_y, dim=(0,1,2,3)).to(device)
norm['y_std'] = torch.std(train_y, dim=(0,1,2,3)).to(device)

energy = torch.sum(data**2, dim=(2,3,4))
delta_energy = energy[:,1:,:] - energy[:,:-1,:]
delta_energy_max = torch.max(delta_energy.reshape(-1, 3), dim=0).values.to(device)
delta_energy_min = torch.min(delta_energy.reshape(-1, 3), dim=0).values.to(device)
norm['delta_energy_max'] = delta_energy_max
norm['delta_energy_min'] = delta_energy_min
with open('norm.pkl', 'wb') as f:
    pickle.dump(norm, f)

size_lst = [(32,33,16)]
length = [4*np.pi, 2, 4*np.pi/3]
pos_lst = get_pos_lst(size_lst, length)
pos_lst = pos_lst[0]

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=config.training.batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=config.training.batch_size, shuffle=False)


# -------------------------- define --------------------------
model = Model(config.model).to(device)

info = f'count_params: {count_params(model)}'
print(info)
with open(os.path.join(save_dirs, 'training_epoch_log.txt'), 'a') as file:
    file.write(info + '\n')

optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.scheduler_step, gamma=config.training.scheduler_gamma)

loss_fn = LpLoss(reduction=False)
# --- 修改处 1: 初始化物理损失 ---
# 这里的 spacing 建议根据 length 和 size_lst 计算得出
# length = [4*np.pi, 2, 4*np.pi/3], size_lst = [(32,33,16)]
dx = length[0] / (size_lst[0][0] - 1)
dy = length[1] / (size_lst[0][1] - 1)
dz = length[2] / (size_lst[0][2] - 1)

from libs.utils import ContinuityLoss 
phys_loss_fn = ContinuityLoss(dx=dx, dy=dy, dz=dz).to(device)

# 设置物理损失的权重 λ (建议起步设为 0.01 或 0.1)
lambda_phys = 0.1 
# -----------------------------------------------------------

train_loss_lst = []
test_loss_lst = []
min_test_loss = 1

# -------------------------- training --------------------------
t0 = default_timer()
for ep in range(config.training.epochs):
    model.train()
    t1 = default_timer()
    for n_iter, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        y_pred = model((x - norm['x_mean']) / norm['x_std'], pos_lst) * norm['y_std'] + norm['y_mean']
        
        # --- 修改处 2: 计算混合损失 ---
        lambda_phys = 100.0  # 你手动设置的初始值
        # 数据驱动损失 (L2)
        loss_data = torch.mean(loss_fn(y_pred, y))
        
        # 物理约束损失 (散度损失)
        # 注意：y_pred 形状应为 (b, nx, ny, nz, 3)，代表 u,v,w
        loss_phys = phys_loss_fn(y_pred)
        
        # 总损失 = 数据损失 + λ * 物理损失
        train_loss = loss_data + lambda_phys * loss_phys
        # 计算贡献比例（用于监控）
        data_contribution = loss_data / train_loss
        phys_contribution = (lambda_phys * loss_phys) / train_loss
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if n_iter % 100 == 0:
            # 建议打印出两个损失的比例，观察谁占据主导
            print(f"Data权重: {data_contribution:.2%}, Phys权重: {phys_contribution:.2%}")
            print(f'Epoch:{ep} Iter:{n_iter} Total:{train_loss.item():.4f} Data:{loss_data.item():.4f} Phys:{loss_phys.item():.6f}')
        
    train_loss_lst.append(train_loss.item())
    scheduler.step()  
    
    with torch.no_grad():
        L2 = []
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            y_pred = model((x - norm['x_mean']) / norm['x_std'], pos_lst) * norm['y_std'] + norm['y_mean']
            
            test_L2 = loss_fn(y_pred, y)
            L2.append(test_L2.cpu())
            
        L2 = torch.mean(torch.cat(L2, dim=0), dim=0)
        
    t2 = default_timer()
    
    info = f'{ep} {(t2-t1):.2f} L2:{L2.item():.4f}'
    print(info)
    with open(os.path.join(save_dirs, 'training_epoch_log.txt'), 'a') as file:
        file.write(info + '\n')
        
    if min_test_loss > L2:
        min_test_loss = L2
        torch.save(model.state_dict(), f'{save_dirs}/checkpoint_best.pth')   
    torch.save(model.state_dict(), f'{save_dirs}/checkpoint_{ep}.pth')