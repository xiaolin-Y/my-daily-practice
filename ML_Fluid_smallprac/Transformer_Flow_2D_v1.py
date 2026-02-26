import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# 1. 数据生成
# --------------------------------
def generate_data(n_points=2500):
    R = 1.0
    U = 1.0
    raw_coords = (torch.rand(n_points * 2, 2) - 0.5) * 6
    r = torch.norm(raw_coords, dim=1)
    mask = r > R
    coords = raw_coords[mask][:n_points]
    
    x, y = coords[:, 0], coords[:, 1]
    r = torch.norm(coords, dim=1)
    theta = torch.atan2(y, x)
    psi = U * (r - R**2 / r) * torch.sin(theta)#流函数
    return coords, psi.unsqueeze(1)

# --------------------------------
# 2. Transformer 组件
# --------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, return_attn=False):
        batch, seq_len, d_model = x.size()
        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(batch, seq_len, d_model)
        output = self.fc(context)
        return (output, attn) if return_attn else output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, weights = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, weights
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FlowTransformer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=3):
        super().__init__()
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, return_attn_at_layer=-1):
        x = self.coord_encoder(x)
        target_attn = None
        for i, block in enumerate(self.blocks):
            if i == return_attn_at_layer:
                x, target_attn = block(x, return_attn=True)
            else:
                x = block(x)
        out = self.head(x)
        return (out, target_attn) if return_attn_at_layer != -1 else out

# --------------------------------
# 3. 训练逻辑
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coords, psi = generate_data(2500)
train_x, train_y = coords.unsqueeze(0).to(device), psi.unsqueeze(0).to(device)

model = FlowTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
criterion = nn.MSELoss()

for epoch in range(601):
    model.train()
    loss = criterion(model(train_x), train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

# --------------------------------
# 4. 可视化
# --------------------------------
def plot_results(model, coords, psi):
    model.eval()
    with torch.no_grad():
        # 不请求注意力权重，只接收一个返回值
        pred = model(coords.unsqueeze(0).to(device), return_attn_at_layer=-1)
        pred = pred.squeeze(0).cpu().numpy()
        
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(coords[:, 0], coords[:, 1], c=psi.flatten(), cmap='RdBu', s=10)
    ax[0].set_title("True $\psi$")
    ax[1].scatter(coords[:, 0], coords[:, 1], c=pred.flatten(), cmap='RdBu', s=10)
    ax[1].set_title("Transformer Pred")

def plot_attention_map(model, coords, query_pos=[-1.0, 0.0]):
    model.eval()
    with torch.no_grad():
        # 明确请求最后一层 (index=2) 的注意力权重
        _, attn = model(coords.unsqueeze(0).to(device), return_attn_at_layer=2)
        attn_map = attn[0].mean(dim=0).cpu().numpy() 
        
    dist = torch.norm(coords - torch.tensor(query_pos), dim=1)
    q_idx = torch.argmin(dist).item()
    weights = attn_map[q_idx, :]

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=weights, cmap='hot_r', s=25)
    plt.scatter(coords[q_idx, 0], coords[q_idx, 1], c='cyan', marker='*', s=150, label='Query')
    plt.colorbar(label='Attention Weight')
    plt.title(f"Attention Map for {query_pos}")
    plt.show()

plot_results(model, coords, psi)
plot_attention_map(model, coords)