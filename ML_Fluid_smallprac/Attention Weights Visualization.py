import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_attention(attention_weights, source_tokens, target_tokens):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, 
                xticklabels=source_tokens,
                yticklabels=target_tokens,
                cmap="YlGnBu")
    plt.xlabel("Source Tokens")
    plt.ylabel("Target Tokens")
    plt.title("Attention Weights Visualization")
    plt.show()

# 示例使用
source = ["The", "cat", "sat", "on", "the", "mat"]
target = ["Le", "chat", "s'est", "assis", "sur", "le", "tapis"]
attention = torch.rand(7, 6)  # 模拟的注意力权重
plot_attention(attention, source, target)