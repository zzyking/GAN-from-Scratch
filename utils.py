import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image

# CIFAR-10 dataloader
def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_conditional_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.classes

# 移动平均便于判断loss变化趋势
def time_weighted_ema(data, alpha):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema

# 可视化
def visualize(g_losses, d_losses, save_dir='loss_curve.png'):

    # 平滑参数
    alpha = 0.005
    g_losses_smoothed = time_weighted_ema(g_losses, alpha)
    d_losses_smoothed = time_weighted_ema(d_losses, alpha)

    # 创建图表
    plt.figure(figsize=(10, 5))
    fig, ax1 = plt.subplots()

    # 绘制生成器损失
    ax1.plot(g_losses, color='#FDB462', alpha=0.3, label='Generator Loss (Actual)')
    ax1.plot(g_losses_smoothed, color='#D95F02', linewidth=2, label='Generator Loss (Smoothed)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Generator Loss', color='#D95F02')
    ax1.tick_params(axis='y', labelcolor='#D95F02')

    # 绘制判别器损失
    ax2 = ax1.twinx()
    ax2.plot(d_losses, color='#B3B3FF', alpha=0.3, label='Discriminator Loss (Actual)')
    ax2.plot(d_losses_smoothed, color='#7570B3', linewidth=2, label='Discriminator Loss (Smoothed)')
    ax2.set_ylabel('Discriminator Loss', color='#7570B3')
    ax2.tick_params(axis='y', labelcolor='#7570B3')

    # 添加标题和图例
    plt.title('Generator and Discriminator Loss Over Epochs')
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # 保存和显示图像
    plt.savefig(save_dir)
    plt.show()