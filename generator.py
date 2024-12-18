import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import numpy as np

# CIFAR-10 dataloader
def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# GAN 生成器
class Generator(nn.Module):
    def __init__(self, noise_dim, img_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_channels * 32 * 32),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), 3, 32, 32)

class CNNGenerator(nn.Module):
    def __init__(self, noise_dim, img_channels):
        super(CNNGenerator, self).__init__()
        self.init_size = 4  # 初始特征图大小
        self.l1 = nn.Sequential(
            nn.Linear(noise_dim, 512 * self.init_size * self.init_size)  # 将噪声映射到 4x4 特征图
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 上采样层 1: 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 上采样层 2: 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 上采样层 3: 16x16 -> 32x32
            nn.ConvTranspose2d(128, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, z):
        out = self.l1(z)  # 先通过全连接层
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)  # 重塑为 [batch_size, channels, height, width]
        img = self.conv_blocks(out)
        return img

# GAN 判别器
class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_channels * 32 * 32, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class CNNDiscriminator(nn.Module):
    def __init__(self, img_channels):
        super(CNNDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # 输入: 32x32 → 下采样: 16x16
            nn.Conv2d(img_channels, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 下采样: 16x16 → 8x8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 下采样: 8x8 → 4x4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 全连接层输出概率
            nn.Flatten(),
            nn.Linear(4 * 4 * 512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 训练 GAN
def train_gan(dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device, epochs):
    generator.train()
    discriminator.train()

    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # 训练判别器
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_images = generator(noise)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            d_optimizer.zero_grad()
            real_loss = criterion(discriminator(real_images), real_labels)
            fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()
            g_loss = criterion(discriminator(fake_images), real_labels)
            g_loss.backward()
            g_optimizer.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # 每 10 epochs 保存一组生成图片
        if (epoch + 1) % 10 == 0:
            if os.path.exists('output') == False:
                os.makedirs('output')
            save_image(fake_images[:25], f'output/epoch_{epoch+1}.png', nrow=5, normalize=True)
    return g_losses, d_losses

# 移动平均便于判断loss变化趋势
def time_weighted_ema(data, alpha):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema

if __name__ == "__main__":

    batch_size = 64
    noise_dim = 100
    img_channels = 3
    epochs = 100
    g_lr = 0.0002
    d_lr = 0.00005

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = get_dataloader(batch_size)

    generator = CNNGenerator(noise_dim, img_channels).to(device)
    discriminator = CNNDiscriminator(img_channels).to(device)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=g_lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)

    g_losses, d_losses = train_gan(dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device, epochs)

    # 可视化

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
    plt.savefig('loss_curve.png')
    plt.show()