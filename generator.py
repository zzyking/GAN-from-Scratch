import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

# DataLoader for CIFAR-10
def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Generator for GAN
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

# Discriminator for GAN
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

# Training GAN
def train_gan(dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device, epochs):
    generator.train()
    discriminator.train()

    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
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

            # Train Generator
            g_optimizer.zero_grad()
            g_loss = criterion(discriminator(fake_images), real_labels)
            g_loss.backward()
            g_optimizer.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # Save some generated images for visualization
        if (epoch + 1) % 10 == 0:
            if os.path.exists('output') == False:
                os.makedirs('output')
            save_image(fake_images[:25], f'output/epoch_{epoch+1}.png', nrow=5, normalize=True)
    return g_losses, d_losses

# time-weighted EMA
def time_weighted_ema(data, alpha):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    noise_dim = 100
    img_channels = 3
    epochs = 200
    lr = 0.00002

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare DataLoader
    dataloader = get_dataloader(batch_size)

    # Initialize models
    generator = Generator(noise_dim, img_channels).to(device)
    discriminator = Discriminator(img_channels).to(device)

    # Loss and optimizers
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # Train GAN
    g_losses, d_losses = train_gan(dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device, epochs)

    # Visualize losses

    # 平滑参数
    alpha = 0.02
    g_losses_smoothed = time_weighted_ema(g_losses, alpha)
    d_losses_smoothed = time_weighted_ema(d_losses, alpha)

    # 创建图表
    plt.figure(figsize=(10, 5))
    fig, ax1 = plt.subplots()

    # 绘制生成器损失
    ax1.fill_between(iterations, g_losses, color='#FDB462', alpha=0.3, label='Generator Loss (Actual)')  # 浅橙色阴影
    ax1.plot(iterations, g_losses_smoothed, color='#D95F02', linewidth=2, label='Generator Loss (Smoothed)')  # 橙色主线
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Generator Loss', color='#D95F02')  # 橙色标签
    ax1.tick_params(axis='y', labelcolor='#D95F02')

    # 绘制判别器损失
    ax2 = ax1.twinx()
    ax2.fill_between(iterations, d_losses, color='#B3B3FF', alpha=0.3, label='Discriminator Loss (Actual)')  # 浅紫色阴影
    ax2.plot(iterations, d_losses_smoothed, color='#7570B3', linewidth=2, label='Discriminator Loss (Smoothed)')  # 紫色主线
    ax2.set_ylabel('Discriminator Loss', color='#7570B3')  # 紫色标签
    ax2.tick_params(axis='y', labelcolor='#7570B3')

    # 添加标题和图例
    plt.title('Generator and Discriminator Loss Over Epochs')
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # 保存和显示图像
    plt.savefig('loss_curve.png')
    plt.show()