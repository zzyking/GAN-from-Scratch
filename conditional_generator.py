import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import numpy as np

from model import ConditionalDiscriminator, ConditionalGenerator
from utils import (
    get_conditional_dataloader,
    time_weighted_ema,
    visualize
)

# 训练 GAN
def train_conditional_gan(dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device, num_classes, epochs):

    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        for i, (real_images, real_labels) in enumerate(dataloader):
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            batch_size = real_images.size(0)

            # 训练判别器
            real_labels_d = torch.ones(batch_size, 1).to(device)
            fake_labels_d = torch.zeros(batch_size, 1).to(device)

            d_optimizer.zero_grad()
            real_loss = criterion(discriminator(real_images, real_labels), real_labels_d)

            noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
            fake_images = generator(noise, real_labels)

            fake_loss = criterion(discriminator(fake_images.detach(), real_labels), fake_labels_d)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()

            noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
            gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            gen_imgs = generator(noise, gen_labels)
            g_loss = criterion(discriminator(gen_imgs, gen_labels), real_labels_d)
            g_loss.backward()
            g_optimizer.step()

            # 记录损失
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
        # 每隔10个epoch生成并保存图片
        if (epoch + 1) % 1 == 0:
            generator.eval()
            # 创建固定的噪声和标签用于观察训练变化
            fixed_n_sample_per_class = 10
            fixed_total_samples = fixed_n_sample_per_class * 10  # 10类
            fixed_noise = torch.randn(fixed_total_samples, noise_dim, 1, 1).to(device)
            fixed_labels = torch.tensor([[i]*fixed_n_sample_per_class for i in range(10)]).flatten().to(device)
            with torch.no_grad():
                gen_imgs = generator(fixed_noise, fixed_labels)
                # 确保输出目录存在
                if not os.path.exists('output'):
                    os.makedirs('output')
                save_image(gen_imgs.data, f"output/epoch_{epoch + 1}.png", nrow=fixed_n_sample_per_class, normalize=True)
            generator.train()
    
    return g_losses, d_losses



if __name__ == "__main__":
    batch_size = 64
    noise_dim = 100
    g_label_dim = 10
    d_label_dim = 50
    img_channels = 3
    epochs = 100
    g_lr = 0.0002
    d_lr = 0.00003

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader, class_names = get_conditional_dataloader(batch_size)
    num_classes = len(class_names)

    generator = ConditionalGenerator(g_label_dim, noise_dim, img_channels, num_classes).to(device)
    discriminator = ConditionalDiscriminator(d_label_dim, img_channels, num_classes).to(device)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=g_lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)

    g_losses, d_losses = train_conditional_gan(
        dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device, num_classes, epochs
    )

    visualize(g_losses, d_losses, save_dir="loss_curve_conditional.png")