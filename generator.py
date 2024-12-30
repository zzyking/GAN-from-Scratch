import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import numpy as np

from model import Generator, CNNGenerator, Discriminator, CNNDiscriminator
from utils import (
    get_dataloader,
    time_weighted_ema,
    visualize
)

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

    '''
    dataloader, class_names = get_conditional_dataloader(batch_size)

    num_classes = len(class_names)
    generator = ConditionalGenerator(noise_dim, img_channels, num_classes).to(device)
    discriminator = ConditionalDiscriminator(img_channels, num_classes).to(device)

    g_losses, d_losses = train_conditional_gan(
        dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device, num_classes, epochs
    )
    '''

    visualize(g_losses, d_losses, save_dir='loss_curve.png')