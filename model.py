import torch
import torch.nn as nn
import os
import numpy as np

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

# GAN 条件生成器
class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim, img_channels, num_classes):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, noise_dim)  # 嵌入标签
        self.init_size = 4
        self.l1 = nn.Sequential(
            nn.Linear(noise_dim * 2, 512 * self.init_size * self.init_size)  # 噪声+标签
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embedding = self.label_embedding(labels)
        input = torch.cat([z, label_embedding], dim=1)  # 拼接噪声和标签
        out = self.l1(input)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# GAN 条件判别器
class ConditionalDiscriminator(nn.Module):
    def __init__(self, img_channels, num_classes):
        super(ConditionalDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, 1 * 32 * 32)
        self.model = nn.Sequential(
            nn.Conv2d(img_channels + 1, 128, 4, 2, 1, bias=False),  # 图像+标签通道
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 4 * 512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # 将标签嵌入为 [batch_size, 1, 32, 32]
        label_embedding = self.label_embedding(labels).view(labels.size(0), 1, 32, 32)
        # 拼接标签嵌入与图像
        input = torch.cat([x, label_embedding], dim=1)
        return self.model(input)