import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
latent_dim = 100
batch_size = 64
epochs = 100
lr = 0.0002
num_classes = 10
save_dir = 'generated_images'

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#cgan生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()

        # 标签嵌入层
        self.label_embedding = nn.Embedding(num_classes, 20)  # 将类别嵌入到10维空间

        # 将嵌入后的标签和噪声连接起来的输入维度
        input_dim = latent_dim + 20

        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # 嵌入标签
        label_embedding = self.label_embedding(labels)
        # 调整标签维度以便与噪声连接
        label_embedding = label_embedding.unsqueeze(2).unsqueeze(3)  # [B, 50, 1, 1]
        # 连接噪声和标签
        z = torch.cat([z, label_embedding], dim=1)
        return self.model(z)

#cgan判别器
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        # 标签嵌入层
        self.label_embedding = nn.Embedding(num_classes, 50)

        # 标签投影层
        self.label_proj = nn.Sequential(
            nn.Linear(50, 32 * 32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model = nn.Sequential(
            # 输入通道为3+1（图像3通道 + 条件信息1通道）
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # 处理标签信息
        embedded_labels = self.label_embedding(labels)
        projected_labels = self.label_proj(embedded_labels)
        # 将投影后的标签重塑为条件通道
        projected_labels = projected_labels.view(-1, 1, 32, 32)

        # 将条件信息与图像连接
        x = torch.cat([x, projected_labels], dim=1)
        return self.model(x).view(-1, 1)


# 损失函数
adversarial_loss = nn.BCELoss()

# 初始化生成器和判别器
generator = Generator(latent_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

# 优化器
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

def train_gan(generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, train_loader, epochs, latent_dim, save_dir, device):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            # 配置真实和假标签
            real_labels_d = torch.ones(imgs.size(0), 1).to(device)
            fake_labels_d = torch.zeros(imgs.size(0), 1).to(device)
            
            # 训练判别器
            optimizer_D.zero_grad()
            
            # 真实图像
            real_imgs = imgs.to(device)
            real_labels = labels.to(device)
            real_validity = discriminator(real_imgs, real_labels)
            d_real_loss = adversarial_loss(real_validity, real_labels_d)
            
            # 假图像
            noise = torch.randn(imgs.size(0), latent_dim, 1, 1).to(device)
            fake_imgs = generator(noise, real_labels)
            fake_validity = discriminator(fake_imgs.detach(), real_labels)
            d_fake_loss = adversarial_loss(fake_validity, fake_labels_d)
            
            # 总判别器损失
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # 训练生成器
            optimizer_G.zero_grad()
            # 生成图像
            noise = torch.randn(imgs.size(0), latent_dim, 1, 1).to(device)
            gen_labels = torch.randint(0, num_classes, (imgs.size(0),)).to(device)
            gen_imgs = generator(noise, gen_labels)
            # 判别器判断生成图像
            validity = discriminator(gen_imgs, gen_labels)
            # 生成器损失
            g_loss = adversarial_loss(validity, real_labels_d)
            g_loss.backward()
            optimizer_G.step()
            
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
        
        # 每10个epoch保存一次生成的图片
        if (epoch + 1) % 1 == 0:
            save_generated_images(generator, epoch, latent_dim, save_dir, device)

def save_generated_images(generator, epoch, latent_dim, save_dir, device, num_classes=10, num_images_per_class=5):
    generator.eval()
    images = []
    with torch.no_grad():
        for class_id in range(num_classes):
            noise = torch.randn(num_images_per_class, latent_dim, 1, 1).to(device)
            labels = torch.full((num_images_per_class,), class_id, dtype=torch.long).to(device)
            generated_imgs = generator(noise, labels)
            images.extend(generated_imgs)
        save_image(images, f'{save_dir}/epoch_{epoch+1}.png', nrow=5, normalize=True)
    print(f"Images saved at epoch {epoch + 1}")
    generator.train()

train_gan(generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, train_loader, epochs, latent_dim, save_dir, device)