import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import inception_v3
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy

os.environ['TORCH_HOME'] = '/raid_sdi/home/zzy/GAN-from-Scratch'

from model import ConditionalDiscriminator, ConditionalGenerator, rcDiscriminator, rcGenerator
from utils import (
    get_conditional_dataloader,
    get_test_dataloader,
    time_weighted_ema,
    visualize_loss,
    plot_fid_scores
)
from FID import get_inception_features, compute_real_features, calculate_fid

# 训练 GAN
def train_conditional_gan(dataloader, test_dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device, num_classes, epochs, output_dir="output"):

    inception = inception_v3(pretrained=True, transform_input=True).to(device)
    inception.eval()
    inception.fc = nn.Identity()

    g_losses = []
    d_losses = []
    fid_scores = []
    eval_samples = 10000

    # 计算或加载测试集特征
    if not os.path.exists('test_features.npy'):
        real_features = compute_real_features(test_dataloader, inception, device)
        np.save('test_features.npy', real_features)
    else:
        real_features = np.load('test_features.npy')

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
        
        # 每隔5个epoch生成并保存图片
        if (epoch + 1) % 5 == 0:
            generator.eval()
            
            # 创建固定的噪声和标签用于观察训练变化
            fixed_n_sample_per_class = 10
            fixed_total_samples = fixed_n_sample_per_class * 10  # 10类
            fixed_noise = torch.randn(fixed_total_samples, noise_dim, 1, 1).to(device)
            fixed_labels = torch.tensor([[i]*fixed_n_sample_per_class for i in range(10)]).flatten().to(device)
            with torch.no_grad():
                gen_imgs = generator(fixed_noise, fixed_labels)
                # 确保输出目录存在
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                save_image(gen_imgs.data, f"{output_dir}/epoch_{epoch + 1}.png", nrow=fixed_n_sample_per_class, normalize=True)
            
            # 计算 FID
            z = torch.randn(eval_samples, noise_dim, 1, 1, device=device)
            random_labels = torch.randint(0, 10, (eval_samples,)).to(device)
            fake_imgs = generator(z, random_labels)
            generated_features = get_inception_features(fake_imgs, inception, device=device)

            torch.cuda.empty_cache()

            fid_score = calculate_fid(real_features, generated_features)
            fid_scores.append((epoch + 1, fid_score))
            print(f"Epoch {epoch + 1}: FID: {fid_score:.2f}")
            torch.cuda.empty_cache()
            
            generator.train()
        torch.cuda.empty_cache()
    return g_losses, d_losses, fid_scores



if __name__ == "__main__":
    batch_size = 64
    noise_dim = 100
    g_label_dim = 5
    img_channels = 3
    epochs = 150
    g_lr = 0.0002
    d_lr = 0.00003
    output_dir = "test_residual_64_128_256"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader, class_names = get_conditional_dataloader(batch_size)
    test_dataloader = get_test_dataloader(batch_size)
    num_classes = len(class_names)

    generator = rcGenerator(g_label_dim, noise_dim, img_channels, num_classes).to(device)
    discriminator = rcDiscriminator(img_channels, num_classes).to(device)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=g_lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)

    g_losses, d_losses, fid_scores = train_conditional_gan(
        dataloader,test_dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device, num_classes, epochs, output_dir
    )
    
    print(f"Minimum FID Score: {min(fid_scores, key=lambda x: x[1])[1]}, Epoch: {min(fid_scores, key=lambda x: x[1])[0]}")
    visualize_loss(g_losses, d_losses, save_dir=f"{output_dir}/loss_curve_conditional.png")
    plot_fid_scores(fid_scores, save_dir=f"{output_dir}/fid_scores.png")