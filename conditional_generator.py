import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# DataLoader for CIFAR-10
def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Conditional Generator for GAN
class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim, img_channels, num_classes):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_channels * 32 * 32),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = self.label_embedding(labels)
        x = torch.cat((noise, labels), dim=1)
        x = self.model(x)
        return x.view(x.size(0), 3, 32, 32)

# Conditional Discriminator for GAN
class ConditionalDiscriminator(nn.Module):
    def __init__(self, img_channels, num_classes):
        super(ConditionalDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_channels * 32 * 32 + num_classes, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        labels = self.label_embedding(labels)
        x = torch.cat((images.view(images.size(0), -1), labels), dim=1)
        return self.model(x)

# Training Conditional GAN
def train_conditional_gan(dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device, epochs, num_classes):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for i, (real_images, labels) in enumerate(dataloader):
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_images = generator(noise, fake_labels)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels_zeros = torch.zeros(batch_size, 1).to(device)

            d_optimizer.zero_grad()
            real_loss = criterion(discriminator(real_images, labels), real_labels)
            fake_loss = criterion(discriminator(fake_images.detach(), fake_labels), fake_labels_zeros)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            g_loss = criterion(discriminator(fake_images, fake_labels), real_labels)
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # Save some generated images for visualization
        if (epoch + 1) % 10 == 0:
            save_image(fake_images[:25], f'output/conditional_epoch_{epoch+1}.png', nrow=5, normalize=True)

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    noise_dim = 100
    img_channels = 3
    num_classes = 10
    epochs = 50
    lr = 0.0002

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare DataLoader
    dataloader = get_dataloader(batch_size)

    # Initialize models
    generator = ConditionalGenerator(noise_dim, img_channels, num_classes).to(device)
    discriminator = ConditionalDiscriminator(img_channels, num_classes).to(device)

    # Loss and optimizers
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # Train Conditional GAN
    train_conditional_gan(dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device, epochs, num_classes)
