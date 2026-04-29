import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


# 生成器
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


def train_gan():
    # 超参数
    latent_dim = 100
    lr = 0.0002
    epochs = 50
    batch_size = 128

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # 损失函数和优化器
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    # 训练
    G_losses = []
    D_losses = []

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            real_imgs = imgs.to(device)

            # 真实和假标签
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ---------------------
            #  训练判别器
            # ---------------------
            optimizer_D.zero_grad()

            # 真实图像的损失
            real_validity = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_validity, real_labels)

            # 假图像的损失
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_validity = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(fake_validity, fake_labels)

            # 总判别器损失
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            #  训练生成器
            # ---------------------
            optimizer_G.zero_grad()

            # 生成假图像
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_imgs = generator(z)

            # 生成器希望判别器认为假图像是真的
            validity = discriminator(gen_imgs)
            g_loss = adversarial_loss(validity, real_labels)

            g_loss.backward()
            optimizer_G.step()

            # 记录损失
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # 每个epoch保存生成的图像
        if epoch % 5 == 0:
            with torch.no_grad():
                test_z = torch.randn(16, latent_dim).to(device)
                generated = generator(test_z).detach().cpu()
                plt.figure(figsize=(4, 4))
                plt.imshow(np.transpose(vutils.make_grid(generated, nrow=4, padding=2, normalize=True), (1, 2, 0)))
                plt.axis('off')
                plt.title(f'Epoch {epoch}')
                plt.savefig(f'gan_epoch_{epoch}.png')
                plt.show()

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss', alpha=0.7)
    plt.plot(D_losses, label='Discriminator Loss', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('gan_losses.png')
    plt.show()

    # 显示最终生成的图像
    with torch.no_grad():
        test_z = torch.randn(64, latent_dim).to(device)
        generated = generator(test_z).detach().cpu()

        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(vutils.make_grid(generated, nrow=8, padding=2, normalize=True), (1, 2, 0)))
        plt.axis('off')
        plt.title('Generated Images')
        plt.savefig('final_generated.png')
        plt.show()


if __name__ == '__main__':
    train_gan()