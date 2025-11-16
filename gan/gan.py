import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import csv


# =====================================================
# Generator / Discriminator 정의
# =====================================================
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=1 * 28 * 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_dim=1 * 28 * 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# 경로 설정 및 폴더 생성
# =====================================================
BASE = Path("/Users/eomjoonseo/models/gan")
DIR_IMAGES = BASE / "gan_images"
DIR_LOGS = BASE / "gan_logs"
DIR_IMAGES.mkdir(parents=True, exist_ok=True)
DIR_LOGS.mkdir(parents=True, exist_ok=True)

csv_path = DIR_LOGS / "gan_training_log.csv"


# =====================================================
# 데이터셋 (MNIST)
# =====================================================
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 100
G = Generator(z_dim).to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))


# =====================================================
# 학습 루프
# =====================================================
def train_step(real_imgs, k):
    batch_size = real_imgs.size(0)

    for _ in range(k):
        z = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = G(z).detach()
        real_flat = real_imgs.view(batch_size, -1).to(device)
        fake_flat = fake_imgs.view(batch_size, -1)

        real_labels = torch.ones((batch_size, 1), device=device)  # 1.0
        fake_labels = torch.zeros((batch_size, 1), device=device)  # 0.0
        labels = torch.cat([real_labels, fake_labels], dim=0)

        pred = torch.cat([D(real_flat), D(fake_flat)], dim=0)
        loss_D = criterion(pred, labels)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

    z = torch.randn(batch_size, z_dim, device=device)
    fake_imgs = G(z)
    D_fake = D(fake_imgs.view(batch_size, -1))

    real_label_for_G = torch.ones(batch_size, 1, device=device)
    loss_G = criterion(D_fake, real_label_for_G)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    D_fake_mean = D_fake.mean().item()

    return loss_D.item(), loss_G.item(), D_fake_mean


# =====================================================
# 학습 실행 + CSV 기록
# =====================================================
epochs = 300
k = 1
D_losses, G_losses = [], []

# CSV 헤더
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "epoch",
            "D_loss",
            "G_loss",
            "D_fake_mean",
        ]
    )

for epoch in range(epochs):
    for real_imgs, _ in loader:
        d_loss, g_loss, d_fake_mean = train_step(real_imgs, k)

    D_losses.append(d_loss)
    G_losses.append(g_loss)

    # 로그 출력
    print(
        f"[{epoch+1}/{epochs}] "
        f"D_loss={d_loss:.4f}, G_loss={g_loss:.4f}, D(G(z))={d_fake_mean:.3f}\n"
    )

    # CSV 저장
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch + 1,
                d_loss,
                g_loss,
                d_fake_mean,
            ]
        )

    # =============================
    # 이미지 샘플 저장
    # =============================
    if epoch < 10 or (epoch + 1) % 100 == 0:
        with torch.no_grad():
            z = torch.randn(32, z_dim, device=device)
            fake = G(z).view(-1, 1, 28, 28)
            grid = utils.make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
            plt.figure(figsize=(8, 8))
            plt.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
            plt.title(f"Epoch {epoch+1}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(DIR_IMAGES / f"epoch_{epoch+1:03d}.png", bbox_inches="tight")
            plt.close()
