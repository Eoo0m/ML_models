import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path


# =====================================================
# 1) Generator / Discriminator
# =====================================================
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
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
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# 2) 경로/환경
# =====================================================
DIR_IMAGES = Path("gan_images")
DIR_IMAGES.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 100

G = Generator(z_dim).to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()

opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# =====================================================
# 데이터 (MNIST)
# =====================================================
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)


# =====================================================
# 3) 학습 Step
# =====================================================
def train_step(real_imgs, k):
    batch_size = real_imgs.size(0)
    real_imgs = real_imgs.view(batch_size, -1).to(device)

    # ----- Train D -----
    for _ in range(k):
        z = torch.randn(batch_size, z_dim, device=device)
        fake = G(z).detach()

        pred_real = D(real_imgs)
        pred_fake = D(fake)

        labels_real = torch.ones(batch_size, 1, device=device)
        labels_fake = torch.zeros(batch_size, 1, device=device)

        loss_D = criterion(pred_real, labels_real) + criterion(pred_fake, labels_fake)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

    # ----- Train G -----
    z = torch.randn(batch_size, z_dim, device=device)
    fake = G(z)
    pred_fake = D(fake)

    loss_G = criterion(pred_fake, torch.ones(batch_size, 1, device=device))

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    return loss_D.item(), loss_G.item()


# =====================================================
# 4) Training Loop
# =====================================================
epochs = 500
k = 1

for epoch in range(epochs):
    for real_imgs, _ in loader:
        d_loss, g_loss = train_step(real_imgs, k)

    print(f"[{epoch+1}/{epochs}] D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")

    # 100 epoch마다 이미지 저장
    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            z = torch.randn(32, z_dim, device=device)
            fake = G(z).view(-1, 1, 28, 28)
            grid = utils.make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))

            plt.figure(figsize=(8, 8))
            plt.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
            plt.title(f"Epoch {epoch+1}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(DIR_IMAGES / f"epoch_{epoch+1:03d}.png")
            plt.close()
