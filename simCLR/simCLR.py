import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random


# --- 1. Reproducibility ---
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# --- 2. Configuration ---
class Config:
    batch_size = 1024
    num_workers = 8
    lr = 1e-3
    epochs = 50
    temperature = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42


# --- 3. Dataset & Transform ---
class SimCLRTransform:
    def __init__(self):
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# --- 4. Optimized Loss (Using CrossEntropy) ---
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # [2B, D]
        z = torch.cat([z1, z2], dim=0)

        # [2B, 2B] similarity matrix
        logits = torch.matmul(z, z.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(mask, float("-inf"))

        # Labels: positive pairs
        # For i in [0, B): positive is i+B
        # For i in [B, 2B): positive is i-B
        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=z.device),
                torch.arange(0, batch_size, device=z.device),
            ]
        )

        return self.criterion(logits, labels)


# --- 5. Model Architecture ---
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super().__init__()
        # SimCLR Paper: Linear -> BN -> ReLU -> Linear
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
        # CIFAR-10 optimization
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.projection = ProjectionMLP(512, 2048, 128)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection(h)
        return h, z


# --- 6. Main Training Loop ---
if __name__ == "__main__":
    cfg = Config()
    set_seed(cfg.seed)

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=SimCLRTransform()
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = SimCLR().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = NTXentLoss(cfg.temperature)
    scaler = torch.cuda.amp.GradScaler()

    print("Starting Training with AMP...")

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for (x1, x2), _ in pbar:
            x1, x2 = x1.to(cfg.device, non_blocking=True), x2.to(
                cfg.device, non_blocking=True
            )

            optimizer.zero_grad()

            # [Optimization] Autocast context
            with torch.cuda.amp.autocast():
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = loss_fn(z1, z2)

            # [Optimization] Scaled backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()  # Update LR

    print("Training Done.")
