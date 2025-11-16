import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# CSV 경로
# =====================================================
csv_path = Path("gan/gan_logs")

# CSV 로드
df = pd.read_csv(csv_path)

plt.style.use("default")
plt.figure(figsize=(10, 6), facecolor="#e8e8e8")  # 전체 배경 밝은 회색

ax = plt.gca()
ax.set_facecolor("#f5f5f5")  # 그래프 영역 더 밝은 회색

# 선 그래프 3개
plt.plot(df["epoch"], df["D_loss"], label="D_loss", color="#d9534f", linewidth=2)
plt.plot(df["epoch"], df["G_loss"], label="G_loss", color="#0275d8", linewidth=2)
plt.plot(
    df["epoch"],
    df["D_fake_mean"],
    label="D(G(z))",
    color="#5cb85c",
    linewidth=2,
    linestyle="--",
)

# 제목 / 라벨
plt.title("GAN Training Metrics", fontsize=16, color="#333333")
plt.xlabel("Epoch", fontsize=12, color="#333333")
plt.ylabel("Value", fontsize=12, color="#333333")

# 축 스타일
ax.tick_params(colors="#333333")
for spine in ax.spines.values():
    spine.set_edgecolor("#aaaaaa")

# 그리드
plt.grid(True, color="#cccccc", linestyle="--", linewidth=0.8)

# 범례
plt.legend(facecolor="#f5f5f5", edgecolor="#aaaaaa", labelcolor="#333333", fontsize=11)

plt.tight_layout()
plt.show()
