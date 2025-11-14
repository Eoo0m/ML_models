import os, gc, json, random, hashlib
from collections import Counter
from typing import List

import numpy as np
import polars as pl
import pyarrow.parquet as pq

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= 설정 =================
DATA_PATH = "data/train.parquet"
SEQ_COL = "seq"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MAX_LEN = 256
MASK_RATIO = 0.15
BATCH_SIZE = 1024
EMB_DIM = 64
HIDDEN = 128
PROJ_DIM = 32
EPOCHS = 1
LR = 3e-4

OUT_MODEL_DIR = "models"
os.makedirs(OUT_MODEL_DIR, exist_ok=True)

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Device: {DEVICE}")

PAD, UNK, MASK = 0, 1, 2


# ================= 유틸 =================
def read_parquet(path: str):
    t = pq.read_table(path)
    return t.combine_chunks().to_pandas()


def parse_seq(s: str) -> List[int]:
    if s is None:
        return []
    return [int(x) for x in str(s).split(",") if x.isdigit()]


def tail_truncate(tokens: List[int], maxlen: int) -> List[int]:
    return tokens[-maxlen:] if len(tokens) > maxlen else tokens


def pad_left(tokens: List[int], maxlen: int, pad_id: int) -> List[int]:
    if len(tokens) >= maxlen:
        return tokens
    return [pad_id] * (maxlen - len(tokens)) + tokens


# ================= Dataset =================
class MaskedSeqDataset(Dataset):
    def __init__(self, df: pl.DataFrame, tok2id: dict, maxlen: int):
        self.seqs = df[SEQ_COL].to_list()
        self.tok2id = tok2id
        self.maxlen = maxlen

    def __len__(self):
        return len(self.seqs)

    def encode(self, toks: List[int]) -> List[int]:
        return [self.tok2id.get(str(t), UNK) for t in toks]

    def __getitem__(self, idx):
        toks = parse_seq(self.seqs[idx])
        toks = tail_truncate(toks, self.maxlen)
        toks = pad_left(toks, self.maxlen, PAD)
        ids = self.encode(toks)
        return np.array(ids, dtype=np.int64)


def collate_masked(batch_ids, mask_ratio: float):
    x = torch.from_numpy(np.stack(batch_ids, axis=0))
    B, L = x.shape
    valid = x != PAD
    mask = torch.zeros_like(x, dtype=torch.bool)

    for b in range(B):
        idx = torch.where(valid[b])[0]
        if len(idx) == 0:
            continue
        k = max(1, int(len(idx) * mask_ratio))
        choice = idx[torch.randperm(len(idx))[:k]]
        mask[b, choice] = True

    x_masked = x.clone()
    x_masked[mask] = MASK
    return x_masked, x, mask


# ================= 모델 =================
class BiGRUMaskedModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, proj_dim, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dec = nn.Linear(2 * hidden, emb_dim)
        self.proj = nn.Linear(2 * hidden, proj_dim)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.gru(emb)
        return out

    def recover_masked(self, h):
        return self.dec(h)

    def pool_and_project(self, h, pad_mask):
        valid = ~pad_mask
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1)
        h_masked = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        pooled = h_masked.sum(dim=1) / denom
        return self.proj(pooled)


# ================= MLM 학습 =================
def train_mlm(model, loader, epochs, lr, device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CosineEmbeddingLoss()

    for ep in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Pretrain {ep}/{epochs}")
        total = 0.0
        for x_masked, x_orig, mask in pbar:
            x_masked = x_masked.to(device)
            x_orig = x_orig.to(device)
            mask = mask.to(device)

            h = model(x_masked)
            pred_emb = model.recover_masked(h)
            with torch.no_grad():
                tgt_emb = model.emb(x_orig)

            if mask.any():
                y = torch.ones(mask.sum().item(), device=device)
                loss = loss_fn(pred_emb[mask], tgt_emb[mask], y)
            else:
                loss = pred_emb.sum() * 0.0

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {ep}  avg_loss={total / len(loader):.4f}")


# ================= seq32 인퍼런스 =================
@torch.no_grad()
def generate_seq32(model, tok2id, in_path, out_path, maxlen, device):
    print(f"🧩 Embedding {in_path} → {out_path}")
    model.eval().to(device)

    df = pl.read_parquet(in_path)
    seqs = df[SEQ_COL].to_list()

    B = 1024
    all_vecs = []

    for i in tqdm(range(0, len(seqs), B)):
        batch = seqs[i : i + B]
        ids_batch = []
        for s in batch:
            toks = tail_truncate(parse_seq(s), maxlen)
            ids = [tok2id.get(str(t), tok2id["<UNK>"]) for t in toks]
            ids = pad_left(ids, maxlen, tok2id["<PAD>"])
            ids_batch.append(ids)

        x = torch.tensor(ids_batch, dtype=torch.long, device=device)
        h = model(x)
        pad_mask = x == tok2id["<PAD>"]
        z = model.pool_and_project(h, pad_mask)
        z = nn.functional.normalize(z, p=2, dim=1)
        all_vecs.append(z.cpu().numpy())

    Z = np.vstack(all_vecs)
    add_cols = {f"seq32_{i:02d}": Z[:, i] for i in range(Z.shape[1])}

    out_df = df.with_columns([pl.Series(k, v) for k, v in add_cols.items()])
    out_df.write_parquet(out_path)
    print(f"✅ Saved {out_path}")


# ================= 메인 =================
def main():
    print("📂 Loading dataset…")
    df = pl.from_pandas(read_parquet(DATA_PATH))
    print("Rows:", len(df))

    # vocab 구축
    cnt = Counter()
    for s in df[SEQ_COL]:
        toks = tail_truncate(parse_seq(s), MAX_LEN)
        cnt.update(toks)

    tok2id = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2}
    next_id = 3
    for tok in cnt.keys():
        tok2id[str(tok)] = next_id
        next_id += 1

    vocab_size = next_id
    print("Vocab size:", vocab_size)

    # 데이터셋
    ds = MaskedSeqDataset(df, tok2id, MAX_LEN)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_masked(b, MASK_RATIO),
        num_workers=0,
    )

    # 모델 학습
    model = BiGRUMaskedModel(vocab_size, EMB_DIM, HIDDEN, PROJ_DIM, PAD)
    train_mlm(model, loader, EPOCHS, LR, DEVICE)

    # 저장
    save_path = os.path.join(OUT_MODEL_DIR, "bigru_seq32.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "tok2id": tok2id,
            "vocab_size": vocab_size,
        },
        save_path,
    )
    print("💾 Saved:", save_path)

    # seq32 생성
    generate_seq32(
        model, tok2id, DATA_PATH, "data/train_seq32.parquet", MAX_LEN, DEVICE
    )


if __name__ == "__main__":
    main()
