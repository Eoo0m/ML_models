import os
import re
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 설정
# ============================================================
CFG = {"BATCH_SIZE": 2048, "EPOCHS": 3, "LEARNING_RATE": 5e-4, "SEED": 42}


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG["SEED"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# metric
# ============================================================
def weighted_logloss_5050(y_true, y_prob, eps: float = 1e-12):
    y = np.asarray(y_true)
    p = np.clip(np.asarray(y_prob), eps, 1 - eps)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0 or neg == 0:
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    w_pos = 0.5 / pos
    w_neg = 0.5 / neg
    w = np.where(y == 1, w_pos, w_neg)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))))


def leaderboard_score(ap, wll):
    return 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))


# ============================================================
# 단일 parquet 로드
# ============================================================
DATA_DIR = "/content/drive/MyDrive/데분/toss"
TRAIN_PATH = os.path.join(DATA_DIR, "train_part_000.parquet")

df_all = pd.read_parquet(TRAIN_PATH)
print("Train shape:", df_all.shape)

# ============================================================
# Train/Valid split
# ============================================================
target_col = "clicked"
seq_col = "seq"

cat_cols = ["inventory_id", "l_feat_4", "l_feat_27", "age_group", "gender"]
special_cols = [f"history_b_{i}" for i in range(1, 31)] + [
    f"history_a_{i}" for i in range(1, 8)
]

df_train, df_valid = train_test_split(
    df_all,
    test_size=0.10,
    random_state=CFG["SEED"],
    stratify=df_all[target_col],
)


# ============================================================
# Categorical encoding
# ============================================================
def fit_safe_encoders(train_df, cat_cols):
    enc = {}
    for c in cat_cols:
        vals = train_df[c].astype(str).fillna("UNK").unique().tolist()
        if "UNK" in vals:
            vals.remove("UNK")
        classes = ["UNK"] + vals
        enc[c] = {v: i for i, v in enumerate(classes)}
    return enc


def transform_with_encoders(df, encoders, cat_cols):
    df = df.copy()
    for c in cat_cols:
        m = encoders[c]
        df[c] = (
            df[c].astype(str).fillna("UNK").map(lambda x: m.get(x, 0)).astype("int64")
        )
    return df


encoders = fit_safe_encoders(df_train, cat_cols)
df_train = transform_with_encoders(df_train, encoders, cat_cols)
df_valid = transform_with_encoders(df_valid, encoders, cat_cols)

num_cols = [
    c for c in df_train.columns if c not in [target_col, seq_col, "ID"] + cat_cols
]


# ============================================================
# seq tokenizer
# ============================================================
def build_tokenizer(df, seq_col):
    freq = defaultdict(int)
    for s in df[seq_col].dropna().astype(str):
        for tok in re.findall(r"\d+", s):
            freq[tok] += 1
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for t, cnt in freq.items():
        vocab[t] = len(vocab)
    return vocab


def tokenize(s, vocab):
    toks = re.findall(r"\d+", str(s))
    return [vocab.get(t, vocab["[UNK]"]) for t in toks] if toks else [0]


vocab = build_tokenizer(df_train, seq_col)


# ============================================================
# Dataset
# ============================================================
class ClickDataset(Dataset):
    def __init__(
        self, df, num_cols, cat_cols, seq_col, target_col, vocab, special_cols
    ):
        self.df = df.reset_index(drop=True)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.seq_col = seq_col
        self.vocab = vocab
        self.special_cols = special_cols

        num_df = (
            df[num_cols]
            .apply(pd.to_numeric, errors="coerce")
            .astype("float32")
            .fillna(0)
        )
        for c in special_cols:
            if c in num_df.columns:
                num_df[c] = np.power(np.abs(num_df[c]), 0.25)

        self.num_X = num_df.values
        self.cat_X = df[cat_cols].values.astype("int64")
        self.seq = df[seq_col].astype(str).fillna("").values
        self.y = df[target_col].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        num_x = torch.tensor(self.num_X[i])
        cat_x = torch.tensor(self.cat_X[i])
        seq = torch.tensor(tokenize(self.seq[i], self.vocab))
        y = torch.tensor(self.y[i])
        return num_x, cat_x, seq, y


def collate_fn(batch):
    num_x, cat_x, seqs, ys = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    seqs_pad = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    lens = torch.tensor([len(s) for s in seqs])
    ys = torch.stack(ys)
    return num_x, cat_x, seqs_pad, lens, ys


# ============================================================
# Model
# ============================================================
class CrossNetworkV2(nn.Module):
    def __init__(self, dim, num_layers=4, low_rank=64):
        super().__init__()
        self.U = nn.ParameterList(
            [nn.Parameter(torch.empty(dim, low_rank)) for _ in range(num_layers)]
        )
        self.V = nn.ParameterList(
            [nn.Parameter(torch.empty(dim, low_rank)) for _ in range(num_layers)]
        )
        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in range(num_layers)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for U, V in zip(self.U, self.V):
            nn.init.xavier_uniform_(U)
            nn.init.xavier_uniform_(V)

    def forward(self, x0):
        x = x0
        for U, V, b in zip(self.U, self.V, self.b):
            v = x @ V
            u = v @ U.t()
            x = x + x0 * (u + b)
        return x


class DeepCrossCTR(nn.Module):
    def __init__(
        self,
        num_features,
        cat_cardinalities,
        vocab_size,
        emb_dim=16,
        lstm_hidden=32,
        seq_emb_dim=16,
    ):
        super().__init__()
        self.emb_layers = nn.ModuleList(
            [nn.Embedding(c, emb_dim) for c in cat_cardinalities]
        )
        self.bn_num = nn.BatchNorm1d(num_features)

        self.seq_emb = nn.Embedding(vocab_size, seq_emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            seq_emb_dim, lstm_hidden, num_layers=2, batch_first=True, bidirectional=True
        )

        in_dim = num_features + len(cat_cardinalities) * emb_dim + lstm_hidden * 2
        self.post_bn = nn.BatchNorm1d(in_dim)
        self.cross = CrossNetworkV2(in_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, num_x, cat_x, seqs, lens):
        num_x = self.bn_num(num_x)
        cat_feat = torch.cat(
            [emb(cat_x[:, i]) for i, emb in enumerate(self.emb_layers)], 1
        )

        seq_emb = self.seq_emb(seqs)
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_emb, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        seq_h = torch.cat([h_n[-2], h_n[-1]], 1)

        z = torch.cat([num_x, cat_feat, seq_h], 1)
        z = self.cross(z)
        z = self.post_bn(z)
        return self.mlp(z).squeeze(1)


# ============================================================
# Evaluate
# ============================================================
def evaluate(model, loader):
    model.eval()
    ys_all, ps_all = [], []

    for num_x, cat_x, seqs, lens, ys in loader:
        num_x, cat_x, seqs, lens, ys = (
            num_x.to(device),
            cat_x.to(device),
            seqs.to(device),
            lens.to(device),
            ys.to(device),
        )

        with torch.no_grad():
            ps = torch.sigmoid(model(num_x, cat_x, seqs, lens))

        ys_all.append(ys.cpu().numpy())
        ps_all.append(ps.cpu().numpy())

    y = np.concatenate(ys_all)
    p = np.concatenate(ps_all)
    ap = average_precision_score(y, p)
    wll = weighted_logloss_5050(y, p)
    score = leaderboard_score(ap, wll)
    return ap, wll, score


# ============================================================
# Train
# ============================================================
def train_model(df_train, df_valid):
    ds_tr = ClickDataset(
        df_train, num_cols, cat_cols, seq_col, target_col, vocab, special_cols
    )
    ds_va = ClickDataset(
        df_valid, num_cols, cat_cols, seq_col, target_col, vocab, special_cols
    )

    dl_tr = DataLoader(
        ds_tr, batch_size=CFG["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn
    )
    dl_va = DataLoader(
        ds_va, batch_size=CFG["BATCH_SIZE"], shuffle=False, collate_fn=collate_fn
    )

    cat_cardinalities = [df_train[c].nunique() + 1 for c in cat_cols]
    model = DeepCrossCTR(len(num_cols), cat_cardinalities, vocab_size=len(vocab)).to(
        device
    )

    pos = df_train[target_col].sum()
    neg = len(df_train) - pos
    pos_w = torch.tensor([neg / max(pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=2)

    best_score = -1e9
    best_path = os.path.join(DATA_DIR, "best_model.pt")

    for epoch in range(1, CFG["EPOCHS"] + 1):
        model.train()
        total_loss = 0

        for num_x, cat_x, seqs, lens, ys in tqdm(dl_tr, desc=f"Train {epoch}"):
            num_x, cat_x, seqs, lens, ys = (
                num_x.to(device),
                cat_x.to(device),
                seqs.to(device),
                lens.to(device),
                ys.to(device),
            )

            opt.zero_grad()
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
            loss.backward()
            opt.step()
            scheduler.step()

            total_loss += loss.item() * len(ys)

        ap, wll, score = evaluate(model, dl_va)
        print(
            f"[Epoch {epoch}] Loss={total_loss/len(ds_tr):.5f}  AP={ap:.5f}  WLL={wll:.5f}  SCORE={score:.5f}"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)

    print("Best score:", best_score)
    return model


# ============================================================
# 실행 (train only)
# ============================================================
model = train_model(df_train, df_valid)
print("학습 종료. best_model.pt 저장됨.")
