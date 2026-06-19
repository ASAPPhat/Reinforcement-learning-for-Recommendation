"""
common.py — tiện ích dùng chung cho pipeline long-term re-rank.

Chống bias (mấu chốt):
  * TIME-SPLIT theo từng user: 20% tương tác CUỐI = test (chỉ để eval).
  * BERT train Cloze CHỈ trên phần 'remaining' (80% đầu) -> test items
    không bao giờ nằm trong chuỗi train -> không leakage.
  * RL lấy reward từ 'rl_window' (20% cuối của remaining), KHÔNG đụng test.
  * Eval cuối: history = remaining, relevant = test (rating làm gain) cho CẢ
    2 model, cùng code, cùng k.
"""
import os, sys, math, random, json
import numpy as np
import pandas as pd
import torch

# import BERT4Rec từ thư mục cha
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Bert4Rec_model import BERT4Rec, gather_indexes, BERT4RecDataset  # noqa: E402


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_env_key(env_path, key="wandb_api_key"):
    if not os.path.exists(env_path):
        return None
    for line in open(env_path, encoding="utf-8"):
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip()
    return None


# ----------------------------------------------------------------------------
# DATA: time-split per user
# ----------------------------------------------------------------------------
def load_and_split(rating_file, test_frac=0.2, min_len=5, max_len=200, seed=42):
    cols = ["UserID", "MovieID", "Rating", "Timestamp"]
    df = pd.read_csv(rating_file, sep="::", engine="python", names=cols)
    df = df.sort_values(["UserID", "Timestamp"]).reset_index(drop=True)

    raw_ids = sorted(df["MovieID"].unique())
    movie2id = {m: i + 1 for i, m in enumerate(raw_ids)}     # 1..N ; 0=pad ; N+1=mask
    n_items = len(raw_ids)
    df["enc"] = df["MovieID"].map(movie2id)
    df["pair"] = list(zip(df["enc"], df["Rating"].astype(float)))
    seqs = df.groupby("UserID")["pair"].apply(list).to_dict()

    users = []           # mỗi user: dict các phần
    bert_train_seqs = [] # chuỗi item (remaining) cho Cloze
    n_inter = n_test = n_remain = 0
    for _, seq in seqs.items():
        if len(seq) < min_len:
            continue
        n = len(seq)
        tsize = max(1, round(test_frac * n))
        remaining = seq[:-tsize]          # 80% đầu
        test = seq[-tsize:]               # 20% cuối
        if len(remaining) < 2:
            continue
        # rl_window: 20% cuối của remaining làm reward target cho RL
        rsize = max(1, round(test_frac * len(remaining)))
        rl_hist = remaining[:-rsize]
        rl_rel = remaining[-rsize:]
        if len(rl_hist) < 1:
            rl_hist = remaining[:1]; rl_rel = remaining[1:] or remaining[-1:]

        rem_items = [m for m, r in remaining][-max_len:]
        users.append({
            "hist_items": rem_items,                                  # eval history
            "test_rel": {int(m): float(r) for m, r in test},          # eval relevant (rating gain)
            "rl_hist_items": [m for m, r in rl_hist][-max_len:],      # RL history
            "rl_rel": {int(m): float(r) for m, r in rl_rel},          # RL reward target
        })
        bert_train_seqs.append(rem_items)
        n_inter += n; n_test += len(test); n_remain += len(remaining)

    stats = {
        "n_users": len(users),
        "n_items": n_items,
        "n_interactions": int(n_inter),
        "n_train_remaining": int(n_remain),
        "n_test": int(n_test),
        "avg_seq_len": round(n_inter / max(1, len(users)), 2),
        "avg_test_len": round(n_test / max(1, len(users)), 2),
        "test_frac": test_frac,
        "max_len": max_len,
    }
    return users, bert_train_seqs, n_items, movie2id, stats


# ----------------------------------------------------------------------------
# METRICS: graded NDCG / DCG (gain = 2^rating - 1), Recall, MeanRating
# ----------------------------------------------------------------------------
def dcg_at_k(ranked_ids, rel, k):
    s = 0.0
    for i, it in enumerate(ranked_ids[:k]):
        g = rel.get(int(it), 0.0)
        if g > 0:
            s += (2.0 ** g - 1.0) / math.log2(i + 2)
    return s

def ndcg_at_k(ranked_ids, rel, k):
    ideal = sorted(rel.values(), reverse=True)
    idcg = sum((2.0 ** g - 1.0) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    if idcg <= 0:
        return 0.0
    return dcg_at_k(ranked_ids, rel, k) / idcg

def recall_at_k(ranked_ids, rel, k):
    if not rel:
        return 0.0
    hit = sum(1 for it in ranked_ids[:k] if int(it) in rel)
    return hit / len(rel)

def mean_rating_at_k(ranked_ids, rel, k):
    """Rating trung bình của các item RELEVANT lọt top-k (chất lượng hit)."""
    vals = [rel[int(it)] for it in ranked_ids[:k] if int(it) in rel]
    return float(np.mean(vals)) if vals else 0.0

def aggregate_metrics(rankings, rels, ks=(5, 10, 20)):
    out = {}
    for k in ks:
        out[f"NDCG@{k}"] = float(np.mean([ndcg_at_k(r, rel, k) for r, rel in zip(rankings, rels)]))
        out[f"Recall@{k}"] = float(np.mean([recall_at_k(r, rel, k) for r, rel in zip(rankings, rels)]))
    out["MeanRating@10"] = float(np.mean([mean_rating_at_k(r, rel, 10) for r, rel in zip(rankings, rels)]))
    return out


# ----------------------------------------------------------------------------
# BERT helpers
# ----------------------------------------------------------------------------
@torch.no_grad()
def encode_states(bert, hist_list, device, max_len=200, bs=256):
    """history (list[list[item]]) -> state THÔ [N,H] (append [mask] ở cuối)."""
    bert.eval()
    outs = []
    for i in range(0, len(hist_list), bs):
        chunk = hist_list[i:i + bs]
        b = len(chunk)
        item_seq = torch.zeros((b, max_len), dtype=torch.long, device=device)
        slen = torch.zeros((b,), dtype=torch.long, device=device)
        for j, s in enumerate(chunk):
            L = min(len(s), max_len)
            slen[j] = max(L, 1)
            if L > 0:
                item_seq[j, :L] = torch.tensor(s[-L:], dtype=torch.long, device=device)
        prep = bert.reconstruct_test_data(item_seq, slen)
        so = bert.forward(prep)
        outs.append(gather_indexes(so, slen - 1))
    return torch.cat(outs, 0)
