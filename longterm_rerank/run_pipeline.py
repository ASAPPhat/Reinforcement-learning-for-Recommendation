"""
run_pipeline.py — Long-term re-rank pipeline (no bias, time-split).

Task 1: Train lại BERT4Rec (kiến trúc gốc) trên time-split (80/20).
Task 2: RL top-N re-rank — Actor-Critic policy-gradient (KHÔNG TD3+BC),
        policy init ngẫu nhiên, reward = graded NDCG (rating gain).
Eval:   CẢ BERT và BERT+RL trên tập test (20% cuối) bằng graded NDCG / Recall / MeanRating.
Log:    wandb (key đọc từ ../.env).
"""
import os, sys, json, math, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (set_seed, load_env_key, load_and_split, encode_states,
                    aggregate_metrics, ndcg_at_k, BERT4Rec, BERT4RecDataset)
from Bert4Rec_model import train_one_epoch

ROOT = os.path.dirname(HERE)
RATING_FILE = os.path.join(ROOT, "Data_Movielens_1m", "ml-1m", "ratings.dat")
ENV_FILE = os.path.join(ROOT, ".env")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 200
set_seed(42)

# ---- wandb (mềm: lỗi cũng không crash train) ----
try:
    import wandb
    key = load_env_key(ENV_FILE)
    if key:
        os.environ["WANDB_API_KEY"] = key
        wandb.login(key=key, relogin=True)
    WANDB = True
except Exception as e:
    print(f"[wandb] tắt ({e})"); WANDB = False

def wb_init(name, cfg):
    if not WANDB: return None
    try:
        return wandb.init(project="bert4rec-longterm-rerank", name=name, config=cfg, reinit=True)
    except Exception as e:
        print(f"[wandb] init fail: {e}"); return None
def wb_log(run, d):
    if run is not None:
        try: wandb.log(d)
        except Exception: pass
def wb_finish(run):
    if run is not None:
        try: wandb.finish()
        except Exception: pass


# =====================================================================
# Eval graded NDCG (full-rank) cho BERT thuần
# =====================================================================
@torch.no_grad()
def eval_bert(bert, users, device, ks=(5,10,20), topk=50, subset=None):
    item_emb = bert.item_embedding.weight[:bert.n_items]   # row i = id i
    bias = bert.output_bias
    idxs = range(len(users)) if subset is None else subset
    hist = [users[i]["hist_items"] for i in idxs]
    rels = [users[i]["test_rel"] for i in idxs]
    states = encode_states(bert, hist, device, MAX_LEN)
    rankings = []
    K = max(ks)
    for i in range(0, states.shape[0], 256):
        sc = states[i:i+256] @ item_emb.T + bias            # [b, n_items]
        sc[:, 0] = -1e9                                      # pad
        top = torch.topk(sc, K, dim=1).indices.cpu().numpy()
        rankings.extend(top.tolist())
    return aggregate_metrics(rankings, rels, ks)


# =====================================================================
# RL re-ranker: Actor (policy) + Critic (value baseline). KHÔNG TD3+BC.
# =====================================================================
class PolicyNet(nn.Module):
    """score(candidate) = MLP([state, cand_emb]).  Init ngẫu nhiên (thuần RL)."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*dim, 256), nn.ReLU(),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, state, cand_emb):           # state [B,H], cand_emb [B,N,H]
        B, N, H = cand_emb.shape
        s = state.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([s, cand_emb], dim=-1)
        return self.net(x).squeeze(-1)            # [B,N]

class ValueNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, state):
        return self.net(state).squeeze(-1)        # [B]


def _cand_gains(cand_ids, rel_dicts):
    """[B,N] gain = 2^rating-1 nếu candidate là relevant, else 0."""
    B, N = cand_ids.shape
    g = np.zeros((B, N), dtype=np.float32)
    cid = cand_ids.cpu().numpy()
    for b in range(B):
        rel = rel_dicts[b]
        if not rel: continue
        for j in range(N):
            r = rel.get(int(cid[b, j]), 0.0)
            if r > 0: g[b, j] = 2.0**r - 1.0
    return torch.tensor(g)


@torch.no_grad()
def _bert_candidates(bert, states, topN):
    item_emb = bert.item_embedding.weight[:bert.n_items]
    bias = bert.output_bias
    sc = states @ item_emb.T + bias
    sc[:, 0] = -1e9
    cand_ids = torch.topk(sc, topN, dim=1).indices           # [B,topN] (item ids)
    cand_emb = item_emb[cand_ids]                            # [B,topN,H]
    return cand_ids, cand_emb


def train_rl(bert, users, device, epochs=30, bs=512, topN=100, K=10,
             lr=1e-4, patience=8, run=None):
    dim = bert.hidden_size
    policy = PolicyNet(dim).to(device); value = ValueNet(dim).to(device)
    popt = optim.Adam(policy.parameters(), lr=lr)
    vopt = optim.Adam(value.parameters(), lr=lr)
    disc = (1.0 / torch.log2(torch.arange(K, device=device).float() + 2)).unsqueeze(0)  # [1,K]

    # precompute RL-train states (BERT đóng băng) 1 lần
    rl_hist = [u["rl_hist_items"] for u in users]
    rl_rel = [u["rl_rel"] for u in users]
    S = encode_states(bert, rl_hist, device, MAX_LEN).cpu()
    N = S.shape[0]
    val_sub = list(range(0, N, max(1, N // 2000)))[:2000]    # subset cho early-stop

    best, best_sd, noimp = -1.0, None, 0
    for ep in range(epochs):
        policy.train(); value.train()
        perm = np.random.permutation(N)
        ep_reward, ep_steps = 0.0, 0
        for i in range(0, N, bs):
            bidx = perm[i:i+bs]
            st = S[bidx].to(device)
            rels = [rl_rel[j] for j in bidx]
            cand_ids, cand_emb = _bert_candidates(bert, st, topN)
            gains = _cand_gains(cand_ids, rels).to(device)           # [B,topN]
            scores = policy(st, cand_emb)                            # [B,topN]

            # ---- sample top-K permutation (Plackett-Luce), REINFORCE ----
            avail = torch.ones_like(scores, dtype=torch.bool)
            logp = torch.zeros(scores.shape[0], device=device)
            chosen_gain = torch.zeros(scores.shape[0], K, device=device)
            for t in range(K):
                logits = scores.masked_fill(~avail, -1e9)
                probs = F.softmax(logits, dim=-1)
                m = torch.distributions.Categorical(probs)
                a = m.sample()
                logp = logp + m.log_prob(a)
                chosen_gain[:, t] = gains.gather(1, a.unsqueeze(1)).squeeze(1)
                avail.scatter_(1, a.unsqueeze(1), False)
            dcg = (chosen_gain * disc).sum(1)
            ideal, _ = torch.sort(gains, dim=1, descending=True)
            idcg = (ideal[:, :K] * disc).sum(1).clamp(min=1e-8)
            reward = dcg / idcg                                       # [B] in [0,1]

            V = value(st)
            adv = (reward - V).detach()
            actor_loss = -(adv * logp).mean()
            critic_loss = F.mse_loss(V, reward)
            popt.zero_grad(); actor_loss.backward(); popt.step()
            vopt.zero_grad(); critic_loss.backward(); vopt.step()

            ep_reward += reward.mean().item() * len(bidx); ep_steps += len(bidx)

        train_ndcg = ep_reward / max(1, ep_steps)
        val_ndcg = eval_rl(bert, policy, [users[j] for j in val_sub], device,
                           target="rl", ks=(10,), topN=topN)["NDCG@10"]
        print(f"[RL] ep {ep+1:02d}/{epochs}  train_reward(NDCG@{K})={train_ndcg:.4f}  VAL NDCG@10={val_ndcg:.4f}"
              + ("  <- best" if val_ndcg > best+1e-5 else f" (best {best:.4f},{noimp+1}/{patience})"))
        wb_log(run, {"rl/train_reward": train_ndcg, "rl/val_ndcg10": val_ndcg, "rl/epoch": ep+1})
        if val_ndcg > best + 1e-5:
            best = val_ndcg; best_sd = {k: v.detach().clone() for k, v in policy.state_dict().items()}; noimp = 0
        else:
            noimp += 1
            if noimp >= patience:
                print(f"[RL] EARLY STOP ep {ep+1}, best VAL NDCG@10={best:.4f}"); break
    if best_sd: policy.load_state_dict(best_sd)
    return policy


@torch.no_grad()
def eval_rl(bert, policy, users, device, target="test", ks=(5,10,20), topN=100):
    """Re-rank BERT top-N bằng policy (greedy), tính graded NDCG vs relevant."""
    policy.eval()
    hist_key = "hist_items" if target == "test" else "rl_hist_items"
    rel_key = "test_rel" if target == "test" else "rl_rel"
    hist = [u[hist_key] for u in users]; rels = [u[rel_key] for u in users]
    S = encode_states(bert, hist, device, MAX_LEN)
    rankings = []
    for i in range(0, S.shape[0], 256):
        st = S[i:i+256]
        cand_ids, cand_emb = _bert_candidates(bert, st, topN)
        sc = policy(st, cand_emb)                                 # [b,topN]
        order = torch.argsort(sc, dim=1, descending=True)         # rerank
        reranked = torch.gather(cand_ids, 1, order)               # ids theo thứ tự policy
        rankings.extend(reranked.cpu().numpy().tolist())
    return aggregate_metrics(rankings, rels, ks)


# =====================================================================
def main():
    os.makedirs(HERE, exist_ok=True)
    print(f"[*] Device {DEVICE}")
    users, bert_seqs, n_real, movie2id, stats = load_and_split(RATING_FILE, max_len=MAX_LEN)
    print("[*] STATS:", json.dumps(stats, ensure_ascii=False))
    json.dump(stats, open(os.path.join(HERE, "data_stats.json"), "w"), ensure_ascii=False, indent=2)

    n_model = n_real + 1   # khớp convention BERT (mask = n_real+1)

    # -------- TASK 1: train BERT4Rec --------
    run = wb_init("bert4rec-retrain", {**stats, "model": "BERT4Rec", "hidden": 64, "layers": 2})
    bert = BERT4Rec(n_items=n_model, max_seq_length=MAX_LEN, hidden_size=64,
                    n_layers=2, n_heads=2, hidden_dropout_prob=0.2,
                    attn_dropout_prob=0.2, loss_type="CE").to(DEVICE)
    ds = BERT4RecDataset(bert_seqs, n_items=n_real, max_seq_len=MAX_LEN,
                         mask_token=n_model, mask_ratio=0.2)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    opt = optim.Adam(bert.parameters(), lr=1e-4)
    EP_B, PAT_B = 80, 10
    val_sub = list(range(0, len(users), max(1, len(users)//2000)))[:2000]
    best_b, best_bsd, noimp = -1.0, None, 0
    for ep in range(EP_B):
        loss, acc = train_one_epoch(bert, dl, opt, DEVICE)
        vnd = eval_bert(bert, users, DEVICE, ks=(10,), topk=10, subset=val_sub)["NDCG@10"]
        print(f"[BERT] ep {ep+1:02d}/{EP_B}  loss={loss:.4f} acc={acc:.2f}%  VAL NDCG@10={vnd:.4f}"
              + ("  <- best" if vnd > best_b+1e-5 else f" (best {best_b:.4f},{noimp+1}/{PAT_B})"))
        wb_log(run, {"bert/loss": loss, "bert/acc": acc, "bert/val_ndcg10": vnd, "bert/epoch": ep+1})
        if vnd > best_b + 1e-5:
            best_b = vnd; best_bsd = {k: v.detach().clone() for k, v in bert.state_dict().items()}; noimp = 0
        else:
            noimp += 1
            if noimp >= PAT_B:
                print(f"[BERT] EARLY STOP ep {ep+1}, best VAL NDCG@10={best_b:.4f}"); break
    if best_bsd: bert.load_state_dict(best_bsd)
    torch.save(bert.state_dict(), os.path.join(HERE, "bert4rec_longterm.pth"))
    wb_finish(run)

    # -------- TASK 2: train RL re-ranker --------
    bert.eval()
    for p in bert.parameters(): p.requires_grad = False
    run = wb_init("rl-rerank-ac", {"algo": "ActorCritic-PG (no TD3/BC)", "topN": 100, "K": 10})
    policy = train_rl(bert, users, DEVICE, epochs=30, bs=512, topN=100, K=10, patience=8, run=run)
    torch.save(policy.state_dict(), os.path.join(HERE, "rl_reranker.pth"))
    wb_finish(run)

    # -------- EVAL cả 2 trên TEST --------
    print("\n[*] ĐÁNH GIÁ TRÊN TEST (graded NDCG / Recall / MeanRating)")
    res_bert = eval_bert(bert, users, DEVICE, ks=(5,10,20))
    res_rl = eval_rl(bert, policy, users, DEVICE, target="test", ks=(5,10,20), topN=100)
    results = {"data_stats": stats,
               "BERT (retrained)": res_bert,
               "BERT + RL re-rank": res_rl}
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("[+] Xong. Lưu results.json")


if __name__ == "__main__":
    main()
