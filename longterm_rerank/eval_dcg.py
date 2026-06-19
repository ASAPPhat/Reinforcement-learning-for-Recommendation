"""Đánh giá 2 pipeline (best model) trên DCG (raw) + NDCG, graded gain=2^rating-1."""
import os, sys, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common, run_pipeline as R
from common import load_and_split, BERT4Rec, dcg_at_k, ndcg_at_k, recall_at_k, encode_states

dev = R.DEVICE; common.set_seed(42)
users, seqs, n_real, m2i, stats = load_and_split(R.RATING_FILE, max_len=R.MAX_LEN)
n_model = n_real + 1
KS = (5, 10, 20)

bert = BERT4Rec(n_items=n_model, max_seq_length=R.MAX_LEN, hidden_size=64, n_layers=2, n_heads=2,
                hidden_dropout_prob=0.2, attn_dropout_prob=0.2, loss_type="CE").to(dev)
bert.load_state_dict(torch.load(os.path.join(HERE, "bert4rec_longterm.pth"), map_location=dev)); bert.eval()
for p in bert.parameters(): p.requires_grad = False
policy = R.PolicyNet(bert.hidden_size).to(dev)
policy.load_state_dict(torch.load(os.path.join(HERE, "rl_reranker.pth"), map_location=dev)); policy.eval()

def rankings_bert():
    item_emb = bert.item_embedding.weight[:bert.n_items]; bias = bert.output_bias
    st = encode_states(bert, [u["hist_items"] for u in users], dev, R.MAX_LEN)
    out = []
    for i in range(0, st.shape[0], 256):
        sc = st[i:i+256] @ item_emb.T + bias; sc[:, 0] = -1e9
        out += torch.topk(sc, max(KS), 1).indices.cpu().numpy().tolist()
    return out

@torch.no_grad()
def rankings_rl():
    st = encode_states(bert, [u["hist_items"] for u in users], dev, R.MAX_LEN)
    out = []
    for i in range(0, st.shape[0], 256):
        cand_ids, cand_emb, cand_logit = R._bert_candidates(bert, st[i:i+256], 100)
        order = torch.argsort(policy(st[i:i+256], cand_emb, cand_logit), 1, descending=True)
        out += torch.gather(cand_ids, 1, order).cpu().numpy().tolist()
    return out

rels = [u["test_rel"] for u in users]
def agg(ranks):
    m = {}
    for k in KS:
        m[f"DCG@{k}"]  = float(np.mean([dcg_at_k(r, rel, k)  for r, rel in zip(ranks, rels)]))
        m[f"NDCG@{k}"] = float(np.mean([ndcg_at_k(r, rel, k) for r, rel in zip(ranks, rels)]))
        m[f"Recall@{k}"] = float(np.mean([recall_at_k(r, rel, k) for r, rel in zip(ranks, rels)]))
    return m

res = {"data_stats": stats, "BERT (best)": agg(rankings_bert()), "BERT+RL re-rank (best)": agg(rankings_rl())}
json.dump(res, open(os.path.join(HERE, "results_dcg.json"), "w"), ensure_ascii=False, indent=2)

COLS = [f"{p}@{k}" for k in KS for p in ("DCG", "NDCG", "Recall")]
print(f"{'Model':<24}" + "".join(f"{c:>11}" for c in COLS))
print("-" * (24 + 11 * len(COLS)))
for name in ("BERT (best)", "BERT+RL re-rank (best)"):
    m = res[name]; print(f"{name:<24}" + "".join(f"{m[c]:>11.4f}" for c in COLS))
print("[+] saved results_dcg.json")
