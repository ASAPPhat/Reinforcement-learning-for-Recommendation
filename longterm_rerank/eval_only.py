"""Eval-only: nạp checkpoint đã train, chấm TEST, dump results.json (ascii-safe)."""
import os, sys, json, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common, run_pipeline as R
from common import load_and_split, BERT4Rec

dev = R.DEVICE
common.set_seed(42)
users, seqs, n_real, m2i, stats = load_and_split(R.RATING_FILE, max_len=R.MAX_LEN)
n_model = n_real + 1

bert = BERT4Rec(n_items=n_model, max_seq_length=R.MAX_LEN, hidden_size=64,
                n_layers=2, n_heads=2, hidden_dropout_prob=0.2,
                attn_dropout_prob=0.2, loss_type="CE").to(dev)
bert.load_state_dict(torch.load(os.path.join(HERE, "bert4rec_longterm.pth"), map_location=dev))
bert.eval()
for p in bert.parameters(): p.requires_grad = False

policy = R.PolicyNet(bert.hidden_size).to(dev)
policy.load_state_dict(torch.load(os.path.join(HERE, "rl_reranker.pth"), map_location=dev))
policy.eval()

print("[*] Eval BERT (full-rank graded NDCG)...")
res_bert = R.eval_bert(bert, users, dev, ks=(5, 10, 20))
print("[*] Eval BERT+RL re-rank...")
res_rl = R.eval_rl(bert, policy, users, dev, target="test", ks=(5, 10, 20), topN=100)

results = {"data_stats": stats,
           "BERT (retrained)": res_bert,
           "BERT + RL re-rank": res_rl}
json.dump(results, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=2)
print(json.dumps(results, ensure_ascii=False, indent=2))
print("[+] saved results.json")
