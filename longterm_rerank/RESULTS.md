# Long-term Re-rank: BERT4Rec vs BERT4Rec + RL (Actor-Critic, no TD3+BC)

**Ngày:** 2026-06-19 · **Dataset:** MovieLens-1M · **Device:** CUDA
**wandb:** [project bert4rec-longterm-rerank](https://wandb.ai/lamgiang-fpt-university/bert4rec-longterm-rerank) — runs: BERT `wdhexav1`, RL `ua4zygqb`

Mục tiêu: đổi sang đánh giá **dài-hạn / nhiều-item theo rating** (graded NDCG), so **BERT4Rec** (retrain) với **BERT4Rec + RL re-rank top-N**. RL **không** dùng TD3+BC.

---

## 1. Chống bias — cách chia dữ liệu (TIME-SPLIT)

Mỗi user (sắp theo thời gian, ≥5 tương tác):
```
[ ───────── remaining (80% đầu) ───────── ][ ── test (20% cuối) ── ]
   dùng train BERT + reward RL                  CHỈ để đánh giá
```
- **BERT** học Cloze **chỉ trên remaining** → item test **không bao giờ** nằm trong chuỗi train ⇒ **không leakage**.
- **RL** lấy reward từ `rl_window` = 20% cuối của *remaining* (history = phần trước đó) ⇒ **không đụng test**.
- **Eval**: history = remaining, relevant = test (rating làm gain) — **cùng code, cùng k** cho cả 2 model ⇒ công bằng tuyệt đối.

### Thống kê dữ liệu đã chia
| | Giá trị |
|---|---|
| #users | 6,040 |
| #items (phim thật) | 3,706 |
| #tương tác | 1,000,209 |
| train (remaining) | 800,193 |
| test | 200,016 |
| avg độ dài chuỗi | 165.6 |
| avg #item test / user | 33.12 |
| test_frac | 0.20 |

---

## 2. Kiến trúc 2 pipeline

### Pipeline A — BERT4Rec (baseline, retrain)
- **Giữ nguyên kiến trúc gốc** (`Bert4Rec_model.py`): 2 lớp Transformer, hidden 64, 2 heads, dropout 0.2, Cloze (mask_ratio 0.2), loss CE.
- Train **từ đầu** trên `remaining` (time-split mới) → tránh leak.
- Chấm điểm gốc: `score = state · item_emb + output_bias` (dot+bias), full-rank.
- Early-stop theo VAL graded NDCG@10. **Best VAL = 0.0400** (early-stop epoch 23).

### Pipeline B — BERT + RL re-rank top-N (Actor-Critic policy-gradient, KHÔNG TD3+BC)
```
history ─► BERT (đóng băng) ─► top-N=100 ứng viên (theo dot+bias)
                                       │
        state ⊕ embedding ứng viên ─► PolicyNet (Actor) ─► điểm mỗi ứng viên ─► sắp lại
                                       │   ValueNet (Critic) = baseline
                  reward = graded NDCG@10 (gain = 2^rating − 1) của thứ tự sinh ra
```
- **Actor** `PolicyNet([state, cand_emb]) → score` — **init ngẫu nhiên** (thuần RL, không neo BERT).
- **Critic** `ValueNet(state) → V` — baseline giảm variance.
- **Thuật toán:** REINFORCE/advantage — `actor_loss = −(reward − V)·logπ`, `critic_loss = MSE(V, reward)`. **Không** twin-critic / delayed / behavior-cloning (khác TD3+BC).
- Sinh thứ tự: lấy mẫu top-K=10 kiểu Plackett-Luce (softmax tuần tự không lặp).
- Reward = chính metric đánh giá (graded NDCG) → RL tối ưu **trực tiếp** thứ ta đo.
- Early-stop theo VAL NDCG@10. **Best VAL = 0.0397** (early-stop epoch 25).

---

## 3. Mấu chốt (key points)
1. **Time-split + reward từ train-window** → không rò rỉ test ở cả 2 pipeline.
2. **BERT đóng băng** trong pipeline B → RL chỉ học sắp xếp, không phá encoder.
3. **Reward = graded NDCG (rating gain)** → RL tối ưu đúng metric (khác các lần trước reward=rating-1-item lệch Hit@k).
4. **Init policy ngẫu nhiên** (theo yêu cầu) → RL phải tự học sắp xếp top-N từ đầu, **không** có sàn ≥ BERT.
5. Cùng tập test, cùng hàm metric → so sánh fair.

---

## 4. Kết quả cuối (TEST, graded NDCG / Recall / MeanRating — full-rank 3706 items)

| Metric | BERT4Rec (retrain) | BERT + RL re-rank | Δ |
|---|---|---|---|
| **NDCG@5**  | **0.0345** | 0.0325 | −5.6% |
| **NDCG@10** | **0.0369** | 0.0348 | −5.8% |
| **NDCG@20** | **0.0487** | 0.0433 | −11.1% |
| **Recall@10** | **0.0220** | 0.0217 | −1.6% |
| **Recall@20** | **0.0474** | 0.0418 | −11.7% |
| **MeanRating@10** | **1.221** | 1.194 | −2.2% |

→ **BERT4Rec thắng nhẹ ở mọi metric.** RL re-rank không vượt.

---

## 5. So sánh & Insight

1. **RL không vượt BERT — kể cả khi tối ưu ĐÚNG metric.**
   Các lần trước RL thua vì reward (rating-1-item) lệch Hit@k. Lần này reward = **chính graded NDCG** — vậy mà RL **vẫn thua sát**. ⇒ Vấn đề sâu hơn cả "lệch mục tiêu".

2. **Lý do gốc: thứ tự top-N của BERT đã quá mạnh.**
   Policy **init ngẫu nhiên** phải học sắp lại 100 ứng viên từ con số 0. BERT đã sắp 100 ứng viên đó theo tín hiệu học sâu; RL từ-đầu học lại → **mất mát**, không bù nổi. Train reward chỉ ~0.05, VAL ~0.04 — RL học được chút nhưng dưới BERT.

3. **Rating tương quan sẵn với next-item của BERT.**
   Người ta thường xem phim mình sẽ thích → thứ tự "dễ xem tiếp" của BERT **đã** đẩy phim rating cao lên khá tốt. Khoảng trống cho RL khai thác rating **rất hẹp**.

4. **Init ngẫu nhiên = không có sàn.**
   Vì policy không neo BERT, nó **không được đảm bảo ≥ BERT** và thực tế tụt dưới. Nếu init = BERT (additive trên logit) thì tệ nhất = BERT — nhưng đó là lựa chọn thiết kế khác.

5. **Baseline BERT (time-split) yếu tuyệt đối** (NDCG@10 ~0.037 full-rank, masked-acc ~0.7%): do time-split khó (chỉ 80% history, 33 item test/user, full-rank 3706) + train ngắn. Tuy yếu, nó vẫn là **baseline công bằng** vì cả 2 dùng chung.

### Kết luận
> Trên đánh giá rating-aware (graded NDCG) với time-split không bias, **RL re-rank top-N (Actor-Critic, init ngẫu nhiên) KHÔNG vượt BERT4Rec** — thua nhẹ ở mọi metric. Ngay cả khi RL tối ưu trực tiếp metric đánh giá, thứ tự top-N của BERT vẫn quá mạnh để một policy học-từ-đầu vượt qua, và rating đã tương quan sẵn với tín hiệu của BERT nên cửa cải thiện rất hẹp.

### Hướng tiếp (nếu muốn RL thắng)
- **Init policy = BERT (additive residual trên logit)** → đảm bảo sàn ≥ BERT, RL chỉ nhích lên.
- **Reward đa mục tiêu**: graded NDCG **+ đa dạng (ILD/genre) + novelty** — thứ BERT *thật sự* không tối ưu → cửa thắng rộng hơn rating.
- Ablation: BERT vs +RL trên từng thành phần reward để tách đóng góp.

---

## 6. File trong folder
- `common.py` — split không-bias, metrics graded NDCG.
- `run_pipeline.py` — train BERT + RL + wandb.
- `eval_only.py` — chấm test → `results.json`.
- `bert4rec_longterm.pth`, `rl_reranker.pth` — best checkpoints.
- `data_stats.json`, `results.json` — số liệu.
- `train_run.log` — log train.
