# Cấu hình thực nghiệm — BERT4Rec (short-term) + Actor-Critic (long-term)

Tóm tắt config để viết phần Thực nghiệm. Kết quả số cập nhật ở cuối sau khi chạy.

## 1. Bài toán
- **BERT4Rec (short-term):** đoán phim user **xem tiếp** (next-item), bất kể rating.
- **RL Actor-Critic (long-term):** giữ ứng viên của BERT nhưng **đẩy phim rating cao (4-5★) lên đầu** → tối ưu chất lượng/hài lòng dài hạn thay vì chỉ "có xem".
- Ví dụ: BERT gợi a→b→c (b,c chỉ 2-3★); RL sắp lại để ưu tiên phim 4-5★.

## 2. Dữ liệu & Split (chống bias)
- Dataset: **MovieLens-1M** — 6,040 users · 3,706 phim · 1,000,209 rating.
- **Time-split / user:** sắp theo timestamp; **20% tương tác cuối = TEST**, 80% đầu = train (`remaining`).
- RL reward lấy từ `rl_window` = **20% cuối của remaining** (history = phần trước); **không** đụng test → không leakage.
- Eval: history = remaining, **relevant = phim trong TEST với gain = 2^rating − 1** (ưu tiên 5★).
- Thống kê: avg seq len ≈ 165.6 ; avg #item test/user ≈ 33.

## 3. BERT4Rec (kiến trúc gốc, giữ nguyên) — `Bert4Rec_model.py`
| Tham số | Giá trị |
|---|---|
| hidden_size | 64 |
| n_layers | 2 |
| n_heads | 2 |
| max_seq_length | 200 |
| dropout (hidden/attn) | 0.2 / 0.2 |
| mask_ratio (Cloze) | 0.2 |
| loss | CE (full-softmax) |
| optimizer | Adam, lr 1e-4 |
| scoring | dot + output_bias |
| epochs / early-stop | ≤80, patience 10 (theo VAL NDCG@10) |
| vai trò | encoder (đóng băng) + sinh top-N candidates |

## 4. Actor-Critic long-term (KHÔNG TD3 / KHÔNG BC)
- **Thuật toán:** REINFORCE + value baseline (Actor-Critic policy-gradient, 1 bước / bandit). Không twin-critic, không delayed, không BC, không bootstrap.
- **Candidate:** BERT top-**N = 50**.
- **Actor (PolicyNet):** `score = bert_logit + β·Δ(state, cand_emb, bert_logit_norm)`, **lớp cuối init = 0 → Δ=0 → khởi đầu = thứ tự BERT** ⇒ **sàn ≥ BERT**. Thêm điểm BERT (chuẩn hoá) làm feature.
- **Critic (ValueNet):** `V(state)` — baseline giảm variance. `advantage = reward − V`.
- **Reward:** graded **NDCG@K (K=10)**, gain = 2^rating − 1 trên `rl_window` → ưu tiên phim cao sao.
- **Sinh slate:** lấy mẫu top-K kiểu Plackett-Luce (softmax tuần tự) lúc train; greedy (argsort) lúc test.
- **Loss:** `actor = −(adv.detach())·logπ` ; `critic = MSE(V, reward)`.
- optimizer Adam lr 1e-4 ; epochs ≤30, patience 8 (theo VAL DCG@10) ; giữ BEST.

## 5. Đánh giá (long-term, DCG)
- Quy trình test: BERT top-50 → Actor sắp lại → đo trên thứ tự cuối.
- **Metric:** DCG@{5,10,20}, NDCG@{5,10,20}, Recall@{5,10,20} ; gain = 2^rating − 1 (graded theo rating).
- **Baseline:** BERT top-K cùng bài (so công bằng, KHÔNG so số paper next-item).

## 6. Log & Tái lập
- wandb project `bert4rec-ac-longterm` (runs: `bert-shortterm`, `actor-critic-longterm`); chart actor/critic loss + VAL.
- seed = 42 (random/numpy/torch + cudnn deterministic).
- best model: `bert4rec.pth`, `actor_longterm.pth`.

## 7. Kết quả (TEST, long-term DCG, gain=2^rating−1)

| Model | DCG@5 | DCG@10 | DCG@20 | NDCG@10 | Recall@10 |
|---|---|---|---|---|---|
| BERT (short-term) | 2.7359 | 3.9998 | 6.3018 | 0.0367 | 0.0220 |
| BERT + Actor-Critic (long-term) | 2.7359 | 3.9998 | 6.3018 | 0.0367 | 0.0220 |

- BERT retrain: early-stop epoch 26, best VAL NDCG@10 = 0.0396 (VAL trên rl-window, không đụng test).
- Actor-Critic: VAL DCG@10 init(=BERT) = **4.0847**; early-stop epoch 8, best = **4.0847** = đúng mức init.
- (2 notebook self-contained, chỉ import `Bert4Rec_model`; số có thể lệch nhẹ so lần chạy module-based do đổi VAL sang rl-window.)

### Nhận định (trung thực)
- **BERT + Actor-Critic = BERT y hệt** trên mọi metric. Sàn init=BERT **giữ vững** (hết hiện tượng tụt như bản init-ngẫu-nhiên trước đó), nhưng Actor **không tìm được cách sắp tốt hơn** BERT → hội tụ về delta≈0 (giữ thứ tự BERT).
- **Lý do:** thứ tự top-50 của BERT **đã tương quan sẵn với rating** (user thường xem phim sẽ thích) → reward rating-NDCG **không còn khe hở** để Actor khai thác trên cùng tập ứng viên đó.
- **Kết luận:** trên DCG rating-aware (long-term) với kiến trúc đúng (init=BERT, no TD3/BC), RL re-rank **hoà BERT** — không vượt được, nhưng cũng không hại. Trần = BERT vẫn đúng cả ở mục tiêu rating.

### Muốn vượt thật (chưa làm)
- **Đa dạng (ILD):** thêm `−λ·độ-giống` vào reward → RL chắc chắn thắng ở ILD (BERT không tối ưu đa dạng). Đây là khe hở rộng nhất còn lại.
- **Relevant = chỉ ≥4★** + Critic dự đoán rating sắc hơn (nhiều feature CF) → có thể nới khe hở 5★.
