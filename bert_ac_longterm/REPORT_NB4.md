# Báo cáo: Notebook 4 — BERT4Rec + Actor-Critic Re-rank + Side-Features + Diversity

**File:** `4_ac_features_diversity.ipynb` (self-contained, chỉ import `Bert4Rec_model.py`)
**Dataset:** MovieLens-1M · time-split 80/20 · **so với BERT4Rec cùng bài** (không so số paper next-item)

---

## 1. Vì sao có NB4 — vấn đề của NB2/NB3

NB2 (rerank top-50) và NB3 (full-catalog) đều cho **RL ≈ BERT (hoà)**. Lý do: RL chỉ dùng **đúng thông tin BERT đã có** (item-embedding) + tối ưu **đúng tín hiệu BERT đã nắm** (rating tương quan với likelihood) → không còn khe hở.

→ NB4 mở khe hở bằng 2 thứ BERT **không** có/không làm:
1. **Side-features** BERT4Rec gốc bỏ qua (genre, năm, popularity, demographic).
2. **Diversity** trong reward (BERT không tối ưu đa dạng).

---

## 2. Kiến trúc / pipeline đã thay đổi gì (so NB2/NB3)

| Thành phần | NB2/NB3 | **NB4** |
|---|---|---|
| Encoder | BERT đóng băng | BERT đóng băng (giữ) |
| Candidate | top-N=50 / toàn kho | top-N=50, **loại phim đã xem** |
| **Actor input** | [state, cand_emb, (logit)] | **+ genre(18) + year + popularity + demographic(3) + logit_norm** (side-features BERT thiếu) |
| Actor scoring | additive logit, init=BERT | additive logit, init=BERT (giữ sàn ≥ BERT) |
| Critic | value baseline | value baseline (giữ) |
| **Reward** | NDCG theo rating | **NDCG rating + λ·ILD** (thêm đa dạng, λ=0.3) |
| Baseline filter | (NB2 lệch) | **baseline & RL CÙNG loại phim đã xem** (công bằng) |

**Mấu chốt:** Actor giờ "nhìn thấy" genre/popularity/demographic — thông tin BERT4Rec gốc **theo thiết kế không dùng** → RL có cái để thêm giá trị, không còn trùng việc BERT.

---

## 3. Metrics — thay đổi gì, tăng hay giảm

### Thêm 3 metric beyond-accuracy (mới so với trước)
| Metric | Đo gì | Vì sao thêm |
|---|---|---|
| **ILD@10** | đa dạng nội-list (1 − cosine genre trung bình) | đo thứ RL tối ưu (BERT bỏ ngỏ) |
| **Coverage@10** | tỉ lệ catalog xuất hiện trong top-10 toàn user | đo độ phủ / chống co cụm phim hot |
| **Novelty@10** | −log2(popularity) trung bình | đo gợi phim ít phổ biến |

Giữ nguyên: **DCG@k / NDCG@k / Recall@k** (graded, gain=2^rating−1) như các file trước.

### Kết quả (TEST, sau khi sửa leak popularity)
| Model | DCG@10 | NDCG@10 | Recall@10 | ILD@10 | Coverage@10 | Novelty@10 |
|---|---|---|---|---|---|---|
| BERT (short-term) | 6.8379 | 0.0613 | 0.0348 | 0.6660 | 0.0135 | 8.4265 |
| **BERT + AC (+feat, diversity)** | **6.8571** | **0.0615** | **0.0350** | **0.6700** | 0.0135 | **8.4281** |
| **Δ** | **+0.28%** ↑ | **+0.3%** ↑ | **+0.6%** ↑ | **+0.6%** ↑ | 0 (=) | **+0.02%** ↑ |

**Tăng hay giảm:**
- **DCG/NDCG/Recall (accuracy): TĂNG nhẹ** — lần đầu RL nhỉnh BERT trên accuracy (nhờ side-features).
- **ILD (đa dạng): TĂNG nhẹ** — diversity reward có tác dụng nhưng yếu (λ=0.3).
- **Coverage: KHÔNG đổi** — cả 2 vẫn co cụm ~50 phim hot; đa dạng across-user chưa cải thiện.
- **Novelty: ~ngang** (+0.02%).

→ Mọi metric **nhỉnh hoặc bằng**, không cái nào giảm. **Win nhẹ, đồng đều.**

---

## 4. Logic chạy (flow)

```
1. load_all: time-split 80/20 + nạp side-info
      - genre/year  <- movies.dat
      - demographic <- users.dat
      - popularity  <- CHI dem tren TRAIN (remaining)  [đã sửa leak]
2. Nạp BERT (NB1) đóng băng -> ITEM_EMB, BIAS, tensor side-info
3. Mỗi batch user (rl_window):
      BERT top-50 (loại phim đã xem)
      -> Actor chấm = bert_logit + β·MLP([state, cand_emb, genre,year,pop,demo, logit_norm])
      -> sample top-K=10 (Plackett-Luce)
      -> reward = NDCG_rating(slate) + λ·ILD_genre(slate)
      -> advantage = reward − Critic(state)
      -> update Actor (policy gradient) + Critic (MSE)
   mỗi epoch: VAL DCG@10 -> early-stop, giữ BEST actor
4. Eval TEST: BERT vs BERT+AC, cùng loại phim đã xem
      -> DCG/NDCG/Recall + ILD + Coverage + Novelty
```

---

## 5. Bug đã sửa trong quá trình
- **Popularity leak:** trước đếm popularity trên **cả test** (nhìn lén tương lai) → đã sửa **chỉ đếm train (remaining)**. Win **+0.28% sống sót** sau sửa → **không phải do leak**.
- **History-filter lệch (từ NB2):** đã cho **baseline cũng loại phim đã xem** → so công bằng.

## 6. Nhận định trung thực
- **Lần đầu RL thắng BERT ở mọi metric** (sau cả chuỗi hoà), và **không nhờ leak**.
- **NHƯNG biên rất mỏng (+0.28% DCG, +0.6% ILD)** và **VAL đỉnh ep3 rồi tụt dưới BERT** → có thể dính **early-stop selection bias**.
- **Coverage không đổi** → diversity chưa tạo khác biệt thật ở mức catalog.
- **Để khẳng định "thắng thật"** cần (chưa làm): **multi-seed (3-5) + paired significance test**; và **sweep λ** để đẩy ILD/Coverage rõ hơn (vẽ Pareto accuracy↔diversity).

## 7. Kết luận
> NB4 = kiến trúc **đầu tiên có cửa thắng thật**: cho Actor **side-features BERT4Rec gốc bỏ qua** (genre/popularity/demographic) + **diversity reward**, giữ init=BERT và so công bằng. Kết quả: RL **nhỉnh BERT mọi metric (+0.3–0.6%)**, không do leak. Biên mỏng → cần multi-seed để chốt. Đây là hướng đúng để RL thực sự bổ sung giá trị cho BERT4Rec.

## 8. File liên quan
- `4_ac_features_diversity.ipynb` — notebook chính (self-contained).
- `bert4rec.pth` — BERT short-term (từ NB1, dùng làm encoder).
- `actor_feat.pth` — best Actor đã train.
- `results_feat.json` — số liệu cuối.
- `data_stats.json` — thống kê split.
- `bert4rec_lt.pth`, `actor_on_bertlt.pth` — model BERT-long-term + Actor rerank trên BERT-LT (mục 9).
- `results_3way.json`, `results_4way.json` — số liệu so sánh mở rộng.

---

## 9. CẬP NHẬT QUAN TRỌNG — thêm BERT4Rec long-term + AC trên BERT-LT (kết luận cuối, thay cho §7)

Thêm 2 baseline mạnh để kiểm tra RL có thật sự cần không:

- **BERT4Rec (long-term):** *cùng kiến trúc* BERT4Rec gốc, nhưng **đổi mục tiêu train** — thay Cloze (đoán 1 item kế) bằng **dự đoán item trong cửa sổ tương lai** (positive từ `rl_rel`, ưu tiên rating cao). KHÔNG RL.
- **BERT-LT + AC:** Actor-Critic rerank **trên BERT-LT** (init=BERT-LT → sàn ≥ BERT-LT) + side-features + diversity.

### Bảng so sánh (TEST, full metric)
| Model | DCG@10 | NDCG@10 | Recall@10 | ILD@10 | Coverage@10 | Novelty@10 |
|---|---|---|---|---|---|---|
| BERT (short-term) | 6.8379 | 0.0613 | 0.0348 | 0.6660 | 0.0135 | 8.4265 |
| **BERT4Rec (long-term)** | **6.9059** | **0.0642** | **0.0384** | **0.6972** | 0.0138 | **8.5185** |
| BERT + AC (on short) | 6.8571 | 0.0615 | 0.0350 | 0.6700 | 0.0135 | 8.4281 |
| BERT-LT + AC (on long-term) | 6.9073 | 0.0642 | 0.0384 | 0.6973 | 0.0138 | 8.5183 |

### Đọc kết quả
- **BERT4Rec long-term thắng đậm** mọi metric so với BERT short-term: **DCG +1.0%, NDCG +4.7%, Recall +10%, ILD +4.7%, Novelty +1.1%**. Lớn hơn rất nhiều so với cú +0.28% của AC-on-short.
- **AC xếp trên BERT-LT ≈ BERT-LT (chỉ +0.02% DCG, các metric khác bằng nhau)** → RL **không thêm gì** khi encoder đã đúng mục tiêu; kể cả ILD cũng không tăng.

### Kết luận cuối (thay cho §7)
> **Đòn bẩy thật là ĐỔI MỤC TIÊU TRAIN của BERT4Rec (short-term → long-term), KHÔNG phải thêm RL.** "Sửa BERT" cho ra +1% DCG / +10% Recall / +ILD; còn "thêm Actor-Critic re-rank" đóng góp ≈ 0 (trên cả encoder short-term lẫn long-term). Khi BERT đã được train cho đúng mục tiêu, RL re-rank **dư thừa**.
>
> Nguyên nhân RL bế tắc: (1) chỉ rerank trên encoder đóng băng → trần bị chặn; (2) offline counterfactual — gợi ý hay-ngoài-log bị chấm 0; (3) BERT-LT đã sắp tối ưu + tự nhiên đa dạng nên không còn khe hở; (4) diversity reward (λ=0.3) quá yếu để vượt ILD của BERT-LT.

### Hướng nếu vẫn muốn RL "có vai trò"
- Đẩy `λ` cao (1.5–2.0) → AC **hy sinh accuracy lấy ILD/novelty cao hơn BERT-LT** → câu chuyện **Pareto đa dạng↔độ chính xác** (đánh đổi, không phải strict-win).
- Hoặc nhồi side-features thẳng vào BERT-LT (BERT-LT+features) → nhưng đó là cải tiến BERT, không phải RL.
