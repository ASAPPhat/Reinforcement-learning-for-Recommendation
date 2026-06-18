# Báo cáo: Sửa & Đánh giá Pipeline BERT4Rec + RL (Actor-Critic)

**Ngày:** 2026-06-18
**File:** `optimzie_bert_rl_pipeline.ipynb`
**Dữ liệu:** MovieLens-1M · **So sánh:** BERT4Rec (Sun et al. 2019)

---

## 1. Mục tiêu & câu hỏi

Ý tưởng ban đầu: dùng BERT4Rec đoán phim, rồi thêm một mạng RL (Actor-Critic) lên trên để **gợi ý tốt hơn** (vượt Hit@k/NDCG của BERT4Rec).

Câu hỏi cốt lõi cần trả lời: **RL có thực sự làm tốt hơn BERT không?**

---

## 2. Pipeline hoạt động thế nào (cách hiểu đơn giản)

```
Lịch sử xem phim ──► BERT4Rec (đóng băng) ──► "state" (1 vector mô tả sở thích user)
                                                   │
                                          Actor (mạng RL nhỏ)
                                                   │
                                              "action" (1 vector)
                                                   │
                              so độ giống action với từng phim ──► xếp hạng gợi ý
```

- **BERT4Rec**: đã train sẵn, **không học thêm** (đóng băng). Nó biến lịch sử xem → 1 vector dự đoán.
- **Actor**: mạng nhỏ, nhận vector của BERT, nhả ra vector mới.
- **Critic**: chấm điểm "vector này đáng giá bao nhiêu" dựa trên **rating** thật user chấm.
- Gợi ý = xếp phim theo độ giống với "action".

**Vấn đề tư duy ngay từ đầu:** vector của BERT (`state`) **đã chính là dự đoán phim kế tiếp rồi**. Actor lấy nó rồi nhả ra vector cũng để đoán phim kế tiếp → **RL và BERT làm CÙNG một việc**. Thêm RL = bắt một mạng chưa học vẽ lại thứ BERT đã làm tốt → dễ làm hỏng.

---

## 3. Các lỗi đã tìm thấy (giải thích đơn giản)

| # | Lỗi | Nói dễ hiểu | Hậu quả |
|---|-----|-------------|---------|
| 1 | **TD3+BC cài sai** | Công thức huấn luyện Actor để "điểm thưởng" (Q) lớn gấp ~10 lần phần "bắt chước item thật" → Actor bỏ bê việc bám phim thật, chạy lung tung. | Model **sụp đổ**, gợi ý ≈ ngẫu nhiên. |
| 2 | **beta quá lớn (0.5)** | Cho phép Actor đẩy vector đi **xa** vector BERT. | Rời khỏi vùng "đẹp" của BERT → kém. |
| 3 | **Rò rỉ dữ liệu** | Tập train chứa luôn phim dùng để test (đáp án). | Điểm bị **thổi phồng**, không trung thực. |
| 4 | **Lỗi padding** | Khi train, chuỗi bị chèn số 0 sai chỗ → vector lúc train **khác** lúc test. | Học một đằng, chấm một nẻo. |
| 5 | **Chấm điểm sai cách** | Lúc đánh giá dùng "cosine" và bỏ `bias`, trong khi BERT gốc chấm bằng "tích vô hướng + bias". | Tự cắt điểm của chính BERT. |
| 6 | **Chậm** | Mỗi epoch encode lại toàn bộ ~958k mẫu qua BERT (dù BERT đóng băng, kết quả không đổi). | Lãng phí thời gian khổng lồ. |
| 7 | **Không có early-stop** | Train cứng 50 epoch, lưu Actor cuối (đã trôi xa). | Lưu nhầm model tệ. |

---

## 4. Đã sửa những gì & sửa thế nào

| # | Trước (sai) | Sau (đã sửa) |
|---|-------------|--------------|
| 1 | `actor_loss = -Q + 2.5·MSE` (Q thô, áp đảo) | **Chuẩn hoá Q**: `lmbda = 2.5 / |Q|trung_bình`; `loss = -lmbda·Q + MSE`. Giờ phần "bắt chước phim thật" cân được với điểm thưởng → Actor không chạy lung tung nữa. |
| 2 | `beta = 0.5` | `beta = 0.05` → Actor chỉ **nhích nhẹ** quanh vector BERT, không bay xa. |
| 3 | Train dùng cả phim cuối | **Leave-one-out**: phim cuối = test, áp cuối = validation, còn lại = train. Không trùng đáp án. |
| 4 | `collate_fn` chèn số 0 → độ dài sai | Bỏ chèn 0, trả lịch sử **thô** → vector lúc train = lúc test. |
| 5 | (đánh giá nội bộ) cosine, bỏ bias | Thêm bảng đánh giá **đúng chuẩn paper**: tích vô hướng + `bias`, negative theo độ phổ biến. |
| 6 | Encode lại mỗi epoch | **Encode 1 lần, lưu cache**; 50 epoch sau chỉ chạy update RL → nhanh hơn ~50 lần/epoch. |
| 7 | Không có | **Early-stop**: mỗi epoch chấm Hit@10 trên validation, giữ **Actor tốt nhất**, dừng nếu 5 epoch liền không cải thiện. |

**Thêm mới:** bộ hàm đánh giá công bằng (`compute_actions`, `eval_full`, `eval_sampled`), tạo 2 tập test (`leave_one_out`, `liked r≥4`), bảng so **BERT vs RL**, và bảng **so với paper**.

---

## 5. Logic/kiến trúc: TRƯỚC vs SAU

**TRƯỚC (hỏng):**
```
mỗi epoch: encode lại 958k mẫu qua BERT  ──►  train Actor với công thức sai (Q áp đảo)
           Actor bay khỏi vùng BERT  ──►  chấm bằng cosine, có rò rỉ, lưu Actor cuối
→ Kết quả: gợi ý ≈ ngẫu nhiên (Hit@10 full-rank = 0.01)
```

**SAU (đúng):**
```
ENCODE 1 LẦN ─► cache (S, A, R, next-S)
        │
   train Actor: TD3+BC chuẩn (Q chuẩn hoá), beta nhỏ ─► Actor bám sát BERT, chỉ nhích nhẹ
        │
   mỗi epoch: chấm VAL Hit@10 ─► giữ BEST ─► EARLY STOP
        │
   đánh giá: (a) cosine nội bộ BERT-vs-RL  (b) đúng chuẩn paper (dot+bias, popularity)
→ Kết quả: số thực, hợp lý, công bằng, tái lập được
```

Điểm mấu chốt: kiến trúc **không đổi bản chất** (vẫn Actor → vector → xếp hạng), nhưng giờ Actor **khởi đầu = BERT** và chỉ nhích nhẹ, cộng cách đo trung thực → ta thấy được **sự thật**.

---

## 6. Kết quả sau khi sửa

### 6.1. Hết "sụp đổ"
- Trước: full-rank Hit@10 = **0.0101** (≈ ngẫu nhiên).
- Sau: full-rank Hit@10 ≈ **0.19**, sampled ≈ **0.79**. Số thật.

### 6.2. Đường validation lúc train (rất quan trọng)
```
VAL Hit@10:  0.1915 → 0.062 → 0.030 → 0.030 → 0.031 → 0.026  → EARLY STOP (epoch 16)
```
→ **Càng train RL, gợi ý càng tệ.** Early-stop giữ lại Actor ở điểm đầu (≈ BERT). Loss nhìn "đẹp" nhưng đánh lừa — chỉ chỉ số validation mới lộ ra sự thật.

### 6.3. So BERT vs RL (đo nội bộ, cosine) — leave-one-out

| | Hit@10 (full-rank) | Hit@10 (sampled-100) |
|---|---|---|
| BERT (không RL) | **0.1932** | **0.7885** |
| RL (TD3+BC) | 0.1911 | 0.7875 |

→ RL **bằng hoặc thua nhẹ** BERT. Không thêm giá trị.

### 6.4. So với PAPER (đúng chuẩn: dot+bias, 100 neg popularity)

| Model | HR@1 | HR@5 | HR@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| **Paper BERT4Rec** | 0.2863 | 0.5876 | **0.6970** | 0.4818 | 0.4254 |
| **BERT (của bạn)** | 0.2775 | 0.5647 | **0.6785** | 0.4658 | 0.4115 |
| RL (TD3+BC) | 0.1055 | 0.3411 | 0.5058 | 0.2783 | 0.2289 |

**Đọc bảng:**
- **BERT của bạn đạt chuẩn paper** (HR@10 0.6785 vs 0.6970, lệch ~2.7% do pool item 3707 vs 3416 + khác seed). → BERT4Rec train **đúng, tốt**.
- RL thấp hẳn (0.5058) ở đây **không phải vì RL kém thật**, mà vì RL nhả ra vector **đã chuẩn hoá (mất độ lớn)**, không khớp cách chấm "dot+bias" của BERT. Tức RL **hạ cấp** cách chấm điểm mạnh của BERT xuống cosine → mất thông tin.

---

## 7. Kết luận

1. **Pipeline giờ đúng đắn, trung thực, tái lập được.** Đã diệt: sụp đổ, rò rỉ, padding sai, chấm sai, chậm; thêm early-stop.
2. **BERT4Rec của bạn ngang paper** (~0.68 vs 0.70 HR@10). Đây là baseline thật.
3. **RL KHÔNG vượt BERT.** Lý do gốc: RL và BERT làm cùng một việc (đoán phim kế tiếp), nên trần của RL chính là BERT. Tệ hơn, việc RL chuẩn hoá vector còn làm mất thông tin "độ lớn + bias" mà BERT đang dùng → kém tương thích.
4. **Bài học:** không thể vượt BERT bằng cách thay cách chấm điểm của nó (dot+bias) bằng cosine.

---

## 8. Muốn THỰC SỰ vượt BERT → đổi hướng (chưa làm)

- **Cách 1 — Rerank trên điểm BERT:** giữ nguyên `điểm = state·item + bias` của BERT, RL chỉ **cộng thêm một chỉnh sửa nhỏ Δ**, khởi đầu Δ = 0 (tức bằng đúng BERT) → đảm bảo **không bao giờ tệ hơn BERT**, chỉ có thể tốt lên.
- **Cách 2 — Đổi mục tiêu:** cho RL tối ưu thứ BERT không làm (giá trị xem dài hạn, độ đa dạng, sắp xếp cả danh sách) — nhưng khi đó dùng thước đo khác, không còn là Hit@k next-item.
