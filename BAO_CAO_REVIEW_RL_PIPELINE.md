# Báo cáo Review Pipeline BERT4Rec + Actor-Critic (Offline RL)

**Ngày:** 2026-06-18
**Phạm vi:** `optimzie_bert_rl_pipeline.ipynb`, `Bert4Rec_model.py`
**Dataset:** MovieLens-1M
**Baseline tham chiếu:** BERT4Rec (Sun et al., 2019)

---

## 1. Bối cảnh & Vấn đề lõi

Kiến trúc hiện tại: BERT4Rec (đóng băng) encode lịch sử → `state` (vector 64-dim), Actor MLP biến `state` → `action`, rồi xếp hạng item bằng độ tương đồng giữa `action` và bảng item embedding.

**Vấn đề bản chất:** `state` mà BERT4Rec sinh ra **đã chính là dự đoán item kế tiếp** (BERT chấm điểm bằng `state · item_embedding + output_bias`). Actor lấy `state` đó rồi sinh ra một vector cũng nằm trong **cùng không gian item embedding**, đánh giá bằng cùng cơ chế truy hồi. Nghĩa là **RL và BERT đang làm chung một task (đoán next-item)**, RL chỉ là một mạng neural thừa đẩy vector "đẹp sẵn" của BERT sang một vùng nhiễu hơn, chưa được học.

Hệ quả: trần hiệu năng của pipeline = BERT4Rec. RL **không thể vượt** BERT trên metric next-item (Hit@k/NDCG) nếu không thay đổi hướng tiếp cận. Tốt nhất chỉ **hòa** BERT, thực tế thường **kém hơn**.

---

## 2. Xác nhận kiến trúc (đã kiểm tra mã nguồn)

- **Tied weights — ĐÚNG.** `Bert4Rec_model.py:468` (`full_sort_predict`): điểm số = `seq_output @ item_embedding.weight[:n_items].T + output_bias`. State sống cùng không gian với item embedding, chấm bằng **dot product + bias**.
- **State = dự đoán BERT.** `forward()` trả vector sau `output_ffn → GELU → LayerNorm`; `get_state` lấy đúng vector này tại vị trí `[mask]`.

---

## 3. Danh sách lỗi (xếp theo mức độ nghiêm trọng)

### 🔴 Nghiêm trọng — làm hỏng kết quả

**Lỗi 1 — Padding nhiễm vào state khi train; train/eval lệch phân phối.**
`collate_fn` pad số 0 vào đuôi history. Vòng lặp train gọi `encoder.get_state()` với list đã pad. Trong `get_state`, `item_seq_len` được tính bằng độ dài **sau khi pad** (vì id 0 không phân biệt được với item thật), nên token `[mask]` bị đặt ở **cuối chuỗi pad** thay vì ngay sau item thật. Position embedding và vị trí mask sai.
- Khi **eval**, history truyền vào là list thô (chưa pad) → `item_seq_len` đúng.
- ⇒ **State lúc train ≠ state lúc eval.** Critic/Actor học trên state hỏng, đánh giá trên state đúng.

**Lỗi 2 — Rò rỉ dữ liệu test (data leakage).**
`load_offline_data_custom` dùng `for i in range(1, len(seq))`, bao gồm `i = len-1` ⇒ target = item cuối cùng. `prepare_rl_test_data` lại lấy chính `seq[-1]` làm test. ⇒ mẫu test nằm trong tập train. Paper dùng leave-one-out: bỏ item cuối (test) + áp cuối (validation).

**Lỗi 3 — Eval bỏ `output_bias` và dùng cosine thay dot product.**
`evaluate_paper_standard` / `evaluate_full_rank` chấm điểm bằng `F.cosine_similarity`, bỏ `output_bias`. BERT4Rec thật chấm `dot + output_bias`. ⇒ Tự cắt điểm; ngay cả khi Actor ≈ identity cũng không tái tạo được hiệu năng BERT thật.

### 🟠 Vừa — lệch mục tiêu / thổi phồng số

**Lỗi 4 — `beta=0.5` quá lớn.**
Actor: `final_action = normalize(state + beta * residual)`. Định nghĩa mặc định `beta=0.05` nhưng khởi tạo thực tế `beta=0.5` (gấp 10 lần). Residual kéo `action` đi xa khỏi `state` (BERT) ⇒ đẩy vào vùng vector nhiễu — đúng vấn đề lõi đã nêu.

**Lỗi 5 — Reward (rating) lệch với metric (next-item).**
`get_reward_from_rating` tối ưu rating cao; eval đo đoán đúng item kế tiếp bất kể rating. Hai mục tiêu kéo ngược nhau ⇒ thành phần RL làm hại Hit@k.

**Lỗi 6 — Negative sampling uniform thay vì theo popularity.**
`evaluate_paper_standard`: `neg = random.randint(...)`. Paper sample 100 negative **theo độ phổ biến (popularity)**. Uniform dễ hơn ⇒ số bị **thổi phồng giả tạo**, không so sánh được với paper.

**Lỗi 7 — `n_items=3707` ≠ paper (3416).**
Kích thước pool item khác ⇒ Hit@k khác bản chất, không so trực tiếp với số paper.

### 🟡 Nhẹ — lãng phí, không sai logic

**Lỗi 8 — Re-push buffer và chạy BERT forward lại mỗi epoch.** Buffer 500k < 970k mẫu nên chỉ là cửa sổ trượt; BERT forward lặp 50 lần vô ích. Nên cache state một lần.

**Lỗi 9 — `CRITIC_WARMUP=4000` + `UPDATE_EVERY=10`.** Actor không cập nhật tới ~epoch 11. Chậm khởi động, không sai.

---

## 4. Đề xuất phương án sửa

### 4.1. Sửa bắt buộc (đúng đắn về phương pháp)

| # | Lỗi | Cách sửa |
|---|-----|----------|
| 1 | Padding nhiễm state | `collate_fn` trả thêm độ dài thật, hoặc không pad trong collate (để `get_state` tự xử lý list thô như lúc eval). Đảm bảo `item_seq_len` = độ dài thật. |
| 2 | Leakage | `for i in range(1, len(seq) - 2)` — chừa item cuối (test) + áp cuối (val). |
| 3 | Eval scoring | `scores = actions @ item_emb.T + output_bias` (dot + bias), bỏ cosine. |
| 4 | beta lớn | Đặt `beta=0.05`. |

### 4.2. Sửa để so sánh công bằng với paper

- **Negative sampling theo popularity** (Lỗi 6): sample 100 negative theo phân phối tần suất tương tác, không uniform.
- **Khớp số lượng item / tiền xử lý** với paper (Lỗi 7), hoặc **tự đo baseline BERT4Rec ngay trong harness này** thay vì lấy số paper. Đây là baseline đúng để RL so kè.

### 4.3. Số tham chiếu paper (ML-1M, popularity-neg, dot+bias)

| Metric | BERT4Rec |
|--------|----------|
| HR@1   | 0.2863 |
| HR@5   | 0.5876 |
| HR@10  | 0.6970 |
| NDCG@5 | 0.4454 |
| NDCG@10| 0.4818 |
| MRR    | 0.4254 |

> Lưu ý: chỉ được so với các số này **khi đã khớp protocol** (leave-one-out + popularity-sampled 100 negatives + dot+bias + cùng pool item). Khác protocol = so sánh vô nghĩa.

---

## 5. Phân tích trần hiệu năng

- **Giữ nguyên hướng (Actor → embedding, eval ranking):** trần = BERT4Rec. Sau khi sửa Lỗi 1–4, kết quả tốt nhất ≈ hòa BERT. RL không thêm giá trị cho metric next-item.
- **Muốn THỰC SỰ vượt BERT:** phải đổi hướng — RL **rerank trên logit BERT** (khởi tạo điều chỉnh = 0 ⇒ init == BERT ⇒ trần ≥ BERT, RL chỉ nhích lên), hoặc đổi sang metric **giá trị/rating dài hạn** (không còn là Hit@k next-item).

---

## 6. Việc cần làm tiếp (checklist)

- [ ] Sửa Lỗi 1 (padding/length) — ưu tiên cao nhất
- [ ] Sửa Lỗi 2 (leakage) — leave-one-out
- [ ] Sửa Lỗi 3 (dot + bias trong eval)
- [ ] Đặt `beta=0.05` (Lỗi 4)
- [ ] Tự đo baseline BERT4Rec trong harness (so kè đúng)
- [ ] (Nếu so paper) popularity negative sampling + khớp pool item
- [ ] Quyết định hướng: rerank-trên-logit (vượt Hit@k) hay metric giá trị dài hạn
