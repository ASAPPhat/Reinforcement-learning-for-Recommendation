# -*- coding: utf-8 -*-
"""
PureBERT4Rec — BERT4Rec độc lập, không phụ thuộc RecBole.
Bám sát hoàn toàn file gốc:
    recbole/model/sequential_recommender/bert4rec.py

Sơ đồ vocab (giữ nguyên RecBole):
    index 0            → PAD  (padding_idx, embedding = 0)
    index 1..n_items   → item thật
    index n_items      → MASK token
    item_embedding     : (n_items + 1, hidden_size)
    output_bias        : (n_items,)   — chỉ item thật

Reference:
    Fei Sun et al. "BERT4Rec: Sequential Recommendation with
    Bidirectional Encoder Representations from Transformer." CIKM 2019.
"""

import math
import random

import torch
import torch.nn as nn


# =============================================================================
# 1.  Attention Mask  (giống get_attention_mask(..., bidirectional=True) RecBole)
# =============================================================================

def get_bidirectional_attention_mask(item_seq: torch.Tensor) -> torch.Tensor:
    """
    Tạo additive float mask [B, 1, L, L].
    PAD (item_id == 0) → -10000.0   |   item thật → 0.0
    Bidirectional: mọi token thật đều thấy nhau.
    """
    attention_mask = (item_seq != 0).long()                      # [B, L]
    extended = attention_mask.unsqueeze(1).unsqueeze(2)          # [B, 1, 1, L]
    extended = extended.expand(-1, -1, item_seq.size(1), -1)     # [B, 1, L, L]
    extended = (1.0 - extended.float()) * -10000.0
    return extended


# =============================================================================
# 2.  gather_indexes  (giống SequentialRecommender.gather_indexes RecBole)
# =============================================================================

def gather_indexes(output: torch.Tensor,
                   gather_index: torch.Tensor) -> torch.Tensor:
    """
    output:       [B, L, H]
    gather_index: [B]        — vị trí cần lấy (item_seq_len - 1)
    returns:      [B, H]
    """
    gather_index = gather_index.view(-1, 1, 1).expand(-1, 1, output.size(-1))
    return output.gather(dim=1, index=gather_index).squeeze(1)


# =============================================================================
# 3.  TransformerEncoder  (tái hiện recbole/model/layers.py::TransformerEncoder)
#     Nhận additive mask [B, 1, L, L], trả về list hidden states.
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention với additive mask — giống RecBole."""

    def __init__(self, hidden_size: int, n_heads: int,
                 attn_dropout_prob: float):
        super().__init__()
        assert hidden_size % n_heads == 0, \
            f"hidden_size ({hidden_size}) phải chia hết cho n_heads ({n_heads})"

        self.n_heads   = n_heads
        self.head_dim  = hidden_size // n_heads
        self.scale     = math.sqrt(self.head_dim)

        self.query     = nn.Linear(hidden_size, hidden_size)
        self.key       = nn.Linear(hidden_size, hidden_size)
        self.value     = nn.Linear(hidden_size, hidden_size)
        self.out_proj  = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(attn_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        B, L, H = hidden.shape
        Q = self._split_heads(self.query(hidden))   # [B, nh, L, hd]
        K = self._split_heads(self.key(hidden))
        V = self._split_heads(self.value(hidden))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, nh, L, L]
        scores = scores + attention_mask                             # additive mask
        attn_w = self.attn_drop(torch.softmax(scores, dim=-1))

        ctx = torch.matmul(attn_w, V)                               # [B, nh, L, hd]
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, H)
        return self.out_proj(ctx)


class FeedForward(nn.Module):
    """Position-wise FFN — giống RecBole."""

    def __init__(self, hidden_size: int, inner_size: int,
                 hidden_dropout_prob: float, hidden_act: str,
                 layer_norm_eps: float):
        super().__init__()
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "swish": nn.SiLU()}
        act_fn  = act_map.get(hidden_act.lower(), nn.GELU())

        self.dense_1    = nn.Linear(hidden_size, inner_size)
        self.act        = act_fn
        self.dense_2    = nn.Linear(inner_size, hidden_size)
        self.LayerNorm  = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout    = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        out = self.dense_2(self.act(self.dense_1(hidden)))
        return self.LayerNorm(hidden + self.dropout(out))


class TransformerLayer(nn.Module):
    """Một Transformer layer (Post-LN) — giống RecBole."""

    def __init__(self, hidden_size, n_heads, inner_size,
                 hidden_dropout_prob, attn_dropout_prob,
                 hidden_act, layer_norm_eps):
        super().__init__()
        self.attn       = MultiHeadSelfAttention(
                              hidden_size, n_heads, attn_dropout_prob)
        self.attn_norm  = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn_drop  = nn.Dropout(hidden_dropout_prob)
        self.ffn        = FeedForward(
                              hidden_size, inner_size,
                              hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(hidden, attention_mask)
        hidden   = self.attn_norm(hidden + self.attn_drop(attn_out))
        hidden   = self.ffn(hidden)
        return hidden


class TransformerEncoder(nn.Module):
    """
    Stack n_layers TransformerLayer.
    Trả về list all_hidden_states — giống RecBole TransformerEncoder
    khi output_all_encoded_layers=True.
    """

    def __init__(self, n_layers, hidden_size, n_heads, inner_size,
                 hidden_dropout_prob, attn_dropout_prob,
                 hidden_act, layer_norm_eps):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, n_heads, inner_size,
                             hidden_dropout_prob, attn_dropout_prob,
                             hidden_act, layer_norm_eps)
            for _ in range(n_layers)
        ])

    def forward(self, hidden: torch.Tensor,
                attention_mask: torch.Tensor) -> list:
        all_hidden = []
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
            all_hidden.append(hidden)
        return all_hidden  # lấy [-1] là output layer cuối


# =============================================================================
# 4.  BERT4Rec  (bám sát bert4rec.py gốc của RecBole)
# =============================================================================

class BERT4Rec(nn.Module):
    """
    BERT4Rec độc lập — không phụ thuộc RecBole.
    Kiến trúc, tên biến, logic tính loss và predict giống hệt file gốc.

    Args:
        n_items (int):
            Số item thật (không gồm PAD, MASK).
            VD: MovieLens-1M → 3706 phim.

        max_seq_length (int):
            Độ dài chuỗi tối đa. Mặc định 50 (giống RecBole ML-1M config).

        hidden_size (int):       Embedding + hidden. Mặc định 64.
        n_layers (int):          Số Transformer layer. Mặc định 2.
        n_heads (int):           Số attention head. Mặc định 2.
        inner_size (int):        FFN inner dim. Mặc định 256.
        hidden_dropout_prob (float): Dropout embedding & FFN. Mặc định 0.2.
        attn_dropout_prob (float):   Dropout attention. Mặc định 0.2.
        hidden_act (str):        "gelu" | "relu" | "swish". Mặc định "gelu".
        layer_norm_eps (float):  LayerNorm eps. Mặc định 1e-12.
        mask_ratio (float):      Tỉ lệ mask khi training. Mặc định 0.2.
        loss_type (str):         "CE" (khuyên dùng) hoặc "BPR".
        initializer_range (float): std khi init weights. Mặc định 0.02.
    """

    def __init__(
        self,
        n_items: int,
        max_seq_length: int    = 50,
        hidden_size: int       = 64,
        n_layers: int          = 2,
        n_heads: int           = 2,
        inner_size: int        = 256,
        hidden_dropout_prob: float = 0.2,
        attn_dropout_prob: float   = 0.2,
        hidden_act: str        = "gelu",
        layer_norm_eps: float  = 1e-12,
        mask_ratio: float      = 0.2,
        loss_type: str         = "CE",
        initializer_range: float = 0.02,
    ):
        super().__init__()

        # ── config ──────────────────────────────────────────────────────────
        self.n_items           = n_items
        self.max_seq_length    = max_seq_length
        self.hidden_size       = hidden_size
        self.mask_ratio        = mask_ratio
        self.loss_type         = loss_type
        self.initializer_range = initializer_range

        # MASK token = n_items  (giống RecBole: self.mask_token = self.n_items)
        self.mask_token        = n_items
        # mask_item_length = int(mask_ratio * max_seq_length)  — giống gốc
        self.mask_item_length  = int(mask_ratio * max_seq_length)

        assert loss_type in ["BPR", "CE"], \
            f"loss_type phải là 'BPR' hoặc 'CE', nhận: '{loss_type}'"

        # ── layers ──────────────────────────────────────────────────────────
        # item_embedding: (n_items + 1, H) — +1 cho MASK token, padding_idx=0
        self.item_embedding = nn.Embedding(
            n_items + 1, hidden_size, padding_idx=0
        )
        # position_embedding: (max_seq_length, H)  — giống gốc
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)

        self.trm_encoder = TransformerEncoder(
            n_layers           = n_layers,
            hidden_size        = hidden_size,
            n_heads            = n_heads,
            inner_size         = inner_size,
            hidden_dropout_prob = hidden_dropout_prob,
            attn_dropout_prob  = attn_dropout_prob,
            hidden_act         = hidden_act,
            layer_norm_eps     = layer_norm_eps,
        )

        self.LayerNorm  = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout    = nn.Dropout(hidden_dropout_prob)

        # output head — giống gốc: Linear → GELU → LayerNorm
        self.output_ffn  = nn.Linear(hidden_size, hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln   = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # output_bias: (n_items,) — chỉ item thật, KHÔNG gồm MASK
        self.output_bias = nn.Parameter(torch.zeros(n_items))

        # ── init ────────────────────────────────────────────────────────────
        self.apply(self._init_weights)

    # -------------------------------------------------------------------------
    # _init_weights — giống hệt RecBole
    # -------------------------------------------------------------------------

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # -------------------------------------------------------------------------
    # reconstruct_test_data — giống hệt RecBole
    # -------------------------------------------------------------------------

    def reconstruct_test_data(self,
                               item_seq: torch.Tensor,
                               item_seq_len: torch.Tensor) -> torch.Tensor:
        """
        Chèn MASK token vào vị trí cuối chuỗi thật để chuẩn bị inference.
        Logic giữ nguyên hoàn toàn so với RecBole.
        """
        padding  = torch.zeros(item_seq.size(0), dtype=torch.long,
                               device=item_seq.device)            # [B]
        item_seq = torch.cat(
            (item_seq, padding.unsqueeze(-1)), dim=-1
        )                                                          # [B, L+1]
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        item_seq = item_seq[:, 1:]                                 # [B, L]
        return item_seq

    # -------------------------------------------------------------------------
    # forward — giống hệt RecBole
    # -------------------------------------------------------------------------

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        item_seq: [B, L]
        returns:  [B, L, H]
        """
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids       = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb           = self.item_embedding(item_seq)
        input_emb          = item_emb + position_embedding
        input_emb          = self.LayerNorm(input_emb)
        input_emb          = self.dropout(input_emb)

        extended_attention_mask = get_bidirectional_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask
        )  # list of [B, L, H]

        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output     = self.output_ln(ffn_output)
        return output  # [B, L, H]

    # -------------------------------------------------------------------------
    # multi_hot_embed — giống hệt RecBole
    # -------------------------------------------------------------------------

    def multi_hot_embed(self,
                        masked_index: torch.Tensor,
                        max_length: int) -> torch.Tensor:
        """
        masked_index: [B, mask_len]
        returns:      [B*mask_len, max_length]

        Ví dụ (giống docstring gốc):
            sequence     : [1 2 3 4 5]
            masked_seq   : [1 mask 3 mask 5]
            masked_index : [1, 3]
            max_length   : 5
            multi_hot    : [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot    = torch.zeros(
            masked_index.size(0), max_length, device=masked_index.device
        )
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    # -------------------------------------------------------------------------
    # calculate_loss — giống hệt RecBole
    # -------------------------------------------------------------------------

    def calculate_loss(self,
                       masked_item_seq: torch.Tensor,
                       pos_items: torch.Tensor,
                       neg_items: torch.Tensor,
                       masked_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masked_item_seq: [B, L]          — chuỗi đã mask
            pos_items:       [B, mask_len]   — item thật tại vị trí mask
            neg_items:       [B, mask_len]   — item âm (dùng với BPR)
            masked_index:    [B, mask_len]   — vị trí bị mask (0=padding)
        Returns:
            loss: scalar
        """
        seq_output = self.forward(masked_item_seq)  # [B, L, H]

        pred_index_map = self.multi_hot_embed(
            masked_index, masked_item_seq.size(-1)
        )  # [B*mask_len, L]
        pred_index_map = pred_index_map.view(
            masked_index.size(0), masked_index.size(1), -1
        )  # [B, mask_len, L]

        # [B, mask_len, L] bmm [B, L, H] → [B, mask_len, H]
        seq_output = torch.bmm(pred_index_map, seq_output)

        if self.loss_type == "BPR":
            pos_items_emb = self.item_embedding(pos_items)  # [B, mask_len, H]
            neg_items_emb = self.item_embedding(neg_items)  # [B, mask_len, H]
            pos_score = (
                torch.sum(seq_output * pos_items_emb, dim=-1)
                + self.output_bias[pos_items]
            )  # [B, mask_len]
            neg_score = (
                torch.sum(seq_output * neg_items_emb, dim=-1)
                + self.output_bias[neg_items]
            )  # [B, mask_len]
            targets = (masked_index > 0).float()
            loss = -torch.sum(
                torch.log(1e-14 + torch.sigmoid(pos_score - neg_score))
                * targets
            ) / torch.sum(targets)
            return loss

        elif self.loss_type == "CE":
            loss_fct     = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
            test_item_emb = self.item_embedding.weight[: self.n_items]  # [n_items, H]
            logits = (
                torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                + self.output_bias
            )  # [B, mask_len, n_items]
            targets = (masked_index > 0).float().view(-1)  # [B*mask_len]
            loss = torch.sum(
                loss_fct(
                    logits.view(-1, test_item_emb.size(0)),  # [B*mask_len, n_items]
                    pos_items.view(-1),                      # [B*mask_len]
                ) * targets
            ) / torch.sum(targets)
            return loss

        else:
            raise NotImplementedError(
                f"loss_type '{self.loss_type}' không hợp lệ. Chọn 'BPR' hoặc 'CE'."
            )

    # -------------------------------------------------------------------------
    # predict — giống hệt RecBole
    # -------------------------------------------------------------------------

    def predict(self,
                item_seq: torch.Tensor,
                item_seq_len: torch.Tensor,
                test_item: torch.Tensor) -> torch.Tensor:
        """
        Score 1 item cụ thể cho mỗi sample.
        item_seq:     [B, L]
        item_seq_len: [B]
        test_item:    [B]
        returns:      [B]
        """
        item_seq   = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)                          # [B, L, H]
        seq_output = gather_indexes(seq_output, item_seq_len - 1)   # [B, H]

        test_item_emb = self.item_embedding(test_item)              # [B, H]
        scores = (
            torch.mul(seq_output, test_item_emb).sum(dim=1)
            + self.output_bias[test_item]
        )  # [B]
        return scores

    # -------------------------------------------------------------------------
    # full_sort_predict — giống hệt RecBole
    # -------------------------------------------------------------------------

    def full_sort_predict(self,
                          item_seq: torch.Tensor,
                          item_seq_len: torch.Tensor) -> torch.Tensor:
        """
        Score toàn bộ catalog item.
        item_seq:     [B, L]
        item_seq_len: [B]
        returns:      [B, n_items]
        """
        item_seq   = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)                         # [B, L, H]
        seq_output = gather_indexes(seq_output, item_seq_len - 1)  # [B, H]

        # Bỏ MASK token (index n_items), chỉ lấy item thật
        test_items_emb = self.item_embedding.weight[: self.n_items]  # [n_items, H]
        scores = (
            torch.matmul(seq_output, test_items_emb.transpose(0, 1))
            + self.output_bias
        )  # [B, n_items]
        return scores


# =============================================================================
# 5.  BERT4RecDataset
#     Xử lý masking NGOÀI model, trong __getitem__  — đúng như RecBole.
#     RecBole thực hiện mask ở BERT4RecDataset (recbole/data/dataset/...),
#     không phải trong model.forward() hay training loop.
# =============================================================================

class BERT4RecDataset(torch.utils.data.Dataset):
    """
    Dataset cho BERT4Rec.
    Mỗi lần __getitem__ được gọi, thực hiện random mask tại chỗ.
    Đây là cách RecBole xử lý — mask được precompute theo từng sample
    trong DataLoader worker, không phải trong model.

    Args:
        sequences   : list of list[int] — mỗi phần tử là 1 chuỗi item
                      (đã được encode, index bắt đầu từ 1).
        n_items     : int  — số item thật.
        max_seq_len : int  — độ dài chuỗi tối đa (right-padding với 0).
        mask_token  : int  — id của MASK token (thường = n_items).
        mask_ratio  : float — tỉ lệ mask. Mặc định 0.2.
    """

    def __init__(
        self,
        sequences: list,
        n_items: int,
        max_seq_len: int   = 50,
        mask_token: int    = None,
        mask_ratio: float  = 0.2,
    ):
        self.sequences    = sequences
        self.n_items      = n_items
        self.max_seq_len  = max_seq_len
        self.mask_token   = mask_token if mask_token is not None else n_items
        self.mask_ratio   = mask_ratio
        self.mask_item_length = int(mask_ratio * max_seq_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Trả về dict:
            item_seq     : [max_seq_len]         — chuỗi gốc (right-padded)
            item_seq_len : scalar                — độ dài thật
            masked_item_seq  : [max_seq_len]
            pos_items        : [mask_item_length]
            neg_items        : [mask_item_length]
            masked_index     : [mask_item_length]
        """
        seq = self.sequences[idx]

        # Cắt nếu quá dài, right-pad nếu ngắn
        seq     = seq[-self.max_seq_len:]
        seq_len = len(seq)

        # Right-padding với 0
        padded  = seq + [0] * (self.max_seq_len - seq_len)

        item_seq     = torch.tensor(padded, dtype=torch.long)
        item_seq_len = torch.tensor(seq_len, dtype=torch.long)

        # ── Masking (giống BERT4RecDataset của RecBole) ─────────────────────
        mask_len   = self.mask_item_length
        # Các vị trí có item thật: [0, 1, ..., seq_len-1]
        real_positions = list(range(seq_len))
        n_mask     = min(mask_len, len(real_positions))
        chosen     = sorted(random.sample(real_positions, n_mask))

        masked_seq   = padded.copy()
        pos_items    = [0] * mask_len
        neg_items    = [0] * mask_len
        masked_index = [0] * mask_len

        for j, pos in enumerate(chosen):
            original_item  = padded[pos]
            masked_seq[pos] = self.mask_token

            pos_items[j]    = original_item
            masked_index[j] = pos

            # Negative sampling — vectorized thay vì while-loop
            neg = random.randint(1, self.n_items)
            if neg == original_item:
                neg = (neg % self.n_items) + 1
            neg_items[j] = neg

        return {
            "item_seq"        : item_seq,
            "item_seq_len"    : item_seq_len,
            "masked_item_seq" : torch.tensor(masked_seq,   dtype=torch.long),
            "pos_items"       : torch.tensor(pos_items,    dtype=torch.long),
            "neg_items"       : torch.tensor(neg_items,    dtype=torch.long),
            "masked_index"    : torch.tensor(masked_index, dtype=torch.long),
        }


# =============================================================================
# 6.  Training Loop mẫu cho MovieLens-1M
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, device):
    """Một epoch training — trả về avg loss và masked token accuracy."""
    model.train()
    total_loss   = 0.0
    total_correct = 0
    total_masked  = 0

    for batch in dataloader:
        masked_item_seq = batch["masked_item_seq"].to(device)
        pos_items       = batch["pos_items"].to(device)
        neg_items       = batch["neg_items"].to(device)
        masked_index    = batch["masked_index"].to(device)

        optimizer.zero_grad()
        loss = model.calculate_loss(
            masked_item_seq, pos_items, neg_items, masked_index
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ── Masked token accuracy (chỉ với CE loss) ─────────────────────────
        if model.loss_type == "CE":
            with torch.no_grad():
                seq_out = model.forward(masked_item_seq)
                pred_map = model.multi_hot_embed(
                    masked_index, masked_item_seq.size(-1)
                ).view(masked_index.size(0), masked_index.size(1), -1)
                gathered = torch.bmm(pred_map, seq_out)  # [B, mask_len, H]

                emb = model.item_embedding.weight[: model.n_items]
                logits = (
                    torch.matmul(gathered, emb.transpose(0, 1))
                    + model.output_bias
                )  # [B, mask_len, n_items]

                preds   = logits.argmax(dim=-1)          # [B, mask_len]
                valid   = (masked_index > 0)             # [B, mask_len]
                correct = ((preds == pos_items) & valid).sum().item()
                total_correct += correct
                total_masked  += valid.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_masked * 100) if total_masked > 0 else 0.0
    return avg_loss, accuracy


# =============================================================================
# 7.  Sanity Check / Quick Demo
# =============================================================================

if __name__ == "__main__":
    print("=== BERT4Rec Standalone — Sanity Check ===\n")

    N_ITEMS = 3706   # MovieLens-1M
    B       = 4
    L       = 50

    model = BERT4Rec(
        n_items            = N_ITEMS,
        max_seq_length     = L,
        hidden_size        = 64,
        n_layers           = 2,
        n_heads            = 2,
        inner_size         = 256,
        hidden_dropout_prob = 0.2,
        attn_dropout_prob  = 0.2,
        hidden_act         = "gelu",
        mask_ratio         = 0.2,
        loss_type          = "CE",
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  mask_token      : {model.mask_token}  (= n_items)")
    print(f"  mask_item_length: {model.mask_item_length}  (= int(0.2 * {L}))")
    print(f"  item_emb size   : {tuple(model.item_embedding.weight.shape)}")
    print(f"  output_bias     : {tuple(model.output_bias.shape)}")
    print(f"  Trainable params: {n_params:,}\n")

    torch.manual_seed(42)
    item_seq     = torch.randint(1, N_ITEMS, (B, L))
    item_seq[:, L // 2:] = 0
    item_seq_len = torch.tensor([L // 2] * B)

    # forward
    model.eval()
    with torch.no_grad():
        out = model.forward(item_seq)
    print(f"[forward]         {tuple(item_seq.shape)} → {tuple(out.shape)}")
    assert out.shape == (B, L, 64)

    # reconstruct_test_data
    ts = model.reconstruct_test_data(item_seq.clone(), item_seq_len)
    n_mask = (ts == model.mask_token).sum(dim=1)
    print(f"[reconstruct]     mask count per sample: {n_mask.tolist()}  (kỳ vọng toàn 1)")
    assert (n_mask == 1).all()

    # full_sort_predict
    with torch.no_grad():
        sc = model.full_sort_predict(item_seq.clone(), item_seq_len)
    print(f"[full_sort]       scores shape: {tuple(sc.shape)}  (kỳ vọng [{B}, {N_ITEMS}])")
    assert sc.shape == (B, N_ITEMS)

    # predict
    ti = torch.randint(1, N_ITEMS, (B,))
    with torch.no_grad():
        ps = model.predict(item_seq.clone(), item_seq_len, ti)
    print(f"[predict]         scores shape: {tuple(ps.shape)}  (kỳ vọng [{B}])")
    assert ps.shape == (B,)

    # Dataset + calculate_loss
    model.train()
    fake_seqs = [list(range(1, 26)) for _ in range(16)]
    ds     = BERT4RecDataset(fake_seqs, N_ITEMS, max_seq_len=L, mask_ratio=0.2)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0)
    batch  = next(iter(loader))

    loss = model.calculate_loss(
        batch["masked_item_seq"],
        batch["pos_items"],
        batch["neg_items"],
        batch["masked_index"],
    )
    print(f"[calculate_loss]  CE loss = {loss.item():.4f}")
    assert loss.item() > 0

    print("\n✅ Tất cả kiểm tra đều PASS!")
    print("\nVocab layout:")
    print(f"  item_embedding : [{N_ITEMS + 1}, 64]")
    print(f"  index 0        : PAD")
    print(f"  index 1..{N_ITEMS} : {N_ITEMS} item thật")
    print(f"  index {N_ITEMS}    : MASK token")
    print(f"  output_bias    : [{N_ITEMS}]  — chỉ item thật")