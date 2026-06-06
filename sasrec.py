# -*- coding: utf-8 -*-
"""
SASRec — Self-Attentive Sequential Recommendation
PyTorch implementation (single-file version).

Reference:
    Wang-Cheng Kang, Julian McAuley (2018).
    "Self-Attentive Sequential Recommendation." ICDM 2018.

Tuned for MovieLens 1M:
    ~6 040 users  |  ~3 706 items  |  sequence length 200

Modules included:
    - positional_encoding
    - LayerNorm
    - MultiHeadAttention
    - PointWiseFeedForward
    - SASRec
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def positional_encoding(dim, sentence_length, dtype=torch.float32):
    """
    Sinusoidal positional encoding.
    Returns tensor of shape [sentence_length, dim].
    """
    encoded_vec = np.array([
        pos / np.power(10000, 2 * i / dim)
        for pos in range(sentence_length)
        for i in range(dim)
    ])
    encoded_vec[::2]  = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return torch.tensor(
        encoded_vec.reshape([sentence_length, dim]), dtype=dtype
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sub-modules
# ══════════════════════════════════════════════════════════════════════════════

class LayerNorm(nn.Module):
    """
    Layer Normalization — exact port of normalize() in the original modules.py.
    Normalises over the last dimension.
    """
    def __init__(self, hidden_units, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(hidden_units))
        self.beta  = nn.Parameter(torch.zeros(hidden_units))

    def forward(self, x):
        mean       = x.mean(dim=-1, keepdim=True)
        variance   = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (variance + self.epsilon).sqrt()
        return self.gamma * normalized + self.beta


class MultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention — port of multihead_attention() in modules.py.

    Key differences vs vanilla MHA
    --------------------------------
    * Key masking  : positions where the key vector is all-zero are masked out
                     (handles padding items whose embedding row is forced to 0).
    * Query masking: attention weights for padding query positions are zeroed.
    * Causal mask  : lower-triangular mask so position t cannot attend to t+1..T.
    * Residual     : outputs += queries  (before the caller applies LayerNorm).
    """

    def __init__(self, hidden_units, num_heads, dropout_rate=0.0):
        super().__init__()
        assert hidden_units % num_heads == 0, \
            "hidden_units must be divisible by num_heads"

        self.num_heads    = num_heads
        self.head_dim     = hidden_units // num_heads
        self.hidden_units = hidden_units

        self.W_Q = nn.Linear(hidden_units, hidden_units, bias=True)
        self.W_K = nn.Linear(hidden_units, hidden_units, bias=True)
        self.W_V = nn.Linear(hidden_units, hidden_units, bias=True)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, causality=True):
        """
        Args:
            queries  : (B, T_q, C)
            keys     : (B, T_k, C)   — for self-attention queries == keys
            causality: mask future positions
        Returns:
            (B, T_q, C)
        """
        B, T_q, _ = queries.shape
        _, T_k, _ = keys.shape
        h = self.num_heads
        d = self.head_dim

        Q = self.W_Q(queries)   # (B, T_q, C)
        K = self.W_K(keys)      # (B, T_k, C)
        V = self.W_V(keys)      # (B, T_k, C)

        # Split into heads → (h*B, T, d)
        def split_heads(t):
            t = t.view(B, -1, h, d).transpose(1, 2)
            return t.contiguous().view(B * h, -1, d)

        Q_ = split_heads(Q)   # (h*B, T_q, d)
        K_ = split_heads(K)   # (h*B, T_k, d)
        V_ = split_heads(V)   # (h*B, T_k, d)

        # Scaled dot-product scores
        scale  = d ** 0.5
        scores = torch.bmm(Q_, K_.transpose(1, 2)) / scale   # (h*B, T_q, T_k)

        NEG_INF = -2 ** 32 + 1   # same sentinel as original TF code

        # --- Key masking (padding items whose key is all-zero) ---
        key_masks = keys.abs().sum(dim=-1).sign()               # (B, T_k)
        key_masks = key_masks.unsqueeze(1).repeat(h, T_q, 1)   # (h*B, T_q, T_k)
        scores = torch.where(key_masks == 0,
                             torch.full_like(scores, NEG_INF),
                             scores)

        # --- Causal mask ---
        if causality:
            causal = torch.tril(torch.ones(T_q, T_k, device=queries.device))
            causal = causal.unsqueeze(0).expand(B * h, -1, -1)
            scores = torch.where(causal == 0,
                                 torch.full_like(scores, NEG_INF),
                                 scores)

        attn = F.softmax(scores, dim=-1)   # (h*B, T_q, T_k)

        # --- Query masking ---
        query_masks = queries.abs().sum(dim=-1).sign()              # (B, T_q)
        query_masks = query_masks.unsqueeze(-1).repeat(h, 1, T_k)  # (h*B, T_q, T_k)
        attn = attn * query_masks

        attn = self.dropout(attn)

        # Weighted sum
        out = torch.bmm(attn, V_)   # (h*B, T_q, d)

        # Merge heads back → (B, T_q, C)
        out = (out.view(B, h, T_q, d)
                  .transpose(1, 2)
                  .contiguous()
                  .view(B, T_q, self.hidden_units))

        # Residual connection (same as original)
        return out + queries


class PointWiseFeedForward(nn.Module):
    """
    Point-wise feed-forward network — port of feedforward() in modules.py.

    Uses two Conv1d(kernel=1) layers (== position-wise Linear),
    with ReLU activation after the first layer.
    Applies dropout after each conv and adds a residual connection.
    """

    def __init__(self, hidden_units, dropout_rate=0.2):
        super().__init__()
        self.conv1   = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.conv2   = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, T, C)
        """
        residual = x
        out = x.transpose(1, 2)                        # (B, C, T)
        out = self.dropout(self.relu(self.conv1(out)))
        out = self.dropout(self.conv2(out))
        out = out.transpose(1, 2)                      # (B, T, C)
        return out + residual


# ══════════════════════════════════════════════════════════════════════════════
# SASRec Model
# ══════════════════════════════════════════════════════════════════════════════

class SASRec(nn.Module):
    """
    SASRec model.

    Parameters
    ----------
    usernum : int — total number of users  (not used in forward, kept for API compat)
    itemnum : int — total number of items  (vocabulary size = itemnum + 1, row 0 = padding)
    args    : namespace with the fields below

    args fields
    -----------
    maxlen          : int   — maximum sequence length
    hidden_units    : int   — embedding / attention dimension (e.g. 50)
    num_blocks      : int   — number of Transformer blocks    (e.g. 2)
    num_heads       : int   — number of attention heads       (e.g. 1)
    dropout_rate    : float — dropout probability             (e.g. 0.2)
    l2_emb          : float — L2 weight-decay on embeddings   (e.g. 0.0)
    """

    def __init__(self, usernum: int, itemnum: int, args):
        super().__init__()

        self.usernum      = usernum
        self.itemnum      = itemnum
        self.hidden_units = args.hidden_units
        self.maxlen       = args.maxlen

        # ── Embeddings ────────────────────────────────────────────────────────
        # vocab_size = itemnum + 1  (index 0 is the padding token)
        self.item_emb = nn.Embedding(itemnum + 1, args.hidden_units, padding_idx=0)
        self.pos_emb  = nn.Embedding(args.maxlen, args.hidden_units)

        # ── Dropout ───────────────────────────────────────────────────────────
        self.emb_dropout = nn.Dropout(args.dropout_rate)

        # ── Transformer blocks ────────────────────────────────────────────────
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers     = nn.ModuleList()
        self.forward_layernorms   = nn.ModuleList()
        self.forward_layers       = nn.ModuleList()

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(LayerNorm(args.hidden_units))
            self.attention_layers.append(
                MultiHeadAttention(
                    hidden_units=args.hidden_units,
                    num_heads=args.num_heads,
                    dropout_rate=args.dropout_rate,
                )
            )
            self.forward_layernorms.append(LayerNorm(args.hidden_units))
            self.forward_layers.append(
                PointWiseFeedForward(
                    hidden_units=args.hidden_units,
                    dropout_rate=args.dropout_rate,
                )
            )

        # ── Final LayerNorm ───────────────────────────────────────────────────
        self.last_layernorm = LayerNorm(args.hidden_units)

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_weights(self):
        """Xavier uniform for linear layers; normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def _encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode an input sequence through the Transformer blocks.

        Args:
            input_seq : (B, T)  — item-id sequences (0 = padding)
        Returns:
            seq       : (B, T, C)  — contextualised representations
        """
        B, T   = input_seq.shape
        device = input_seq.device

        # Padding mask — (B, T, 1), float, 1 = real token
        mask = (input_seq != 0).float().unsqueeze(-1)

        # Item embedding  (scaled by sqrt(d) — same as scale=True in original)
        seq = self.item_emb(input_seq) * (self.hidden_units ** 0.5)

        # Positional embedding
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        seq = seq + self.pos_emb(positions)

        # Embedding dropout + zero out padding
        seq = self.emb_dropout(seq) * mask

        # Transformer blocks
        for ln_attn, attn, ln_ffn, ffn in zip(
            self.attention_layernorms,
            self.attention_layers,
            self.forward_layernorms,
            self.forward_layers,
        ):
            # Pre-LN self-attention  (LayerNorm on queries only)
            q   = ln_attn(seq)
            seq = attn(queries=q, keys=seq, causality=True) * mask

            # Pre-LN feed-forward
            seq = ffn(ln_ffn(seq)) * mask

        return self.last_layernorm(seq)   # (B, T, C)

    # ── Forward pass (training) ───────────────────────────────────────────────

    def forward(
        self,
        input_seq: torch.Tensor,   # (B, T)
        pos_seqs:  torch.Tensor,   # (B, T)  positive next-items
        neg_seqs:  torch.Tensor,   # (B, T)  sampled negative items
    ):
        """
        Compute binary cross-entropy loss and AUC estimate for one batch.

        Returns
        -------
        loss : scalar tensor
        auc  : scalar tensor  (approximate, computed over all positions)
        """
        seq_emb = self._encode(input_seq)           # (B, T, C)

        pos_emb = self.item_emb(pos_seqs)           # (B, T, C)
        neg_emb = self.item_emb(neg_seqs)           # (B, T, C)

        pos_logits = (seq_emb * pos_emb).sum(dim=-1)   # (B, T)
        neg_logits = (seq_emb * neg_emb).sum(dim=-1)   # (B, T)

        # Mask padding positions (where the positive label is 0)
        istarget = (pos_seqs != 0).float()

        loss = (
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ).sum() / istarget.sum()

        auc = (
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ).sum() / istarget.sum()

        return loss, auc

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        input_seq:       torch.Tensor,   # (B, T)
        candidate_items: torch.Tensor,   # (K,)
    ) -> torch.Tensor:
        """
        Score a set of candidate items using the last position of each sequence.

        Args:
            input_seq        : (B, T)  — item-id sequences
            candidate_items  : (K,)    — item ids to score (1 positive + K-1 negatives)
        Returns:
            logits : (B, K)
        """
        seq_emb  = self._encode(input_seq)          # (B, T, C)
        final    = seq_emb[:, -1, :]                # (B, C)

        cand_emb = self.item_emb(candidate_items)   # (K, C)
        return final.matmul(cand_emb.T)             # (B, K)


# ══════════════════════════════════════════════════════════════════════════════
# Smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--maxlen",       default=200,  type=int)
    parser.add_argument("--hidden_units", default=50,   type=int)
    parser.add_argument("--num_blocks",   default=2,    type=int)
    parser.add_argument("--num_heads",    default=1,    type=int)
    parser.add_argument("--dropout_rate", default=0.2,  type=float)
    parser.add_argument("--l2_emb",       default=0.0,  type=float)
    parser.add_argument("--lr",           default=1e-3, type=float)
    args = parser.parse_args([])

    USERNUM = 6040
    ITEMNUM = 3706
    BATCH   = 4
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SASRec(USERNUM, ITEMNUM, args).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    inp = torch.randint(1, ITEMNUM + 1, (BATCH, args.maxlen)).to(device)
    pos = torch.randint(1, ITEMNUM + 1, (BATCH, args.maxlen)).to(device)
    neg = torch.randint(1, ITEMNUM + 1, (BATCH, args.maxlen)).to(device)

    model.train()
    loss, auc = model(inp, pos, neg)
    print(f"[train] loss={loss.item():.4f}  auc={auc.item():.4f}")

    model.eval()
    with torch.no_grad():
        candidates = torch.randint(1, ITEMNUM + 1, (101,)).to(device)
        logits = model.predict(inp, candidates)
    print(f"[eval]  logits shape={logits.shape}")   # expect (4, 101)
    print("Smoke-test passed.")
