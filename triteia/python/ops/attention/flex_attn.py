import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from attn_gym.masks import generate_sliding_window
from einops import rearrange


def checkerboard(score, batch, head, token_q, token_kv):
    score = torch.where(torch.abs(token_kv - token_q) % 2 == 1, score * 0.5, score)
    score = torch.where(torch.abs(token_kv - token_q) % 2 == 0, score * 2.0, score)
    return score


def flex_attention(qkv):
    print(f"qkv: {qkv.shape}")
    # (6300, 3, 24, 128)
    q, k, v = rearrange(qkv, "(b s) t h d -> t b h s d", b=1)
    print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
    with torch.autocast("cuda", enabled=False):
        out = flex_attention(q, k, v, score_mod=checkerboard)
        return rearrange(out, "b h s d -> s (b h d)")


triteia_flex_attention = torch.compile(flex_attention)
