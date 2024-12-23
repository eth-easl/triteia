import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from einops import rearrange


def noop(score, b, h, q_idx, kv_idx):
    return score


from torch.nn.attention.flex_attention import create_block_mask

SLIDING_WINDOW = 1024


def sliding_window_causal(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW
    return causal_mask & window_mask


block_mask = create_block_mask(
    sliding_window_causal, B=None, H=None, Q_LEN=1024, KV_LEN=1024
)


def vdit_attention(qkv):
    # (6300, 3, 24, 128)
    q, k, v = rearrange(qkv, "(b s) t h d -> t b h s d", b=1)
    with torch.autocast("cuda", enabled=False):
        out = flex_attention(q, k, v, block_mask=block_mask)
        return rearrange(out, "b h s d -> s (b h d)")


triteia_vdit_attention = torch.compile(vdit_attention)
# triteia_vdit_attention = vdit_attention
