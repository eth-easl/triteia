column_chunking_modules = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
]
pack_modules = {
    "self_attn.q_proj": "self_attn.qkv_proj:0",
    "self_attn.k_proj": "self_attn.qkv_proj:1",
    "self_attn.v_proj": "self_attn.qkv_proj:2",
    "mlp.gate_proj": "mlp.gate_up_proj:0",
    "mlp.up_proj": "mlp.gate_up_proj:1",
}
row_chunking_modules = [
    "self_attn.o_proj",
    "mlp.down_proj",
]
uncompressed_row_chunking_modules = [
    "embed_tokens",
    "lm_head",
]