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
llama_shapes = {
    "7B": [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
    "13B": [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
    # "33B": [(6656, 3 * 6656), (6656, 6656), (6656, 2 * 17664), (17664, 6656)],
    # "70B": [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
}
