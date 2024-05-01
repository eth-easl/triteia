import bitblas
from triteia.ao.utils.dtypes import QUANTIZED_DTYPE, BITBLAS_STORAGE_DTYPE, DTYPES_BIT
from triteia.ao.utils.bitblas_utils import get_or_create_bitblas_operator
from multiprocessing import Pool
from tqdm import tqdm
# llama 7b tp=2

intermediate_size = 11008
vocab_size = 32000
hidden_size = 4096
tp = 2

Ms = [1]
Ns = [2048, 5504, 4096, 16000]
Ks = Ns

# llama 70b

bitwidth = 4
configs = []
for M in Ms:
    for N in Ns:
        for K in Ks:
            matmul_config = bitblas.MatmulConfig(
                    M=M,
                    N=N,
                    K=K,
                    fast_decoding=True,
                    A_dtype="float16",
                    W_dtype=QUANTIZED_DTYPE[bitwidth],
                    accum_dtype="float16",
                    out_dtype="float16",
                    layout="nt",
                    with_bias=False,
                    group_size=K,
                    with_scaling=True,
                    with_zeros=True,
                    zeros_mode="quantized",
            )
            configs.append(matmul_config)

for config in tqdm(configs):
    get_or_create_bitblas_operator(config, enable_tuning=True)
