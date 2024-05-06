import bitblas
from triteia.ao.utils.dtypes import QUANTIZED_DTYPE, BITBLAS_STORAGE_DTYPE, DTYPES_BIT
from triteia.ao.utils.bitblas_utils import get_or_create_bitblas_operator
from multiprocessing import Pool
from tqdm import tqdm

configs = {
    'llama-7b': {
        'intermediate_size': 11008,
        'vocab_size': 32000,
        'hidden_size': 4096,
    },
    'llama-70b': {
        'intermediate_size': 28672,
        'vocab_size': 32000,
        'hidden_size': 8192,
    },
    'llama-13b': {
        'intermediate_size': 13824,
        'vocab_size': 32000,
        'hidden_size': 5120,
    },
}

def get_MNKs(intermediate_size, vocab_size, hidden_size, tp):
    Ms = [1,2,3,4,5,6,7,8]
    Ns = [1024, 2048, 4096]
    Ns += [hidden_size, hidden_size // tp, intermediate_size, intermediate_size // tp]
    Ks = Ns.copy()
    Ks += [vocab_size, vocab_size // tp]
    return Ms, Ns, Ks

def get_MNKs_test():
    Ms = [1,2,3,4,8,16,32]
    Ns = [2048, 4096]
    Ks = Ns
    return Ms, Ns, Ks

# llama 7b tp=2
config = configs['llama-7b']
tp_size = 2
bitwidth = 4

Ms, Ns, Ks = get_MNKs(
    config['intermediate_size'],
    config['vocab_size'],
    config['hidden_size'],
    tp=tp_size,
)
# Ms, Ns, Ks = get_MNKs_test()

configs = []
# reverse Ns to start with the largest N
# Ns = Ns[::-1]
for N in Ns:
    for K in Ks:
        matmul_config = bitblas.MatmulConfig(
                M=Ms,
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
