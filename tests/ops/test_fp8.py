import pytest
import torch
from triteia.python.ops import to_float8, fp8_mm
import torch.nn.functional as F

from triteia.python.utils.gpu import is_hopper


@pytest.mark.parametrize("m", [512])
@pytest.mark.parametrize("n", [512])
@pytest.mark.parametrize("k", [512])
def test_fp8_mm(m, n, k, dtype=torch.float8_e4m3fn):
    if not is_hopper():
        pytest.skip("Test is only for Hopper")
    else:
        x = torch.randn((1, m, n), dtype=torch.float16, device="cuda")
        w = torch.randn((n, k), dtype=torch.float16, device="cuda").t()
        x_f8, x_inv_s = to_float8(x, dtype=dtype)
        w_f8, w_inv_s = to_float8(w)
        y_fp8 = fp8_mm(x, w_f8, out_dtype=torch.float16, scale_b=w_inv_s)
        y_fp16 = torch.matmul(x, w)
        cos_sim = F.cosine_similarity(y_fp8.reshape(-1), y_fp16.reshape(-1), dim=0)
        assert cos_sim > 0.95
