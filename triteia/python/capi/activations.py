import torch
import triteia_cuda


class QuickGELU:
    def __init__(self):
        super().__init__()
        self.op = triteia_cuda.gelu_quick

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return x * torch.sigmoid(1.702 * x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self.op(out, x)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_cuda(x)
