"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch._custom_ops as torch_custom_ops

from vllm._C import ops


@torch_custom_ops.custom_op("vllm::rms")
def rms(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    raise NotImplementedError()


@torch_custom_ops.impl("vllm::rms", device_types="cuda")
def rms_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    out = torch.empty_like(hidden_states)
    ops.rms_norm(
        out,
        hidden_states,
        weight,
        eps,
    )
    return out


@torch_custom_ops.impl_abstract("vllm::rms")
def rms_abstract(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


@torch_custom_ops.custom_op("vllm::fused_add_rms")
def fused_add_rms(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError()


@torch_custom_ops.impl("vllm::fused_add_rms", device_types="cuda")
def fused_add_rms_impl(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # FIXME: The custom ops is in-place.
    ops.fused_add_rms_norm(
        hidden_states,
        residual,
        weight,
        eps,
    )
    return hidden_states, residual


@torch_custom_ops.impl_abstract("vllm::fused_add_rms")
def fused_add_rms_abstract(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(hidden_states), torch.empty_like(residual)


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        do_compile: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            if do_compile:
                x, residual = torch.ops.vllm.fused_add_rms(
                    x, residual, self.weight.data, self.variance_epsilon)
            else:
                # NOTE(woosuk): Fused RMSNorm is in-place operation.
                ops.fused_add_rms_norm(
                    x,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                )
            return x, residual

        if do_compile:
            out = torch.ops.vllm.rms(x, self.weight.data,
                                     self.variance_epsilon)
        else:
            out = torch.empty_like(x)
            ops.rms_norm(
                out,
                x,
                self.weight.data,
                self.variance_epsilon,
            )
        return out
