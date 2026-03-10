import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.overrides import has_torch_function_variadic

from typing import Callable, List, Optional, Tuple, Union

Tensor = torch.Tensor

class _Loss(nn.Module):

    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class TripletSoftMarginWithDistanceLoss(_Loss):

    __constants__ = ['margin', 'swap', 'reduction']
    margin: float
    swap: bool

    def __init__(self, *, distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
                 margin: float = 1.0, swap: bool = False, reduction: str = 'mean'):
        super().__init__(size_average=None, reduce=None, reduction=reduction)
        self.distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = \
            distance_function if distance_function is not None else PairwiseDistance()
        self.margin = margin
        self.swap = swap

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:

        return triplet_margin_with_distance_loss(
            anchor, positive, negative,
            distance_function=self.distance_function,
            margin=self.margin,
            swap=self.swap,
            reduction=self.reduction
        )

def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> Tensor:

    if torch.jit.is_scripting():
        raise NotImplementedError(
            "F.triplet_margin_with_distance_loss does not support JIT scripting: "
            "functions requiring Callables cannot be scripted."
        )

    if has_torch_function_variadic(anchor, positive, negative):
        return handle_torch_function(
            triplet_margin_with_distance_loss,
            (anchor, positive, negative),
            anchor,
            positive,
            negative,
            distance_function=distance_function,
            margin=margin,
            swap=swap,
            reduction=reduction,
        )

    # Check validity of reduction mode
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")

    # Check dimensions
    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim

    if not (a_dim == p_dim and p_dim == n_dim):
        raise RuntimeError(
            f"The anchor, positive, and negative tensors are expected to have "
            f"the same number of dimensions, but got: anchor {a_dim}D, "
            f"positive {p_dim}D, and negative {n_dim}D inputs")

    # Calculate loss
    if distance_function is None:
        distance_function = torch.pairwise_distance

    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)

    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = torch.minimum(dist_neg, dist_swap)

    # Apply soft margin
    # dist = margin + dist_pos - dist_neg
    if math.isclose(margin, 1.0, abs_tol=0.01):
        dist2 = torch.log1p(torch.exp(dist_pos - dist_neg))
    else:
        dist2 = torch.log(margin + torch.exp(dist_pos - dist_neg))
    loss = torch.clamp_min(dist2, 0)

    # Apply reduction
    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    else:
        return loss