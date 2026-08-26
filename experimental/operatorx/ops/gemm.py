from __future__ import annotations

from dataclasses import dataclass

from operatorx.core.op import OpSpec
from operatorx.core.op_registry import register


@dataclass(frozen=True)
class GemmArgs:
    m: int
    n: int
    k: int
    dtype_a: str = "bf16"
    dtype_b: str = "bf16"
    dtype_out: str = "bf16"
    bias: bool = False
    activation: str | None = None


GEMM = OpSpec(
    type="gemm",
    arg_schema=GemmArgs,
    description="C = activation(A[M,K] @ B[K,N] + bias)",
)

register(GEMM)


@dataclass(frozen=True)
class GroupedGemmArgs:
    """One ragged grouped GEMM: X[M_total, K] @ W[G, K, N] -> Y[M_total, N].

    `group_sizes` is the per-group split of the token dim (len == G, may
    contain zeros); M_total is its sum. Row segment g of X multiplies W[g].
    """
    n: int
    k: int
    group_sizes: tuple[int, ...] | list
    dtype_a: str = "bf16"
    dtype_b: str = "bf16"
    dtype_out: str = "bf16"


GROUPED_GEMM = OpSpec(
    type="grouped_gemm",
    arg_schema=GroupedGemmArgs,
    description="Y[M_total,N] = concat_g(X_g @ W[g]) over per-group row segments of X",
)

register(GROUPED_GEMM)
