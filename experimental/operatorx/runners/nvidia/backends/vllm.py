"""vLLM GEMM backend — exercises vLLM's own high-level linear/quantization
entrypoints, including the weight-only-quant paths real deployments use
(bf16 activation x fp8 / mxfp4 weight).

No kernels are implemented here. Every path calls a public vLLM API:
  bf16 x bf16   -> torch.nn.functional.linear (vLLM's UnquantizedLinearMethod path)
  bf16 x fp8    -> prepare_fp8_layer_for_marlin + apply_fp8_marlin_linear  (W8A16)
  fp8  x fp8    -> Fp8LinearMethod / cutlass_scaled_mm                     (W8A8)
  nvfp4 x nvfp4 -> scaled_fp4_quant + cutlass_scaled_fp4_mm                (W4A4)

grouped_gemm (ragged X[M_total,K] @ W[G,K,N] with an explicit per-group split):
  bf16 x bf16   -> torch._grouped_mm (the shared PyTorch grouped entrypoint;
                   vLLM exposes no bf16 grouped mm of its own)
  bf16/fp8 x fp8   -> scaled_fp8_quant + cutlass_moe_mm
  bf16/nvfp4 x nvfp4 -> scaled_fp4_experts_quant + cutlass_fp4_moe_mm
  bf16/mxfp4 x mxfp4 -> mxfp4_experts_quant + cutlass_mxfp4_moe_mm

Quantization of the weights happens in prepare(), outside the timed region.
For weight-only pairs (bf16 activation x quantized weight) the per-call
activation quantization stays INSIDE the timed region — that is real per-call
serving work; symmetric pairs pre-quantize both sides and time the mm alone.
"""
from __future__ import annotations

import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("vllm", "torch")


class _Layer(torch.nn.Module):
    """Minimal stand-in for a vLLM linear layer: the marlin helpers only touch
    .weight/.weight_scale/.input_size/.output_size and friends."""

    def __init__(self, k: int, n: int):
        super().__init__()
        self.input_size = k
        self.output_size = n
        self.input_size_per_partition = k
        self.output_size_per_partition = n
        self.orig_dtype = torch.bfloat16


def _bf16(m: int, k: int) -> torch.Tensor:
    return torch.randn(m, k, dtype=torch.bfloat16, device="cuda")


def _prepare_gemm(op: Op) -> dict:
    a = op.args
    if a.get("activation") is not None:
        raise UnsupportedOpError(
            f"vllm gemm backend has no fused activation; got activation={a['activation']!r}")
    m, n, k = a["m"], a["n"], a["k"]
    if m <= 0 or n <= 0 or k <= 0:
        raise UnsupportedOpError(f"degenerate gemm shape m={m} n={n} k={k}")
    da, db = a["dtype_a"], a["dtype_b"]
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") if a.get("bias") else None

    # ---- bf16 x bf16 : unquantized ------------------------------------------
    if da == "bf16" and db == "bf16":
        return {"kind": "bf16", "x": _bf16(m, k),
                "w": torch.randn(n, k, dtype=torch.bfloat16, device="cuda"), "bias": bias}

    # ---- bf16 in x fp8 weight -------------------------------------------------
    # This is what a real vLLM deployment runs on FP8-native hardware: the layer
    # takes bf16 activations and dynamically quantizes them per token to fp8,
    # then calls the native cutlass scaled MM (W8A8). The activation quant is
    # kept INSIDE the timed region because it is real per-call serving work.
    # Marlin's weight-only (W8A16) path is only vLLM's fallback for GPUs without
    # native FP8, so it is used here only if cutlass is unavailable.
    if da == "bf16" and db == "fp8":
        fp8 = torch.float8_e4m3fn
        w_hi = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        wscale = (w_hi.abs().amax() / 448.0).clamp(min=1e-6).to(torch.float32)
        wq = (w_hi / wscale).to(fp8)
        x = _bf16(m, k)
        # Prefer vLLM's cutlass scaled MM; some SKUs (e.g. sm103) report
        # cutlass_fp8_supported() == True but ship no compiled kernel, so the
        # only reliable test is to actually call it once here, outside the
        # timed region. Fall back to torch's native fp8 MM, then to Marlin
        # weight-only (what vLLM itself uses on non-FP8-native GPUs).
        try:
            from vllm import _custom_ops as vops
            xq, xs = vops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
            vops.cutlass_scaled_mm(xq, wq.t(), xs, wscale.reshape(1, 1),
                                   torch.bfloat16, None)
            return {"kind": "fp8_w8a8_cutlass", "x": x, "w_t": wq.t(),
                    "wscale": wscale.reshape(1, 1), "bias": bias, "ops": vops}
        except Exception:
            pass
        try:
            from vllm import _custom_ops as vops
            xq, xs = vops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=False)
            torch._scaled_mm(xq, wq.t(), scale_a=xs.reshape(1, 1),
                             scale_b=wscale.reshape(1, 1), out_dtype=torch.bfloat16)
            return {"kind": "fp8_w8a8_torch", "x": x, "w_t": wq.t(),
                    "wscale": wscale.reshape(1, 1), "bias": bias, "ops": vops}
        except Exception:
            pass
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
        layer = _Layer(k, n)
        layer.weight = torch.nn.Parameter(wq, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(wscale.reshape(1), requires_grad=False)
        layer.input_scale = None
        prepare_fp8_layer_for_marlin(layer, size_k_first=False)
        return {"kind": "fp8_marlin", "x": x, "layer": layer, "bias": bias,
                "apply": apply_fp8_marlin_linear, "n": n, "k": k}

    # ---- fp8 x fp8 : W8A8 via vLLM's cutlass scaled mm ----------------------
    if da == "fp8" and db == "fp8":
        from vllm import _custom_ops as vops
        fp8 = torch.float8_e4m3fn
        x_hi = _bf16(m, k)
        w_hi = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        # Scales must be float32 and 2-D: amax() on a bf16 tensor stays bf16, and
        # cutlass_scaled_mm's dispatch rejects both a bf16 scale and a 1-D one.
        xs = (x_hi.abs().amax().float() / 448.0).clamp(min=1e-6)
        ws = (w_hi.abs().amax().float() / 448.0).clamp(min=1e-6)
        return {"kind": "fp8_w8a8", "x": (x_hi / xs).to(fp8),
                "w": (w_hi / ws).to(fp8).t(),
                "xs": xs.reshape(1, 1), "ws": ws.reshape(1, 1),
                "bias": bias, "mm": vops.cutlass_scaled_mm}

    # ---- nvfp4 x nvfp4 : W4A4 via vLLM's cutlass scaled fp4 mm -------------
    # Both operands are pre-quantized in prepare(), matching the fp8 x fp8
    # convention above: the timed region is the mm alone.
    #
    # nvfp4 = e2m1 values packed two per byte, one fp8(e4m3) scale per group of
    # 16 along K, plus a per-tensor global scale. scaled_fp4_quant emits the
    # swizzled block-scale layout the cutlass kernel expects; `alpha` folds the
    # two global scales back out of the accumulator.
    if da == "nvfp4" and db == "nvfp4":
        from vllm import _custom_ops as vops
        if k % 16:
            raise UnsupportedOpError(f"nvfp4 needs k % 16 == 0; got k={k}")
        if bias is not None:
            raise UnsupportedOpError("cutlass_scaled_fp4_mm takes no bias")
        # e4m3 max * e2m1 max -- vLLM's ModelOpt global-scale convention.
        gmax = 448.0 * 6.0
        x_hi = _bf16(m, k)
        w_hi = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        gs_x = (gmax / x_hi.abs().amax().float()).clamp(min=1e-6)
        gs_w = (gmax / w_hi.abs().amax().float()).clamp(min=1e-6)
        xq, xs = vops.scaled_fp4_quant(x_hi, gs_x)
        wq, ws = vops.scaled_fp4_quant(w_hi, gs_w)
        alpha = (1.0 / (gs_x * gs_w)).to(torch.float32)
        # Some SKUs report fp4 support but ship no compiled kernel; the only
        # reliable check is to call it once here, outside the timed region.
        try:
            vops.cutlass_scaled_fp4_mm(xq, wq, xs, ws, alpha, torch.bfloat16)
        except Exception as e:
            raise UnsupportedOpError(f"cutlass_scaled_fp4_mm unavailable: {e}") from e
        return {"kind": "nvfp4_w4a4", "x": xq, "xs": xs, "w": wq, "ws": ws,
                "alpha": alpha, "mm": vops.cutlass_scaled_fp4_mm}

    # ---- bf16 x fp4-family, and mxfp4 anywhere ------------------------------
    # vLLM exposes no plain-Linear entrypoint for these: every mxfp4-capable op
    # in this build is MoE-only (cutlass_mxfp4_moe_mm, mxfp4_experts_quant), and
    # the nvfp4 W4A16 method routes through Marlin rather than a dense kernel.
    if db in ("mxfp4", "nvfp4", "fp4"):
        raise UnsupportedOpError(
            f"vllm has no plain-Linear kernel for {da}/{db} (mxfp4 is MoE-only; "
            f"nvfp4 requires a quantized activation -- use nvfp4/nvfp4)")

    raise UnsupportedOpError(f"vllm gemm backend: unsupported dtype pair {da}/{db}")


def _kernel_gemm(ctx: dict) -> None:
    kind = ctx["kind"]
    if kind == "bf16":
        ctx["out"] = torch.nn.functional.linear(ctx["x"], ctx["w"], ctx["bias"])
    elif kind == "fp8_marlin":
        layer = ctx["layer"]
        ctx["out"] = ctx["apply"](
            input=ctx["x"], weight=layer.weight, weight_scale=layer.weight_scale,
            workspace=layer.workspace, size_n=ctx["n"], size_k=ctx["k"], bias=ctx["bias"])
    elif kind == "fp8_w8a8_cutlass":
        ops = ctx["ops"]
        xq, xs = ops.scaled_fp8_quant(ctx["x"], None, use_per_token_if_dynamic=True)
        ctx["out"] = ops.cutlass_scaled_mm(xq, ctx["w_t"], xs, ctx["wscale"],
                                           torch.bfloat16, ctx["bias"])
    elif kind == "fp8_w8a8_torch":
        ops = ctx["ops"]
        xq, xs = ops.scaled_fp8_quant(ctx["x"], None, use_per_token_if_dynamic=False)
        ctx["out"] = torch._scaled_mm(xq, ctx["w_t"], scale_a=xs.reshape(1, 1),
                                      scale_b=ctx["wscale"], out_dtype=torch.bfloat16,
                                      bias=ctx["bias"])
    elif kind == "fp8_w8a8":
        ctx["out"] = ctx["mm"](ctx["x"], ctx["w"], ctx["xs"], ctx["ws"],
                               torch.bfloat16, ctx["bias"])
    elif kind == "nvfp4_w4a4":
        ctx["out"] = ctx["mm"](ctx["x"], ctx["w"], ctx["xs"], ctx["ws"],
                               ctx["alpha"], torch.bfloat16)


def _grouped_sizes(a: dict) -> tuple[list[int], int, int]:
    gs = [int(x) for x in a["group_sizes"]]
    if not gs or any(x < 0 for x in gs):
        raise UnsupportedOpError(f"bad group_sizes (len={len(gs)})")
    m_total = sum(gs)
    if m_total <= 0:
        raise UnsupportedOpError("group_sizes sum to 0")
    return gs, m_total, len(gs)


def _synth_topk(gs: list[int]) -> torch.Tensor:
    """topk_ids (M_total, 1): token i's expert = its segment. Tokens are laid
    out pre-sorted by group, so vLLM's a_map/c_map come out as identity."""
    return torch.repeat_interleave(
        torch.arange(len(gs), device="cuda", dtype=torch.int32),
        torch.tensor(gs, device="cuda")).unsqueeze(1).to(torch.int32)


def _prepare_grouped_gemm(op: Op) -> dict:
    a = op.args
    gs, m_total, g = _grouped_sizes(a)
    n, k = a["n"], a["k"]
    da, db = a["dtype_a"], a["dtype_b"]
    dev = "cuda"

    # ---- bf16 x bf16: torch's grouped mm (offs = per-group cumulative ends) --
    if da == "bf16" and db == "bf16":
        x = torch.randn(m_total, k, dtype=torch.bfloat16, device=dev)
        w = torch.randn(g, k, n, dtype=torch.bfloat16, device=dev)
        offs = torch.cumsum(torch.tensor(gs, device=dev), 0).to(torch.int32)
        torch._grouped_mm(x, w, offs=offs)  # availability check, untimed
        return {"kind": "grouped_bf16", "x": x, "w": w, "offs": offs}

    # ---- (bf16|fp8) x fp8: cutlass_moe_mm ------------------------------------
    if db == "fp8" and da in ("bf16", "fp8"):
        from vllm import _custom_ops as vops
        fp8 = torch.float8_e4m3fn
        w_hi = torch.randn(g, n, k, dtype=torch.bfloat16, device=dev)
        ws = (w_hi.abs().amax(dim=(1, 2)).float() / 448.0).clamp(min=1e-6)
        wq = (w_hi / ws.view(g, 1, 1)).to(fp8)
        x = torch.randn(m_total, k, dtype=torch.bfloat16, device=dev)
        offs = torch.tensor([0] + list(torch.tensor(gs).cumsum(0)[:-1]),
                            device=dev, dtype=torch.int64)  # per-group starts
        # The c3x grouped kernel schedules with swapped A/B when total M <= 64
        # and then REQUIRES problem rows as (n, m, k); unswapped rows make it
        # read out of bounds (verified empirically -- mirrors vLLM's own
        # swap_ab in run_cutlass_moe_fp8).
        swap = m_total <= 64
        ps = torch.tensor([[n, m, k] if swap else [m, n, k] for m in gs],
                          device=dev, dtype=torch.int32)
        ab_strides = torch.full((g,), k, device=dev, dtype=torch.int64)
        c_strides = torch.full((g,), n, device=dev, dtype=torch.int64)
        ctx = {"kind": "grouped_fp8", "w": wq, "ws": ws.view(g, 1),
               "offs": offs, "ps": ps, "ab": ab_strides, "c": c_strides,
               "out": torch.empty(m_total, n, dtype=torch.bfloat16, device=dev),
               "ops": vops, "act_quant": da == "bf16"}
        if da == "bf16":
            ctx["x"] = x
        else:
            xq, xs = vops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
            ctx["xq"], ctx["xs"] = xq, xs
        _kernel_grouped_gemm(ctx)  # availability + layout check, untimed
        return ctx

    # ---- (bf16|nvfp4) x nvfp4: cutlass_fp4_moe_mm ----------------------------
    if db == "nvfp4" and da in ("bf16", "nvfp4"):
        from vllm import _custom_ops as vops
        if k % 16:
            raise UnsupportedOpError(f"nvfp4 grouped needs k % 16 == 0; got k={k}")
        gmax = 448.0 * 6.0
        w_hi = torch.randn(g, n, k, dtype=torch.bfloat16, device=dev)
        gs_w = (gmax / w_hi.abs().amax(dim=(1, 2)).float()).clamp(min=1e-6)
        # SKUs without fp4 tensor cores (e.g. sm90) fail here, not at the mm;
        # surface that as unsupported rather than an error.
        try:
            vops.scaled_fp4_quant(w_hi[0], gs_w[0])
        except Exception as e:
            raise UnsupportedOpError(
                f"nvfp4 quant/mm unavailable on this SKU: {e}") from e
        wq_l, wbs_l = [], []
        for e in range(g):
            q, bs = vops.scaled_fp4_quant(w_hi[e], gs_w[e])
            wq_l.append(q)
            wbs_l.append(bs)
        wq = torch.stack(wq_l)
        wbs = torch.stack(wbs_l)
        x = torch.randn(m_total, k, dtype=torch.bfloat16, device=dev)
        gs_x = (gmax / x.abs().amax().float()).clamp(min=1e-6).expand(g).contiguous()
        alphas = (1.0 / (gs_x * gs_w)).to(torch.float32)
        topk = _synth_topk(gs)
        expert_offsets = torch.empty(g + 1, dtype=torch.int32, device=dev)
        blockscale_offsets = torch.empty(g + 1, dtype=torch.int32, device=dev)
        ps1 = torch.empty(g, 3, dtype=torch.int32, device=dev)
        ps2 = torch.empty(g, 3, dtype=torch.int32, device=dev)
        a_map = torch.empty(topk.numel(), dtype=torch.int32, device=dev)
        c_map = torch.empty(topk.numel(), dtype=torch.int32, device=dev)
        vops.get_cutlass_moe_mm_data(topk, expert_offsets, ps1, ps2, a_map,
                                     c_map, g, n, k, blockscale_offsets,
                                     is_gated=False)
        ctx = {"kind": "grouped_nvfp4", "w": wq, "wbs": wbs, "alphas": alphas,
               "gs_x": gs_x, "eo": expert_offsets, "bo": blockscale_offsets,
               "ps": ps1,
               "out": torch.empty(m_total, n, dtype=torch.bfloat16, device=dev),
               "ops": vops, "act_quant": da == "bf16"}
        if da == "bf16":
            ctx["x"] = x
        else:
            xq, xbs = vops.scaled_fp4_experts_quant(
                x, gs_x, expert_offsets, blockscale_offsets, 1)
            ctx["xq"], ctx["xbs"] = xq, xbs
        _kernel_grouped_gemm(ctx)  # availability + layout check, untimed
        return ctx

    # ---- (bf16|mxfp4) x mxfp4: cutlass_mxfp4_moe_mm --------------------------
    # Weights are quantized here to the kernel's linear layout: e2m1 packed two
    # per byte with one e8m0 (power-of-two) scale per 32-block along K.
    # Activations go through vllm's mxfp4_experts_quant. Both-sides-mxfp4 has
    # coarser scales than nvfp4, so vs a bf16 reference segments land at
    # cos ~0.94 -- validated as quant noise, not a mapping error.
    if db == "mxfp4" and da in ("bf16", "mxfp4"):
        from vllm import _custom_ops as vops
        if k % 32:
            raise UnsupportedOpError(f"mxfp4 grouped needs k % 32 == 0; got k={k}")
        if not vops.mxfp4_experts_quant_supported(
                torch.cuda.get_device_capability()[0] * 10
                + torch.cuda.get_device_capability()[1]):
            raise UnsupportedOpError("mxfp4_experts_quant unsupported on this SKU")
        w_hi = torch.randn(g, n, k, dtype=torch.bfloat16, device=dev)
        kb = k // 32
        grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=dev)
        wq = torch.empty(g, n, k // 2, dtype=torch.uint8, device=dev)
        wbs = torch.empty(g, n, kb, dtype=torch.uint8, device=dev)
        for e in range(g):  # per-expert loop bounds peak memory
            wb = w_hi[e].float().view(n, kb, 32)
            ex = torch.ceil(torch.log2(wb.abs().amax(-1).clamp(min=1e-9) / 6.0)) \
                .clamp(-127, 127)
            sc = (wb / torch.pow(2.0, ex).unsqueeze(-1))
            idx = (sc.abs().unsqueeze(-1) - grid).abs().argmin(-1).to(torch.uint8)
            code = idx | ((sc < 0).to(torch.uint8) << 3)
            flat = code.view(n, k)
            wq[e] = flat[..., 0::2] | (flat[..., 1::2] << 4)
            wbs[e] = (ex + 127).to(torch.uint8).view(n, kb)
        x = torch.randn(m_total, k, dtype=torch.bfloat16, device=dev)
        topk = _synth_topk(gs)
        expert_offsets = torch.empty(g + 1, dtype=torch.int32, device=dev)
        blockscale_offsets = torch.empty(g + 1, dtype=torch.int32, device=dev)
        ps1 = torch.empty(g, 3, dtype=torch.int32, device=dev)
        ps2 = torch.empty(g, 3, dtype=torch.int32, device=dev)
        a_map = torch.empty(topk.numel(), dtype=torch.int32, device=dev)
        c_map = torch.empty(topk.numel(), dtype=torch.int32, device=dev)
        vops.get_cutlass_moe_mm_data(topk, expert_offsets, ps1, ps2, a_map,
                                     c_map, g, n, k, blockscale_offsets,
                                     is_gated=False)
        ctx = {"kind": "grouped_mxfp4", "w": wq, "wbs": wbs,
               "eo": expert_offsets, "bo": blockscale_offsets, "ps": ps1,
               "g": g,
               "out": torch.empty(m_total, n, dtype=torch.bfloat16, device=dev),
               "ops": vops, "act_quant": da == "bf16"}
        if da == "bf16":
            ctx["x"] = x
        else:
            xq, xbs = vops.mxfp4_experts_quant(x, expert_offsets,
                                               blockscale_offsets, g, 1)
            ctx["xq"], ctx["xbs"] = xq, xbs
        _kernel_grouped_gemm(ctx)  # availability + layout check, untimed
        return ctx

    raise UnsupportedOpError(f"vllm grouped_gemm: unsupported dtype pair {da}/{db}")


def _kernel_grouped_gemm(ctx: dict) -> None:
    kind = ctx["kind"]
    if kind == "grouped_bf16":
        ctx["out"] = torch._grouped_mm(ctx["x"], ctx["w"], offs=ctx["offs"])
    elif kind == "grouped_fp8":
        ops = ctx["ops"]
        if ctx["act_quant"]:
            xq, xs = ops.scaled_fp8_quant(ctx["x"], None,
                                          use_per_token_if_dynamic=True)
        else:
            xq, xs = ctx["xq"], ctx["xs"]
        ops.cutlass_moe_mm(ctx["out"], xq, ctx["w"], xs, ctx["ws"], ctx["offs"],
                           ctx["ps"], ctx["ab"], ctx["ab"], ctx["c"],
                           True, False)
    elif kind == "grouped_nvfp4":
        ops = ctx["ops"]
        if ctx["act_quant"]:
            xq, xbs = ops.scaled_fp4_experts_quant(
                ctx["x"], ctx["gs_x"], ctx["eo"], ctx["bo"], 1)
        else:
            xq, xbs = ctx["xq"], ctx["xbs"]
        ops.cutlass_fp4_moe_mm(ctx["out"], xq, ctx["w"], xbs, ctx["wbs"],
                               ctx["alphas"], ctx["ps"], ctx["eo"][:-1],
                               ctx["bo"][:-1])
    elif kind == "grouped_mxfp4":
        ops = ctx["ops"]
        if ctx["act_quant"]:
            xq, xbs = ops.mxfp4_experts_quant(ctx["x"], ctx["eo"], ctx["bo"],
                                              ctx["g"], 1)
        else:
            xq, xbs = ctx["xq"], ctx["xbs"]
        ops.cutlass_mxfp4_moe_mm(ctx["out"], xq, ctx["w"], xbs, ctx["wbs"],
                                 ctx["ps"], ctx["eo"][:-1], ctx["bo"][:-1])


IMPLS = [
    BackendImpl(op_type="gemm", prepare=_prepare_gemm, kernel=_kernel_gemm),
    BackendImpl(op_type="grouped_gemm", prepare=_prepare_grouped_gemm,
                kernel=_kernel_grouped_gemm),
]
