"""
kairos_hybrid_block.py  (v3 — 4B scale)

Kairos-4B Hybrid Block — scaled to ≥4B total parameters.

v3 scale changes (v2 → v3):
  SCALE  MoE: 8 experts → 64 experts (10× more routing capacity)
  SCALE  MoE: d_ff=2730 → moe_d_ff=5460 (2× wider expert layers)
  SCALE  Separation: moe_d_ff (MoE experts) vs d_ff (text/decoder FFN) are now distinct
  PARAM  Per-block MoE: 66.9M → 1,073.5M  |  3-block core: 243M → 3,263M

Architecture rules (CLAUDE.md — updated for 4B scale):
  d_model=1024, d_ff=2730, moe_d_ff=5460, 64 experts, Top-2, BF16, ZeRO-3 compatible
  stop-grad on h_mamba and h_cfc, z-loss on MoE router,
  checkpoint prefix s3://kairos-emr-assets/checkpoints/kairos-4b/
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func  # type: ignore
    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False

from torch.utils.checkpoint import checkpoint as _grad_ckpt

BUILD_ID = "zero3-debug-bypass-v21"
print(f"[kairos] {BUILD_ID}", flush=True)

# torch.compile guard — set after first call
_SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")


def _debug_shapes_enabled() -> bool:
    return (
        os.environ.get("KAIROS_DEBUG_SHAPES", "0") == "1"
        and int(os.environ.get("RANK", "0")) == 0
    )


def _assert_same_dtype_for_inplace(
    name: str, dst: torch.Tensor, src: torch.Tensor
) -> None:
    """Raise if dst/src dtype or device differ — gated by KAIROS_DEBUG_DTYPE=1."""
    if dst.dtype != src.dtype or dst.device != src.device:
        raise RuntimeError(
            f"[dtype/inplace] {name}: "
            f"dst dtype/device={dst.dtype}/{dst.device}, "
            f"src dtype/device={src.dtype}/{src.device}"
        )


def _safe_index_add_(
    dst: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    *,
    name: str = "",
) -> torch.Tensor:
    """index_add_ with automatic dtype/device coercion of src to match dst."""
    if os.environ.get("KAIROS_DEBUG_DTYPE", "0") == "1" \
            and int(os.environ.get("RANK", "0")) == 0 \
            and (dst.dtype != src.dtype or dst.device != src.device):
        print(
            f"[dtype/index_add_] {name}: "
            f"dst={dst.dtype}/{dst.device} src={src.dtype}/{src.device} "
            f"— casting src",
            flush=True,
        )
    return dst.index_add_(dim, index.to(device=dst.device), src.to(device=dst.device, dtype=dst.dtype))


def _safe_scatter_add_(
    dst: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    *,
    name: str = "",
) -> torch.Tensor:
    """scatter_add_ with automatic dtype/device coercion of src to match dst."""
    if os.environ.get("KAIROS_DEBUG_DTYPE", "0") == "1" \
            and int(os.environ.get("RANK", "0")) == 0 \
            and (dst.dtype != src.dtype or dst.device != src.device):
        print(
            f"[dtype/scatter_add_] {name}: "
            f"dst={dst.dtype}/{dst.device} src={src.dtype}/{src.device} "
            f"— casting src",
            flush=True,
        )
    return dst.scatter_add_(dim, index.to(device=dst.device), src.to(device=dst.device, dtype=dst.dtype))


def _flash_attn_safe(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
    window_size: Optional[Tuple[int, int]] = None,
) -> Optional[torch.Tensor]:
    """Call flash_attn_func with optional window_size; returns None on API mismatch."""
    if not _HAS_FLASH:
        return None
    try:
        if window_size is not None:
            return flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
            )
        return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
    except TypeError as e:
        if "window_size" in str(e):
            print(
                "[attn] flash_attn_func does not support window_size; "
                "falling back to SDPA",
                flush=True,
            )
            return None
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KairosConfig:
    # ── core ──────────────────────────────────────────────────────────────────
    d_model: int = 1024

    # ── Mamba-2 ───────────────────────────────────────────────────────────────
    d_state: int = 64        # SSM state size N
    d_conv: int = 4          # depthwise conv kernel width
    mamba_expand: int = 2    # d_inner = expand × d_model  →  2048
    dt_rank: int = 64        # Δ projection rank (⌈d_model/16⌉); 1 in v1 was wrong
    mamba_chunk: int = 64    # chunkwise scan chunk size; keeps Q above float floor

    # ── Sliding-Window Attention ───────────────────────────────────────────────
    num_heads_q: int = 16
    num_heads_kv: int = 4    # GQA: 4 KV heads, each serves 4 Q heads
    attn_window: int = 1024
    max_seq_len: int = 4096  # RoPE cache upper bound

    # ── MoE ───────────────────────────────────────────────────────────────────
    num_experts: int = 64
    top_k: int = 2
    d_ff: int = 2730         # ⌊8/3 × d_model⌋ — used by text encoder & S2FT decoder
    moe_d_ff: int = 5460     # ⌊16/3 × d_model⌋ — MoE expert width (2× iso-param)
    moe_bias_lr: float = 1e-3    # γ for aux-free bias update
    z_loss_coeff: float = 1e-3   # router logit entropy regulariser (CLAUDE.md rule)

    # ── Loop structure ─────────────────────────────────────────────────────────
    num_loops: int = 4
    num_blocks: int = 3      # unique blocks; loop_block_map = [0,1,2,0]

    # ── Adaptive early exit (inference only) ──────────────────────────────────
    exit_threshold: float = 0.5

    # ── Regularisation ────────────────────────────────────────────────────────
    loop_dropout: float = 0.0

    # ── Activation checkpointing ──────────────────────────────────────────────
    use_grad_checkpoint: bool = True   # checkpoint each hybrid block call during training

    # ── Dense MoE fallback (ZeRO-3 safe smoke testing) ────────────────────────
    # When True, MoeSwiGLUFFN uses a dense weighted-sum over all experts instead
    # of sparse sort-by-expert dispatch.  Eliminates variable-length expert slices
    # that cause ZeRO-3 shape mismatches when expert token counts differ across ranks.
    # Auto-enabled by kairos_train.py when --ultra_smoke_core_loss=True.
    dense_moe_fallback: bool = False

    # ── Module isolation flags for ZeRO-3 backward debug ─────────────────────
    # When True, the corresponding sub-block is replaced with an identity pass.
    # Use --core_debug_bypass_* CLI flags to isolate which sub-block fails ZeRO-3.
    # Logs on rank-0 exactly which components are disabled.  False in production.
    core_debug_bypass_mamba: bool = False   # Mamba-2 → identity (x unchanged)
    core_debug_bypass_cfc:   bool = False   # CfC (IMU) → identity (x unchanged)
    core_debug_bypass_swa:   bool = False   # Sliding-Window Attention → identity
    core_debug_bypass_moe:   bool = False   # MoE FFN → identity (_z_loss_val=None)
    # Limit hybrid core to first N loop iterations (0 = run all num_loops).
    # Combine with bypass flags to binary-search the ZeRO-3 failure point.
    core_debug_layers: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """RMS Layer Norm.  Weight multiply kept in float32 for BF16 precision."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms_inv = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        # Keep in float32 until the final cast so weight precision is not lost
        return (x_f * rms_inv * self.weight.float()).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    q: torch.Tensor,    # (B, T, H_q, head_dim)
    k: torch.Tensor,    # (B, T, H_kv, head_dim)
    cos: torch.Tensor,  # (T, head_dim)
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, :, None, :]   # (1, T, 1, head_dim)
    sin = sin[None, :, None, :]
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised Mamba selective scan (chunkwise prefix scan — no Python loop over T)
# ──────────────────────────────────────────────────────────────────────────────

def _selective_scan_chunked(
    Abar:   torch.Tensor,    # (B, T, di, N) — discrete A coefficients ∈ (0,1]
    Bbar_x: torch.Tensor,    # (B, T, di, N) — Δ·B·x (input already folded in)
    C_ssm:  torch.Tensor,    # (B, T, N)     — read-out projection
    h_init: torch.Tensor,    # (B, di, N)    — initial hidden state
    chunk:  int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the selective SSM in O(T) time without a Python loop over T.

    Recurrence:  h_t = Abar_t ⊙ h_{t-1}  +  Bbar_x_t
                 y_t = Σ_n C_t[n] · h_t[:,:,n]

    Within each chunk of length C the closed-form solution is:
        Q_t   = ∏_{j=0}^{t} Abar_j           (cumulative product, log-space)
        h_t   = Q_t · h_prev  +  Q_t · Σ_{k=0}^{t} (Bbar_x_k / Q_k)

    Chunking with C=64 keeps Q above numerical floor for any realistic Abar.

    Returns:
        y        (B, T, di)   — SSM output at every timestep
        h_final  (B, di, N)   — hidden state after the last timestep (detached
                                 by the caller for stop-grad handoff)
    """
    B, T, di, N = Abar.shape
    h_t = h_init
    ys: List[torch.Tensor] = []

    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        Ab = Abar[:, start:end]       # (B, C, di, N)
        Bb = Bbar_x[:, start:end]     # (B, C, di, N)
        Cs = C_ssm[:, start:end]      # (B, C, N)

        # Q: cumulative product of Abar within the chunk (log-space for stability)
        log_Q = Ab.float().clamp(min=1e-38).log().cumsum(dim=1)  # (B,C,di,N)
        Q = log_Q.exp().to(Ab.dtype)                              # (B,C,di,N)

        # Σ_{k=0}^{t} (Bbar_x_k / Q_k)  — normalised cumsum
        Bb_norm = Bb / Q.clamp(min=torch.finfo(Bb.dtype).tiny)   # (B,C,di,N)
        norm_cs = Bb_norm.cumsum(dim=1)                           # (B,C,di,N)

        # h_t_at_each_step = Q · (h_prev + norm_cumsum)
        h_chunk = Q * (h_t.unsqueeze(1) + norm_cs)               # (B,C,di,N)

        # y_t = Σ_n C_t[n] · h_chunk[:,:,:,n]
        y_chunk = (h_chunk * Cs.unsqueeze(2)).sum(-1)            # (B,C,di)
        ys.append(y_chunk)

        h_t = h_chunk[:, -1]   # carry state to next chunk (B,di,N)

    return torch.cat(ys, dim=1), h_t   # (B,T,di), (B,di,N)


# ──────────────────────────────────────────────────────────────────────────────
# IMU delta_t alignment utility
# ──────────────────────────────────────────────────────────────────────────────

def _align_imu_delta_t(
    imu_delta_t: Optional[torch.Tensor],
    B: int,
    n_imu: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Ensure imu_delta_t has shape (B, n_imu) on the correct device and dtype.

    Called before CfC processing when delta_t is expressed as a compact
    (B, n_imu) tensor rather than the full-sequence (B, T) form.
    Returns None if n_imu == 0 (no IMU tokens this frame).
    """
    if n_imu == 0:
        return None
    default_dt = 1.0 / 30.0
    if imu_delta_t is None:
        return torch.full((B, n_imu), default_dt, device=device, dtype=dtype)
    imu_delta_t = imu_delta_t.to(device=device, dtype=dtype)
    if imu_delta_t.ndim == 1:
        imu_delta_t = imu_delta_t.unsqueeze(0).expand(B, -1)
    if imu_delta_t.shape[0] != B:
        imu_delta_t = imu_delta_t[:1].expand(B, -1)
    if imu_delta_t.shape[1] < n_imu:
        pad = torch.full((B, n_imu - imu_delta_t.shape[1]), default_dt,
                         device=device, dtype=dtype)
        imu_delta_t = torch.cat([imu_delta_t, pad], dim=1)
    elif imu_delta_t.shape[1] > n_imu:
        imu_delta_t = imu_delta_t[:, :n_imu]
    return imu_delta_t


# ──────────────────────────────────────────────────────────────────────────────
# Sub-block 1 — Mamba-2 Selective SSM
# ──────────────────────────────────────────────────────────────────────────────

class Mamba2Block(nn.Module):
    """
    Mamba-2 selective SSM.

    Forward path:
      RMSNorm → Linear(d→2·di) → Conv1D(k=4)+SiLU
             → Selective-SSM (chunked parallel scan)
             → ×SiLU(z) gate → Out-proj(→d) → ⊕ residual

    SSM equations (input-dependent A, B, C — no fixed dynamics):
      h_t = Ā(Δ) ⊙ h_{t-1}  +  B̄(Δ)·x_t
      y_t = C_t · h_t  +  D ⊙ x_t

    Key v2 fix: dt_rank=64 (was 1). Using rank-1 Δ projection collapses all
    channels to a single time-constant; dt_rank=64 lets each group of ~32
    d_inner channels have an independent Δ, matching the original Mamba design.
    """

    def __init__(self, cfg: KairosConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.d_inner = cfg.mamba_expand * d   # 2048
        self.d_state = cfg.d_state            # 64
        self.dt_rank = cfg.dt_rank            # 64
        self.chunk = cfg.mamba_chunk          # 64
        di = self.d_inner

        # x branch + z gate
        self.in_proj = nn.Linear(d, 2 * di, bias=False)

        # Causal depthwise conv (padding trimmed to T in forward)
        self.conv1d = nn.Conv1d(
            di, di,
            kernel_size=cfg.d_conv,
            padding=cfg.d_conv - 1,
            groups=di,
            bias=True,
        )

        # x → (B_ssm[d_state], C_ssm[d_state], dt[dt_rank])
        self.x_proj = nn.Linear(di, 2 * cfg.d_state + cfg.dt_rank, bias=False)

        # Δ: dt_rank → d_inner (rank expansion with bias for stable initial dt)
        self.dt_proj = nn.Linear(cfg.dt_rank, di, bias=True)
        # Initialise dt_proj bias so initial Δ ≈ softplus(log(0.01)) ≈ 0.01
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
        nn.init.constant_(self.dt_proj.bias, math.log(0.01))

        # Diagonal A in log-space; always negative → stable discretisation
        A_init = (
            torch.arange(1, cfg.d_state + 1, dtype=torch.float32)
            .unsqueeze(0).expand(di, -1).clone()
        )
        self.A_log = nn.Parameter(torch.log(A_init))   # (di, d_state)

        self.D = nn.Parameter(torch.ones(di))

        # Scale output projection down by 1/√(2·num_loops) for residual stability
        self.out_proj = nn.Linear(di, d, bias=False)
        nn.init.normal_(self.out_proj.weight, std=d ** -0.5)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                  # (B, T, d)
        h: Optional[torch.Tensor],        # (B, di, d_state) | None
        norm: RMSNorm,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns  (residual + out,  new_h.detach())."""
        residual = x
        x = norm(x)

        B, T, _ = x.shape
        di = self.d_inner

        # ── Input projection ─────────────────────────────────────────────────
        xz = self.in_proj(x)                          # (B, T, 2·di)
        x_ssm, z = xz.chunk(2, dim=-1)               # each (B, T, di)

        # ── Causal depthwise conv (trim right-side padding) ──────────────────
        x_ssm = (
            self.conv1d(x_ssm.transpose(1, 2))[..., :T]
            .transpose(1, 2)
        )
        x_ssm = F.silu(x_ssm)                        # (B, T, di)

        # ── SSM parameter projections ─────────────────────────────────────────
        bcd = self.x_proj(x_ssm)                      # (B, T, 2·d_state + dt_rank)
        B_ssm = bcd[..., : self.d_state]              # (B, T, d_state)
        C_ssm = bcd[..., self.d_state : 2 * self.d_state]
        dt_raw = bcd[..., 2 * self.d_state :]         # (B, T, dt_rank)

        # Δ > 0 via softplus; shape (B, T, di)
        delta = F.softplus(self.dt_proj(dt_raw))

        # ── Discrete A and B ─────────────────────────────────────────────────
        # Ā(Δ) = exp(Δ·A) — computed in float32, always in (0,1]
        A = -torch.exp(self.A_log.float())             # (di, d_state), <0
        # (B,T,di,1) * (1,1,di,d_state) → (B,T,di,d_state)
        Abar = torch.exp(
            delta.unsqueeze(-1).float() * A[None, None]
        ).to(x.dtype)

        # B̄(Δ) = Δ · B_ssm;  fold x_ssm in for the scan
        # (B,T,di,1)*(B,T,1,d_state) → (B,T,di,d_state)
        Bbar = delta.unsqueeze(-1) * B_ssm.unsqueeze(2)
        # fold x_ssm: Bbar_x[b,t,i,n] = Bbar[b,t,i,n] * x_ssm[b,t,i]
        Bbar_x = Bbar * x_ssm.unsqueeze(-1)           # (B, T, di, d_state)

        # ── Initialise hidden state ───────────────────────────────────────────
        if h is None:
            h = x_ssm.new_zeros(B, di, self.d_state)

        # ── Chunkwise parallel scan ───────────────────────────────────────────
        y, h_new = _selective_scan_chunked(Abar, Bbar_x, C_ssm, h, self.chunk)
        # y: (B,T,di);  h_new: (B,di,d_state)

        y = y + x_ssm * self.D                        # D skip connection
        y = y * F.silu(z)                             # output gate
        out = self.out_proj(y)                         # (B,T,d)

        return residual + out, h_new.detach()


# ──────────────────────────────────────────────────────────────────────────────
# Sub-block 2 — CfC (Liquid Neural Network) for IMU tokens
# ──────────────────────────────────────────────────────────────────────────────

class CfCBlock(nn.Module):
    """
    Closed-form Continuous-time (CfC) unit applied only to IMU tokens.

    Forward path:
      Extract IMU by mask → RMSNorm + concat(Δt) → CfC closed-form
                         → Out-proj → index_add residual → ⊕

    CfC equation (one closed-form step per IMU token):
      x(t) = σ(-f·Δt) · h_prev  +  [1 - σ(-f·Δt)] · A(x)
      f     = urgency MLP(x_norm ‖ Δt)  →  softplus  →  f > 0
      τ     = 1/f  (input-dependent time constant)

    v2 fixes:
      - delta_t cast to x.dtype (was dtype mismatch)
      - Residual via index_add (no clone+assign)
      - Persisted state = last IMU token in the sequence (most recent reading)
        rather than mean of all IMU tokens (corrects temporal semantics)
    """

    def __init__(self, cfg: KairosConfig) -> None:
        super().__init__()
        d = cfg.d_model

        # f: urgency MLP  [x_norm ‖ Δt] → d, softplus → f > 0
        self.f_mlp = nn.Sequential(
            nn.Linear(d + 1, d // 2, bias=True),
            nn.SiLU(),
            nn.Linear(d // 2, d, bias=True),
        )
        # Attractor: x_norm → A
        self.A_proj = nn.Linear(d, d, bias=True)
        # Output projection (scaled down for residual stability)
        self.out_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.out_proj.weight, std=d ** -0.5)

        # Initialise so f starts near zero → gate ≈ 0.5 (balanced mixing)
        nn.init.zeros_(self.f_mlp[-1].weight)
        nn.init.constant_(self.f_mlp[-1].bias, -2.0)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,              # (B, T, d)
        imu_mask: torch.Tensor,       # (B, T) bool
        delta_t: torch.Tensor,        # (B, T) float — Δt per token (0 non-IMU)
        h_cfc: Optional[torch.Tensor],  # (B, d) | None — persisted state
        norm: RMSNorm,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns  (output (B,T,d),  new_h_cfc (B,d).detach())."""
        B, T, d = x.shape
        # Normalise delta_t dtype/device at the block boundary so every downstream
        # op (cat, sigmoid, index_add) sees a uniform dtype. This is the final
        # safety net even if the caller already cast delta_t_seq before hybrid_core.
        delta_t = delta_t.to(device=x.device, dtype=x.dtype)
        if imu_mask.shape != (B, T):
            raise RuntimeError(
                f"CfCBlock imu_mask shape mismatch: got {tuple(imu_mask.shape)}, "
                f"expected ({B}, {T})"
            )
        if delta_t.shape != (B, T):
            raise RuntimeError(
                f"CfCBlock delta_t shape mismatch: got {tuple(delta_t.shape)}, "
                f"expected ({B}, {T})"
            )
        x_flat = x.reshape(B * T, d)
        mask_flat = imu_mask.reshape(B * T)                        # (B*T,) bool
        dt_flat   = delta_t.reshape(B * T)                         # already cast above
        if _debug_shapes_enabled():
            print(f"[shape/cfc] CfCBlock N_imu={int(mask_flat.sum().item())}", flush=True)

        if not mask_flat.any():
            dummy = x.new_zeros(B, d)
            return x, dummy.detach()

        # ── Extract IMU tokens ────────────────────────────────────────────────
        imu_idx = mask_flat.nonzero(as_tuple=True)[0]              # (N_imu,)
        x_imu = x_flat[imu_idx]                                    # (N_imu, d)
        dt_imu = dt_flat[imu_idx].unsqueeze(-1)                    # (N_imu, 1)
        batch_of_imu = imu_idx // T                                # (N_imu,)
        if x_imu.shape[0] != dt_imu.shape[0]:
            raise RuntimeError(
                f"CfCBlock IMU/delta length mismatch: "
                f"x_imu={x_imu.shape[0]} dt_imu={dt_imu.shape[0]}"
            )

        # ── CfC closed-form update ────────────────────────────────────────────
        x_normed = norm(x_imu)
        x_with_dt = torch.cat([x_normed, dt_imu], dim=-1)
        f = F.softplus(self.f_mlp(x_with_dt))                     # (N_imu, d), >0
        A_attr = self.A_proj(x_normed)                             # (N_imu, d)

        h_prev = (
            x.new_zeros(len(imu_idx), d) if h_cfc is None
            else h_cfc[batch_of_imu]   # (N_imu, d)
        )

        gate = torch.sigmoid(-f * dt_imu)                          # (N_imu, d)
        x_new = gate * h_prev + (1.0 - gate) * A_attr             # (N_imu, d)
        out_imu = self.out_proj(x_new)                             # (N_imu, d)

        # ── Residual via index_add (no clone, zero extra allocation) ─────────
        # Defensive dtype cast: under DeepSpeed BF16, x_flat and out_imu can
        # diverge in dtype (e.g. x_flat float32 from input path, out_imu BF16
        # from linear projection) causing:
        #   RuntimeError: index_add_(): self (Float) and source (BFloat16)
        output_flat = _safe_index_add_(
            x_flat.clone(), 0, imu_idx, out_imu, name="cfc_residual"
        )                                                          # (B*T, d)

        # ── Persisted state: last IMU token per batch item (fully vectorised) ───
        # imu_idx comes from ascending nonzero() scan, so local indices 0..N-1
        # are also sorted ascending within each batch.  amax therefore picks the
        # rightmost (most recent) IMU token per batch item — no Python loop needed.
        local_pos = torch.arange(len(imu_idx), device=x.device, dtype=torch.long)

        # scatter_reduce_ "amax", include_self=True: self starts at -1 (< any valid
        # local_pos), so updated positions become max(local_pos), un-updated stay -1.
        last_local = torch.full((B,), -1, dtype=torch.long, device=x.device)
        last_local.scatter_reduce_(0, batch_of_imu, local_pos, reduce="amax")

        has_imu = last_local >= 0                                     # (B,) bool
        safe_last = last_local.clamp(min=0)                           # guard -1 index
        new_h = torch.where(has_imu.unsqueeze(-1), x_new[safe_last], x.new_zeros(B, d))

        return output_flat.view(B, T, d), new_h.detach()


# ──────────────────────────────────────────────────────────────────────────────
# Sub-block 3 — Sliding-Window Attention (GQA + RoPE-1D + Flash Attention 2)
# ──────────────────────────────────────────────────────────────────────────────

class SlidingWindowAttention(nn.Module):
    """
    Grouped-Query Attention with RoPE-1D and causal sliding window.

    Forward path:
      RMSNorm → GQA(16Q/4KV) → RoPE-1D → Flash-Attn / SDPA / manual → ⊕

    v2 improvements:
      - RoPE cos/sin cached as a buffer (pre-computed up to max_seq_len)
      - Fallback 1: F.scaled_dot_product_attention (fused kernel in PyTorch 2+)
      - Fallback 2: manual matmul with memory-efficient expand (not repeat_interleave)
    """

    def __init__(self, cfg: KairosConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.num_heads_q == 0
        assert cfg.num_heads_q % cfg.num_heads_kv == 0

        d = cfg.d_model
        self.H_q = cfg.num_heads_q        # 16
        self.H_kv = cfg.num_heads_kv      # 4
        self.head_dim = d // self.H_q     # 64
        self.window = cfg.attn_window     # 1024
        self.groups = self.H_q // self.H_kv  # 4
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d, self.H_q * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.H_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.H_kv * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.out_proj.weight, std=d ** -0.5)

        # ── Pre-compute RoPE cache ─────────────────────────────────────────────
        hd = self.head_dim
        inv_freq = 1.0 / (
            10_000.0 ** (
                torch.arange(0, hd, 2, dtype=torch.float32) / hd
            )
        )
        t = torch.arange(cfg.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)         # (max_T, hd)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,   # (B, T, d)
        norm: RMSNorm,
    ) -> torch.Tensor:
        residual = x
        x = norm(x)

        B, T, _ = x.shape
        hd = self.head_dim

        # ── Q, K, V projections ───────────────────────────────────────────────
        q = self.q_proj(x).view(B, T, self.H_q, hd)    # (B,T,H_q,hd)
        k = self.k_proj(x).view(B, T, self.H_kv, hd)
        v = self.v_proj(x).view(B, T, self.H_kv, hd)

        # ── RoPE (sliced from buffer, cast to x.dtype) ────────────────────────
        cos = self.rope_cos[:T].to(x.dtype)             # (T, hd)
        sin = self.rope_sin[:T].to(x.dtype)
        q, k = _apply_rope(q, k, cos, sin)

        # ── Attention ─────────────────────────────────────────────────────────
        attn_out: Optional[torch.Tensor] = None
        if _HAS_FLASH:
            attn_out = _flash_attn_safe(
                q, k, v,
                causal=True,
                window_size=(self.window - 1, 0),
            )                                            # (B,T,H_q,hd) or None
        if attn_out is None:
            if _SDPA_AVAILABLE:
                attn_out = self._sdpa_attn(q, k, v, T)
            else:
                attn_out = self._manual_attn(q, k, v, T)

        out = self.out_proj(attn_out.reshape(B, T, -1)) # (B,T,d)
        return residual + out

    # ------------------------------------------------------------------
    def _sdpa_attn(self, q, k, v, T: int) -> torch.Tensor:
        """F.scaled_dot_product_attention with GQA expansion and sliding mask."""
        # Expand K/V: (B,T,H_kv,hd) → (B,T,H_q,hd) using expand (no copy)
        k_exp = k.unsqueeze(3).expand(-1, -1, -1, self.groups, -1) \
                 .reshape(k.shape[0], T, self.H_q, self.head_dim)
        v_exp = v.unsqueeze(3).expand(-1, -1, -1, self.groups, -1) \
                 .reshape(v.shape[0], T, self.H_q, self.head_dim)

        # SDPA expects (B, H, T, hd)
        q_t = q.transpose(1, 2).contiguous()
        k_t = k_exp.transpose(1, 2).contiguous()
        v_t = v_exp.transpose(1, 2).contiguous()

        # Build additive mask only when T > window (otherwise full causal is fine)
        attn_mask = None
        if T > self.window:
            attn_mask = _causal_sliding_window_mask(T, self.window, q.device, q.dtype)

        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=attn_mask,
            is_causal=(attn_mask is None),  # let SDPA handle full causal mask
        )                                              # (B,H_q,T,hd)
        return out.transpose(1, 2)                     # (B,T,H_q,hd)

    # ------------------------------------------------------------------
    def _manual_attn(self, q, k, v, T: int) -> torch.Tensor:
        """Fallback: pure-PyTorch GQA with causal sliding-window bias."""
        # Memory-efficient: reshape Q to groups then matmul against unexpanded K
        # (B, H_kv, groups, T, hd) × (B, H_kv, 1, hd, T) → (B, H_kv, groups, T, T)
        q_g = q.transpose(1, 2).view(q.shape[0], self.H_kv, self.groups, T, self.head_dim)
        k_t = k.transpose(1, 2).unsqueeze(2)                      # (B,H_kv,1,T,hd)
        v_t = v.transpose(1, 2).unsqueeze(2)                      # (B,H_kv,1,T,hd)

        attn = torch.matmul(q_g, k_t.transpose(-2, -1)) * self.scale  # (B,H_kv,g,T,T)

        mask = _causal_sliding_window_mask(T, self.window, q.device, attn.dtype)
        attn = attn + mask[None, None, None]
        attn = attn.float().softmax(dim=-1).to(q.dtype)

        out = torch.matmul(attn, v_t)              # (B,H_kv,g,T,hd)
        out = out.reshape(q.shape[0], self.H_q, T, self.head_dim)
        return out.transpose(1, 2)                 # (B,T,H_q,hd)


def _causal_sliding_window_mask(
    T: int, window: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Additive bias: 0 for valid, -inf for masked. Token i attends to [i-w+1..i]."""
    causal = torch.triu(torch.full((T, T), float("-inf"), device=device, dtype=dtype), 1)
    too_far = torch.tril(torch.full((T, T), float("-inf"), device=device, dtype=dtype),
                         diagonal=-window)
    return causal + too_far


# ──────────────────────────────────────────────────────────────────────────────
# Sub-block 4 — MoE SwiGLU FFN (8 experts, Top-2, aux-free + z-loss)
# ──────────────────────────────────────────────────────────────────────────────

class MoeSwiGLUFFN(nn.Module):
    """
    MoE FFN with SwiGLU activation and aux-free load balancing.

    Forward path:
      RMSNorm → Router+bias(Top-2) → 64 SwiGLU experts → Σ g_i·E_i(x) → ⊕

    SwiGLU(x) = SiLU(x@W1) ⊙ (x@W3) @ W2   (no bias; d_ff=2730)

    v2 performance change — sort-by-expert dispatch:
      All (N × top_k) token-expert assignments are sorted once by expert index.
      Each expert then processes a contiguous slice with a single torch.mm call.
      Eliminates nested Python loops and per-iteration .nonzero() calls (which
      could cause implicit CPU-GPU sync in PyTorch <= 2.3).

    v2 accuracy addition — z-loss (CLAUDE.md rule):
      z_loss = mean(logsumexp(router_logits)²) × z_loss_coeff
      Penalises large router logits, preventing over-confident routing collapse.
      Stored as self._z_loss_val (live tensor, no detach) each training forward.
      KairosHybridCore aggregates these into self._z_loss_for_backward.
      Training script: loss = task_loss + cfg.z_loss_coeff * core._z_loss_for_backward

    NOTE (multi-GPU): expert_bias has no gradient and must be all-reduced manually
    after each optimizer step in DDP/ZeRO-3:
        dist.all_reduce(block.moe_ffn.expert_bias, op=dist.ReduceOp.AVG)
    """

    # Expert names not enumerated — 64 experts specialise through learned routing.

    def __init__(self, cfg: KairosConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.E = cfg.num_experts    # 64
        self.top_k = cfg.top_k     # 2
        self.d_ff = cfg.moe_d_ff   # 5460  (expert width, distinct from cfg.d_ff)
        self.γ = cfg.moe_bias_lr   # aux-free bias update rate
        self.dense_moe_fallback = getattr(cfg, 'dense_moe_fallback', False)

        # Expert weights — batched for efficient indexing
        self.W1 = nn.Parameter(torch.empty(self.E, d, cfg.moe_d_ff))  # gate branch
        self.W2 = nn.Parameter(torch.empty(self.E, cfg.moe_d_ff, d))  # down-proj
        self.W3 = nn.Parameter(torch.empty(self.E, d, cfg.moe_d_ff))  # linear branch

        # Router projection (bias injected per-loop from PerLoopParams)
        self.router_proj = nn.Linear(d, self.E, bias=False)

        # Load-balance bias — no gradient, updated post-forward via sign rule
        self.register_buffer("expert_bias", torch.zeros(self.E), persistent=True)
        with torch.no_grad():
            self.expert_bias[:40] += 0.05
            self.expert_bias[56:] -= 0.05
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # NCCL requires all tensors to be on CUDA.
            # expert_bias is freshly allocated on CPU at __init__ time (model not yet
            # moved to device), so we must move it before the broadcast.
            if torch.distributed.get_backend() == "nccl":
                self.expert_bias.data = self.expert_bias.data.to(
                    torch.device("cuda", torch.cuda.current_device())
                )
            print(
                f"[moe] BUILD_ID=moe-broadcast-fix-v14  "
                f"expert_bias device={self.expert_bias.device}",
                flush=True,
            )
            torch.distributed.broadcast(self.expert_bias, src=0)
        self.register_buffer("last_expert_load", torch.zeros(self.E), persistent=False)

        # z-loss: plain Python attribute so the computation graph is NOT severed.
        # Set to None in __init__; populated as a live tensor during training forward.
        # Training script reads core._z_loss_for_backward (see KairosHybridCore).
        self._z_loss_val: Optional[torch.Tensor] = None

        self._init_weights(d, cfg.moe_d_ff)

    def _init_weights(self, d: int, d_ff: int) -> None:
        # W1, W3: (E, d, d_ff) — fan_in=d, fan_out=d_ff
        nn.init.kaiming_uniform_(self.W1.reshape(self.E * d, d_ff), a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W3.reshape(self.E * d, d_ff), a=math.sqrt(5))
        # W2: (E, d_ff, d) — fan_in=d_ff, fan_out=d
        nn.init.kaiming_uniform_(self.W2.reshape(self.E * d_ff, d), a=math.sqrt(5))
        # Router: smaller init → more uniform initial routing
        nn.init.normal_(self.router_proj.weight, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                      # (B, T, d)
        norm: RMSNorm,
        router_bias: Optional[torch.Tensor],  # (E,) per-loop bias | None
    ) -> torch.Tensor:
        residual = x
        x_normed = norm(x)

        B, T, d = x_normed.shape
        N = B * T
        x_flat = x_normed.reshape(N, d)       # (N, d)

        # ── Router ────────────────────────────────────────────────────────────
        logits = self.router_proj(x_flat)     # (N, E)

        # z-loss: penalise large logits BEFORE adding balance bias.
        # Stored as a live tensor (no .detach()) so gradients flow to router_proj.
        if self.training and torch.is_grad_enabled():
            z = torch.logsumexp(logits.float(), dim=-1)  # (N,)
            self._z_loss_val = z.pow(2).mean()            # keep in graph
        else:
            self._z_loss_val = None

        bias = self.expert_bias
        if router_bias is not None:
            bias = bias + router_bias
        scores = logits + bias               # (N, E)

        # ── Dense fallback (ZeRO-3 safe) ─────────────────────────────────────────
        # When dense_moe_fallback=True, avoid all sparse sort-by-expert dispatch.
        # Uses a soft weighted sum over all E experts: no variable-length slices,
        # no rank-dependent token distributions → no ZeRO-3 shape mismatches.
        # z-loss (above) still flows to router_proj in both modes.
        if self.dense_moe_fallback:
            gate_all = F.softmax(scores.float(), dim=-1).to(x.dtype)  # (N, E)
            # Batched expert GEMM: (N, E, d_ff) via einsum over all experts at once
            h1 = torch.einsum('nd,edm->nem', x_flat, self.W1)     # (N, E, d_ff)
            h3 = torch.einsum('nd,edm->nem', x_flat, self.W3)     # (N, E, d_ff)
            h_sg = F.silu(h1) * h3                                 # (N, E, d_ff)
            dense_out = torch.einsum('nem,emd->ned', h_sg, self.W2)  # (N, E, d)
            output = (dense_out * gate_all.unsqueeze(-1)).sum(1)   # (N, d)
            if self.training:
                self._last_forward_N = 0  # skip aux-free bias update in dense mode
            if os.environ.get("KAIROS_DEBUG_MOE", "0") == "1" \
                    and int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"[moe_debug] dense_fallback=True N={N} E={self.E} "
                    f"gate_max={float(gate_all.max().item()):.4f}",
                    flush=True,
                )
            return residual + output.view(B, T, d)

        # Top-2 per token (sparse dispatch path)
        top_vals, top_idx = scores.topk(self.top_k, dim=-1)   # (N,2)
        # Gate: softmax over selected expert scores only
        top_gates = F.softmax(top_vals.float(), dim=-1).to(x.dtype)  # (N,2)

        # ── Sort-by-expert dispatch ────────────────────────────────────────────
        # Flatten all (token, expert) assignments: (N·top_k,)
        token_rep = torch.arange(N, device=x.device).unsqueeze(1).expand_as(top_idx)
        all_tokens = token_rep.reshape(-1)       # (N·K,) token indices
        all_experts = top_idx.reshape(-1)        # (N·K,) expert indices
        all_gates = top_gates.reshape(-1)        # (N·K,) gate values

        # Sort by expert for contiguous GEMM slices
        order = all_experts.argsort(stable=True)
        sorted_tokens = all_tokens[order]        # (N·K,)
        sorted_experts = all_experts[order]
        sorted_gates = all_gates[order]
        sorted_x = x_flat[sorted_tokens]        # (N·K, d)

        # Expert boundary counts — all CPU ints after a single .tolist() sync
        counts = sorted_experts.bincount(minlength=self.E)   # (E,) on GPU
        ends = counts.cumsum(0)                               # (E,) on GPU
        ends_list: List[int] = ends.tolist()                 # one sync, then pure Python
        starts_list: List[int] = [0] + ends_list[:-1]

        # Track load for bias update (stays on GPU)
        actual_load = counts.float()                         # (E,)
        with torch.no_grad():
            self.last_expert_load.copy_(actual_load.detach())

        # ── Expert GEMMs ──────────────────────────────────────────────────────
        # Build per-expert weighted outputs as a list, then torch.cat in sorted
        # order.  Tokens are already sorted by expert index, so the concatenation
        # of per-expert results equals the full sorted output tensor.
        # This replaces the previous in-place output_sorted[s:t_] = ... pattern,
        # which created backward-graph ambiguity under ZeRO-3 when expert token
        # counts differ across ranks (zero-size slice vs non-zero gradient).
        if os.environ.get("KAIROS_DEBUG_MOE", "0") == "1" \
                and int(os.environ.get("RANK", "0")) == 0:
            _n_active = sum(1 for s_, t__ in zip(starts_list, ends_list) if t__ > s_)
            print(
                f"[moe_debug] N={N} top_k={self.top_k} "
                f"active_experts={_n_active}/{self.E} "
                f"min_count={int(counts.min().item())} "
                f"max_count={int(counts.max().item())} "
                f"zero_experts={int((counts == 0).sum().item())}",
                flush=True,
            )

        expert_pieces: List[torch.Tensor] = []
        for e in range(self.E):
            s, t_ = starts_list[e], ends_list[e]
            if s >= t_:
                continue

            tok_e = sorted_x[s:t_]                  # (N_e, d)
            gate_e = sorted_gates[s:t_]              # (N_e,)

            # SwiGLU(x) = SiLU(x@W1) ⊙ (x@W3) @ W2
            expert_out = (
                F.silu(tok_e @ self.W1[e]) * (tok_e @ self.W3[e])
            ) @ self.W2[e]                           # (N_e, d)

            expert_pieces.append(expert_out * gate_e.unsqueeze(-1))

        # cat of per-expert pieces equals output_sorted (same sorted order)
        if expert_pieces:
            output_sorted = torch.cat(expert_pieces, dim=0)   # (N·K, d)
        else:
            output_sorted = sorted_x.new_zeros(N * self.top_k, d)

        # ── Scatter-accumulate back to original token positions ───────────────
        output = torch.zeros_like(x_flat)
        _safe_scatter_add_(
            output, 0,
            sorted_tokens.unsqueeze(-1).expand_as(output_sorted),
            output_sorted,
            name="moe_scatter",
        )

        # Store N for bias update called by KairosHybridCore OUTSIDE checkpoint.
        # The in-place update must not live inside the checkpointed _run_block
        # function: if it did, expert_bias would differ between original forward
        # and recompute, producing different-sized dispatch slices → CheckpointError.
        if self.training:
            self._last_forward_N = N

        return residual + output.view(B, T, d)

    def _update_bias_from_last_load(self) -> None:
        """Aux-free bias step — called by KairosHybridCore after _grad_ckpt returns."""
        if not self.training:
            return
        N = getattr(self, '_last_forward_N', 0)
        if N == 0:
            return
        target    = float(N * self.top_k) / self.E
        load_diff = target - self.last_expert_load.float()
        self.expert_bias.data.add_(self.γ * torch.sign(load_diff))


# ──────────────────────────────────────────────────────────────────────────────
# Per-loop parameters — NOT shared across loop iterations
# ──────────────────────────────────────────────────────────────────────────────

class PerLoopParams(nn.Module):
    """Four RMSNorm layers + one router-bias vector (length = num_experts) per loop."""

    def __init__(self, cfg: KairosConfig) -> None:
        super().__init__()
        self.mamba_norm = RMSNorm(cfg.d_model)
        self.cfc_norm   = RMSNorm(cfg.d_model)
        self.attn_norm  = RMSNorm(cfg.d_model)
        self.moe_norm   = RMSNorm(cfg.d_model)
        self.router_bias = nn.Parameter(torch.zeros(cfg.num_experts))


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive early-exit gate (richer v2 version)
# ──────────────────────────────────────────────────────────────────────────────

class ExitGate(nn.Module):
    """
    Predicts scene complexity (easy/hard) from mean + max + last-token features.

    v2: 3× richer input feature (mean ‖ max ‖ last) + 2-layer MLP.
    Initialised conservatively so all loops run at the start of training.
    Used at inference only; training runs all loops for clean gradients.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * d, d // 4, bias=True),
            nn.SiLU(),
            nn.Linear(d // 4, 1, bias=True),
        )
        # Start conservative: near-zero logits → sigmoid ≈ 0.5 (don't exit)
        nn.init.zeros_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) → confidence (B, 1)."""
        feat = torch.cat([x.mean(1), x.max(1).values, x[:, -1]], dim=-1)  # (B, 3d)
        return torch.sigmoid(self.mlp(feat))


# ──────────────────────────────────────────────────────────────────────────────
# KairosHybridBlock — shared weights for one block (norms + router_bias injected)
# ──────────────────────────────────────────────────────────────────────────────

class KairosHybridBlock(nn.Module):
    """
    One hybrid block: Mamba-2, CfC, SWA, MoE-FFN.
    Core weights are shared when this block is reused across loops.
    Per-loop norms and router bias are injected at call time via `loop_params`.
    """

    def __init__(self, cfg: KairosConfig) -> None:
        super().__init__()
        self.mamba2  = Mamba2Block(cfg)
        self.cfc     = CfCBlock(cfg)
        self.swa     = SlidingWindowAttention(cfg)
        self.moe_ffn = MoeSwiGLUFFN(cfg)
        # Bypass flags — set at construction from cfg; identity pass when True.
        self._bypass_mamba  = cfg.core_debug_bypass_mamba
        self._bypass_cfc    = cfg.core_debug_bypass_cfc
        self._bypass_swa    = cfg.core_debug_bypass_swa
        self._bypass_moe    = cfg.core_debug_bypass_moe
        self._bypass_logged = False   # one-shot rank-0 log per block instance

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        h_mamba: Optional[torch.Tensor],
        h_cfc: Optional[torch.Tensor],
        imu_mask: torch.Tensor,
        delta_t: torch.Tensor,
        loop_params: PerLoopParams,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _rank0 = int(os.environ.get("RANK", "0")) == 0
        if not self._bypass_logged and _rank0 and any([
            self._bypass_mamba, self._bypass_cfc, self._bypass_swa, self._bypass_moe,
        ]):
            bypassed = [n for n, f in [
                ("mamba", self._bypass_mamba), ("cfc", self._bypass_cfc),
                ("swa",   self._bypass_swa),   ("moe", self._bypass_moe),
            ] if f]
            print(f"[core_debug] HybridBlock bypassed sub-blocks: {bypassed}", flush=True)
            self._bypass_logged = True

        if self._bypass_mamba:
            pass   # identity: x unchanged, h_mamba stays zero (initialised by core)
        else:
            x, h_mamba = self.mamba2(x, h_mamba, loop_params.mamba_norm)

        if self._bypass_cfc:
            pass   # identity: x unchanged, h_cfc stays zero (initialised by core)
        else:
            x, h_cfc   = self.cfc(x, imu_mask, delta_t, h_cfc, loop_params.cfc_norm)

        if not self._bypass_swa:
            x           = self.swa(x, loop_params.attn_norm)

        if self._bypass_moe:
            # Reset stale tracking so the post-forward bias update and z-loss
            # accumulation in KairosHybridCore see a clean slate for this block.
            self.moe_ffn._z_loss_val = None
            self.moe_ffn._last_forward_N = 0
        else:
            x           = self.moe_ffn(x, loop_params.moe_norm, loop_params.router_bias)

        return x, h_mamba, h_cfc


# ──────────────────────────────────────────────────────────────────────────────
# KairosHybridCore — 3 unique blocks × 4 loops with adaptive early exit
# ──────────────────────────────────────────────────────────────────────────────

class KairosHybridCore(nn.Module):
    """
    Applies 3 unique KairosHybridBlocks in a 4-loop schedule.

    Loop → block map:  [0, 1, 2, 0]
      loop 0 uses block 0  (first pass)
      loop 1 uses block 1
      loop 2 uses block 2
      loop 3 uses block 0  (weight reuse — cross-loop state flows via h_mamba)

    Per-loop (not shared): PerLoopParams — 4 RMSNorms + router bias per iteration.
    Per-block (shared):    Mamba-2, CfC, SWA, MoE expert weight matrices.

    Adaptive early exit:
      Training: all loops run (clean gradient flow).
      Inference: stop when every sample's ExitGate confidence ≥ threshold.

    MoE z-loss:
      Aggregated from all blocks/loops into self._z_loss_for_backward (live tensor).
      Training script: loss = task_loss + cfg.z_loss_coeff * core._z_loss_for_backward
    """

    LOOP_BLOCK_MAP: List[int] = [0, 1, 2, 0]

    def __init__(self, cfg: KairosConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.blocks      = nn.ModuleList([KairosHybridBlock(cfg) for _ in range(cfg.num_blocks)])
        self.loop_params = nn.ModuleList([PerLoopParams(cfg) for _ in range(cfg.num_loops)])
        self.exit_gates  = nn.ModuleList([ExitGate(cfg.d_model) for _ in range(cfg.num_loops)])

        self.loop_drop = (
            nn.Dropout(cfg.loop_dropout) if cfg.loop_dropout > 0.0 else nn.Identity()
        )

        # _z_loss_for_backward: live tensor (stays in graph) populated each training
        # forward.  Training script usage:
        #     loss = task_loss + cfg.z_loss_coeff * core._z_loss_for_backward
        #     loss.backward()
        # Initialised to None; never accessed in eval mode.
        self._z_loss_for_backward: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,           # (B, T, d)
        imu_mask: torch.Tensor,    # (B, T) bool
        delta_t: torch.Tensor,     # (B, T) float — Δt per position (0 for non-IMU)
    ) -> torch.Tensor:
        cfg = self.cfg
        B, T, d = x.shape
        if imu_mask.shape != (B, T):
            raise RuntimeError(
                f"KairosHybridCore imu_mask shape mismatch: got {tuple(imu_mask.shape)}, "
                f"expected ({B}, {T})"
            )
        if delta_t.shape != (B, T):
            raise RuntimeError(
                f"KairosHybridCore delta_t shape mismatch: got {tuple(delta_t.shape)}, "
                f"expected ({B}, {T})"
            )

        # Guard against old 328-token layout (cam256 + lidar64 + query8, no IMU)
        if T == 328:
            raise RuntimeError(
                "[layout] Old 328-token fused layout detected: [cam256|lidar64|query8]. "
                "IMU tokens are missing — expected 336 [cam256|lidar64|imu8|query8]. "
                "Check that n_imu=8 is consistent across KairosModelConfig, "
                "FusionConfig, and IMUEncoderConfig."
            )

        # One-shot shape log (every-forward with KAIROS_DEBUG_SHAPES=1).
        # The .item() here is a GPU→CPU sync — printing it on every forward adds
        # a hot-loop stall plus one CloudWatch line per train/val step.
        _rank0 = int(os.environ.get("RANK", "0")) == 0
        if _rank0 and (
            not getattr(self, "_core_shape_logged", False)
            or os.environ.get("KAIROS_DEBUG_SHAPES", "0") == "1"
        ):
            self._core_shape_logged = True
            n_imu_in_seq = int(imu_mask[0].long().sum().item())
            print(
                f"[shape/core] x={tuple(x.shape)}  imu_mask={tuple(imu_mask.shape)}  "
                f"delta_t={tuple(delta_t.shape)}  imu_tokens_per_sample={n_imu_in_seq}",
                flush=True,
            )

        # Pre-allocate zero hidden states so checkpoint receives real tensors
        # (avoids None inputs which complicate use_reentrant=False checkpointing).
        # Semantics are identical: Mamba2Block/CfCBlock zero-init when h is None.
        h_mamba: List[torch.Tensor] = [
            x.new_zeros(B, self.blocks[i].mamba2.d_inner, self.blocks[i].mamba2.d_state)
            for i in range(cfg.num_blocks)
        ]
        h_cfc: List[torch.Tensor] = [x.new_zeros(B, d) for _ in range(cfg.num_blocks)]

        # z-loss accumulator: Optional so the first assignment avoids a 0.0 leaf
        z_loss_accum: Optional[torch.Tensor] = None

        # Module-isolation: optionally limit loop iterations for ZeRO-3 debug.
        # core_debug_layers=N runs only the first N iterations of LOOP_BLOCK_MAP.
        _n_debug = cfg.core_debug_layers
        if _n_debug > 0 and _n_debug < len(self.LOOP_BLOCK_MAP):
            _active_loops = list(enumerate(self.LOOP_BLOCK_MAP))[:_n_debug]
            if _rank0:
                print(
                    f"[core_debug] core_debug_layers={_n_debug}: "
                    f"running first {_n_debug} of {len(self.LOOP_BLOCK_MAP)} loop iterations",
                    flush=True,
                )
        else:
            _active_loops = list(enumerate(self.LOOP_BLOCK_MAP))

        for loop_idx, block_idx in _active_loops:
            x = self.loop_drop(x)

            if self.training and cfg.use_grad_checkpoint:
                # Capture loop-local refs to avoid Python closure-over-loop-var bug
                _blk = self.blocks[block_idx]
                _lp  = self.loop_params[loop_idx]

                # Snapshot expert_bias so the RECOMPUTE (during backward) uses
                # the same bias as the original forward.  Without this, the
                # post-forward bias update shifts the bias between forward and
                # recompute, changing which expert each token routes to.
                # Different-sized dispatch slices → CheckpointError.
                # bias_snap is passed as a tensor arg so _grad_ckpt saves and
                # restores it on replay.
                _bias_snap = _blk.moe_ffn.expert_bias.detach().clone()

                def _run_block(x_, hm, hc, bias_snap, imu_mask_, delta_t_,
                              blk=_blk, lp=_lp):
                    # Temporarily use the snapshotted bias for deterministic routing;
                    # restore the live bias (which may have been updated) after.
                    # imu_mask_ and delta_t_ are explicit tensor args so the
                    # checkpoint mechanism saves and replays them correctly during
                    # the backward recompute — no closure-capture ambiguity.
                    _live = blk.moe_ffn.expert_bias.data
                    blk.moe_ffn.expert_bias.data = bias_snap
                    out = blk(x_, hm, hc, imu_mask_, delta_t_, lp)
                    blk.moe_ffn.expert_bias.data = _live
                    return out

                x, h_mamba[block_idx], h_cfc[block_idx] = _grad_ckpt(
                    _run_block,
                    x, h_mamba[block_idx], h_cfc[block_idx], _bias_snap,
                    imu_mask, delta_t,
                    use_reentrant=False,
                )
            else:
                x, h_mamba[block_idx], h_cfc[block_idx] = self.blocks[block_idx](
                    x, h_mamba[block_idx], h_cfc[block_idx],
                    imu_mask, delta_t,
                    self.loop_params[loop_idx],
                )

            # Bias update runs once per forward, outside the checkpointed region.
            self.blocks[block_idx].moe_ffn._update_bias_from_last_load()

            # Accumulate z-loss only from the block that just ran this iteration.
            # Iterating all self.blocks would pick up stale _z_loss_val tensors
            # from blocks not yet called this forward (set during the previous
            # batch's graph — already freed — corrupting the current grad graph).
            if self.training and torch.is_grad_enabled():
                zlv = self.blocks[block_idx].moe_ffn._z_loss_val
                if zlv is not None:
                    z_loss_accum = zlv if z_loss_accum is None else z_loss_accum + zlv

            # Adaptive early exit (inference only, not on final loop)
            if not self.training and loop_idx < cfg.num_loops - 1:
                conf = self.exit_gates[loop_idx](x)   # (B, 1)
                if (conf >= cfg.exit_threshold).all():
                    break

        # Store as a live graph tensor — NO detach() — training script adds to loss.
        if self.training and torch.is_grad_enabled():
            self._z_loss_for_backward = (
                z_loss_accum if z_loss_accum is not None
                else x.new_zeros(1).squeeze()
            )
        elif self.training:
            self._z_loss_for_backward = None

        return x

    # ------------------------------------------------------------------
    def compile_forward(self, mode: str = "reduce-overhead") -> "KairosHybridCore":
        """
        JIT-compile the forward pass with torch.compile.

        Call this from the training script AFTER DDP/ZeRO wrapping:
            core = KairosHybridCore(cfg).to(device)
            core = DDP(core)              # or DeepSpeed init
            core.module.compile_forward() # compile the inner module

        Why not in __init__?  torch.compile() called inside __init__ wraps the
        method before DDP/ZeRO hooks are attached, which breaks gradient hooks.

        mode="reduce-overhead" is best for fixed-shape training loops.
        Use mode="default" if you see recompilation warnings about dynamic shapes.

        Note: do NOT use @torch.jit.script on _selective_scan_chunked or any
        sub-function — TorchScript and torch.compile use different tracing
        backends and combining them prevents Inductor from fusing the loops.
        """
        self.forward = torch.compile(self.forward, mode=mode)  # type: ignore[method-assign]
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def count_params(module: nn.Module) -> str:
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return f"total={total/1e6:.1f}M  trainable={trainable/1e6:.1f}M"


def sync_moe_bias(core: KairosHybridCore) -> None:
    """
    All-reduce expert_bias across ranks after each optimizer step (DDP / ZeRO-3).
    Call from the training loop:
        sync_moe_bias(core)  # after optimizer.step()
    """
    import torch.distributed as dist
    if not dist.is_initialized():
        return
    backend = dist.get_backend()
    for block in core.blocks:
        bias = block.moe_ffn.expert_bias
        if backend == "nccl" and not bias.is_cuda:
            block.moe_ffn.expert_bias.data = bias.data.to(
                torch.device("cuda", torch.cuda.current_device())
            )
            bias = block.moe_ffn.expert_bias
        dist.all_reduce(bias, op=dist.ReduceOp.AVG)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test (shape + dtype only — no GPU required)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    cfg = KairosConfig()

    print("Building KairosHybridCore (v2 — optimized) …")
    core = KairosHybridCore(cfg)
    print(f"  {count_params(core)}")

    B, T, d = 2, 336, cfg.d_model

    x = torch.randn(B, T, d, dtype=torch.bfloat16)

    # IMU tokens at positions 320-327 (last 8 slots after cam[256]+lidar[64])
    imu_mask = torch.zeros(B, T, dtype=torch.bool)
    imu_mask[:, 320:328] = True

    # Δt = 1/30 s (30 Hz IMU); 0 for non-IMU tokens
    delta_t = torch.zeros(B, T)
    delta_t[:, 320:328] = 1.0 / 30.0

    print(f"  input : {tuple(x.shape)}  dtype={x.dtype}")

    # ── inference forward (early exit active) ──────────────────────────────────
    core.eval()
    with torch.no_grad():
        out = core(x, imu_mask, delta_t)
    assert out.shape == (B, T, d), f"shape mismatch: {out.shape}"
    print(f"  output: {tuple(out.shape)}  dtype={out.dtype}")

    # ── training forward — verify z-loss is live (gradient flows to router) ──────
    x_grad = x.float().requires_grad_(True)
    core.train()
    out_tr = core(x_grad.to(torch.bfloat16), imu_mask, delta_t)
    assert out_tr.shape == (B, T, d)

    assert core._z_loss_for_backward is not None, "_z_loss_for_backward not set"
    assert core._z_loss_for_backward.requires_grad, \
        "BUG: z-loss is detached — router_proj will not receive gradients"

    # Simulate the training-script loss composition
    fake_task_loss = out_tr.float().mean()
    total_loss = fake_task_loss + cfg.z_loss_coeff * core._z_loss_for_backward
    total_loss.backward()

    router_grad = core.blocks[0].moe_ffn.router_proj.weight.grad
    assert router_grad is not None, "router_proj.weight has no gradient"
    assert router_grad.abs().sum() > 0, "router_proj gradient is all-zero"
    print(f"  z-loss: {core._z_loss_for_backward.item():.4f}  "
          f"router grad norm: {router_grad.norm().item():.4e}")

    print("Smoke-test passed.")
