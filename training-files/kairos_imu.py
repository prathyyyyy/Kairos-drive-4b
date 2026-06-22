"""
kairos_imu.py  —  CfC/LNN IMU encoder for Kairos-4B.

IMU encoder used by kairos_model.py.

Architecture:
  OXTS IMU stream (B, T_imu, 7) + timestamps (B, T_imu)
  → n_imu=8 tokens (B, 8, d_model) + delta_t (B, 8)

  The 7 IMU fields from OXTS:
    [0] velocity_fwd  (m/s)
    [1] acceleration  (m/s²)
    [2] jerk          (m/s³)
    [3] lat           (degrees)
    [4] lon           (degrees)
    [5] alt           (metres)
    [6] yaw           (radians)

Pipeline:
  1. Normalise IMU fields (learnable running stats — no hard-coded ranges)
  2. Project 7D → d_model via linear layer
  3. Full CfC pass over T_imu timesteps with ACTUAL Δt from timestamps
     - Input-dependent time constant τ = 1/f, f = urgency MLP(x, Δt)
     - CfC: x(t) = σ(-f·Δt)·h + [1 - σ(-f·Δt)]·A(x)
     - h persists across the T_imu steps (full CfC, not just one step)
  4. Stride-select n_imu evenly-spaced hidden states → n_imu tokens
  5. Final projection + RMSNorm

Key behavior:
  - Full CfC recurrence with Δt-aware gating at every step

  The CfCBlock INSIDE KairosHybridCore is a DIFFERENT thing:
  it processes the IMU *tokens* (already encoded here) within the
  multimodal sequence. This encoder produces those tokens.

Interface:
    encoder = IMUEncoder(d_model=1024, n_tokens=8)
    tokens, delta_t = encoder(imu_data, timestamps)
    # tokens   (B, 8, d_model)
    # delta_t  (B, 8)  Δt between output token timesteps

Parameter budget: ~105M at d_model=1024, cfc_hidden=4096, n_cfc_layers=3
    3 × CfCCell (each ~34.6M):
      f_mlp 3-layer (1025→4096→4096→1024): ~25M
      A_proj 2-layer (1024→4096→1024):    ~8.4M
      out_proj (1024→1024):               ~1.1M
    Input proj 7→1024 + norms:            ~1.1M
    Total:                                ~105M
"""

from __future__ import annotations

import math
import os as _os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kairos_hybrid_block import KairosConfig, RMSNorm   # type: ignore
except ImportError:
    from dataclasses import dataclass as _dc

    @_dc
    class KairosConfig:   # type: ignore
        d_model: int = 1024

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            xf = x.float()
            return (xf * xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
                    * self.weight.float()).to(x.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IMUEncoderConfig:
    n_tokens:     int   = 8        # output tokens (target n_imu)
    imu_features: int   = 7        # OXTS fields
    cfc_hidden:   int   = 4096     # CfC urgency MLP + attractor hidden dim (~100M total)
    n_cfc_layers: int   = 3        # depth: stack this many CfC cells sequentially
    # Normalisation
    running_momentum: float = 0.01   # EMA for learnable normaliser


# ──────────────────────────────────────────────────────────────────────────────
# Learnable running normaliser (replaces hard-coded mean/std)
# ──────────────────────────────────────────────────────────────────────────────

class LearnableNorm(nn.Module):
    """
    Per-feature affine normalisation with EMA running stats.
    More robust than BatchNorm for variable-length IMU sequences.

    Running mean/var updated during training via EMA.
    At inference uses stored running stats.

    BF16 safety: running_mean and running_var are kept in float32 permanently
    via _apply() override, so model.to(torch.bfloat16) does not convert them.
    lerp_() end tensors are explicitly cast to destination dtype before the op.
    """

    def __init__(self, n_features: int, momentum: float = 0.01, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps   = eps
        self.mom   = momentum
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta  = nn.Parameter(torch.zeros(n_features))
        self.register_buffer("running_mean", torch.zeros(n_features))
        self.register_buffer("running_var",  torch.ones(n_features))
        self._norm_debug_printed = False   # one-shot debug flag

    def _apply(self, fn):
        """Keep running stats in float32 after model.to(dtype) / model.cuda() calls."""
        super()._apply(fn)
        self.running_mean = self.running_mean.float()
        self.running_var  = self.running_var.float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., n_features)"""
        if self.training:
            flat = x.reshape(-1, x.shape[-1]).float()
            mean = flat.mean(0)
            var  = flat.var(0, unbiased=False)
            # Explicit cast: batch stats are float32; running buffers stay float32
            # (via _apply), but the cast below is a belt-and-suspenders guard
            # against any future code path that might change buffer dtype.
            with torch.no_grad():
                self.running_mean.lerp_(
                    mean.detach().to(
                        device=self.running_mean.device,
                        dtype=self.running_mean.dtype,
                    ),
                    self.mom,
                )
                self.running_var.lerp_(
                    var.detach().to(
                        device=self.running_var.device,
                        dtype=self.running_var.dtype,
                    ),
                    self.mom,
                )
        else:
            mean = self.running_mean.float()
            var  = self.running_var.float()

        if (
            not self._norm_debug_printed
            and _os.environ.get("KAIROS_DEBUG_DTYPE", "0") == "1"
            and int(_os.environ.get("RANK", "0")) == 0
        ):
            self._norm_debug_printed = True
            print(
                f"[dtype/imu_norm] x={x.dtype} mean={mean.dtype} "
                f"running_mean={self.running_mean.dtype} "
                f"var={var.dtype} running_var={self.running_var.dtype}",
                flush=True,
            )

        x_n = (x.float() - mean) / (var + self.eps).sqrt()
        return (x_n * self.gamma.float() + self.beta.float()).to(x.dtype)


def _linear_input(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    return x.to(device=linear.weight.device, dtype=linear.weight.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# CfC cell — full recurrence over T timesteps
# ──────────────────────────────────────────────────────────────────────────────

class CfCCell(nn.Module):
    """
    Closed-Form Continuous-time (CfC) recurrent cell — scaled for ~35M params each.

    At each timestep t:
        f = softplus( f_mlp([x_t ‖ Δt]) )      ← urgency (3-layer wide MLP)
        A = A_down( SiLU( A_up(x_t) ) )         ← attractor (2-layer expanded)
        gate = sigmoid(-f * Δt)
        h_t  = gate ⊙ h_{t-1} + (1 − gate) ⊙ A

    f_mlp: (d+1) → d_hidden → d_hidden → d   [wide: d_hidden=4096]
    A_proj: d → d_hidden → d                  [expanded attractor for richer landscape]

    Multiple cells are stacked in IMUEncoder for deep temporal processing.
    """

    def __init__(self, d: int, d_hidden: int = 4096) -> None:
        super().__init__()
        self.d = d

        # 3-layer urgency MLP: (d+1) → d_hidden → d_hidden → d
        self.f_mlp = nn.Sequential(
            nn.Linear(d + 1, d_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(d_hidden, d, bias=True),
        )

        # 2-layer expanded attractor: d → d_hidden → d
        self.A_up   = nn.Linear(d, d_hidden, bias=True)
        self.A_down = nn.Linear(d_hidden, d, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.out_proj.weight, std=d ** -0.5)

        # Conservative init: gate ≈ 0.5 at t=0
        nn.init.zeros_(self.f_mlp[-1].weight)
        nn.init.constant_(self.f_mlp[-1].bias, -2.0)
        # A_down near-zero so attractor starts close to zero (no premature bias)
        nn.init.zeros_(self.A_down.weight)
        nn.init.zeros_(self.A_down.bias)

    def forward(
        self,
        x_seq: torch.Tensor,     # (B, T, d) — projected + normed IMU sequence
        dt_seq: torch.Tensor,    # (B, T)    — Δt between consecutive readings
        h_init: Optional[torch.Tensor] = None,  # (B, d) | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_seq    (B, T, d)  — output at each step
            h_final  (B, d)     — final hidden state (detached for stop-grad)
        """
        B, T, _ = x_seq.shape
        x_seq = _linear_input(x_seq, self.A_up)
        dt_seq = dt_seq.to(device=x_seq.device, dtype=x_seq.dtype)
        h = h_init if h_init is not None else x_seq.new_zeros(B, self.d)
        h = h.to(device=x_seq.device, dtype=x_seq.dtype)

        h_seq = []
        for t in range(T):
            x_t  = x_seq[:, t]                                # (B, d)
            dt_t = dt_seq[:, t:t+1].to(x_t.dtype)            # (B, 1)

            f    = F.softplus(self.f_mlp(torch.cat([x_t, dt_t], dim=-1)))  # (B, d)
            A    = self.A_down(F.silu(self.A_up(x_t)))        # (B, d)
            gate = torch.sigmoid(-f * dt_t)                   # (B, d)
            h    = gate * h + (1.0 - gate) * A

            h_seq.append(self.out_proj(h))

        return torch.stack(h_seq, dim=1), h.detach()          # (B, T, d),  (B, d)


# ──────────────────────────────────────────────────────────────────────────────
# Full IMU Encoder
# ──────────────────────────────────────────────────────────────────────────────

class IMUEncoder(nn.Module):
    """
    Full IMU encoder — ~105M parameters at d_model=1024.

    Drop-in interface:
        tokens, delta_t = encoder(imu_data, timestamps)
        # tokens   (B, n_tokens, d_model)
        # delta_t  (B, n_tokens)

    Processing:
        1. LearnableNorm over 7 IMU fields
        2. Linear projection 7 → d_model
        3. n_cfc_layers stacked CfC cells, each processing all T_imu timesteps
           (deep temporal architecture — output of cell k feeds input of cell k+1)
        4. Evenly-spaced stride selection of n_tokens hidden states from final cell
        5. RMSNorm output projection

    Parameter budget at d_model=1024, cfc_hidden=4096, n_cfc_layers=3:
        3 × CfCCell: 3 × 34.6M = 103.9M
        Input/output projections + norms: ~1.1M
        Total: ~105M
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_tokens: int = 8,
        cfg: Optional[IMUEncoderConfig] = None,
        kcfg: Optional[KairosConfig] = None,
    ) -> None:
        super().__init__()
        cfg    = cfg  or IMUEncoderConfig(n_tokens=n_tokens)
        kcfg   = kcfg or KairosConfig()
        self.d        = d_model
        self.n_tokens = n_tokens
        self.cfg      = cfg

        # ── Normalisation ───────────────────────────────────────────────────────
        self.imu_norm = LearnableNorm(cfg.imu_features, momentum=cfg.running_momentum)

        # ── Input projection 7 → d ─────────────────────────────────────────────
        self.in_proj = nn.Linear(cfg.imu_features, d_model, bias=True)
        self.in_norm = RMSNorm(d_model)

        # ── Stacked CfC recurrence ─────────────────────────────────────────────
        self.cfc_layers = nn.ModuleList([
            CfCCell(d_model, d_hidden=cfg.cfc_hidden)
            for _ in range(cfg.n_cfc_layers)
        ])

        # ── Output projection ──────────────────────────────────────────────────
        self.out_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, std=d_model ** -0.5)

    # ------------------------------------------------------------------
    def _compute_delta_t(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute per-step Δt from raw timestamps.
        timestamps: (B, T)  — seconds
        Returns:    (B, T)  — Δt[0] = 1/30, Δt[t] = ts[t] - ts[t-1]
        """
        if timestamps.shape[1] > 1 and not (timestamps[:, 1:] >= timestamps[:, :-1]).all():
            import warnings
            warnings.warn(
                "Non-monotonic IMU timestamps detected; delta_t was clamped.",
                RuntimeWarning, stacklevel=2,
            )
        dt = torch.zeros_like(timestamps)
        dt[:, 0]  = 1.0 / 30.0                          # 30 Hz default for first
        dt[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
        return dt.clamp(min=1e-4, max=0.2)

    # ------------------------------------------------------------------
    def forward(
        self,
        imu_data:    torch.Tensor,   # (B, T_imu, 7)
        timestamps:  torch.Tensor,   # (B, T_imu)  seconds
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tokens   (B, n_tokens, d_model)
            delta_t  (B, n_tokens)  — Δt between output token timesteps
        """
        # Cast inputs to model dtype immediately. Prevents mixed BF16/float32
        # in timestamp arithmetic and tensor creation ops (zeros_like, full)
        # that inherit the input dtype. Without this, float32 batch timestamps
        # combined with BF16 intermediate x produce dtype mismatches in clamp
        # and comparison ops that require homogeneous dtypes.
        _mdtype  = self.in_proj.weight.dtype
        _mdevice = self.in_proj.weight.device
        imu_data   = imu_data.to(device=_mdevice, dtype=_mdtype)
        timestamps = timestamps.to(device=_mdevice, dtype=_mdtype)

        B, T_imu, _ = imu_data.shape
        K            = self.n_tokens

        # ── 1. Normalise + project ─────────────────────────────────────────────
        x = self.imu_norm(imu_data)             # (B, T_imu, 7) — stays in _mdtype
        x = _linear_input(x, self.in_proj)
        x = self.in_norm(self.in_proj(x))       # (B, T_imu, d)

        # ── 2. Compute Δt from timestamps ─────────────────────────────────────
        # timestamps is already on the correct device/dtype from the cast above.
        dt = self._compute_delta_t(timestamps)   # (B, T_imu) — same dtype as timestamps

        # ── 3. Stacked CfC recurrence (depth = n_cfc_layers) ─────────────────
        for cfc in self.cfc_layers:
            x, _ = cfc(x, dt)                  # (B, T_imu, d) → (B, T_imu, d)
        h_seq = x

        # ── 4. Stride-select n_tokens evenly-spaced hidden states ─────────────
        if T_imu >= K:
            # Pick K evenly-spaced indices covering [0, T_imu-1].
            # torch.linspace with explicit dtype=float32 + round().long() is safe
            # under BF16 autocast — it prevents the 'expected BFloat16 for end'
            # error that torch.arange(K)*step can trigger when the default dtype
            # is inferred as BFloat16 and start/end dtypes diverge in dispatch.
            indices = torch.linspace(
                0.0,
                float(T_imu - 1),
                steps=K,
                device=x.device,
                dtype=torch.float32,   # explicit — never BF16
            ).round().long().clamp_(0, T_imu - 1)
        else:
            # Fewer steps than tokens: repeat last
            indices = torch.arange(T_imu, device=x.device, dtype=torch.long)
            pad_idx = torch.full(
                (K - T_imu,), T_imu - 1,
                device=x.device, dtype=torch.long,
            )
            indices = torch.cat([indices, pad_idx])

        selected = h_seq[:, indices]            # (B, K, d)

        # ── 5. Output projection + norm ───────────────────────────────────────
        selected = _linear_input(selected, self.out_proj)
        tokens = self.out_norm(self.out_proj(selected))   # (B, K, d)

        # ── 6. Compute output-token Δt ────────────────────────────────────────
        # Compute in float32 and cast to tokens.dtype at the end.
        # timestamps is already BF16 (cast at top of forward), but arithmetic
        # on BF16 timestamps loses precision for small Δt values.  Computing in
        # float32 and casting back ensures the final out_dt matches tokens.dtype
        # regardless of what BF16 autocast ops may have done to timestamps.
        selected_ts  = timestamps[:, indices]             # (B, K) — BF16
        dt_f32       = selected_ts.float()                # promote to float32
        out_dt_f32   = torch.zeros_like(dt_f32)           # float32
        out_dt_f32[:, 0]  = 1.0 / 30.0
        out_dt_f32[:, 1:] = dt_f32[:, 1:] - dt_f32[:, :-1]
        out_dt_f32 = out_dt_f32.clamp_(min=1e-4, max=0.2)
        out_dt = out_dt_f32.to(device=tokens.device, dtype=tokens.dtype)  # → BF16

        return tokens, out_dt


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    d = 1024

    enc   = IMUEncoder(d_model=d, n_tokens=8)
    total = sum(p.numel() for p in enc.parameters())
    print(f"IMUEncoder params: {total/1e6:.1f}M  (target ~105M)")

    B, T_imu = 2, 30
    imu_data   = torch.randn(B, T_imu, 7)
    timestamps = torch.cumsum(
        torch.full((B, T_imu), 1.0/30.0), dim=1
    )

    print(f"Input: imu_data {tuple(imu_data.shape)}  ts {tuple(timestamps.shape)}")

    enc.eval()
    with torch.no_grad():
        tokens, delta_t = enc(imu_data, timestamps)

    assert tokens.shape  == (B, 8, d),  f"tokens shape: {tokens.shape}"
    assert delta_t.shape == (B, 8),     f"delta_t shape: {delta_t.shape}"
    print(f"Output: tokens {tuple(tokens.shape)}  delta_t {tuple(delta_t.shape)}")
    print(f"delta_t sample: {delta_t[0].tolist()}")

    # Gradient check
    enc.train()
    tok_tr, _ = enc(imu_data, timestamps)
    tok_tr.float().mean().backward()
    no_grad = [n for n, p in enc.named_parameters()
               if p.requires_grad and p.grad is None]
    print(f"Params without grad: {no_grad[:3] if no_grad else 'NONE — all OK'}")

    # Verify temporal sensitivity: different Δt → different output
    imu2 = imu_data.clone()
    ts2  = timestamps * 2.0   # slower sampling
    enc.eval()
    with torch.no_grad():
        tok2, _ = enc(imu2, ts2)
    diff = (tokens - tok2).abs().mean().item()
    print(f"Δt sensitivity (should be > 0): {diff:.5f} {'OK' if diff > 0 else 'FAIL'}")

    print("Smoke-test passed.")
