"""
kairos_lidar.py  —  PointMamba LiDAR encoder for Kairos-4B.

LiDAR encoder used by kairos_model.py.

Architecture:
  Two velodyne frames (t, t-1) → fused into 64 LiDAR tokens + 64 centroids.

  1. Pre-processing
       Concatenate both frames → (B, 2N, 4)
       Remove ground plane (z < -1.5 m) and behind-vehicle points (x < 0)
       Random subsample to MAX_PTS=16384 if needed

  2. Farthest Point Sampling (FPS)
       Select 64 seed points maximally spread in 3D space
       Each seed defines one LiDAR token cluster

  3. Ball query grouping
       For each seed: gather K=32 nearest points within radius r=2.0 m
       Normalize relative to seed centroid → local geometry

  4. PointNet feature extraction per cluster
       Shared MLP: (3+1) → 64 → 128 → d_model per point
       Max-pool over K points → (B, 64, d_model) local feature

  5. Temporal motion features
       Compute nearest-neighbour flow: for each t cluster,
       find closest t-1 cluster → append displacement (Δx,Δy,Δz)
       Project (d_model + 3) → d_model

  6. Mamba SSM over Morton-ordered clusters
       Sort 64 clusters by Morton code (Z-order in 3D)
       Apply one Mamba-2 SSM layer for long-range context

  Output:
    tokens    (B, 64, d_model)  — LiDAR token embeddings
    centroids (B, 64, 3)        — cluster XYZ in velodyne frame (for calib gate)

Interface:
    encoder = PointMambaEncoder(d_model=1024, n_tokens=64)
    tokens, centroids = encoder(pts_t, pts_t1)

Parameter budget: ~101M at d_model=1024
    PointNet cluster MLP (4→128→256→1024) + norm: ~0.3M
    Temporal motion projector (1024+3→1024) + norm: ~1.1M
    8 × _LiDARMambaLayer (d_inner=2048, d_state=64): ~54.6M
    LiDAR MoE FFN (8 experts × 3 × 1024 × 1820): ~44.7M
    Misc norms: ~0.3M
"""

from __future__ import annotations

import math
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

try:
    from torch_cluster import fps as _tc_fps   # type: ignore
    _HAS_TC = True
except ImportError:
    _HAS_TC = False


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def _linear_input(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    return x.to(device=linear.weight.device, dtype=linear.weight.dtype)


import os as _os


def _safe_assign_(dst: "torch.Tensor", index: object, src: "torch.Tensor", *, name: str = "") -> None:
    """Index/slice assignment with automatic dtype/device coercion of src to match dst."""
    if _os.environ.get("KAIROS_DEBUG_DTYPE", "0") == "1" \
            and int(_os.environ.get("RANK", "0")) == 0 \
            and (dst.dtype != src.dtype or dst.device != src.device):
        print(
            f"[dtype/assign] {name}: "
            f"dst={dst.dtype}/{dst.device} src={src.dtype}/{src.device} "
            f"— casting src",
            flush=True,
        )
    dst[index] = src.to(device=dst.device, dtype=dst.dtype)


def _safe_linspace_indices(
    start: float,
    end: float,
    steps: int,
    *,
    device: torch.device,
    max_index: int,
    name: str = "",
) -> torch.Tensor:
    """
    Create evenly-spaced long index tensor in [0, max_index] — immune to BF16 autocast.

    Under DeepSpeed BF16 / torch.autocast(dtype=bfloat16) the default floating-point
    dtype can be inferred as BFloat16 for some PyTorch dispatch paths.  If torch.linspace
    receives start as BFloat16 (inferred) and end as float32, it raises:
        RuntimeError: expected dtype c10::BFloat16 for 'end' but got dtype float

    Wrapping both endpoints with float() forces Python-native scalars, and the explicit
    dtype=torch.float32 override prevents any autocast promotion of the result before
    we convert to long via .round().long().
    """
    if int(_os.environ.get("RANK", "0")) == 0 \
            and _os.environ.get("KAIROS_DEBUG_DTYPE", "0") == "1":
        print(
            f"[dtype/linspace] {name}: start={start} end={end} "
            f"steps={steps} device={device}",
            flush=True,
        )
    idx = torch.linspace(
        float(start),
        float(end),
        steps=int(steps),
        device=device,
        dtype=torch.float32,   # explicit — prevents BF16 autocast promotion
    ).round().long()
    return idx.clamp_(0, int(max_index))


@dataclass
class LiDAREncoderConfig:
    n_tokens:     int   = 64       # number of LiDAR clusters / output tokens
    n_points:     int   = 16_384   # points sampled per frame pair
    n_neighbors:  int   = 32       # ball-query neighbors per cluster
    ball_radius:  float = 2.0      # metres
    pn_hidden:    int   = 256      # PointNet hidden dim (wider for richer features)
    # Ground/behind filter
    z_min:        float = -1.5     # drop points below this height
    x_min:        float = 0.0      # drop points behind vehicle
    # Mamba SSM stack
    d_state:      int   = 64
    mamba_chunk:  int   = 16       # chunkwise scan chunk for 64 tokens
    n_mamba_layers: int = 8        # depth: stack this many Mamba layers (~55M)
    # MoE FFN after Mamba stack
    moe_experts:  int   = 8        # routing experts in the LiDAR-specific MoE
    moe_d_ff:     int   = 1820     # expert FFN width (8×3×1024×1820 ≈ 44.7M)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry utilities
# ──────────────────────────────────────────────────────────────────────────────

def _farthest_point_sample(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Farthest point sampling.

    Uses torch_cluster.fps (GPU-native, ~10× faster) when available;
    falls back to the iterative O(n_samples × N) loop otherwise.

    Args:
        xyz       (B, N, 3)
        n_samples int
    Returns:
        idx       (B, n_samples) — indices into dim-1 of xyz
    """
    B, N, _ = xyz.shape
    device   = xyz.device
    if N == 0:
        raise ValueError("_farthest_point_sample requires at least one point")

    if _HAS_TC and N >= n_samples:
        # torch_cluster.fps expects (B*N, 3) + per-point batch index
        xyz_flat  = xyz.reshape(B * N, 3)
        batch_vec = torch.arange(B, device=device).repeat_interleave(N)
        ratio     = n_samples / N          # ceil(ratio*N) = n_samples when N%n_samples==0
        sel       = _tc_fps(xyz_flat, batch_vec, ratio=ratio, random_start=True)
        # sel: (B*n_samples,) global indices into xyz_flat, sorted by batch
        return (sel % N).reshape(B, n_samples)

    # Iterative fallback — also handles N < n_samples via repeat.
    # Compute distances in float32 explicitly:
    #   • pow(2) and sum() are in PyTorch's autocast float32-preserve list, so they
    #     return float32 when the model runs under torch.autocast(dtype=bfloat16).
    #   • torch.minimum(bfloat16, float32) raises a dtype mismatch error.
    #   • Explicitly keeping dist in float32 and using xyz.float() avoids the mismatch
    #     under both autocast and non-autocast contexts.
    dist    = torch.full((B, N), float("inf"), device=device, dtype=torch.float32)
    idx     = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    current = torch.randint(0, N, (B,), device=device)

    for i in range(n_samples):
        idx[:, i] = current
        cur_xyz  = xyz[torch.arange(B, device=device), current].unsqueeze(1)
        new_dist = (xyz.float() - cur_xyz.float()).pow(2).sum(-1)  # float32
        dist     = torch.minimum(dist, new_dist)                    # float32 ✓
        current  = dist.argmax(dim=1)

    return idx   # (B, n_samples)


def _ball_query(
    xyz_query: torch.Tensor,   # (B, K, 3) seed points
    xyz_all:   torch.Tensor,   # (B, N, 3) all points
    radius:    float,
    n_sample:  int,
    _chunk:    int = 16,       # clusters per memory chunk; tune to VRAM budget
) -> torch.Tensor:
    """
    For each query point, find up to n_sample neighbors within radius.
    If fewer than n_sample found, repeat the closest one.

    Memory-efficient vs. naive: processes _chunk clusters at a time so peak
    allocation is (B, _chunk, N, 3) instead of (B, K, N, 3).
    Uses topk (O(N)) instead of full argsort (O(N log N)) for ranking.

    Returns:
        idx  (B, K, n_sample)  indices into xyz_all
    """
    B, K, _ = xyz_query.shape
    device  = xyz_query.device
    N = xyz_all.shape[1]
    if N == 0:
        raise ValueError("_ball_query requires at least one point")
    idx_map: Optional[torch.Tensor] = None
    if N > 20_000:
        # Downsample xyz_all to 16 384 points for memory efficiency.
        # _safe_linspace_indices wraps endpoints in float() and uses explicit
        # dtype=float32, preventing the BF16 autocast promotion that causes:
        #   RuntimeError: expected dtype c10::BFloat16 for 'end' but got dtype float
        idx_map = _safe_linspace_indices(
            0, N - 1, 16_384,
            device=device, max_index=N - 1,
            name="ball_query_downsample",
        )
        xyz_all = xyz_all.index_select(1, idx_map)
        N = xyz_all.shape[1]
    topk_n = min(n_sample, N)

    parts = []
    for start in range(0, K, _chunk):
        end  = min(start + _chunk, K)
        ck   = end - start

        # (B, ck, N, 3) — only ck clusters materialised at once
        diff  = xyz_query[:, start:end].unsqueeze(2) - xyz_all.unsqueeze(1)
        dist2 = diff.pow(2).sum(-1)                              # (B, ck, N)

        # topk nearest: O(N), avoids full sort
        top_dists, top_idx = dist2.topk(topk_n, dim=-1, largest=False, sorted=False)
        if idx_map is not None:
            top_idx = idx_map[top_idx]
        in_ball = top_dists <= radius ** 2                       # (B, ck, topk_n)

        # Pad out-of-ball slots with the closest point
        closest = top_idx[:, :, :1].expand(B, ck, topk_n)
        selected = torch.where(in_ball, top_idx, closest)
        if topk_n < n_sample:
            pad = closest[:, :, :1].expand(B, ck, n_sample - topk_n)
            selected = torch.cat([selected, pad], dim=-1)
        parts.append(selected)

    return torch.cat(parts, dim=1)   # (B, K, n_sample)


def _morton_sort_idx(xyz: torch.Tensor) -> torch.Tensor:
    """
    Sort point indices by 3D Morton code (Z-order curve).
    Gives spatial locality for the Mamba SSM scan.

    Args:
        xyz  (B, K, 3)  float — will be quantized to 10-bit integers
    Returns:
        order (B, K)  sorted indices
    """
    B, K, _ = xyz.shape
    # Normalise to [0, 1023]
    mn = xyz.amin(dim=1, keepdim=True)
    mx = xyz.amax(dim=1, keepdim=True)
    rng = (mx - mn).clamp(min=1e-4)
    q = ((xyz - mn) / rng * 1023).long().clamp(0, 1023)  # (B, K, 3)

    # Interleave bits: x→bit0, y→bit1, z→bit2
    # Efficient 10-bit interleave
    def _spread(v: torch.Tensor) -> torch.Tensor:
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v <<  8)) & 0x0300F00F
        v = (v | (v <<  4)) & 0x030C30C3
        v = (v | (v <<  2)) & 0x09249249
        return v

    mx_code = _spread(q[:, :, 0]) | (_spread(q[:, :, 1]) << 1) | (_spread(q[:, :, 2]) << 2)
    return mx_code.argsort(dim=1)   # (B, K)


# ──────────────────────────────────────────────────────────────────────────────
# PointNet per-cluster feature extractor
# ──────────────────────────────────────────────────────────────────────────────

class PointNetCluster(nn.Module):
    """
    Shared-weight MLP applied to each of K neighbors inside a cluster,
    then max-pooled to a single (d_model,) feature vector.

    Input per cluster: (B, K_clusters, n_nbrs, 4)
      4 = [Δx, Δy, Δz, intensity]  (relative to cluster centroid)
    Output: (B, K_clusters, d_model)
    """

    def __init__(self, d_model: int, d_hidden: int = 128) -> None:
        super().__init__()
        # Shared across all points in all clusters
        self.mlp = nn.Sequential(
            nn.Linear(4,        d_hidden,  bias=False),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden * 2, bias=False),
            nn.LayerNorm(d_hidden * 2),
            nn.SiLU(),
            nn.Linear(d_hidden * 2, d_model, bias=False),
        )
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        feats: torch.Tensor,   # (B, K_clusters, n_nbrs, 4)
    ) -> torch.Tensor:
        B, K, n, C = feats.shape
        # Flatten to apply shared MLP: (B*K*n, C)
        f = feats.reshape(B * K * n, C).float()
        f = _linear_input(f, self.mlp[0])

        f = self.mlp(f)                          # (B*K*n, d_model)
        f = f.reshape(B, K, n, -1)               # (B, K, n, d_model)

        # Max-pool over neighbors
        f = f.max(dim=2).values                  # (B, K, d_model)
        return self.norm(f.to(feats.dtype))


# ──────────────────────────────────────────────────────────────────────────────
# Mamba SSM layer (single, lightweight — reuses chunked scan from hybrid block)
# ──────────────────────────────────────────────────────────────────────────────

class _LiDARMambaLayer(nn.Module):
    """
    One Mamba-2 SSM pass over K=64 spatially-ordered LiDAR tokens.
    Provides long-range geometric context across the point cloud.

    Lighter than the full Mamba2Block in hybrid_block.py:
      - No conv1d (K=64 is tiny; conv overhead not worth it)
      - dt_rank=32 (smaller than core's 64; sufficient for 64 tokens)
      - Single layer, no loop
    """

    def __init__(self, d: int, d_state: int = 64, chunk: int = 16) -> None:
        super().__init__()
        di          = d * 2         # d_inner
        self.d      = d
        self.di     = di
        self.d_state = d_state
        self.chunk  = chunk
        dt_rank     = 32

        self.norm    = RMSNorm(d)
        self.in_proj = nn.Linear(d, 2 * di, bias=False)
        self.x_proj  = nn.Linear(di, 2 * d_state + dt_rank, bias=False)
        self.dt_proj = nn.Linear(dt_rank, di, bias=True)

        A_init = torch.arange(1, d_state + 1, dtype=torch.float32) \
                      .unsqueeze(0).expand(di, -1).clone()
        self.A_log  = nn.Parameter(torch.log(A_init))
        self.D      = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d, bias=False)
        nn.init.normal_(self.out_proj.weight, std=d ** -0.5)
        nn.init.constant_(self.dt_proj.bias, math.log(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, d) → (B, K, d)"""
        x = _linear_input(x, self.in_proj)
        residual = x
        x = self.norm(x)
        B, K, _ = x.shape
        di, N = self.di, self.d_state

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        bcd     = self.x_proj(x_ssm)
        B_ssm   = bcd[..., :N]
        C_ssm   = bcd[..., N:2*N]
        dt_raw  = bcd[..., 2*N:]
        delta   = F.softplus(self.dt_proj(dt_raw))

        A      = -torch.exp(self.A_log.float())
        Abar   = torch.exp(delta.unsqueeze(-1).float() * A[None, None]).to(x.dtype)
        Bbar_x = (delta.unsqueeze(-1) * B_ssm.unsqueeze(2)) * x_ssm.unsqueeze(-1)

        h = x.new_zeros(B, di, N)
        ys = []
        for start in range(0, K, self.chunk):
            end    = min(start + self.chunk, K)
            Ab     = Abar[:, start:end]
            Bb     = Bbar_x[:, start:end]
            Cs     = C_ssm[:, start:end]
            log_Q  = Ab.float().clamp(min=1e-38).log().cumsum(1)
            Q      = log_Q.exp().to(Ab.dtype)
            Bb_n   = Bb / Q.clamp(min=torch.finfo(Bb.dtype).tiny)
            h_c    = Q * (h.unsqueeze(1) + Bb_n.cumsum(1))
            ys.append((h_c * Cs.unsqueeze(2)).sum(-1))
            h      = h_c[:, -1]

        y   = torch.cat(ys, 1) + x_ssm * self.D
        y   = y * F.silu(z)
        out = self.out_proj(y)
        return residual + out


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight MoE FFN for LiDAR cluster features
# ──────────────────────────────────────────────────────────────────────────────

class _LiDARMoEFFN(nn.Module):
    """
    Lightweight Mixture-of-Experts SwiGLU FFN for LiDAR token enrichment.

    Simpler than the hybrid-core MoE: top-1 routing, no z-loss, no aux-free bias.
    Top-1 is sufficient for the 64-token LiDAR sequence (each cluster sees the
    most relevant expert without the overhead of top-k scatter-accumulate).

    Parameter budget at n_experts=8, d_ff=1820, d=1024:
        W1 + W2 + W3 = 3 × 8 × 1024 × 1820 ≈ 44.7M
    """

    def __init__(self, d: int, n_experts: int = 8, d_ff: int = 1820) -> None:
        super().__init__()
        self.E = n_experts
        self.W1 = nn.Parameter(torch.empty(n_experts, d, d_ff))
        self.W2 = nn.Parameter(torch.empty(n_experts, d_ff, d))
        self.W3 = nn.Parameter(torch.empty(n_experts, d, d_ff))
        self.router = nn.Linear(d, n_experts, bias=False)
        self.norm   = RMSNorm(d)

        nn.init.kaiming_uniform_(self.W1.reshape(n_experts * d, d_ff), a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2.reshape(n_experts * d_ff, d), a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W3.reshape(n_experts * d, d_ff), a=math.sqrt(5))
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, d) → (B, K, d)"""
        x = _linear_input(x, self.router)
        residual = x
        x_n  = self.norm(x)
        B, K, d = x_n.shape
        x_flat = x_n.reshape(B * K, d)                        # (N, d)

        # Top-1 routing.
        # F.softmax is in PyTorch's autocast float32-preserve list: it returns
        # float32 even when logits is bfloat16.  Cast back to x_flat.dtype
        # (following the same pattern as MoeSwiGLUFFN in kairos_hybrid_block.py)
        # so that gate_e stays bfloat16 and output_sorted[s:t] = out_e * gate_e
        # does not raise "Index put requires dtypes match".
        logits     = self.router(x_flat)                                        # (N, E)
        expert_idx = logits.argmax(dim=-1)                                      # (N,)
        gates      = F.softmax(logits.float(), dim=-1).to(x_flat.dtype)        # (N, E)
        top_gates  = gates.gather(1, expert_idx.unsqueeze(1)).squeeze(1)       # (N,)

        # Sort by expert for contiguous GEMM slices
        order          = expert_idx.argsort(stable=True)
        sorted_x       = x_flat[order]
        sorted_experts = expert_idx[order]
        sorted_gates   = top_gates[order]

        counts     = sorted_experts.bincount(minlength=self.E)
        ends_list  = counts.cumsum(0).tolist()
        starts_list = [0] + ends_list[:-1]

        output_sorted = torch.zeros_like(sorted_x)
        for e in range(self.E):
            s, t = starts_list[e], ends_list[e]
            if s >= t:
                continue
            tok_e  = sorted_x[s:t]
            gate_e = sorted_gates[s:t]
            out_e  = (F.silu(tok_e @ self.W1[e]) * (tok_e @ self.W3[e])) @ self.W2[e]
            # Defensive cast: F.silu / matmul may return float32 under autocast;
            # output_sorted is zeros_like(sorted_x) = model dtype (BF16 after .to(bf16)).
            _safe_assign_(output_sorted, slice(s, t), out_e * gate_e.unsqueeze(-1),
                          name=f"lidar_moe_e{e}")

        # Unsort (top-1: each token appears exactly once)
        output = torch.empty_like(x_flat)
        # Defensive cast: output_sorted and output must share dtype.
        _safe_assign_(output, order, output_sorted, name="lidar_moe_unsort")

        return residual + output.view(B, K, d)


# ──────────────────────────────────────────────────────────────────────────────
# Full PointMamba encoder
# ──────────────────────────────────────────────────────────────────────────────

class PointMambaEncoder(nn.Module):
    """
    Full LiDAR encoder used by KairosModel.

    Drop-in interface:
        tokens, centroids = encoder(pts_t, pts_t1)
        # tokens    (B, 64, d_model)
        # centroids (B, 64, 3)

    Architecture:
        FPS → ball-query grouping → PointNet per cluster → temporal motion
        → Morton-sort → Mamba SSM → (tokens, centroids)

    ~101M parameters at d_model=1024.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_tokens: int = 64,
        cfg: Optional[LiDAREncoderConfig] = None,
    ) -> None:
        super().__init__()
        cfg            = cfg or LiDAREncoderConfig(n_tokens=n_tokens)
        self.cfg       = cfg
        self.d         = d_model
        self.n_tokens  = n_tokens

        # ── PointNet per cluster ────────────────────────────────────────────────
        self.pointnet = PointNetCluster(d_model, d_hidden=cfg.pn_hidden)

        # ── Temporal motion projector ───────────────────────────────────────────
        self.motion_proj = nn.Linear(d_model + 3, d_model, bias=False)
        self.motion_norm = RMSNorm(d_model)

        # ── Stacked Mamba SSM over Morton-ordered tokens ────────────────────────
        self.mamba_layers = nn.ModuleList([
            _LiDARMambaLayer(d_model, d_state=cfg.d_state, chunk=cfg.mamba_chunk)
            for _ in range(cfg.n_mamba_layers)
        ])

        # ── MoE FFN for feature enrichment after Mamba stack ───────────────────
        self.moe_ffn = _LiDARMoEFFN(d_model,
                                     n_experts=cfg.moe_experts,
                                     d_ff=cfg.moe_d_ff)

    # ------------------------------------------------------------------
    def _preprocess(
        self,
        pts_t:  torch.Tensor,   # (B, N, 4)
        pts_t1: torch.Tensor,   # (B, N, 4)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter ground/behind points, subsample to n_points, return both frames.
        Returns: pts_curr (B, n_pts, 4), pts_prev (B, n_pts, 4)
        """
        cfg = self.cfg

        def _filter_sub(pts: torch.Tensor) -> torch.Tensor:
            B, N, C = pts.shape
            device   = pts.device
            n        = cfg.n_points

            # Valid points: forward hemisphere (x > x_min) and above ground (z > z_min)
            mask = (pts[:, :, 0] > cfg.x_min) & (pts[:, :, 2] > cfg.z_min)  # (B, N)

            # Sort so valid points come first — fully vectorised, no Python loop over B.
            # argsort(descending) puts mask=1 indices before mask=0 indices.
            order       = mask.long().argsort(dim=1, descending=True)         # (B, N)
            pts_sorted  = pts.gather(1, order.unsqueeze(-1).expand(B, N, C))  # (B, N, C)
            mask_sorted = mask.gather(1, order)                                # (B, N)

            # Truncate to n_points, padding with zeros if N < n_points
            if N >= n:
                result     = pts_sorted[:, :n].clone()
                mask_valid = mask_sorted[:, :n]
            else:
                pad_n      = n - N
                result     = torch.cat([pts_sorted,
                                        pts.new_zeros(B, pad_n, C)], dim=1)
                mask_valid = torch.cat([mask_sorted,
                                        torch.zeros(B, pad_n, dtype=torch.bool,
                                                    device=device)], dim=1)

            # Zero out invalid entries so downstream (FPS / ball-query) see 0.0
            return result * mask_valid.unsqueeze(-1).to(pts.dtype)   # (B, n, C)

        return _filter_sub(pts_t), _filter_sub(pts_t1)

    # ------------------------------------------------------------------
    def _temporal_motion(
        self,
        tokens_curr: torch.Tensor,   # (B, K, d)
        cent_curr:   torch.Tensor,   # (B, K, 3)
        cent_prev:   torch.Tensor,   # (B, K, 3)
    ) -> torch.Tensor:
        """
        Append nearest-cluster motion vector to each token, then project back.
        """
        B, K, _ = cent_curr.shape
        # Pairwise distances between current and previous centroids
        diff   = cent_curr.unsqueeze(2) - cent_prev.unsqueeze(1)  # (B,K,K,3)
        dist2  = diff.pow(2).sum(-1)                               # (B,K,K)
        nn_idx = dist2.argmin(dim=2)                               # (B,K) prev-cluster index
        nn_cent = cent_prev.gather(
            1, nn_idx.unsqueeze(-1).expand(B, K, 3)
        )                                                          # (B,K,3)
        motion  = cent_curr - nn_cent                             # (B,K,3) Δxyz

        # Concat and project
        x = torch.cat([tokens_curr, motion.to(tokens_curr.dtype)], dim=-1)  # (B,K,d+3)
        x = _linear_input(x, self.motion_proj)
        return self.motion_norm(self.motion_proj(x))

    # ------------------------------------------------------------------
    def forward(
        self,
        pts_t:  torch.Tensor,   # (B, N, 4)  current frame
        pts_t1: torch.Tensor,   # (B, N, 4)  previous frame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tokens    (B, n_tokens, d_model)
            centroids (B, n_tokens, 3)
        """
        # Cast inputs to model dtype immediately. Prevents BF16/float32 mixing
        # in all downstream geometry ops (_farthest_point_sample, _ball_query,
        # _morton_sort_idx, _temporal_motion) which produce intermediate tensors
        # using the input dtype. Without this, float32 batch inputs combined with
        # BF16 linear layer outputs cause dtype mismatches in ops like
        # torch.minimum and torch.linspace that require homogeneous dtypes.
        _mdtype  = self.pointnet.mlp[0].weight.dtype
        _mdevice = self.pointnet.mlp[0].weight.device
        pts_t  = pts_t.to(device=_mdevice, dtype=_mdtype)
        pts_t1 = pts_t1.to(device=_mdevice, dtype=_mdtype)

        B = pts_t.shape[0]
        K = self.n_tokens
        cfg = self.cfg

        # ── 1. Pre-process ────────────────────────────────────────────────────
        curr, prev = self._preprocess(pts_t, pts_t1)  # each (B, n_pts, 4)

        # ── 2. FPS on current frame ───────────────────────────────────────────
        xyz_curr = curr[:, :, :3]   # (B, n_pts, 3)

        # Guard: replace zero-padded invalid entries with the valid-point centroid
        # so FPS never selects them as "farthest" points.
        def _fps_safe(xyz: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """FPS with invalid (zero-padded) points replaced by valid centroid."""
            valid = (xyz.norm(dim=-1) > 1e-3)   # (B, n_pts) bool
            # Compute centroid in float32 explicitly:
            #   sum() is in PyTorch's autocast float32-preserve list — it returns
            #   float32 even when the input is bfloat16. Using .float() here makes
            #   the behaviour identical under autocast and non-autocast, and we
            #   cast vmean back to xyz.dtype before the index assignment so that
            #   xyz_safe[~valid] = vmean does not hit a BF16/float32 mismatch.
            vsum  = (xyz.float() * valid.float().unsqueeze(-1)).sum(1)  # (B, 3) float32
            vcnt  = valid.float().sum(1, keepdim=True).clamp(min=1.0)  # (B, 1) float32
            vmean = (vsum / vcnt).to(xyz.dtype)                         # (B, 3) → xyz.dtype
            xyz_safe          = xyz.clone()
            xyz_safe[~valid]  = vmean.unsqueeze(1).expand_as(xyz)[~valid]  # same dtype ✓
            idx               = _farthest_point_sample(xyz_safe, k)    # (B, k)
            centroids = xyz.gather(1, idx.unsqueeze(-1).expand(B, k, 3))
            return idx, centroids

        fps_idx,   cent_curr = _fps_safe(xyz_curr, K)   # (B,K), (B,K,3)

        # FPS on previous frame (same guard)
        xyz_prev = prev[:, :, :3]
        _fps_idx_prev, cent_prev = _fps_safe(xyz_prev, K)

        # ── 3. Ball query grouping ────────────────────────────────────────────
        nbr_idx = _ball_query(
            cent_curr, xyz_curr,
            radius=cfg.ball_radius, n_sample=cfg.n_neighbors
        )                                                 # (B, K, n_nbrs)

        # Gather neighbor features: (B, K, n_nbrs, 4)
        nbr_pts = curr.gather(
            1,
            nbr_idx.reshape(B, -1).unsqueeze(-1).expand(B, K * cfg.n_neighbors, 4)
        ).reshape(B, K, cfg.n_neighbors, 4)

        # Normalize relative to cluster centroid — cast centroid to nbr_pts dtype
        # to avoid in-place subtraction dtype mismatch (e.g. BF16 -= float32).
        nbr_pts[..., :3] -= cent_curr.unsqueeze(2).to(dtype=nbr_pts.dtype, device=nbr_pts.device)

        # ── 4. PointNet per cluster ───────────────────────────────────────────
        tokens = self.pointnet(nbr_pts)                   # (B, K, d)

        # ── 5. Temporal motion features ───────────────────────────────────────
        tokens = self._temporal_motion(tokens, cent_curr, cent_prev)

        # ── 6. Morton sort + Mamba SSM ────────────────────────────────────────
        order     = _morton_sort_idx(cent_curr)           # (B, K) sorted indices
        inv_order = order.argsort(dim=1)                  # to unsort after SSM

        # Sort tokens and centroids
        tokens_sorted = tokens.gather(
            1, order.unsqueeze(-1).expand(B, K, self.d)
        )
        cent_sorted = cent_curr.gather(
            1, order.unsqueeze(-1).expand(B, K, 3)
        )

        # ── Stacked Mamba SSM ────────────────────────────────────────────────
        tokens_ctx = tokens_sorted
        for mamba in self.mamba_layers:
            tokens_ctx = mamba(tokens_ctx)                # (B, K, d) each

        # ── MoE feature enrichment ────────────────────────────────────────────
        tokens_ctx = self.moe_ffn(tokens_ctx)             # (B, K, d)

        # Unsort back to original FPS order
        tokens_out = tokens_ctx.gather(
            1, inv_order.unsqueeze(-1).expand(B, K, self.d)
        )
        centroids_out = cent_sorted.gather(
            1, inv_order.unsqueeze(-1).expand(B, K, 3)
        )

        return tokens_out, centroids_out


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    d = 1024

    enc = PointMambaEncoder(d_model=d, n_tokens=64)
    total = sum(p.numel() for p in enc.parameters())
    print(f"PointMambaEncoder params: {total/1e6:.1f}M  (target ~100M)")

    B, N = 2, 30_000
    pts_t  = torch.randn(B, N, 4)
    pts_t1 = torch.randn(B, N, 4)

    print(f"Input: pts_t {tuple(pts_t.shape)}")
    enc.eval()
    with torch.no_grad():
        tokens, centroids = enc(pts_t, pts_t1)

    assert tokens.shape    == (B, 64, d), f"tokens shape: {tokens.shape}"
    assert centroids.shape == (B, 64, 3), f"centroids shape: {centroids.shape}"
    print(f"Output: tokens {tuple(tokens.shape)}  centroids {tuple(centroids.shape)}")

    # Gradient check
    enc.train()
    tok_tr, _ = enc(pts_t, pts_t1)
    tok_tr.float().mean().backward()
    no_grad = [n for n, p in enc.named_parameters()
               if p.requires_grad and p.grad is None]
    print(f"Params without grad: {no_grad[:3] if no_grad else 'NONE — all OK'}")
    print("Smoke-test passed.")
