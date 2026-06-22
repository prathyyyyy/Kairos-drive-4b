"""
kairos_fusion.py

Calibration-Aware Fusion Gate for Kairos-4B.

What this module does:
  Takes the four separately-encoded modality streams and produces the single
  (B, T, d) token sequence that enters KairosHybridCore.

  The geometric step — projecting LiDAR cluster centroids into camera space
  using KITTI calibration matrices — drives a spatially-grounded attention
  bias.  Camera token i attends strongly to LiDAR token j when j's centroid
  projects near camera patch i.  A learned sigmoid gate then controls how
  much of that LiDAR context each camera token absorbs (and vice-versa).

  This is the architectural detail that most multimodal driving models miss:
  explicit geometric alignment before learned fusion.

Sequence layout produced:
  [cam_fused | lidar_fused | imu | query]   lengths: 256 + 64 + N_imu + 8
  = 336 tokens for the typical KITTI setup (N_imu = 8)

KITTI calibration matrices required (per frame, read from calib_*.txt files):
  P2            (3, 4) — camera projection  (image_02 left colour)
  R0_rect       (3, 3) — camera rectification
  Tr_velo_to_cam (3,4) — velodyne → camera extrinsics [R | t]

  Full projection:  P_full = P2 @ R0_4x4 @ Tr_4x4   → (3, 4)
  For point x_velo  (4-vec homogeneous):
      px  = P_full @ x_velo          →  (3,)
      u   = px[0] / px[2],  v = px[1] / px[2]   (pixel coords, original resolution)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kairos_hybrid_block import KairosConfig, RMSNorm   # type: ignore
except ImportError:
    @dataclass
    class KairosConfig:                                       # type: ignore
        d_model: int = 1024

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_f = x.float()
            return (x_f * x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
                    * self.weight.float()).to(x.dtype)


def _linear_input(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    return x.to(device=linear.weight.device, dtype=linear.weight.dtype)


def _debug_shapes_enabled() -> bool:
    return (
        os.environ.get("KAIROS_DEBUG_SHAPES", "0") == "1"
        and int(os.environ.get("RANK", "0")) == 0
    )


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionConfig:
    # ── Token layout (must match KairosHybridCore expectations) ───────────────
    n_cam_tokens:   int = 256
    n_lidar_tokens: int = 64
    n_imu_tokens:   int = 8
    n_query_tokens: int = 8

    # ── Camera patch grid (from VisionEncoderConfig) ───────────────────────────
    patch_rows: int = 8     # enc_h  / patch_size = 112 / 14
    patch_cols: int = 32    # enc_w  / patch_size = 448 / 14

    # ── Original KITTI image resolution (for calibration projection) ───────────
    orig_h: int = 375
    orig_w: int = 1242
    # Encoder resolution (after resize in KairosVisionEncoder._preprocess)
    enc_h:  int = 112
    enc_w:  int = 448

    # ── Geometry bias ─────────────────────────────────────────────────────────
    sigma_init: float = 2.0   # initial spatial bandwidth in patch units; learnable
    min_depth:  float = 0.5   # metres — LiDAR points closer than this are ignored

    # ── Cross-modal attention ─────────────────────────────────────────────────
    fusion_heads: int = 8     # cross-modal attention heads (less than main 16)

    @property
    def cam_start(self) -> int:
        return 0

    @property
    def cam_end(self) -> int:
        return self.n_cam_tokens

    @property
    def lidar_start(self) -> int:
        return self.cam_end

    @property
    def lidar_end(self) -> int:
        return self.lidar_start + self.n_lidar_tokens

    @property
    def imu_start(self) -> int:
        return self.lidar_end

    @property
    def imu_end(self) -> int:
        return self.imu_start + self.n_imu_tokens

    @property
    def query_start(self) -> int:
        return self.imu_end

    @property
    def query_end(self) -> int:
        return self.query_start + self.n_query_tokens

    @property
    def total_tokens(self) -> int:
        return self.query_end


# ──────────────────────────────────────────────────────────────────────────────
# Calibration matrices — batched dataclass (not nn.Module; no learnable params)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibMatrices:
    """
    Batched KITTI calibration matrices for B samples.

    All tensors should be float32 on the same device as the model inputs.
    Call `.to(device)` or `.to(dtype)` to move them.

    Shapes:
      P2              (B, 3, 4)  — camera projection (image_02)
      R0_rect         (B, 3, 3)  — rectification rotation
      Tr_velo_to_cam  (B, 3, 4)  — velodyne → camera  [R | t]
    """
    P2:             torch.Tensor   # (B, 3, 4)
    R0_rect:        torch.Tensor   # (B, 3, 3)
    Tr_velo_to_cam: torch.Tensor   # (B, 3, 4)

    @property
    def P_full(self) -> torch.Tensor:
        """
        Combined projection matrix P2 @ R0_4x4 @ Tr_4x4 → (B, 3, 4).

        Apply to homogeneous velodyne point x_v (4-vec) as: P_full @ x_v → (3,)
        Then u = out[0]/out[2], v = out[1]/out[2].
        """
        B, device, dtype = self.P2.shape[0], self.P2.device, self.P2.dtype

        # R0_rect (B, 3, 3) → R0_4x4 (B, 4, 4): pad to homogeneous
        R0 = torch.zeros(B, 4, 4, device=device, dtype=dtype)
        R0[:, :3, :3] = self.R0_rect
        R0[:, 3, 3]   = 1.0

        # Tr_velo_to_cam (B, 3, 4) → Tr_4x4 (B, 4, 4): add [0,0,0,1] row
        Tr = torch.zeros(B, 4, 4, device=device, dtype=dtype)
        Tr[:, :3, :]  = self.Tr_velo_to_cam
        Tr[:, 3, 3]   = 1.0

        # P_full = P2 (B,3,4) @ [R0 (B,4,4) @ Tr (B,4,4)] → (B,3,4)
        return torch.bmm(self.P2, torch.bmm(R0, Tr))

    def to(self, *args, **kwargs) -> "CalibMatrices":
        return CalibMatrices(
            P2             = self.P2.to(*args, **kwargs),
            R0_rect        = self.R0_rect.to(*args, **kwargs),
            Tr_velo_to_cam = self.Tr_velo_to_cam.to(*args, **kwargs),
        )

    @staticmethod
    def stack(items: List["CalibMatrices"]) -> "CalibMatrices":
        """Collate a list of single-sample CalibMatrices into a batch."""
        return CalibMatrices(
            P2             = torch.stack([c.P2.squeeze(0)             for c in items]),
            R0_rect        = torch.stack([c.R0_rect.squeeze(0)        for c in items]),
            Tr_velo_to_cam = torch.stack([c.Tr_velo_to_cam.squeeze(0) for c in items]),
        )


# ──────────────────────────────────────────────────────────────────────────────
# KITTI calibration file parsers
# ──────────────────────────────────────────────────────────────────────────────

def parse_kitti_calib(
    cam_to_cam_path: str,
    velo_to_cam_path: str,
) -> CalibMatrices:
    """
    Parse one KITTI calibration file pair into a single-sample CalibMatrices.

    cam_to_cam_path: path to calib_cam_to_cam.txt
      Extracts: P_rect_02 → P2 (3×4), R_rect_00 → R0_rect (3×3)

    velo_to_cam_path: path to calib_velo_to_cam.txt
      Extracts: R (3×3) and T (3,) → Tr_velo_to_cam (3×4) = [R | T]

    Returns CalibMatrices with unsqueezed batch dim (B=1).
    """
    def _read_matrix(path: str, key: str, rows: int, cols: int) -> torch.Tensor:
        with open(path, "r") as f:
            for line in f:
                if line.startswith(key + ":"):
                    vals = list(map(float, line.split(":")[1].split()))
                    return torch.tensor(vals, dtype=torch.float32).view(rows, cols)
        raise KeyError(f"Key '{key}' not found in {path}")

    P2      = _read_matrix(cam_to_cam_path, "P_rect_02", 3, 4)
    R0_rect = _read_matrix(cam_to_cam_path, "R_rect_00", 3, 3)

    with open(velo_to_cam_path, "r") as f:
        content = f.read()
    lines = {l.split(":")[0].strip(): l.split(":")[1].strip()
             for l in content.splitlines() if ":" in l}
    R_velo = torch.tensor(list(map(float, lines["R"].split())),
                          dtype=torch.float32).view(3, 3)
    T_velo = torch.tensor(list(map(float, lines["T"].split())),
                          dtype=torch.float32).view(3, 1)
    Tr = torch.cat([R_velo, T_velo], dim=1)  # (3, 4) = [R | t]

    return CalibMatrices(
        P2             = P2.unsqueeze(0),      # (1, 3, 4)
        R0_rect        = R0_rect.unsqueeze(0), # (1, 3, 3)
        Tr_velo_to_cam = Tr.unsqueeze(0),      # (1, 3, 4)
    )


def parse_kitti_calib_combined(path: str) -> CalibMatrices:
    """
    Parse a single combined calib.txt (used in KITTI object detection splits).
    Keys: P2, R0_rect, Tr_velo_to_cam.
    """
    def _read(key: str, rows: int, cols: int) -> torch.Tensor:
        with open(path, "r") as f:
            for line in f:
                if line.startswith(key + ":"):
                    vals = list(map(float, line.split(":")[1].split()))
                    return torch.tensor(vals, dtype=torch.float32).view(rows, cols)
        raise KeyError(f"'{key}' not in {path}")

    return CalibMatrices(
        P2             = _read("P2",            3, 4).unsqueeze(0),
        R0_rect        = _read("R0_rect",       3, 3).unsqueeze(0),
        Tr_velo_to_cam = _read("Tr_velo_to_cam",3, 4).unsqueeze(0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Geometry utilities — vectorised, no Python loops over B or N
# ──────────────────────────────────────────────────────────────────────────────

def _project_to_patch(
    xyz_velo: torch.Tensor,    # (B, N, 3) LiDAR centroids in velodyne frame
    P_full:   torch.Tensor,    # (B, 3, 4) combined projection matrix
    fcfg:     FusionConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project N 3D LiDAR centroids to 2D camera patch indices.

    Pipeline (all vectorised):
      x_h   = [x, y, z, 1]^T          homogeneous velodyne point
      px    = P_full @ x_h             (3,) image coords (unnormalised)
      u, v  = px[0]/px[2], px[1]/px[2] pixel in original 1242×375 image
      u_enc = u * (enc_w / orig_w)     pixel in 448×112 encoder image
      patch = floor(u_enc/14), floor(v_enc/14)

    Returns:
      patch_idx  (B, N) int64  — flat patch index ∈ [0, n_patches); -1 if OOF
      visible    (B, N) bool   — True if point projects within camera FOV
    """
    B, N, _ = xyz_velo.shape
    device, dtype = xyz_velo.device, xyz_velo.dtype

    # Homogeneous coordinates (B, N, 4)
    ones  = torch.ones(B, N, 1, device=device, dtype=dtype)
    xyz_h = torch.cat([xyz_velo, ones], dim=-1)              # (B, N, 4)

    # Project: P_full (B,3,4) @ xyz_h^T (B,4,N) → (B,3,N)
    px = torch.bmm(P_full.to(dtype), xyz_h.transpose(1, 2))  # (B, 3, N)

    depth = px[:, 2, :]                                       # (B, N) metres
    u = px[:, 0, :] / depth.clamp(min=1e-6)                  # (B, N) pixel x
    v = px[:, 1, :] / depth.clamp(min=1e-6)                  # (B, N) pixel y

    # Scale from original resolution to encoder resolution
    u_enc = u * (fcfg.enc_w / fcfg.orig_w)                   # (B, N)
    v_enc = v * (fcfg.enc_h / fcfg.orig_h)                   # (B, N)

    # Visibility: positive depth, inside encoder image bounds
    visible = (
        (depth > fcfg.min_depth)
        & (u_enc >= 0.0) & (u_enc < fcfg.enc_w)
        & (v_enc >= 0.0) & (v_enc < fcfg.enc_h)
    )                                                         # (B, N) bool

    # Patch indices (clamp for safe indexing even for OOF points)
    col = u_enc.div(14, rounding_mode="floor").long().clamp(0, fcfg.patch_cols - 1)
    row = v_enc.div(14, rounding_mode="floor").long().clamp(0, fcfg.patch_rows - 1)
    patch_idx = row * fcfg.patch_cols + col                   # (B, N) ∈ [0, 255]

    # Mark out-of-frame points
    patch_idx = torch.where(visible, patch_idx, patch_idx.new_full((), -1))

    return patch_idx, visible


def _geometry_bias(
    patch_idx:  torch.Tensor,   # (B, N_lidar) ∈ [-1, 255]
    visible:    torch.Tensor,   # (B, N_lidar) bool
    n_cam:      int,            # 256
    patch_cols: int,            # 32
    log_sigma:  torch.Tensor,   # () scalar learnable parameter
) -> torch.Tensor:
    """
    Build additive spatial attention bias for cam→lidar cross-attention.

    bias[b, i, j]  ∝  –dist²( camera_patch_i, projected_position_of_lidar_j )
                         / (2 σ²)

    Invisible LiDAR tokens get bias = –1e9 (suppressed, avoids NaN from –inf).

    Returns:
      bias  (B, n_cam, N_lidar)  — additive logit bias, same dtype as patch_idx
    """
    B, N_lidar = patch_idx.shape
    device = patch_idx.device
    patch_rows = n_cam // patch_cols   # 8

    # ── Fixed camera grid coordinates ─────────────────────────────────────────
    ids      = torch.arange(n_cam, device=device)             # (n_cam,)
    cam_row  = (ids // patch_cols).float()                    # (n_cam,)
    cam_col  = (ids  % patch_cols).float()

    # ── Projected LiDAR positions (clamp -1 → 0, masked below) ───────────────
    safe_idx = patch_idx.clamp(min=0)                         # (B, N_lidar)
    lidar_row = (safe_idx // patch_cols).float()              # (B, N_lidar)
    lidar_col = (safe_idx  % patch_cols).float()

    # ── Pairwise 2D squared distance: (B, n_cam, N_lidar) ────────────────────
    # cam_row[None, :, None]   →  (1, n_cam, 1)
    # lidar_row[:, None, :]    →  (B,    1, N_lidar)
    dr   = cam_row[None, :, None] - lidar_row[:, None, :]    # (B, n_cam, N_lidar)
    dc   = cam_col[None, :, None] - lidar_col[:, None, :]
    dist2 = dr.pow(2) + dc.pow(2)                            # (B, n_cam, N_lidar)

    # ── Gaussian bias (learnable bandwidth) ───────────────────────────────────
    sigma2 = torch.exp(log_sigma).pow(2).clamp(min=0.25)     # σ² ≥ 0.25 patch²
    bias   = -dist2 / (2.0 * sigma2)                         # (B, n_cam, N_lidar)

    # ── Mask invisible LiDAR tokens ───────────────────────────────────────────
    # invisible[:, None, :] broadcasts to (B, n_cam, N_lidar)
    bias = bias.masked_fill(~visible[:, None, :], -1e9)

    return bias.to(torch.float32)   # keep in fp32 for attention stability


# ──────────────────────────────────────────────────────────────────────────────
# Cross-modal attention with optional geometry bias
# ──────────────────────────────────────────────────────────────────────────────

class GeometryBiasedCrossAttn(nn.Module):
    """
    Standard multi-head cross-attention where Q comes from one modality and
    K/V from another, with an optional additive geometry bias on the logits.

    bias shape: (B, n_q, n_kv) — broadcast to (B, n_heads, n_q, n_kv).
    """

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        assert d % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d // n_heads

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.o_proj.weight, std=d ** -0.5)

    def forward(
        self,
        q_tok:  torch.Tensor,                    # (B, n_q, d)
        kv_tok: torch.Tensor,                    # (B, n_kv, d)
        bias:   Optional[torch.Tensor] = None,   # (B, n_q, n_kv) fp32 or None
    ) -> torch.Tensor:
        B, n_q, _  = q_tok.shape
        n_kv       = kv_tok.shape[1]
        H, hd      = self.n_heads, self.head_dim
        q_tok = _linear_input(q_tok, self.q_proj)
        kv_tok = _linear_input(kv_tok, self.k_proj)

        Q = self.q_proj(q_tok).view(B, n_q,  H, hd).transpose(1, 2)  # (B,H,n_q,hd)
        K = self.k_proj(kv_tok).view(B, n_kv, H, hd).transpose(1, 2)
        V = self.v_proj(kv_tok).view(B, n_kv, H, hd).transpose(1, 2)

        # bias: (B, n_q, n_kv) → (B, 1, n_q, n_kv) — broadcast over heads
        attn_mask = bias.unsqueeze(1).to(Q.dtype) if bias is not None else None

        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, n_q, -1)
        return self.o_proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# Per-token sigmoid gate
# ──────────────────────────────────────────────────────────────────────────────

class ModalityGate(nn.Module):
    """
    Scalar sigmoid gate per token: g ∈ (0, 1).
    Controls how much cross-modal context to blend into each token.

    Initialised at –2 → sigmoid(–2) ≈ 0.12: gate starts mostly closed,
    opening gradually as training progresses.  This prevents early training
    instability from large random cross-modal signals.

    Usage:
        g     = gate(x_tok)              # (B, T, 1)
        x_out = x_tok + g * cross_modal  # residual blend
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d)
        self.proj = nn.Linear(d, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(self.norm(x)))   # (B, T, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration-Aware Fusion Gate
# ──────────────────────────────────────────────────────────────────────────────

class KairosCalibrationGate(nn.Module):
    """
    Geometric alignment + learned modality gating.

    Inputs (separately-encoded modality streams):
      cam    (B, 256, d)  — from KairosVisionEncoder
      lidar  (B,  64, d)  — from PointMambaEncoder (to be implemented)
      imu    (B, N_imu,d) — from CfCEncoder
      query  (B,   8, d)  — from text/byte encoder

    Geometry inputs:
      lidar_xyz  (B, 64, 3)  — LiDAR cluster centroids in velodyne frame
                               (output of PointMamba's FPS step, before encoding)
      calib      CalibMatrices  — batch of KITTI calibration matrices

    Processing:
      1. Project lidar_xyz → camera patch indices via P_full  [vectorised]
      2. Build (B, 256, 64) Gaussian geometry bias matrix      [vectorised]
      3. Camera ← LiDAR  cross-attention, geometry-biased
         g_cam    = sigmoid gate(cam)       ∈ (0,1) per cam token
         cam_out  = cam + g_cam * attn(cam, lidar, bias)
      4. LiDAR  ← Camera cross-attention, transposed bias
         g_lidar  = sigmoid gate(lidar)     ∈ (0,1) per lidar token
         lidar_out= lidar + g_lidar * attn(lidar, cam, bias.T)
      5. Concatenate: [cam_out | lidar_out | imu | query]  → (B, T, d)

    IMU and query tokens are left unchanged here; the hybrid block's Mamba-2
    and sliding-window attention handle their temporal cross-modal context.

    Output:
      x  (B, T, d)  — T = 256 + 64 + N_imu + 8 = 336 for typical KITTI batches
                       This tensor goes directly into KairosHybridCore.forward()
    """

    def __init__(
        self,
        fcfg: Optional[FusionConfig] = None,
        kcfg: Optional[KairosConfig] = None,
    ) -> None:
        super().__init__()
        fcfg = fcfg or FusionConfig()
        kcfg = kcfg or KairosConfig()
        self.fcfg = fcfg
        self.d    = kcfg.d_model

        # ── Learnable spatial bandwidth (log σ) ───────────────────────────────
        # log_sigma=log(2.0) → σ=2 patches at init; unconstrained so model adapts
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(fcfg.sigma_init), dtype=torch.float32)
        )

        # ── Pre-fusion norms (pre-norm residual architecture) ──────────────────
        self.norm_cam   = RMSNorm(self.d)
        self.norm_lidar = RMSNorm(self.d)

        # ── Camera ← LiDAR cross-attention ────────────────────────────────────
        self.cam_from_lidar = GeometryBiasedCrossAttn(self.d, fcfg.fusion_heads)
        self.gate_cam       = ModalityGate(self.d)

        # ── LiDAR ← Camera cross-attention ────────────────────────────────────
        self.lidar_from_cam = GeometryBiasedCrossAttn(self.d, fcfg.fusion_heads)
        self.gate_lidar     = ModalityGate(self.d)

    # ------------------------------------------------------------------
    def forward(
        self,
        cam:       torch.Tensor,        # (B, 256, d)
        lidar:     torch.Tensor,        # (B,  64, d)
        imu:       torch.Tensor,        # (B, N_imu, d)  — may be 0-length
        query:     torch.Tensor,        # (B,   8, d)
        lidar_xyz: torch.Tensor,        # (B,  64, 3)  velodyne centroids
        calib:     CalibMatrices,
    ) -> torch.Tensor:
        """
        Returns (B, T, d) — full fused sequence for KairosHybridCore.
        """
        # ── 1. Calibration projection ─────────────────────────────────────────
        P_full = calib.P_full.to(dtype=cam.dtype, device=cam.device)   # (B, 3, 4)

        patch_idx, visible = _project_to_patch(lidar_xyz, P_full, self.fcfg)
        # patch_idx: (B, 64) ∈ [-1, 255]; visible: (B, 64) bool

        # ── 2. Geometry bias: (B, 256, 64) ───────────────────────────────────
        bias_c2l = _geometry_bias(
            patch_idx, visible,
            n_cam      = self.fcfg.n_cam_tokens,
            patch_cols = self.fcfg.patch_cols,
            log_sigma  = self.log_sigma,
        )                                                                # (B, 256, 64)

        # ── 3. Camera ← LiDAR (geometry-biased) ──────────────────────────────
        cam_q      = self.norm_cam(cam)
        lidar_ctx  = self.cam_from_lidar(cam_q, lidar, bias=bias_c2l)  # (B, 256, d)
        g_cam      = self.gate_cam(cam)                                  # (B, 256, 1)
        cam_fused  = cam + g_cam * lidar_ctx                            # residual

        # ── 4. LiDAR ← Camera (transposed bias) ──────────────────────────────
        lidar_q    = self.norm_lidar(lidar)
        cam_ctx    = self.lidar_from_cam(
            lidar_q, cam, bias=bias_c2l.transpose(-1, -2)
        )                                                                # (B, 64, d)
        g_lidar    = self.gate_lidar(lidar)                             # (B, 64, 1)
        lidar_fused = lidar + g_lidar * cam_ctx                         # residual

        # ── 5. Assemble full sequence ─────────────────────────────────────────
        # [cam_fused | lidar_fused | imu | query] → (B, T, d)
        fused_tokens = torch.cat([cam_fused, lidar_fused, imu, query], dim=1)
        if _debug_shapes_enabled():
            print("[shape] fused_tokens", fused_tokens.shape, flush=True)
        return fused_tokens

    # ------------------------------------------------------------------
    def fusion_stats(
        self,
        cam:       torch.Tensor,   # (B, 256, d)
        lidar:     torch.Tensor,   # (B,  64, d)
        lidar_xyz: torch.Tensor,   # (B,  64, 3)
        calib:     CalibMatrices,
    ) -> Dict[str, float]:
        """
        Diagnostic helper: returns interpretability statistics for one batch.
        Call after forward() during eval; useful for logging / debugging.

        Returns dict with:
          visible_frac    — fraction of LiDAR tokens that project into camera FOV
          sigma_patches   — current spatial bandwidth in patch units
          gate_cam_mean   — mean cam gate value (how open the cam←lidar gate is)
          gate_lidar_mean — mean lidar gate value (how open the lidar←cam gate is)
        """
        P_full = calib.P_full.to(dtype=cam.dtype, device=cam.device)
        _, visible = _project_to_patch(lidar_xyz, P_full, self.fcfg)
        g_cam   = self.gate_cam(cam)
        g_lidar = self.gate_lidar(lidar)   # fixed: pass lidar tokens, not cam slice
        return {
            "visible_frac":    visible.float().mean().item(),
            "sigma_patches":   torch.exp(self.log_sigma).item(),
            "gate_cam_mean":   g_cam.mean().item(),
            "gate_lidar_mean": g_lidar.mean().item(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    torch.manual_seed(0)

    fcfg = FusionConfig()
    kcfg = KairosConfig()
    B, d = 2, kcfg.d_model

    print("Building KairosCalibrationGate ...")
    gate = KairosCalibrationGate(fcfg, kcfg)
    n_params = sum(p.numel() for p in gate.parameters())
    print(f"  params: {n_params / 1e6:.2f}M")

    # ── Synthetic inputs ──────────────────────────────────────────────────────
    cam   = torch.randn(B, 256, d)
    lidar = torch.randn(B,  64, d)
    imu   = torch.randn(B,   8, d)    # 8 IMU tokens (variable in practice)
    query = torch.randn(B,   8, d)

    # LiDAR centroids: ~30 m in front of car, spread laterally
    # Velodyne frame: X=forward, Y=left, Z=up (approx)
    lidar_xyz = torch.zeros(B, 64, 3)
    lidar_xyz[:, :, 0] = torch.rand(B, 64) * 60.0 - 5.0   # X: -5 to 55 m
    lidar_xyz[:, :, 1] = torch.rand(B, 64) * 20.0 - 10.0  # Y: ±10 m lateral
    lidar_xyz[:, :, 2] = torch.rand(B, 64) *  2.0 -  1.0  # Z: ±1 m height

    # Approximate real KITTI calibration matrices (2011_09_26 drive)
    P2_val = torch.tensor([
        [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
        [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
        [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],
    ])
    R0_val = torch.tensor([
        [ 9.999239e-01,  9.837760e-03, -7.445048e-03],
        [-9.869795e-03,  9.999421e-01, -4.278459e-03],
        [ 7.402527e-03,  4.351614e-03,  9.999631e-01],
    ])
    Tr_val = torch.tensor([
        [ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
        [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
        [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
    ])

    calib = CalibMatrices(
        P2             = P2_val.unsqueeze(0).repeat(B, 1, 1),
        R0_rect        = R0_val.unsqueeze(0).repeat(B, 1, 1),
        Tr_velo_to_cam = Tr_val.unsqueeze(0).repeat(B, 1, 1),
    )

    # ── Verify P_full shape ────────────────────────────────────────────────────
    P_full = calib.P_full
    assert P_full.shape == (B, 3, 4), f"P_full shape: {P_full.shape}"
    print(f"  P_full shape: {tuple(P_full.shape)} OK")

    # ── Verify projection ─────────────────────────────────────────────────────
    patch_idx, visible = _project_to_patch(lidar_xyz, P_full, fcfg)
    vis_pct = visible.float().mean().item() * 100
    print(f"  LiDAR visible in cam FOV: {vis_pct:.1f}%  (expect ~25-60% for KITTI)")
    assert patch_idx.shape == (B, 64)
    assert visible.shape   == (B, 64)

    # ── Verify bias shape and values ──────────────────────────────────────────
    bias = _geometry_bias(patch_idx, visible, 256, 32, gate.log_sigma)
    assert bias.shape == (B, 256, 64)
    assert not bias.isnan().any(), "NaN in geometry bias"
    print(f"  Geometry bias: {tuple(bias.shape)}  min={bias[bias>-1e8].min():.2f}"
          f"  max={bias.max():.2f}")

    # ── Forward pass ──────────────────────────────────────────────────────────
    gate.eval()
    with torch.no_grad():
        x = gate(cam, lidar, imu, query, lidar_xyz, calib)

    T = 256 + 64 + 8 + 8
    assert x.shape == (B, T, d), f"Output shape mismatch: {x.shape}"
    print(f"  Output: {tuple(x.shape)}  (cam256 + lidar64 + imu8 + q8 = {T} tokens)")

    # ── Gradient check: gate and sigma must receive gradients ──────────────────
    gate.train()
    x_tr = gate(cam, lidar, imu, query, lidar_xyz, calib)
    x_tr.sum().backward()

    assert gate.log_sigma.grad is not None and gate.log_sigma.grad.abs() > 0, \
        "log_sigma has no gradient"
    assert gate.gate_cam.proj.weight.grad is not None, \
        "gate_cam.proj.weight has no gradient"
    print(f"  log_sigma grad: {gate.log_sigma.grad.item():.4e}  OK")
    print(f"  sigma (patches): {torch.exp(gate.log_sigma).item():.2f}")

    print("Smoke-test passed.")
