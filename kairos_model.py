"""
kairos_model.py

Full Kairos-4B model assembly.

Connects every sub-module into one nn.Module:

    Vision encoder  (DINOv2-L + LoRA + 3-frame fusion)   → cam  [256, d]
    LiDAR encoder   (PointMamba SSM)                      → lidar [ 64, d]  + xyz
    IMU encoder     (CfC/LNN projection)                  → imu  [  N, d]  + Δt
    Text encoder    (byte tokenizer + Perceiver queries)  → query[  8, d]
            │
            ▼
    CalibGate  (P2 · R0 · Tr projection + sigmoid gating)
            │  (B, 336, d)
            ▼
    KairosHybridCore  (3 blocks × 4 loops, Mamba+CfC+SWA+MoE)
            │  (B, 336, d)
            ┌────────────────────────────────┐
            ▼                                ▼
    S2FTDecoder                    ObjectDetectionHead
    (4-layer causal, byte-vocab)   (DETR-style, 50 queries)

Training usage (training script does the optimizer step):
    output = model(batch)
    output.total_loss.backward()   # aggregates s2ft + det + z-loss
"""

from __future__ import annotations

import math
import os as _os
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt

from kairos_hybrid_block import KairosConfig, KairosHybridCore, RMSNorm
from kairos_encoders import KairosVisionEncoder, VisionEncoderConfig
from kairos_fusion import CalibMatrices, FusionConfig, KairosCalibrationGate
from kairos_lidar import PointMambaEncoder, LiDAREncoderConfig
from kairos_imu   import IMUEncoder, IMUEncoderConfig

# ──────────────────────────────────────────────────────────────────────────────
# Canonical token layout (single source of truth for all slice offsets)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TokenLayout:
    """
    Immutable descriptor of the fused token sequence fed into KairosHybridCore.

    Canonical layout: [cam | lidar | imu | query]
      cam:   0 : n_cam
      lidar: n_cam : n_cam + n_lidar
      imu:   n_cam + n_lidar : n_cam + n_lidar + n_imu
      query: n_cam + n_lidar + n_imu : total

    Default KITTI layout: n_cam=256, n_lidar=64, n_imu=8, n_query=8 → total=336.
    """
    n_cam:   int
    n_lidar: int
    n_imu:   int
    n_query: int

    @property
    def cam_slice(self) -> slice:
        return slice(0, self.n_cam)

    @property
    def lidar_slice(self) -> slice:
        return slice(self.n_cam, self.n_cam + self.n_lidar)

    @property
    def imu_start(self) -> int:
        return self.n_cam + self.n_lidar

    @property
    def imu_end(self) -> int:
        return self.imu_start + self.n_imu

    @property
    def imu_slice(self) -> slice:
        return slice(self.imu_start, self.imu_end)

    @property
    def query_start(self) -> int:
        return self.imu_end

    @property
    def query_end(self) -> int:
        return self.query_start + self.n_query

    @property
    def query_slice(self) -> slice:
        return slice(self.query_start, self.query_end)

    @property
    def total(self) -> int:
        return self.n_cam + self.n_lidar + self.n_imu + self.n_query


# ──────────────────────────────────────────────────────────────────────────────
# Optional per-component CUDA memory trace (KAIROS_MEM_TRACE=1)
# ──────────────────────────────────────────────────────────────────────────────
_MEM_TRACE = _os.environ.get("KAIROS_MEM_TRACE", "0") == "1"


def _debug_shapes_enabled() -> bool:
    return (
        _os.environ.get("KAIROS_DEBUG_SHAPES", "0") == "1"
        and int(_os.environ.get("RANK", "0")) == 0
    )


def _shape_log(*args) -> None:
    if _debug_shapes_enabled():
        print(*args, flush=True)


def _dtype_log(tag: str, **tensors) -> None:
    """Print dtype/device of named tensors when KAIROS_DEBUG_DTYPE=1 (rank-0 only)."""
    if _os.environ.get("KAIROS_DEBUG_DTYPE", "0") != "1":
        return
    if int(_os.environ.get("RANK", "0")) != 0:
        return
    for name, t in tensors.items():
        if isinstance(t, torch.Tensor):
            print(
                f"[dtype/{tag}] {name}: dtype={t.dtype} device={t.device} "
                f"shape={tuple(t.shape)}",
                flush=True,
            )


def _as_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Cast src to ref's dtype and device.  Use before index/slice assignments."""
    return src.to(device=ref.device, dtype=ref.dtype)


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

    Wraps endpoints in float() and forces dtype=torch.float32 so that DeepSpeed BF16
    training cannot promote either endpoint to BFloat16, which would raise:
        RuntimeError: expected dtype c10::BFloat16 for 'end' but got dtype float
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
        dtype=torch.float32,
    ).round().long()
    return idx.clamp_(0, int(max_index))


def _assert_same_dtype(tag: str, ref: torch.Tensor, **others: torch.Tensor) -> None:
    """Assert all tensors share ref's dtype/device; print diagnostics if KAIROS_DEBUG_DTYPE=1."""
    mismatches = []
    for name, t in others.items():
        if t.dtype != ref.dtype or t.device != ref.device:
            mismatches.append(
                f"{name}: dtype={t.dtype} device={t.device} "
                f"(expected dtype={ref.dtype} device={ref.device})"
            )
    if mismatches and _os.environ.get("KAIROS_DEBUG_DTYPE", "0") == "1":
        if int(_os.environ.get("RANK", "0")) == 0:
            print(
                f"[dtype_mismatch/{tag}] ref dtype={ref.dtype} device={ref.device}; "
                + "  |  ".join(mismatches),
                flush=True,
            )


def _trace_mem(tag: str) -> None:
    """Print rank-0 CUDA memory stats after each major forward component."""
    if not _MEM_TRACE or not torch.cuda.is_available():
        return
    if int(_os.environ.get("RANK", "0")) != 0:
        return
    print(
        f"[mem_trace] {tag}  "
        f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB  "
        f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB  "
        f"max_alloc={torch.cuda.max_memory_allocated()/1e9:.2f}GB",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KairosModelConfig:
    """Top-level config.  Sub-configs compose the full architecture."""

    # ── Core components ────────────────────────────────────────────────────────
    kcfg:       KairosConfig        = field(default_factory=KairosConfig)
    vcfg:       VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    fcfg:       FusionConfig        = field(default_factory=FusionConfig)
    lidar_cfg:  LiDAREncoderConfig  = field(default_factory=LiDAREncoderConfig)
    imu_cfg:    IMUEncoderConfig    = field(default_factory=IMUEncoderConfig)

    # ── Token layout (must be consistent across fcfg + hybrid block) ──────────
    n_cam:   int = 256
    n_lidar: int = 64
    n_imu:   int = 8     # target IMU tokens per frame; actual may vary
    n_query: int = 8

    # ── S2FT decoder ──────────────────────────────────────────────────────────
    byte_vocab: int  = 258     # 256 bytes + BOS(256) + EOS(257)
    bos_id:     int  = 256
    eos_id:     int  = 257
    decoder_layers: int = 6    # 6-layer decoder → ~101M
    decoder_heads:  int = 16
    max_gen_len:    int = 512  # max reasoning chain + answer length

    # ── Detection head ────────────────────────────────────────────────────────
    max_det:   int = 50
    n_det_cls: int = 9         # KITTI: Car,Van,Truck,Ped,Cyclist,Misc×4

    # ── Multi-task weights ────────────────────────────────────────────────────
    w_s2ft: float = 1.0
    w_det:  float = 0.0        # set > 0 to enable detection loss

    # ── Regularisation (mirrors KairosConfig.z_loss_coeff) ───────────────────
    z_loss_coeff: float = 1e-3

    # ── Activation checkpointing ──────────────────────────────────────────────
    use_grad_checkpoint: bool = True

    # ── Text encoder depth ────────────────────────────────────────────────────
    # 8 layers → ~106M params; reduce to 2 for ultra_smoke (~26M)
    n_text_enc_layers: int = 8

    # ── Smoke/ultra_smoke forward flags ──────────────────────────────────────
    # Set no_grad_vision=True (+ freeze all vision_encoder params) when using
    # --ultra_smoke_dino_no_grad; disables LoRA gradient flow for plumbing-only smoke.
    no_grad_vision: bool = False

    # Keep returned training outputs lean unless an explicit debug run asks for
    # large tensors. Eval/inference behavior is unchanged.
    return_debug_tensors: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# I/O containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KairoBatch:
    """All inputs for one forward pass — training or inference."""

    # Camera frames (B, 3, H, W) ∈ [0, 1] — raw KITTI PNG resolution
    img_t:  torch.Tensor
    img_t1: torch.Tensor
    img_t2: torch.Tensor

    # LiDAR point clouds (B, N_pts, 4) — x, y, z, intensity  (velodyne frame)
    lidar_t:  torch.Tensor
    lidar_t1: torch.Tensor   # t-1 frame; concat with lidar_t before encoding

    # IMU / GPS stream from OXTS (B, T_imu, 7):
    #   [velocity_fwd, acceleration, jerk, lat, lon, alt, reserved]
    imu_data:       torch.Tensor
    imu_timestamps: torch.Tensor   # (B, T_imu)  seconds since epoch

    # Calibration (per-sample, batched by dataloader collate_fn)
    calib: CalibMatrices

    # Byte-tokenised prompt (B, L) — system_prompt + user_prompt.
    # None during inference when the model only uses visual/LiDAR context.
    text_bytes: Optional[torch.Tensor]

    # Training-only: target reasoning chain + answer as byte tokens (B, T_tgt)
    target_bytes: Optional[torch.Tensor] = None
    # True for positions that contribute to the S2FT loss (mask out prompt prefix)
    loss_mask:    Optional[torch.Tensor] = None


@dataclass
class KairosOutput:
    """Outputs for one forward pass."""

    # S2FT language modelling
    logits:    Optional[torch.Tensor] = None   # (B, T_tgt, vocab)  — training
    generated: Optional[torch.Tensor] = None   # (B, T_gen)         — inference

    # Per-component losses (training only; None during inference)
    s2ft_loss:   Optional[torch.Tensor] = None
    det_loss:    Optional[torch.Tensor] = None
    moe_z_loss:  Optional[torch.Tensor] = None
    total_loss:  Optional[torch.Tensor] = None

    # Detection (when detection head is enabled)
    det_boxes:  Optional[torch.Tensor] = None   # (B, max_det, 7)
    det_scores: Optional[torch.Tensor] = None   # (B, max_det, n_cls)

    # Debug / visualisation
    hidden: Optional[torch.Tensor] = None       # (B, T, d) hybrid core output


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Byte text encoder
# ──────────────────────────────────────────────────────────────────────────────

class _TextEncoderLayer(nn.Module):
    """One transformer encoder layer for the byte text encoder."""

    def __init__(self, d: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm_sa  = RMSNorm(d)
        self.norm_ffn = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.o.weight, std=d ** -0.5)

        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up   = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d,  bias=False)
        nn.init.normal_(self.down.weight, std=d ** -0.5)

        self.n_heads  = n_heads
        self.head_dim = d // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        H, hd = self.n_heads, self.head_dim

        # Self-attention (bidirectional — encoder, no causal mask)
        xn = self.norm_sa(x)
        Q = self.q(xn).view(B, T, H, hd).transpose(1, 2)
        K = self.k(xn).view(B, T, H, hd).transpose(1, 2)
        V = self.v(xn).view(B, T, H, hd).transpose(1, 2)
        attn = F.scaled_dot_product_attention(Q, K, V)
        x = x + self.o(attn.transpose(1, 2).reshape(B, T, d))

        # SwiGLU FFN
        xn = self.norm_ffn(x)
        x = x + self.down(F.silu(self.gate(xn)) * self.up(xn))
        return x


class ByteTextEncoder(nn.Module):
    """
    Byte tokenizer: raw byte sequence (B, L) → 8 learnable query tokens (B, 8, d).

    Architecture:
      1. Byte embedding: 258-vocab (256 bytes + BOS + EOS) → d
      2. 2 transformer encoder layers (bidirectional attention over all bytes)
      3. 8 learnable query tokens compress the byte sequence via cross-attention
         (Perceiver-style: same cross-attention as TemporalFusionBlock)

    For long prompts (system_prompt + user_prompt ~ 800 bytes), this efficiently
    distils the question into 8 dense tokens that enter the hybrid block.
    """

    def __init__(
        self,
        d: int,
        vocab: int = 258,
        n_enc_layers: int = 8,   # 8 layers → ~106M at d=1024
        n_heads: int = 8,
        n_queries: int = 8,
        d_ff: int = 2730,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, d)
        self.pos_emb   = nn.Embedding(max_len, d)  # learnable positional
        self.max_len   = max_len

        self.enc_layers = nn.ModuleList(
            [_TextEncoderLayer(d, n_heads, d_ff) for _ in range(n_enc_layers)]
        )

        # Learnable query tokens that compress the byte sequence
        self.query_tokens = nn.Parameter(torch.randn(1, n_queries, d) * d ** -0.5)

        # Cross-attention: queries attend to the encoded bytes
        self.xattn_norm_q  = RMSNorm(d)
        self.xattn_norm_kv = RMSNorm(d)
        self.xq = nn.Linear(d, d, bias=False)
        self.xk = nn.Linear(d, d, bias=False)
        self.xv = nn.Linear(d, d, bias=False)
        self.xo = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.xo.weight, std=d ** -0.5)

        self.out_norm = RMSNorm(d)
        self.n_heads  = n_heads
        self.head_dim = d // n_heads

    def _cross_attn(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, nq, d = q.shape
        nkv = kv.shape[1]
        H, hd = self.n_heads, self.head_dim
        Q = self.xq(q) .view(B, nq,  H, hd).transpose(1, 2)
        K = self.xk(kv).view(B, nkv, H, hd).transpose(1, 2)
        V = self.xv(kv).view(B, nkv, H, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        return self.xo(out.transpose(1, 2).reshape(B, nq, d))

    def forward(
        self,
        byte_ids: Optional[torch.Tensor],   # (B, L) int64 | None
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Returns (B, n_queries, d) query tokens."""
        # Fallback when no text query is provided: return learned query tokens.
        if byte_ids is None:
            if batch_size is None:
                raise ValueError("batch_size is required when byte_ids is None")
            q = self.query_tokens.expand(batch_size, -1, -1)
            return self.out_norm(q)

        B, L = byte_ids.shape
        L_trunc = min(L, self.max_len)
        ids = byte_ids[:, :L_trunc]

        # Byte embedding + positional
        pos = torch.arange(L_trunc, device=ids.device).unsqueeze(0)
        x = self.embedding(ids) + self.pos_emb(pos)     # (B, L_trunc, d)

        # 2-layer bidirectional encoder
        for layer in self.enc_layers:
            x = layer(x)

        # 8 learnable queries compress the encoded bytes
        q  = self.query_tokens.expand(B, -1, -1)         # (B, 8, d)
        q  = self.xattn_norm_q(q)
        kv = self.xattn_norm_kv(x)
        q  = q + self._cross_attn(q, kv)                 # (B, 8, d)

        return self.out_norm(q)


# ──────────────────────────────────────────────────────────────────────────────
# S2FT decoder
# ──────────────────────────────────────────────────────────────────────────────

class _S2FTDecoderLayer(nn.Module):
    """
    One causal decoder layer:
      1. Causal self-attention over generated tokens
      2. Cross-attention to scene memory (KairosHybridCore output)
      3. SwiGLU FFN

    Pre-norm (RMSNorm before each sub-block) for training stability.
    """

    def __init__(self, d: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d // n_heads

        self.norm_sa   = RMSNorm(d)
        self.norm_ca   = RMSNorm(d)
        self.norm_kv   = RMSNorm(d)
        self.norm_ffn  = RMSNorm(d)

        # Self-attention projections
        self.sq = nn.Linear(d, d, bias=False)
        self.sk = nn.Linear(d, d, bias=False)
        self.sv = nn.Linear(d, d, bias=False)
        self.so = nn.Linear(d, d, bias=False)

        # Cross-attention projections
        self.cq = nn.Linear(d, d, bias=False)
        self.ck = nn.Linear(d, d, bias=False)
        self.cv = nn.Linear(d, d, bias=False)
        self.co = nn.Linear(d, d, bias=False)

        # SwiGLU FFN
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up   = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

        nn.init.normal_(self.so.weight, std=d ** -0.5)
        nn.init.normal_(self.co.weight, std=d ** -0.5)
        nn.init.normal_(self.down.weight, std=d ** -0.5)

    def _mha(
        self,
        q_proj, k_proj, v_proj, o_proj,
        q: torch.Tensor,
        kv: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        B, nq, d = q.shape
        nkv = kv.shape[1]
        H, hd = self.n_heads, self.head_dim
        Q = q_proj(q) .view(B, nq,  H, hd).transpose(1, 2)
        K = k_proj(kv).view(B, nkv, H, hd).transpose(1, 2)
        V = v_proj(kv).view(B, nkv, H, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
        return o_proj(out.transpose(1, 2).reshape(B, nq, d))

    def forward(
        self,
        x:      torch.Tensor,   # (B, T_dec, d) — decoder sequence
        memory: torch.Tensor,   # (B, T_mem, d) — scene representation
    ) -> torch.Tensor:
        # 1. Causal self-attention
        xn = self.norm_sa(x)
        x = x + self._mha(self.sq, self.sk, self.sv, self.so,
                           xn, xn, causal=True)

        # 2. Cross-attention to scene memory
        xn  = self.norm_ca(x)
        mem = self.norm_kv(memory)
        x = x + self._mha(self.cq, self.ck, self.cv, self.co,
                           xn, mem, causal=False)

        # 3. SwiGLU FFN
        xn = self.norm_ffn(x)
        x = x + self.down(F.silu(self.gate(xn)) * self.up(xn))
        return x


class S2FTDecoder(nn.Module):
    """
    System-2 Fine-Tuning decoder.

    Generates the chain-of-thought reasoning chain + final answer autoregressively
    at byte level, conditioned on the scene representation from KairosHybridCore.

    Training (teacher forcing):
        logits = decoder(memory, target_bytes[:, :-1])   # (B, T-1, vocab)
        loss   = CE(logits[loss_mask], target_bytes[:, 1:][loss_mask])

    Inference (greedy / sampling):
        tokens = decoder.generate(memory, prompt_ids, max_len=512)
    """

    def __init__(
        self,
        d: int,
        vocab: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_len: int,
        bos_id: int,
        eos_id: int,
        tie_weights: bool = True,
        use_grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_len = max_len
        self.use_grad_checkpoint = use_grad_checkpoint

        self.embedding = nn.Embedding(vocab, d)
        self.pos_emb   = nn.Embedding(max_len, d)

        self.layers = nn.ModuleList(
            [_S2FTDecoderLayer(d, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm_out = RMSNorm(d)
        self.lm_head  = nn.Linear(d, vocab, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedding.weight

        nn.init.normal_(self.embedding.weight, std=d ** -0.5)
        # Zero-init LM head logit bias (if untied, embedding init handles it)

    # ------------------------------------------------------------------
    def forward(
        self,
        memory: torch.Tensor,          # (B, T_mem, d) — hybrid core output
        decoder_input: torch.Tensor,   # (B, T_dec) byte ids — teacher-forced input
    ) -> torch.Tensor:
        """Returns logits (B, T_dec, vocab) for computing the S2FT loss."""
        B, T_dec = decoder_input.shape
        pos = torch.arange(T_dec, device=decoder_input.device).unsqueeze(0)
        x   = self.embedding(decoder_input) + self.pos_emb(pos)    # (B, T_dec, d)

        for layer in self.layers:
            if self.training and self.use_grad_checkpoint:
                x = _grad_ckpt(layer, x, memory, use_reentrant=False)
            else:
                x = layer(x, memory)

        return self.lm_head(self.norm_out(x))                       # (B, T_dec, vocab)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,          # (B, T_mem, d) — scene context
        prompt_ids: torch.Tensor,      # (B, L_prompt) byte ids
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-p (nucleus) sampling.
        Returns (B, T_gen) byte token IDs.  Stops at EOS or max_gen_len.
        """
        B = memory.shape[0]
        ids = prompt_ids.clone()
        done = torch.zeros(B, dtype=torch.bool, device=memory.device)

        for _ in range(self.max_len):
            logits = self.forward(memory, ids)[:, -1, :]   # (B, vocab)

            if temperature > 0:
                logits = logits / temperature
                probs  = F.softmax(logits, dim=-1)
                # Top-p filtering
                sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
                cum_p = sorted_p.cumsum(dim=-1)
                remove = (cum_p - sorted_p) > top_p
                sorted_p[remove] = 0.0
                sorted_p /= sorted_p.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                next_tok = sorted_idx.gather(-1, torch.multinomial(sorted_p, 1))
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)

            ids = torch.cat([ids, next_tok], dim=1)
            done |= (next_tok.squeeze(-1) == self.eos_id)
            if done.all():
                break

        return ids


# ──────────────────────────────────────────────────────────────────────────────
# Object detection head
# ──────────────────────────────────────────────────────────────────────────────

class ObjectDetectionHead(nn.Module):
    """
    DETR-style 3D object detector on the fused scene representation.

    max_det learnable object queries cross-attend to the camera + LiDAR tokens
    (first n_cam + n_lidar = 320 positions), then MLP heads predict:
      - 3D box: (cx, cy, cz, l, w, h, yaw)  — 7 values per object
      - class:  n_det_cls + 1 (background)  — softmax logits

    Loss (when targets provided): Hungarian matching + L1 box + CE class.
    Loss computation is left to the training script for flexibility.
    """

    def __init__(
        self,
        d: int,
        n_det_cls: int,
        max_det: int,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.max_det   = max_det
        self.n_det_cls = n_det_cls

        # Learnable object queries
        self.queries = nn.Parameter(torch.randn(1, max_det, d) * d ** -0.5)

        # Cross-attention to scene tokens
        self.norm_q  = RMSNorm(d)
        self.norm_kv = RMSNorm(d)
        self.q_proj  = nn.Linear(d, d, bias=False)
        self.k_proj  = nn.Linear(d, d, bias=False)
        self.v_proj  = nn.Linear(d, d, bias=False)
        self.o_proj  = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.o_proj.weight, std=d ** -0.5)

        self.n_heads  = n_heads
        self.head_dim = d // n_heads

        # Prediction heads
        self.box_head = nn.Sequential(
            RMSNorm(d),
            nn.Linear(d, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, 7),      # cx, cy, cz, l, w, h, yaw
        )
        self.cls_head = nn.Sequential(
            RMSNorm(d),
            nn.Linear(d, n_det_cls + 1),
        )

    def forward(
        self,
        scene: torch.Tensor,   # (B, T, d) — hybrid core output
        n_scene_tokens: int,   # attend only to first n_scene_tokens (cam+lidar)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns boxes (B, max_det, 7), scores (B, max_det, n_cls+1)."""
        B = scene.shape[0]
        H, hd = self.n_heads, self.head_dim
        n_q = self.max_det

        kv = scene[:, :n_scene_tokens]   # (B, 320, d)

        q_tok  = self.norm_q (self.queries.expand(B, -1, -1))
        kv_tok = self.norm_kv(kv)

        Q = self.q_proj(q_tok) .view(B, n_q,             H, hd).transpose(1, 2)
        K = self.k_proj(kv_tok).view(B, n_scene_tokens,  H, hd).transpose(1, 2)
        V = self.v_proj(kv_tok).view(B, n_scene_tokens,  H, hd).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(Q, K, V)
        obj = self.o_proj(attn_out.transpose(1, 2).reshape(B, n_q, -1))   # (B,max_det,d)

        boxes  = self.box_head(obj)     # (B, max_det, 7)
        scores = self.cls_head(obj)     # (B, max_det, n_cls+1)
        return boxes, scores


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────

class KairosModel(nn.Module):
    """
    Kairos-4B — full model assembly.

    Sub-module layout:
      vision_encoder   KairosVisionEncoder   (DINOv2-L + LoRA + temporal fusion)
      lidar_encoder    PointMambaEncoder
      imu_encoder      IMUEncoder
      text_encoder     ByteTextEncoder
      calib_gate       KairosCalibrationGate
      hybrid_core      KairosHybridCore
      s2ft_decoder     S2FTDecoder
      det_head         ObjectDetectionHead

    Checkpointing:
      Only the trainable sub-modules are saved.
      DINOv2 backbone weights are reloaded from HuggingFace each time.
      Call model.trainable_state_dict() for the training loop checkpoint.

    Distributed training:
      Compatible with DDP and DeepSpeed ZeRO-3.
      After optimizer.step(), call sync_moe_expert_bias(model) to all-reduce
      the non-gradient expert load-balance biases.
    """

    def __init__(self, cfg: Optional[KairosModelConfig] = None) -> None:
        super().__init__()
        cfg = cfg or KairosModelConfig()
        self.cfg = cfg
        self._validate_config(cfg)
        d = cfg.kcfg.d_model

        # Single canonical layout — used for all slice/offset computations
        self.layout = TokenLayout(cfg.n_cam, cfg.n_lidar, cfg.n_imu, cfg.n_query)
        if self.layout.total != 336:
            print(
                f"[layout] TokenLayout total={self.layout.total} "
                f"(full KITTI is 336); non-standard config",
                flush=True,
            )

        # ── Encoders ────────────────────────────────────────────────────────────
        self.vision_encoder = KairosVisionEncoder(cfg.vcfg, cfg.kcfg)
        self.lidar_encoder  = PointMambaEncoder(d_model=d, n_tokens=cfg.n_lidar,
                                                 cfg=cfg.lidar_cfg)
        self.imu_encoder    = IMUEncoder(d_model=d, n_tokens=cfg.n_imu,
                                          cfg=cfg.imu_cfg)
        self.text_encoder   = ByteTextEncoder(
            d=d, vocab=cfg.byte_vocab,
            n_queries=cfg.n_query,
            n_enc_layers=cfg.n_text_enc_layers,
            d_ff=cfg.kcfg.d_ff,
        )

        # ── Fusion gate ──────────────────────────────────────────────────────────
        self.calib_gate = KairosCalibrationGate(cfg.fcfg, cfg.kcfg)

        # ── Hybrid core ──────────────────────────────────────────────────────────
        self.hybrid_core = KairosHybridCore(cfg.kcfg)

        # ── Task heads ───────────────────────────────────────────────────────────
        # Propagate checkpoint flag to hybrid core and S2FT decoder
        cfg.kcfg.use_grad_checkpoint = cfg.use_grad_checkpoint

        self.s2ft_decoder = S2FTDecoder(
            d=d, vocab=cfg.byte_vocab,
            n_layers=cfg.decoder_layers, n_heads=cfg.decoder_heads,
            d_ff=cfg.kcfg.d_ff, max_len=cfg.max_gen_len,
            bos_id=cfg.bos_id, eos_id=cfg.eos_id,
            use_grad_checkpoint=cfg.use_grad_checkpoint,
        )
        self.det_head = ObjectDetectionHead(
            d=d, n_det_cls=cfg.n_det_cls, max_det=cfg.max_det,
        )

        # Plumbing-only loss anchor — a tiny trainable scalar that provides a
        # valid DeepSpeed backward path when _smoke_skip_decoder_loss=True,
        # completely disconnected from the hybrid_core/MoE expert GEMM graph.
        # Falls into the 'decoder' param_groups() bucket (no special name match).
        self.smoke_loss_anchor = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _validate_config(cfg: KairosModelConfig) -> None:
        patch = cfg.vcfg.dinov2_patch_size
        if cfg.vcfg.enc_h % patch != 0 or cfg.vcfg.enc_w % patch != 0:
            raise ValueError(
                f"Vision encoder size ({cfg.vcfg.enc_h}, {cfg.vcfg.enc_w}) "
                f"must be divisible by patch size {patch}"
            )
        patch_rows = cfg.vcfg.enc_h // patch
        patch_cols = cfg.vcfg.enc_w // patch
        n_patches = patch_rows * patch_cols
        checks = {
            "vcfg.n_patches": (cfg.vcfg.n_patches, n_patches),
            "cfg.n_cam": (cfg.n_cam, n_patches),
            "fcfg.n_cam_tokens": (cfg.fcfg.n_cam_tokens, n_patches),
            "fcfg.patch_rows": (cfg.fcfg.patch_rows, patch_rows),
            "fcfg.patch_cols": (cfg.fcfg.patch_cols, patch_cols),
            "fcfg.enc_h": (cfg.fcfg.enc_h, cfg.vcfg.enc_h),
            "fcfg.enc_w": (cfg.fcfg.enc_w, cfg.vcfg.enc_w),
            "cfg.n_lidar": (cfg.n_lidar, cfg.lidar_cfg.n_tokens),
            "fcfg.n_lidar_tokens": (cfg.fcfg.n_lidar_tokens, cfg.n_lidar),
            "fcfg.n_imu_tokens": (cfg.fcfg.n_imu_tokens, cfg.n_imu),
            "fcfg.n_query_tokens": (cfg.fcfg.n_query_tokens, cfg.n_query),
            "fcfg.total_tokens": (
                cfg.fcfg.total_tokens,
                cfg.n_cam + cfg.n_lidar + cfg.n_imu + cfg.n_query,
            ),
            "imu_cfg.n_tokens": (cfg.imu_cfg.n_tokens, cfg.n_imu),
        }
        mismatches = [
            f"{name}={actual} expected {expected}"
            for name, (actual, expected) in checks.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError("KairosModelConfig shape mismatch: " + "; ".join(mismatches))

    # ── Internal helper ────────────────────────────────────────────────────────

    def _build_sequence_masks(
        self,
        cam:      torch.Tensor,   # (B, 256, d)
        lidar:    torch.Tensor,   # (B, 64, d)
        imu:      torch.Tensor,   # (B, N_imu, d)
        query:    torch.Tensor,   # (B, 8, d)
        imu_dt:   torch.Tensor,   # (B, N_imu) — Δt per IMU token
        skip_core_imu: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct (imu_mask, delta_t_seq) in the full-sequence layout:
          [cam | lidar | imu | query]
          [  0 |   256 | 320 | 328  ]   (default start indices)
        """
        B    = cam.shape[0]
        N_imu = imu.shape[1]
        T     = cam.shape[1] + lidar.shape[1] + N_imu + query.shape[1]

        imu_mask  = cam.new_zeros(B, T, dtype=torch.bool)
        delta_t   = cam.new_zeros(B, T)

        imu_start = self.cfg.fcfg.imu_start
        imu_end   = self.cfg.fcfg.imu_end
        if imu_end - imu_start != N_imu:
            raise RuntimeError(
                f"IMU layout mismatch: fused IMU span is {imu_end - imu_start} "
                f"tokens but imu_tokens has {N_imu}"
            )
        if skip_core_imu:
            if int(_os.environ.get("RANK", "0")) == 0:
                print(
                    "[ultra_smoke] skip_imu=True -> hybrid core CfC disabled; "
                    "imu_mask true count=0",
                    flush=True,
                )
            return imu_mask, delta_t

        imu_mask[:, imu_start:imu_end] = True
        # Explicit cast: imu_dt may be float32 if the IMU encoder ran before
        # the model-level target_dtype normalisation, or if any autocast op
        # promoted it.  _as_like guarantees same dtype as delta_t (BF16).
        delta_t[:, imu_start:imu_end] = _as_like(imu_dt, delta_t)

        return imu_mask, delta_t

    @staticmethod
    def _match_imu_delta_t(
        imu_tokens: torch.Tensor,
        imu_delta_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return IMU delta_t with shape (B, N_imu), creating/padding as needed."""
        B, N_imu, _ = imu_tokens.shape
        default_dt = 1.0 / 30.0
        if imu_delta_t is None:
            return torch.full(
                (B, N_imu),
                default_dt,
                device=imu_tokens.device,
                dtype=imu_tokens.dtype,
            )

        imu_delta_t = imu_delta_t.to(device=imu_tokens.device, dtype=imu_tokens.dtype)
        if imu_delta_t.dim() == 3 and imu_delta_t.shape[-1] == 1:
            imu_delta_t = imu_delta_t.squeeze(-1)
        if imu_delta_t.dim() != 2 or imu_delta_t.shape[0] != B:
            raise ValueError(
                f"imu_delta_t must have shape ({B}, N_imu), got {tuple(imu_delta_t.shape)}"
            )
        if imu_delta_t.shape[1] == N_imu:
            return imu_delta_t
        if imu_delta_t.shape[1] > N_imu:
            print(
                f"[warn] imu_delta_t has {imu_delta_t.shape[1]} steps but imu_tokens "
                f"has {N_imu}; trimming delta_t",
                flush=True,
            )
            return imu_delta_t[:, :N_imu]

        print(
            f"[warn] imu_delta_t has {imu_delta_t.shape[1]} steps but imu_tokens "
            f"has {N_imu}; padding delta_t",
            flush=True,
        )
        pad = torch.full(
            (B, N_imu - imu_delta_t.shape[1]),
            default_dt,
            device=imu_tokens.device,
            dtype=imu_tokens.dtype,
        )
        return torch.cat([imu_delta_t, pad], dim=1)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, batch: KairoBatch) -> KairosOutput:
        """
        Full forward pass.  Handles both training (with target_bytes) and
        inference (without).  Returns KairosOutput with populated fields.
        """
        cfg = self.cfg
        d   = cfg.kcfg.d_model
        B   = batch.img_t.shape[0]

        # ── 1. Encode all modalities ───────────────────────────────────────────
        # no_grad_vision: set when ultra_smoke_dino_no_grad=True; saves DINO
        # activation memory and removes LoRA grad overhead for plumbing-only smoke.
        if cfg.no_grad_vision:
            with torch.no_grad():
                cam = self.vision_encoder(batch.img_t, batch.img_t1, batch.img_t2)
        else:
            cam = self.vision_encoder(batch.img_t, batch.img_t1, batch.img_t2)
        _trace_mem("after_vision_encoder")
        if cam.shape != (B, cfg.n_cam, d):
            raise ValueError(
                f"VisionEncoder output: expected ({B}, {cfg.n_cam}, {d}), "
                f"got {tuple(cam.shape)}"
            )

        # _smoke_skip_lidar / _smoke_no_grad_lidar: set by kairos_train.py on the
        # model instance before deepspeed.initialize() when the encoder is frozen.
        _skip_lidar    = getattr(self, '_smoke_skip_lidar',    False)
        _no_grad_lidar = getattr(self, '_smoke_no_grad_lidar', False)
        if _skip_lidar:
            lidar     = torch.zeros(B, cfg.n_lidar, d, device=cam.device, dtype=cam.dtype)
            lidar_xyz = torch.zeros(B, cfg.n_lidar, 3, device=cam.device, dtype=cam.dtype)
        elif _no_grad_lidar:
            with torch.no_grad():
                lidar, lidar_xyz = self.lidar_encoder(batch.lidar_t, batch.lidar_t1)
        else:
            lidar, lidar_xyz = self.lidar_encoder(batch.lidar_t, batch.lidar_t1)
        _trace_mem("after_lidar_encoder")
        if lidar.shape != (B, cfg.n_lidar, d):
            raise ValueError(
                f"LiDAREncoder output: expected ({B}, {cfg.n_lidar}, {d}), "
                f"got {tuple(lidar.shape)}"
            )

        _skip_imu    = getattr(self, '_smoke_skip_imu',    False)
        _no_grad_imu = getattr(self, '_smoke_no_grad_imu', False)
        if _skip_imu:
            imu    = torch.zeros(B, cfg.n_imu, d, device=cam.device, dtype=cam.dtype)
            imu_dt = torch.zeros(B, cfg.n_imu,    device=cam.device, dtype=cam.dtype)
        elif _no_grad_imu:
            with torch.no_grad():
                imu, imu_dt = self.imu_encoder(batch.imu_data, batch.imu_timestamps)
        else:
            imu, imu_dt = self.imu_encoder(batch.imu_data, batch.imu_timestamps)
        _trace_mem("after_imu_encoder")
        if imu.shape[0] != B or imu.shape[2] != d:
            raise ValueError(
                f"IMUEncoder output: expected (B={B}, *, d={d}), "
                f"got {tuple(imu.shape)}"
            )
        if imu.shape[1] != cfg.n_imu:
            print(
                f"[warn] IMU token count {imu.shape[1]} does not match cfg.n_imu="
                f"{cfg.n_imu}; padding/trimming IMU tokens",
                flush=True,
            )
            if imu.shape[1] == 0:
                imu = torch.zeros(B, cfg.n_imu, d, device=cam.device, dtype=cam.dtype)
            elif imu.shape[1] < cfg.n_imu:
                pad = torch.zeros(
                    B,
                    cfg.n_imu - imu.shape[1],
                    d,
                    device=imu.device,
                    dtype=imu.dtype,
                )
                imu = torch.cat([imu, pad], dim=1)
            else:
                imu = imu[:, :cfg.n_imu]
        imu_dt = self._match_imu_delta_t(imu, imu_dt)

        query = self.text_encoder(batch.text_bytes, batch_size=B)
        _trace_mem("after_text_encoder")
        if query.shape != (B, cfg.n_query, d):
            raise ValueError(
                f"TextEncoder output: expected ({B}, {cfg.n_query}, {d}), "
                f"got {tuple(query.shape)}"
            )

        target_dtype = cam.dtype
        target_device = cam.device
        lidar = lidar.to(device=target_device, dtype=target_dtype)
        lidar_xyz = lidar_xyz.to(device=target_device, dtype=target_dtype)
        imu = imu.to(device=target_device, dtype=target_dtype)
        imu_dt = imu_dt.to(device=target_device, dtype=target_dtype)
        query = query.to(device=target_device, dtype=target_dtype)
        calib = batch.calib.to(dtype=target_dtype, device=target_device)

        _dtype_log(
            "post_encode",
            cam=cam, lidar=lidar, lidar_xyz=lidar_xyz,
            imu=imu, imu_dt=imu_dt, query=query,
        )
        _assert_same_dtype(
            "post_encode", cam,
            lidar=lidar, lidar_xyz=lidar_xyz,
            imu=imu, imu_dt=imu_dt, query=query,
        )

        _shape_log("[shape] cam_tokens", cam.shape)
        _shape_log("[shape] lidar_tokens", lidar.shape)
        _shape_log("[shape] imu_tokens", imu.shape)
        _shape_log("[shape] query_tokens", query.shape)
        _shape_log("[shape] imu_delta_t", imu_dt.shape)

        # ── 2. Calibration-aware fusion gate ──────────────────────────────────
        x = self.calib_gate(cam, lidar, imu, query, lidar_xyz, calib)
        _trace_mem("after_calib_gate")
        expected_T = cfg.n_cam + cfg.n_lidar + cfg.n_imu + cfg.n_query
        if x.shape[1] != expected_T:
            raise RuntimeError(
                f"Bad fused token length: got {x.shape[1]}, expected {expected_T}. "
                f"cam={cam.shape}, lidar={lidar.shape}, imu={imu.shape}, query={query.shape}"
            )
        if x.shape != (B, expected_T, d):
            raise RuntimeError(
                f"FusionGate output: expected ({B}, {expected_T}, {d}), "
                f"got {tuple(x.shape)}"
            )
        _shape_log("[shape] fused_tokens", x.shape)

        # Detect and reject old 328-token layout before hybrid core.
        # This layout had [cam256|lidar64|query8] without IMU tokens and would
        # produce (B,0,d) imu_x when the CfC block extracts imu_slice.
        if x.shape[1] == 328:
            raise RuntimeError(
                "[layout] Old 328-token fused layout detected: "
                "[cam256|lidar64|query8] — IMU tokens are missing. "
                f"Expected {self.layout.total} = "
                f"cam{cfg.n_cam}+lidar{cfg.n_lidar}+imu{cfg.n_imu}+query{cfg.n_query}. "
                "Ensure n_imu is set consistently across KairosModelConfig, "
                "FusionConfig, and IMUEncoderConfig."
            )

        _shape_log(
            f"[shape/core] expected_total={self.layout.total}  "
            f"imu_start={self.layout.imu_start}  imu_end={self.layout.imu_end}  "
            f"imu_delta_t={imu_dt.shape}"
        )

        # ── 3. Build IMU masks for hybrid core ────────────────────────────────
        imu_mask, delta_t_seq = self._build_sequence_masks(
            cam, lidar, imu, query, imu_dt, skip_core_imu=_skip_imu
        )

        # Guarantee delta_t_seq matches the fused sequence dtype/device before
        # it enters KairosHybridCore → CfCBlock where it drives gate arithmetic.
        # _build_sequence_masks uses cam.new_zeros() so it should already match,
        # but an explicit cast here is the final safety net.
        delta_t_seq = delta_t_seq.to(device=x.device, dtype=x.dtype)
        _dtype_log(
            "pre_core",
            x=x, delta_t_seq=delta_t_seq,
        )
        _assert_same_dtype("pre_core", x, delta_t_seq=delta_t_seq)

        # ── 4. Hybrid core (Mamba + CfC + SWA + MoE) ─────────────────────────
        _skip_core = getattr(self, '_smoke_skip_core', False)
        if _skip_core:
            # Plumbing-only: bypass the entire hybrid core.  Use together with
            # _smoke_skip_decoder_loss=True so the anchor loss provides the only
            # backward path — no MoE sparse dispatch, no CfC, no Mamba backward.
            if int(_os.environ.get("RANK", "0")) == 0:
                print(
                    "[ultra_smoke] skip_core=True -> hybrid core bypassed",
                    flush=True,
                )
            moe_z = None
        else:
            x = self.hybrid_core(x, imu_mask, delta_t_seq)
            _trace_mem("after_hybrid_core")
            # x: (B, T, d);  hybrid_core._z_loss_for_backward set during training
            moe_z = self.hybrid_core._z_loss_for_backward if self.training else None

        # ── 5. Task heads ─────────────────────────────────────────────────────

        # ── 5a. Object detection ──────────────────────────────────────────────
        det_boxes, det_scores = None, None
        if (not self.training) or cfg.w_det > 0:
            n_scene = cam.shape[1] + lidar.shape[1]   # attend only to cam+lidar tokens
            det_boxes, det_scores = self.det_head(x, n_scene_tokens=n_scene)
            _trace_mem("after_det_head")

        # ── 5b. S2FT reasoning decoder ────────────────────────────────────────
        logits    = None
        s2ft_loss = None

        _skip_dec_loss = getattr(self, '_smoke_skip_decoder_loss', False)

        if batch.target_bytes is not None and _skip_dec_loss:
            # Plumbing-only loss: anchor-only, completely disconnected from the
            # hybrid_core/MoE expert GEMM backward graph.  Using x.pow(2) here
            # would still traverse the sparse MoE dispatch backward under ZeRO-3,
            # causing shape mismatches when expert token counts vary across ranks.
            anchor = self.smoke_loss_anchor.float()
            s2ft_loss = anchor * 0.0 + 1e-8 * anchor.pow(2)
            _trace_mem("after_dummy_decoder_loss")
        elif batch.target_bytes is not None:
            if batch.loss_mask is not None:
                if batch.loss_mask.shape != batch.target_bytes.shape:
                    raise ValueError(
                        f"loss_mask shape {tuple(batch.loss_mask.shape)} must match "
                        f"target_bytes shape {tuple(batch.target_bytes.shape)}"
                    )

            # Teacher forcing: input = BOS + target[:-1]; label = target
            dec_input = batch.target_bytes[:, :-1].clamp(min=0)  # PAD/ignore -> PAD
            dec_label = batch.target_bytes[:, 1:].clone()        # (B, T_tgt-1)
            if batch.loss_mask is not None:
                dec_label = dec_label.masked_fill(~batch.loss_mask[:, 1:].bool(), -1)
            logits    = self.s2ft_decoder(x, dec_input)      # (B, T_tgt-1, vocab)
            _trace_mem("after_s2ft_decoder")

            loss_flat = F.cross_entropy(
                logits.reshape(-1, cfg.byte_vocab),
                dec_label.reshape(-1),
                ignore_index=-1,
                label_smoothing=0.1,
                reduction="none",
            )
            if batch.loss_mask is not None:
                mask = batch.loss_mask[:, 1:].reshape(-1).float()
                s2ft_loss = (loss_flat * mask).sum() / mask.sum().clamp(min=1)
            else:
                valid = dec_label.reshape(-1) != -1
                s2ft_loss = loss_flat[valid].mean() if valid.any() else loss_flat.mean()

        # ── 5c. Detection loss (optional) ────────────────────────────────────
        # Hungarian matching + regression + CE — implemented in training script
        det_loss = None   # training script adds this via a DetectionLoss module

        # ── 6. Aggregate total loss ───────────────────────────────────────────
        _ultra_core_loss = getattr(self, '_ultra_smoke_core_loss', False)

        if _skip_dec_loss and not _ultra_core_loss:
            # Plumbing-only: anchor dummy is the sole loss source.
            # Deliberately excludes moe_z_loss and det_loss — both traverse the
            # hybrid_core backward graph.  Under ZeRO-3, sparse MoE dispatch can
            # produce different expert token counts per rank, causing shape errors
            # when ZeRO-3 tries to reduce-scatter the resulting gradients.
            total_loss = s2ft_loss  # set to anchor dummy above; None if no targets
            if total_loss is not None and int(_os.environ.get("RANK", "0")) == 0:
                print(
                    "[ultra_smoke] skip_decoder_loss=True -> total_loss uses "
                    "smoke_loss_anchor only; core graph (moe_z, det_loss) detached",
                    flush=True,
                )
        elif _ultra_core_loss and not _skip_core:
            # Safe core backward test: use x.pow(2).mean() as total loss.
            # This flows backward through the hybrid core — use only with
            # dense_moe_fallback=True (in cfg.kcfg) to avoid ZeRO-3 sparse dispatch
            # shape mismatches.  Decoder CE and moe_z_loss are excluded.
            total_loss = x.float().pow(2).mean() * 1e-4
            if int(_os.environ.get("RANK", "0")) == 0:
                print(
                    "[ultra_smoke] ultra_smoke_core_loss=True -> "
                    "total_loss = x.pow(2).mean()*1e-4 (safe core backward test); "
                    "decoder CE and moe_z excluded",
                    flush=True,
                )
        else:
            total = x.new_zeros(())
            has_loss = False
            if s2ft_loss is not None:
                total = total + cfg.w_s2ft * s2ft_loss
                has_loss = True
            if det_loss is not None and cfg.w_det > 0:
                total = total + cfg.w_det * det_loss
                has_loss = True
            # Z-loss included here so the training script can use output.total_loss
            # directly and z-loss gradients are guaranteed to reach router_proj.
            if moe_z is not None and cfg.z_loss_coeff > 0:
                total = total + cfg.z_loss_coeff * moe_z
                has_loss = True
            total_loss = total if has_loss else None

        keep_debug = (not self.training) or cfg.return_debug_tensors

        return KairosOutput(
            logits     = logits if keep_debug else None,
            s2ft_loss  = s2ft_loss,
            det_loss   = det_loss,
            moe_z_loss = moe_z,
            total_loss = total_loss,
            det_boxes  = det_boxes,
            det_scores = det_scores,
            hidden     = x if keep_debug else None,
        )

    # ── Generation ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        batch: KairoBatch,
        prompt_ids: Optional[torch.Tensor] = None,   # (B, L) byte ids | None → BOS
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> KairosOutput:
        """Inference: encode scene, generate reasoning chain + answer."""
        self.eval()
        out = self.forward(batch)   # populates out.hidden, out.det_boxes, out.det_scores
        B = out.hidden.shape[0]

        if prompt_ids is None:
            prompt_ids = torch.full(
                (B, 1), self.cfg.bos_id,
                dtype=torch.long, device=out.hidden.device
            )

        generated = self.s2ft_decoder.generate(
            out.hidden, prompt_ids, temperature=temperature, top_p=top_p
        )
        return KairosOutput(
            generated  = generated,
            det_boxes  = out.det_boxes,
            det_scores = out.det_scores,
            hidden     = out.hidden,
        )

    # ── Parameter management ────────────────────────────────────────────────────

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Checkpoint only trainable weights.
        Frozen DINOv2 backbone (~307M) is excluded — reloaded from HuggingFace.
        """
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in trainable}

    def load_trainable_state_dict(
        self, state: Dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        missing, unexpected = self.load_state_dict(
            {**self.state_dict(), **state}, strict=False
        )
        if strict and unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:5]}")

    def param_groups(
        self,
        lr_backbone_lora: float = 1e-4,
        lr_encoders:      float = 3e-4,
        lr_core:          float = 3e-4,
        lr_decoder:       float = 3e-4,
        weight_decay:     float = 0.01,
    ) -> List[Dict]:
        """
        Per-component learning rate groups for AdamW.

        Component LR rationale:
          backbone_lora  1e-4 — adapts pre-trained DINOv2; needs small LR
          encoders       3e-4 — LiDAR/IMU/text encoders train from scratch
          core           3e-4 — hybrid core: large, trains from scratch
          decoder        3e-4 — S2FT decoder: trains from scratch

        Usage:
          optimizer = AdamW(model.param_groups(...), weight_decay=0.01)
        """
        lora, encoders, core, decoder = [], [], [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                lora.append(param)
            elif any(k in name for k in ("lidar_encoder", "imu_encoder",
                                          "text_encoder", "vision_encoder",
                                          "calib_gate")):
                encoders.append(param)
            elif "hybrid_core" in name:
                core.append(param)
            else:
                decoder.append(param)

        groups = [
            {"params": lora,     "lr": lr_backbone_lora, "name": "backbone_lora"},
            {"params": encoders, "lr": lr_encoders,      "name": "encoders"},
            {"params": core,     "lr": lr_core,          "name": "hybrid_core"},
            {"params": decoder,  "lr": lr_decoder,       "name": "decoder"},
        ]
        seen: Dict[int, str] = {}
        for group in groups:
            unique_params = []
            for param in group["params"]:
                pid = id(param)
                assert pid not in seen, (
                    f"Trainable parameter appears in both {seen[pid]} "
                    f"and {group['name']} optimizer groups"
                )
                seen[pid] = group["name"]
                unique_params.append(param)
            group["params"] = unique_params

        trainable = {id(p) for p in self.parameters() if p.requires_grad}
        assert set(seen) == trainable, (
            f"param_groups() covers {len(seen)} trainable tensors, "
            f"expected {len(trainable)}"
        )
        return groups

    def count_params(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"total={total/1e6:.1f}M  "
            f"trainable={trainable/1e6:.1f}M  "
            f"frozen(DINOv2)={(total-trainable)/1e6:.1f}M"
        )


# ──────────────────────────────────────────────────────────────────────────────
# DDP / ZeRO-3 utility
# ──────────────────────────────────────────────────────────────────────────────

def sync_moe_expert_bias(model: KairosModel) -> None:
    """
    All-reduce the non-gradient MoE expert load-balance biases after each
    optimizer step.  Required for DDP / ZeRO-3 training where each rank
    sees a different data shard.

        optimizer.step()
        sync_moe_expert_bias(model)   # ← call here
        optimizer.zero_grad()

    No-op when torch.distributed is not initialised (single-GPU training).
    """
    import torch.distributed as dist
    import torch
    if not dist.is_initialized():
        return
    backend = dist.get_backend()
    m = model.module if hasattr(model, "module") else model
    for blk in m.hybrid_core.blocks:
        bias = blk.moe_ffn.expert_bias
        if backend == "nccl" and not bias.is_cuda:
            blk.moe_ffn.expert_bias.data = bias.data.to(
                torch.device("cuda", torch.cuda.current_device())
            )
            bias = blk.moe_ffn.expert_bias
        dist.all_reduce(bias, op=dist.ReduceOp.AVG)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    torch.manual_seed(0)

    cfg = KairosModelConfig()
    print("Building KairosModel ...")
    model = KairosModel(cfg)
    print(f"  {model.count_params()}")

    # ── Synthetic batch (KITTI-like dimensions) ────────────────────────────────
    B = 2
    batch = KairoBatch(
        img_t  = torch.rand(B, 3, 375, 1242),
        img_t1 = torch.rand(B, 3, 375, 1242),
        img_t2 = torch.rand(B, 3, 375, 1242),
        lidar_t  = torch.rand(B, 30_000, 4),
        lidar_t1 = torch.rand(B, 30_000, 4),
        imu_data       = torch.randn(B, 30, 7),      # 30 IMU samples at 30 Hz = 1s
        imu_timestamps = torch.arange(30).float().unsqueeze(0).repeat(B, 1) / 30.,
        calib = CalibMatrices(
            P2 = torch.tensor([[
                [7.215377e+02, 0., 6.095593e+02, 4.485728e+01],
                [0., 7.215377e+02, 1.728540e+02, 2.163791e-01],
                [0., 0., 1., 2.745884e-03],
            ]]).repeat(B, 1, 1),
            R0_rect = torch.eye(3).unsqueeze(0).repeat(B, 1, 1),
            Tr_velo_to_cam = torch.tensor([[
                [ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
                [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
            ]]).repeat(B, 1, 1),
        ),
        text_bytes = torch.randint(0, 256, (B, 80)),  # short prompt
        # Training targets
        target_bytes = torch.randint(0, 256, (B, 64)),
        loss_mask    = torch.ones(B, 64, dtype=torch.bool),
    )
    # Mask prompt prefix from loss (first 16 bytes)
    batch.loss_mask[:, :16] = False

    # ── Inference forward (eval, no targets) ──────────────────────────────────
    model.eval()
    with torch.no_grad():
        batch_inf = KairoBatch(
            **{k: v for k, v in vars(batch).items()
               if k not in ("target_bytes", "loss_mask")},
            target_bytes=None, loss_mask=None,
        )
        out_inf = model(batch_inf)

    assert out_inf.hidden.shape == (B, 256 + 64 + cfg.n_imu + cfg.n_query, cfg.kcfg.d_model)
    assert out_inf.det_boxes.shape  == (B, cfg.max_det, 7)
    assert out_inf.det_scores.shape == (B, cfg.max_det, cfg.n_det_cls + 1)
    print(f"  Inference hidden: {tuple(out_inf.hidden.shape)}")
    print(f"  Detection boxes : {tuple(out_inf.det_boxes.shape)}")

    # ── Training forward (with targets) ───────────────────────────────────────
    model.train()
    out_tr = model(batch)
    assert out_tr.s2ft_loss is not None
    assert out_tr.moe_z_loss is not None

    # Simulate training step — use model-aggregated total_loss (includes z-loss)
    total_loss = out_tr.total_loss
    assert total_loss is not None, "total_loss is None — forward() should set it during training"
    total_loss.backward()

    # Verify gradients flow to key parameters
    checks = {
        "S2FT decoder":  model.s2ft_decoder.layers[0].gate.weight,
        "LoRA A":        next(
            p for n, p in model.named_parameters() if "lora_A" in n
        ),
        "Hybrid MoE W1": model.hybrid_core.blocks[0].moe_ffn.W1,
        "Calib log_sigma": model.calib_gate.log_sigma,
    }
    for name, param in checks.items():
        assert param.grad is not None and param.grad.abs().sum() > 0, \
            f"No gradient for {name}"
        print(f"  grad {name}: norm={param.grad.norm():.3e}  OK")

    # ── Param groups ──────────────────────────────────────────────────────────
    groups = model.param_groups()
    names  = [g["name"] for g in groups]
    total_pg = sum(p.numel() for g in groups for p in g["params"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_pg == trainable, "param_groups() doesn't cover all trainable params"
    print(f"  Param groups: {names}  total={total_pg/1e6:.1f}M  OK")

    print("Smoke-test passed.")
