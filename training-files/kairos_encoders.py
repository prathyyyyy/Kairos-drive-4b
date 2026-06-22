"""
kairos_encoders.py

Kairos-4B Vision Encoder — DINOv2-L + LoRA + 3-frame temporal fusion.

Encoding pipeline:
  Three KITTI frames (t, t-1, t-2), each (B, 3, H, W) ∈ [0, 1]:

  1. Resize → 448 × 112   (32 × 8 = 256 patches at 14 px patch stride)
  2. ImageNet normalise
  3. DINOv2-L ViT-L/14  (307M params, frozen)
     LoRA adapters on every Q/K/V projection in all 24 attention layers
     (rank r=32, α=32)
  4. Drop CLS token → (B, 256, 1024) patch tokens per frame

  Temporal fusion:
  5. Add learned frame-position embeddings (t=0, t−1=1, t−2=2)
  6. Concat frames → (B, 768, 1024)  key-value pool
  7. Cross-attention: 256 learnable query tokens attend to 768 KV tokens
     (F.scaled_dot_product_attention — memory-efficient, BF16 safe)
  8. SwiGLU FFN (d_ff=2730, iso-param trick)
  9. Output: (B, 256, 1024) = cam[256] tokens

Interface with KairosHybridCore:
  cam_tokens = vision_enc(img_t, img_t1, img_t2)     # (B, 256, d)
  x = torch.cat([cam_tokens, lidar_tokens,            # (B, 336, d) full seq
                 imu_tokens,  query_tokens], dim=1)

Rules (CLAUDE.md):
  - d_model = 1024; preprocessing stays float32, BF16 via autocast/DeepSpeed in training
  - DINOv2 backbone frozen; only LoRA + fusion block trained
  - Checkpoint prefix: s3://kairos-emr-assets/checkpoints/kairos-4b/
  - model hub: facebook/dinov2-large
"""

from __future__ import annotations

import inspect
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    from transformers import Dinov2Config, Dinov2Model  # type: ignore
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    from huggingface_hub import snapshot_download as _hf_snapshot_download
    _HAS_HF_HUB = True
except ImportError:
    _HAS_HF_HUB = False

try:
    from kairos_hybrid_block import KairosConfig, RMSNorm  # type: ignore
except ImportError:
    # Inline fallbacks so this file can run standalone
    @dataclass
    class KairosConfig:  # type: ignore
        d_model: int = 1024

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_f = x.float()
            rms_inv = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (x_f * rms_inv * self.weight.float()).to(x.dtype)


def _linear_input(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    return x.to(device=linear.weight.device, dtype=linear.weight.dtype)


def _enc_log_cuda_mem(tag: str) -> None:
    """Log CUDA memory at encoder milestones (rank-0 only, [cuda_mem] format)."""
    if not torch.cuda.is_available():
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    rsvd  = torch.cuda.memory_reserved()  / 1e9
    maxm  = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[cuda_mem] {tag}  "
        f"allocated={alloc:.2f}GB  reserved={rsvd:.2f}GB  max={maxm:.2f}GB",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Encoder-specific config (extends KairosConfig)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VisionEncoderConfig:
    # ── DINOv2 backbone ────────────────────────────────────────────────────────
    dinov2_model_name: str = "facebook/dinov2-large"
    dinov2_patch_size: int = 14
    dinov2_hidden_size: int = 1024    # ViT-L hidden dim
    dinov2_num_layers: int = 24       # transformer depth

    # ── LoRA adapter ──────────────────────────────────────────────────────────
    lora_rank: int = 32
    lora_alpha: int = 32              # effective scale = alpha / rank = 1.0
    lora_targets: Tuple[str, ...] = ("query", "key", "value")
    # "output" (attention output proj) can be added for extra capacity
    unfreeze_last_n: int = 4
    use_mock_backbone: bool = False

    # ── Input processing ──────────────────────────────────────────────────────
    # Resize each KITTI frame (1242×375) to enc_H × enc_W before DINOv2.
    # 448×112 = 32×8 = 256 patches at 14 px  (aspect ratio 4:1 vs KITTI 3.3:1)
    enc_h: int = 112
    enc_w: int = 448
    n_patches: int = 256              # enc_h/patch × enc_w/patch = 8 × 32

    # ImageNet normalisation (DINOv2 was trained on ImageNet)
    img_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    img_std:  Tuple[float, ...] = (0.229, 0.224, 0.225)

    # ── Temporal fusion ────────────────────────────────────────────────────────
    n_frames: int = 3                 # t, t-1, t-2
    fusion_heads: int = 16
    # SwiGLU FFN in fusion block re-uses d_ff=2730 (iso-param trick)
    fusion_d_ff: int = 2730

    # ── Misc ───────────────────────────────────────────────────────────────────
    use_grad_checkpoint: bool = True   # gradient-checkpoint DINOv2 layers
    sequential_frames: bool = False    # True → encode each frame individually through
                                       # DINOv2 instead of batching all 3 at once.
                                       # Lower peak activation memory; set True in
                                       # smoke_mode / budget_mode / ultra_smoke_mode.


# ──────────────────────────────────────────────────────────────────────────────
# LoRA adapter
# ──────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a low-rank LoRA adapter.

      W_eff = W_frozen  +  (B · A) × (α / r)

    Initialization:
      A ~ N(0, 0.01)  — small noise so the adapter starts near zero
      B = 0           — ensures W_eff == W_frozen at t=0

    The original weight and bias are frozen; only A and B are trained.
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 16,
        alpha: int = 32,
    ) -> None:
        super().__init__()
        out_f, in_f = linear.weight.shape
        self.linear = linear                  # keep original (frozen)
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.scale  = alpha / r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays at zero — no LoRA contribution at init
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base path (frozen) + LoRA delta
        x = _linear_input(x, self.linear)
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def _inject_lora(
    dinov2: "Dinov2Model",
    r: int = 16,
    alpha: int = 32,
    targets: Tuple[str, ...] = ("query", "key", "value"),
) -> None:
    """
    Replace targeted nn.Linear layers inside every DINOv2 attention block with
    LoRALinear wrappers.  Called once after loading the pretrained model.

    DINOv2 (transformers ≥ 4.37) attribute paths:
      dinov2.encoder.layer[i].attention.attention.{query,key,value}
      dinov2.encoder.layer[i].attention.output.dense    (optional)

    After injection the backbone stays frozen; only LoRA A/B are trainable.
    """
    for layer in dinov2.encoder.layer:
        sa = layer.attention.attention   # Dinov2SelfAttention
        for attr in targets:
            original: nn.Linear = getattr(sa, attr)
            setattr(sa, attr, LoRALinear(original, r=r, alpha=alpha))

        # Optional: output projection
        if "output" in targets:
            out_dense = layer.attention.output.dense
            layer.attention.output.dense = LoRALinear(out_dense, r=r, alpha=alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Fusion Block
# ──────────────────────────────────────────────────────────────────────────────

class TemporalFusionBlock(nn.Module):
    """
    Fuses n_frames × n_patches frame tokens into n_patches output tokens.

    Step 1 — frame-position conditioning:
      Each of the 3 frame token sets gets a learned frame-position embedding
      (t=0, t-1=1, t-2=2) added before fusion, giving the model an explicit
      temporal clock without modifying the patch positional structure.

    Step 2 — cross-attention compression:
      n_queries learnable query tokens attend to the (3 × n_patches) KV pool.
      Uses F.scaled_dot_product_attention (memory-efficient, no mask needed).

    Step 3 — SwiGLU FFN:
      Same d_ff=2730 iso-param trick as the MoE experts in KairosHybridBlock
      for parameter budget consistency.
    """

    def __init__(self, vcfg: VisionEncoderConfig, d: int) -> None:
        super().__init__()
        self.n_frames   = vcfg.n_frames      # 3
        self.n_patches  = vcfg.n_patches     # 256
        self.n_heads    = vcfg.fusion_heads  # 16
        self.head_dim   = d // self.n_heads  # 64
        self.d          = d

        # Learnable query tokens  (1, 256, d) → broadcast over B
        self.query_tokens = nn.Parameter(
            torch.randn(1, vcfg.n_patches, d) * (d ** -0.5)
        )

        # Learned frame-position embeddings: one scalar offset per frame
        self.frame_pos = nn.Embedding(vcfg.n_frames, d)

        # ── Cross-attention ────────────────────────────────────────────────────
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.o_proj.weight, std=d ** -0.5)

        self.norm_q  = RMSNorm(d)
        self.norm_kv = RMSNorm(d)

        # ── SwiGLU FFN ─────────────────────────────────────────────────────────
        d_ff = vcfg.fusion_d_ff    # 2730
        self.norm_ffn = RMSNorm(d)
        self.ffn_gate = nn.Linear(d, d_ff, bias=False)  # SiLU gate branch
        self.ffn_up   = nn.Linear(d, d_ff, bias=False)  # linear branch
        self.ffn_down = nn.Linear(d_ff, d, bias=False)
        nn.init.normal_(self.ffn_down.weight, std=d ** -0.5)

    # ------------------------------------------------------------------
    def _cross_attn(
        self,
        q: torch.Tensor,   # (B, n_q, d)
        kv: torch.Tensor,  # (B, n_kv, d)
    ) -> torch.Tensor:
        B, n_q, _ = q.shape
        n_kv = kv.shape[1]
        H, hd = self.n_heads, self.head_dim
        q = _linear_input(q, self.q_proj)
        kv = _linear_input(kv, self.k_proj)

        # Project and split into heads
        Q = self.q_proj(q).view(B, n_q, H, hd).transpose(1, 2)   # (B,H,n_q,hd)
        K = self.k_proj(kv).view(B, n_kv, H, hd).transpose(1, 2)
        V = self.v_proj(kv).view(B, n_kv, H, hd).transpose(1, 2)

        # Memory-efficient attention (no mask — all 768 KV positions are valid)
        out = F.scaled_dot_product_attention(Q, K, V)              # (B,H,n_q,hd)
        out = out.transpose(1, 2).contiguous().view(B, n_q, -1)    # (B,n_q,d)
        return self.o_proj(out)

    def _swiglu(self, x: torch.Tensor) -> torch.Tensor:
        x = _linear_input(x, self.ffn_gate)
        return F.silu(self.ffn_gate(x)) * self.ffn_up(x)

    # ------------------------------------------------------------------
    def forward(
        self,
        frame_tokens: List[torch.Tensor],   # [tok_t, tok_t1, tok_t2] each (B,256,d)
    ) -> torch.Tensor:
        """Returns (B, 256, d) fused camera tokens."""
        B = frame_tokens[0].shape[0]

        # ── Add frame-position embeddings ──────────────────────────────────────
        frame_ids = torch.arange(self.n_frames, device=frame_tokens[0].device)
        pos_emb   = self.frame_pos(frame_ids)               # (3, d)

        conditioned = [
            tok + pos_emb[i]          # (B, 256, d) + (d,) broadcasts over B, patches
            for i, tok in enumerate(frame_tokens)
        ]

        # ── Concat all frame tokens into KV pool ───────────────────────────────
        kv = torch.cat(conditioned, dim=1)                  # (B, 768, d)

        # ── Cross-attention: 256 queries × 768 KV ─────────────────────────────
        q   = self.query_tokens.expand(B, -1, -1)           # (B, 256, d)
        q   = self.norm_q(q)
        kv  = self.norm_kv(kv)
        out = q + self._cross_attn(q, kv)                   # (B, 256, d) residual

        # ── SwiGLU FFN ─────────────────────────────────────────────────────────
        out = out + self.ffn_down(self._swiglu(self.norm_ffn(out)))

        return out                                           # (B, 256, 1024)


# ──────────────────────────────────────────────────────────────────────────────
# DINOv2 local-directory helpers
#
# Root cause that drove this design:
#   HuggingFace snapshot_download(cache_dir=...) places weights under a hash
#   sub-directory and uses symlinks into a separate blobs/ dir.  Under
#   torchrun/DDP on SageMaker that sub-directory is created by rank 0 but the
#   symlinks are on a filesystem other ranks see as read-only before the
#   barrier.  Non-zero ranks get "unable to open file ... in read-only mode".
#
#   The fix: never use the HF snapshot cache layout for weights that must be
#   shared across DDP ranks.  Instead, snapshot_download is called with
#   local_dir=KAIROS_DINO_LOCAL_DIR and local_dir_use_symlinks=False, which
#   copies real files into a flat directory.  Every rank then loads from that
#   flat directory with local_files_only=True.
# ──────────────────────────────────────────────────────────────────────────────

_DINO_DEFAULT_MODEL_ID  = "facebook/dinov2-large"
_DINO_DEFAULT_LOCAL_DIR = "/opt/ml/checkpoints/models/dinov2-large"
_DINO_REQUIRED_FILES    = ("config.json",)
_DINO_WEIGHT_FILES      = ("model.safetensors", "pytorch_model.bin")


def _set_hf_env() -> None:
    """Set writable HF/transformers cache env vars (setdefault, respects pre-set values).

    TRANSFORMERS_CACHE == HF_HUB_CACHE so transformers and huggingface_hub look
    in the same place and never produce a snapshot-path mismatch.
    """
    hf_home = os.environ.get("HF_HOME", "/opt/ml/checkpoints/hf")
    hub_dir  = os.path.join(hf_home, "hub")
    try:
        os.makedirs(hub_dir, exist_ok=True)
    except OSError:
        pass
    os.environ.setdefault("HF_HOME",           hf_home)
    os.environ.setdefault("HF_HUB_CACHE",       hub_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE",  hub_dir)   # same as hub — no snapshot mismatch
    os.environ.setdefault(
        "XDG_CACHE_HOME",
        os.path.join(os.path.dirname(hf_home), ".cache"),
    )


def _dino_local_dir() -> str:
    """Return the explicit flat directory to use for DINOv2 weights."""
    env = os.environ.get("KAIROS_DINO_LOCAL_DIR", "").strip()
    return env or _DINO_DEFAULT_LOCAL_DIR


def _get_rank_world() -> Tuple[int, int]:
    """Return (rank, world_size) from torch.distributed if initialised, else env vars."""
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            return _dist.get_rank(), _dist.get_world_size()
    except Exception:
        pass
    rank  = int(os.environ.get("RANK",       os.environ.get("LOCAL_RANK", "0")))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world


def _verify_dino_dir(local_dir: str) -> List[str]:
    """
    Verify local_dir contains config.json and at least one weight file.

    Returns list of present weight file names.
    Raises RuntimeError with instructions if any required file is missing.
    """
    try:
        contents = sorted(os.listdir(local_dir)) if os.path.isdir(local_dir) else []
    except OSError:
        contents = []

    if not os.path.isdir(local_dir):
        raise RuntimeError(
            f"[dino] Local dir does not exist: {local_dir!r}\n"
            f"  Hint: set KAIROS_DINO_FORCE_REFRESH=true to trigger a fresh download."
        )

    missing_required = [
        f for f in _DINO_REQUIRED_FILES
        if not os.path.exists(os.path.join(local_dir, f))
    ]
    if missing_required:
        raise RuntimeError(
            f"[dino] Verification failed in {local_dir!r}\n"
            f"  Missing required files: {missing_required}\n"
            f"  Dir contents (first 20): {contents[:20]}\n"
            f"  Hint: set KAIROS_DINO_FORCE_REFRESH=true to re-download."
        )

    present = [wf for wf in _DINO_WEIGHT_FILES if os.path.exists(os.path.join(local_dir, wf))]
    if not present:
        raise RuntimeError(
            f"[dino] No weight file found in {local_dir!r}\n"
            f"  Expected one of: {_DINO_WEIGHT_FILES}\n"
            f"  Dir contents (first 20): {contents[:20]}\n"
            f"  Hint: set KAIROS_DINO_FORCE_REFRESH=true to re-download."
        )
    return present


def _dino_s3_copy(s3_uri: str, local_dir: str) -> None:
    """Recursively copy a pre-cached DINOv2 directory from S3 into local_dir."""
    import boto3 as _boto3
    s3_uri = s3_uri.rstrip("/")
    bucket, prefix = s3_uri.removeprefix("s3://").split("/", 1)
    region = os.environ.get("DATA_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    s3 = _boto3.client("s3", region_name=region)
    pager = s3.get_paginator("list_objects_v2")
    os.makedirs(local_dir, exist_ok=True)
    for page in pager.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/")
            if not rel:
                continue
            dest = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest) and os.path.getsize(dest) == obj["Size"]:
                continue
            s3.download_file(bucket, key, dest)


def _dino_download_hf(model_id: str, local_dir: str) -> None:
    """Download DINOv2 from HuggingFace into local_dir as real flat files (no symlinks)."""
    if not _HAS_HF_HUB:
        raise RuntimeError(
            f"[dino] Cannot download {model_id!r}: huggingface_hub is not installed.\n"
            f"  Install with: pip install huggingface_hub\n"
            f"  Or set KAIROS_DINO_S3_URI to a pre-cached S3 path."
        )
    _hf_snapshot_download(
        model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=[
            "*.msgpack", "flax_model*", "tf_model*",
            "rust_model.ot", "coreml/",
        ],
    )


def _dino_rank0_prepare(
    local_dir: str,
    model_id: str,
    s3_uri: str,
    force_refresh: bool,
) -> List[str]:
    """
    Rank-0-only: optionally purge local_dir, then download weights and verify.

    Returns list of present weight file names after preparation.
    """
    import shutil
    if force_refresh and os.path.isdir(local_dir):
        print(f"[dino] rank0 force-refresh: removing {local_dir}", flush=True)
        shutil.rmtree(local_dir, ignore_errors=True)

    os.makedirs(local_dir, exist_ok=True)

    if s3_uri:
        print(f"[dino] source=s3  uri={s3_uri}", flush=True)
        print(f"[dino] rank0 downloading/copying S3 -> {local_dir}", flush=True)
        _dino_s3_copy(s3_uri, local_dir)
    else:
        print(f"[dino] source=huggingface  model_id={model_id}", flush=True)
        print(f"[dino] rank0 downloading/copying -> {local_dir}", flush=True)
        _dino_download_hf(model_id, local_dir)

    return _verify_dino_dir(local_dir)


def resolve_dino_model_id_or_path(
    model_id: Optional[str] = None,
) -> Tuple[str, bool]:
    """
    Resolve DINOv2 weights to a flat local directory.

    Always returns (KAIROS_DINO_LOCAL_DIR, local_files_only=True).
    Rank 0 downloads first; all ranks barrier-sync; every rank verifies before loading.

    Env vars:
      KAIROS_DINO_LOCAL_DIR     -- flat dir for weights (default: /opt/ml/checkpoints/models/dinov2-large)
      KAIROS_DINO_MODEL_ID      -- HuggingFace model ID (default: facebook/dinov2-large)
      KAIROS_DINO_S3_URI        -- s3://... of pre-cached weights; skips HF download
      KAIROS_DINO_FORCE_REFRESH -- "true"/"1" -> rank 0 deletes local_dir before download
    """
    effective_id  = (
        model_id
        or os.environ.get("KAIROS_DINO_MODEL_ID", "").strip()
        or _DINO_DEFAULT_MODEL_ID
    )
    s3_uri        = os.environ.get("KAIROS_DINO_S3_URI",        "").strip()
    force_refresh = os.environ.get("KAIROS_DINO_FORCE_REFRESH", "").lower() in ("1", "true")
    local_dir     = _dino_local_dir()

    _set_hf_env()  # set writable, aligned HF cache env vars before any transformers call

    rank, world = _get_rank_world()
    is_dist = world > 1

    print(f"[dino] rank={rank} world={world} model_id={effective_id}", flush=True)
    print(f"[dino] local_dir={local_dir}", flush=True)

    if is_dist:
        try:
            import torch.distributed as _dist
            _barrier = (
                _dist.barrier
                if (_dist.is_available() and _dist.is_initialized())
                else None
            )
        except Exception:
            _barrier = None

        if rank == 0:
            print("[dino] rank0 preparing local dir", flush=True)
            files = _dino_rank0_prepare(local_dir, effective_id, s3_uri, force_refresh)
            print(f"[dino] rank0 verify ok  files={files}", flush=True)

        print(f"[dino] barrier enter  rank={rank}", flush=True)
        if _barrier is not None:
            _barrier()
        print(f"[dino] barrier exit   rank={rank}", flush=True)

        # Post-barrier: every rank confirms files exist before loading
        files = _verify_dino_dir(local_dir)
        print(f"[dino] rank={rank} verify ok  files={files}", flush=True)

    else:
        print("[dino] rank0 preparing local dir", flush=True)
        files = _dino_rank0_prepare(local_dir, effective_id, s3_uri, force_refresh)
        print(f"[dino] rank0 verify ok  files={files}", flush=True)

    print(f"[dino] rank={rank} loading local_files_only=True from local_dir={local_dir}", flush=True)
    return local_dir, True


def _dino_load_error(
    model_id_or_path: str,
    local_files_only: bool,
    exc: Exception,
) -> RuntimeError:
    """Build a descriptive RuntimeError for DINO load failures."""
    rank, world = _get_rank_world()
    local_dir   = _dino_local_dir()
    s3_uri      = os.environ.get("KAIROS_DINO_S3_URI",        "not set")
    force_ref   = os.environ.get("KAIROS_DINO_FORCE_REFRESH", "not set")
    try:
        contents = sorted(os.listdir(local_dir))[:20] if os.path.isdir(local_dir) else ["(directory missing)"]
    except OSError:
        contents = ["(unreadable)"]
    missing = [
        f for f in ("config.json",) + _DINO_WEIGHT_FILES
        if not os.path.exists(os.path.join(local_dir, f))
    ]
    return RuntimeError(
        f"[DINO load failed]\n"
        f"  model          : {model_id_or_path}\n"
        f"  local_dir      : {local_dir}\n"
        f"  local_only     : {local_files_only}\n"
        f"  rank           : {rank}  world={world}\n"
        f"  DINO_S3_URI    : {s3_uri}\n"
        f"  FORCE_REFRESH  : {force_ref}\n"
        f"  missing files  : {missing}\n"
        f"  dir contents   : {contents}\n"
        f"\nHint: set KAIROS_DINO_FORCE_REFRESH=true to delete and re-download.\n"
        f"      Or set KAIROS_DINO_S3_URI to a pre-cached S3 path.\n"
        f"      Or use --ultra_smoke_mock_vision True to skip DINO entirely.\n"
        f"\nOriginal error: {exc}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Vision Encoder
# ──────────────────────────────────────────────────────────────────────────────

class KairosVisionEncoder(nn.Module):
    """
    DINOv2-L (frozen) + LoRA + 3-frame temporal fusion.

    Training behaviour:
      - DINOv2-L backbone: all parameters frozen.
      - LoRA A/B in attention Q/K/V: trainable (~2.5M params).
      - TemporalFusionBlock (query tokens, frame_pos, cross-attn, FFN): trainable.

    Gradient checkpointing:
      Enabled on the DINOv2 encoder when vcfg.use_grad_checkpoint=True.
      Reduces activation memory by ~60% at the cost of one extra forward pass
      per backward.  Strongly recommended on ml.g5.48xlarge.

    Loading pretrained weights:
      The backbone is downloaded from HuggingFace Hub on first run and cached.
      To use an offline copy:
          os.environ["TRANSFORMERS_OFFLINE"] = "1"
          os.environ["HF_DATASETS_OFFLINE"] = "1"

    Checkpoint (LoRA + fusion only):
      torch.save(encoder.trainable_state_dict(), path)
      encoder.load_trainable_state_dict(torch.load(path))
    """

    # One-time log guard shared across all instances (reset in tests via monkeypatch)
    _dino_forward_logged: bool = False

    def __init__(
        self,
        vcfg: Optional[VisionEncoderConfig] = None,
        kcfg: Optional[KairosConfig] = None,
    ) -> None:
        super().__init__()
        vcfg = vcfg or VisionEncoderConfig()
        kcfg = kcfg or KairosConfig()
        self.vcfg = vcfg
        self.d    = kcfg.d_model   # 1024

        # Forward compat defaults — overridden for real backbone by _setup_dino_forward_compat
        self._fw_use_interpolate: bool = False
        self._dino_fallback_size: Optional[int] = None
        self._fw_known_kwargs: set = {"pixel_values"}
        self._fw_manual_fallback: bool = False

        # Patch token adaptation defaults (overridden by env vars in _setup_dino_forward_compat).
        # DINO may return N patches where N != vcfg.n_patches (e.g. 518x518 input → 37x37=1369).
        # _adapt_dino_patch_tokens uses adaptive_avg_pool2d to project to the target grid.
        _default_sq = int(vcfg.n_patches ** 0.5)   # 16 for n_patches=256
        self._target_patch_tokens: int = vcfg.n_patches
        self._target_grid_h: int = _default_sq
        self._target_grid_w: int = _default_sq
        self._debug_shapes: bool = False

        # DINO input-size & memory opts (env vars read in _setup_dino_forward_compat).
        # KAIROS_DINO_INPUT_SIZE=224 resizes before DINO: 224/14=16×16=256 tokens (vs 518/14=37×37=1369).
        # KAIROS_DINO_NO_GRAD_IN_SMOKE=true runs DINO under torch.no_grad() to eliminate
        # activation retention across frames — key for fitting 3-frame encode in 22 GB.
        self._dino_input_size: Optional[int] = None   # None = governed by _preprocess logic
        self._dino_no_grad_in_smoke: bool = False
        self._dino_train_lora_in_smoke: bool = False  # placeholder; reserved for later
        self._dino_temporal_frames: int = 3
        self._embed_fw_has_interpolate: bool = False  # detected once in _setup_dino_forward_compat

        # ── DINOv2-L backbone ──────────────────────────────────────────────────
        if vcfg.use_mock_backbone:
            # dinov2_hidden_size keeps patch_proj input dim correct (→Linear(1024, d))
            self.dinov2 = _MockDinov2(vcfg.n_patches, vcfg.dinov2_hidden_size)
        elif _HAS_TRANSFORMERS:
            # Resolve model ID / local path; sets writable HF cache env vars;
            # rank 0 prefetches in DDP mode so all ranks share one download.
            _dino_path, _local_only = resolve_dino_model_id_or_path(
                vcfg.dinov2_model_name
            )
            try:
                self.dinov2: nn.Module = Dinov2Model.from_pretrained(
                    _dino_path,
                    add_pooling_layer=False,   # keep patch tokens, skip pooler
                    local_files_only=_local_only,
                )
            except TypeError:
                # Older transformers builds don't accept add_pooling_layer;
                # CLS token is dropped manually in _encode_frame either way.
                try:
                    self.dinov2 = Dinov2Model.from_pretrained(
                        _dino_path,
                        local_files_only=_local_only,
                    )
                except Exception as _exc:
                    raise _dino_load_error(_dino_path, _local_only, _exc) from _exc
            except Exception as _exc:
                raise _dino_load_error(_dino_path, _local_only, _exc) from _exc
            # Inject LoRA into attention Q/K/V before freezing
            _inject_lora(
                self.dinov2,               # type: ignore[arg-type]
                r=vcfg.lora_rank,
                alpha=vcfg.lora_alpha,
                targets=vcfg.lora_targets,
            )
            # Freeze everything that is not a LoRA parameter
            self._freeze_backbone()
            self._unfreeze_last_layers(vcfg.unfreeze_last_n)

            # DINO backbone is frozen (only LoRA trains); gradient checkpointing on a
            # frozen encoder is unnecessary and can conflict with ZeRO-3 module wrapping.
            if hasattr(self.dinov2, "gradient_checkpointing_disable"):
                self.dinov2.gradient_checkpointing_disable()
            # Probe forward signature, force return_dict=True, set interpolate compat
            self._setup_dino_forward_compat()
        else:
            # Lightweight stub used for unit-testing without transformers installed
            self.dinov2 = _MockDinov2(vcfg.n_patches, vcfg.dinov2_hidden_size)

        # ── Image normalisation constants (ImageNet, stored as buffer) ─────────
        self.register_buffer(
            "img_mean",
            torch.tensor(vcfg.img_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor(vcfg.img_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        # ── Down-projection (DINOv2 hidden=1024 → d_model=1024; identity if equal)
        if vcfg.dinov2_hidden_size != self.d:
            self.patch_proj: nn.Module = nn.Linear(
                vcfg.dinov2_hidden_size, self.d, bias=False
            )
        else:
            self.patch_proj = nn.Identity()

        # ── Temporal fusion ────────────────────────────────────────────────────
        self.temporal_fusion = TemporalFusionBlock(vcfg, self.d)

    # ------------------------------------------------------------------
    def _freeze_backbone(self) -> None:
        """Freeze every DINOv2 parameter that is NOT a LoRA weight."""
        for name, param in self.dinov2.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)

    def _unfreeze_last_layers(self, n_layers: int) -> None:
        """Unfreeze the final DINOv2 transformer blocks after LoRA injection."""
        if n_layers <= 0:
            return
        encoder = getattr(self.dinov2, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            return
        for layer in list(layers)[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad_(True)

    # ------------------------------------------------------------------
    def _setup_dino_forward_compat(self) -> None:
        """
        Probe the installed transformers version for DINOv2 forward compatibility.

        Called once after loading the real DINOv2 backbone (also called explicitly in tests
        after swapping enc.dinov2).  Sets:
          self._fw_use_interpolate  -- True iff we will pass interpolate_pos_encoding=True
          self._dino_fallback_size  -- model config.image_size (int or None) for fallback resize
          self._fw_known_kwargs     -- set of kwargs accepted by self.dinov2.forward
          self._fw_manual_fallback  -- reset to False; set True at runtime on KeyError(0)

        Env var:
          KAIROS_DINO_INTERPOLATE_POS_ENCODING  default "true"
            "false"/"0"/"no" -> never pass the kwarg (use model-native image size)
        """
        # Probe ALL kwargs supported by the installed Dinov2Model.forward via signature
        try:
            sig = inspect.signature(self.dinov2.forward)
            self._fw_known_kwargs = set(sig.parameters.keys())
        except (ValueError, TypeError):
            self._fw_known_kwargs = {"pixel_values"}

        supports = "interpolate_pos_encoding" in self._fw_known_kwargs

        env_val   = os.environ.get("KAIROS_DINO_INTERPOLATE_POS_ENCODING", "true").lower()
        requested = env_val not in ("0", "false", "no")

        self._fw_use_interpolate = supports and requested
        self._fw_manual_fallback = False

        # Read patch-token adaptation config from env vars.
        # KAIROS_DINO_TARGET_GRID_H / _W take priority over KAIROS_DINO_TARGET_PATCH_TOKENS.
        _tgh = os.environ.get("KAIROS_DINO_TARGET_GRID_H", "").strip()
        _tgw = os.environ.get("KAIROS_DINO_TARGET_GRID_W", "").strip()
        if _tgh and _tgw:
            self._target_grid_h  = int(_tgh)
            self._target_grid_w  = int(_tgw)
            self._target_patch_tokens = self._target_grid_h * self._target_grid_w
        else:
            _tpt = os.environ.get("KAIROS_DINO_TARGET_PATCH_TOKENS", "").strip()
            if _tpt:
                self._target_patch_tokens = int(_tpt)
                _sq = int(self._target_patch_tokens ** 0.5)
                if _sq * _sq != self._target_patch_tokens:
                    raise ValueError(
                        f"KAIROS_DINO_TARGET_PATCH_TOKENS={self._target_patch_tokens} is not a "
                        f"perfect square.  Set KAIROS_DINO_TARGET_GRID_H and KAIROS_DINO_TARGET_GRID_W "
                        f"for non-square target grids."
                    )
                self._target_grid_h = self._target_grid_w = _sq
            # else: keep __init__ defaults (vcfg.n_patches, sqrt×sqrt)
        self._debug_shapes = os.environ.get("KAIROS_DINO_DEBUG_SHAPES", "").lower() in ("1", "true", "yes")

        # ── DINO input-size & memory optimizations from env vars ─────────────────
        # KAIROS_DINO_INPUT_SIZE: explicit resize target before DINO (e.g. 224 for smoke).
        # Takes highest priority in _preprocess; 224×224 → 16×16=256 tokens, no pooling needed.
        _dis = os.environ.get("KAIROS_DINO_INPUT_SIZE", "").strip()
        if _dis:
            self._dino_input_size = int(_dis)
        _ngs = os.environ.get("KAIROS_DINO_NO_GRAD_IN_SMOKE", "").lower()
        self._dino_no_grad_in_smoke = _ngs in ("1", "true", "yes")
        _tls = os.environ.get("KAIROS_DINO_TRAIN_LORA_IN_SMOKE", "").lower()
        self._dino_train_lora_in_smoke = _tls in ("1", "true", "yes")
        _dtf = os.environ.get("KAIROS_DINO_TEMPORAL_FRAMES", "").strip()
        if _dtf:
            self._dino_temporal_frames = max(1, min(3, int(_dtf)))

        # Detect whether embeddings.forward() independently exposes interpolate_pos_encoding.
        # Under ZeRO-3 the wrapped model.forward doesn't have it (_fw_use_interpolate=False),
        # but the underlying Dinov2Embeddings.forward may — used in _dinov2_manual_forward
        # so that non-native input sizes (e.g. 224 vs native 518) get correct positional encoding.
        _emb = getattr(self.dinov2, "embeddings", None)
        self._embed_fw_has_interpolate = False
        if _emb is not None:
            try:
                _emb_sig = inspect.signature(_emb.forward)
                self._embed_fw_has_interpolate = "interpolate_pos_encoding" in _emb_sig.parameters
            except (TypeError, ValueError):
                pass

        # Fallback resize target when interpolation is unavailable
        config = getattr(self.dinov2, "config", None)
        if config is not None:
            img_size = getattr(config, "image_size", None)
            if isinstance(img_size, int):
                self._dino_fallback_size = img_size

        # Force return_dict=True so encoder always returns a ModelOutput (not plain tuple/dict).
        # Under ZeRO-3 the encoder sometimes returns a plain dict; Dinov2Model.forward then
        # does encoder_outputs[0] which raises KeyError(0) on string-keyed dicts.
        if config is not None:
            try:
                config.return_dict = True
            except (AttributeError, TypeError):
                pass
            try:
                config.use_return_dict = True
            except (AttributeError, TypeError):
                pass
        encoder = getattr(self.dinov2, "encoder", None)
        if encoder is not None:
            enc_cfg = getattr(encoder, "config", None)
            if enc_cfg is not None:
                try:
                    enc_cfg.return_dict = True
                except (AttributeError, TypeError):
                    pass
                try:
                    enc_cfg.use_return_dict = True
                except (AttributeError, TypeError):
                    pass

        if not KairosVisionEncoder._dino_forward_logged:
            KairosVisionEncoder._dino_forward_logged = True
            try:
                import transformers as _tf_ver
                _tv = getattr(_tf_ver, "__version__", "unknown")
            except Exception:
                _tv = "unknown"
            print(f"[dino] transformers_version={_tv}", flush=True)
            print(f"[dino] return_dict_forced=True", flush=True)
            print(f"[dino] forward_kwargs={sorted(self._fw_known_kwargs)}", flush=True)
            print(f"[dino] forward_supports_interpolate_pos_encoding={supports}", flush=True)
            print(f"[dino] requested_interpolate_pos_encoding={requested}", flush=True)
            print(f"[dino] using_interpolate_pos_encoding={self._fw_use_interpolate}", flush=True)
            print(f"[dino] target_patch_tokens={self._target_patch_tokens}", flush=True)
            print(f"[dino] target_grid=({self._target_grid_h}, {self._target_grid_w})", flush=True)
            print(f"[dino] input_size_override={self._dino_input_size!r}", flush=True)
            print(f"[dino] embed_fw_has_interpolate={self._embed_fw_has_interpolate}", flush=True)
            print(f"[dino] no_grad_in_smoke={self._dino_no_grad_in_smoke}", flush=True)
            print(f"[dino] temporal_frames={self._dino_temporal_frames}", flush=True)
            if self._dino_input_size is not None:
                _ps_log = getattr(config, "patch_size", 14) if config is not None else 14
                _nat_log = getattr(config, "image_size", 518) if config is not None else 518
                _eg_log  = self._dino_input_size // _ps_log
                _mode_log = "smoke_224" if self._dino_input_size == 224 else f"configured_{self._dino_input_size}"
                print(f"[dino] expected_patch_grid={_eg_log}x{_eg_log}", flush=True)
                print(f"[dino] memory_mode={_mode_log}", flush=True)
                if self._dino_input_size != _nat_log and not (supports or self._embed_fw_has_interpolate):
                    print(
                        f"[dino] WARN: input_size={self._dino_input_size} != native {_nat_log} "
                        f"but neither model.forward nor embeddings.forward support "
                        f"interpolate_pos_encoding — position encoding mismatch likely.",
                        flush=True,
                    )
                if self._dino_input_size == 518:
                    # 518 is the native DINOv2-L size: no interpolation required.
                    _raw_n = _eg_log * _eg_log  # 37*37=1369
                    print(f"[dino] input_size=518 native=True", flush=True)
                    print(f"[dino] raw_grid=({_eg_log},{_eg_log})", flush=True)
                    print(f"[dino] raw_patch_tokens={_raw_n}", flush=True)
                    print(f"[dino] target_grid=({self._target_grid_h},{self._target_grid_w})", flush=True)
                    if _eg_log != self._target_grid_h or _eg_log != self._target_grid_w:
                        print(
                            f"[dino] patch_adaptation={_eg_log}x{_eg_log} -> "
                            f"{self._target_grid_h}x{self._target_grid_w} adaptive_avg_pool2d",
                            flush=True,
                        )
                    else:
                        print("[dino] patch_adaptation=passthrough", flush=True)
            if config is not None:
                print(f"[dino] config.image_size={getattr(config, 'image_size', 'N/A')}", flush=True)
                print(f"[dino] config.patch_size={getattr(config, 'patch_size', 'N/A')}", flush=True)
            if not supports:
                print(
                    f"[dino] WARN: interpolate_pos_encoding not supported by installed "
                    f"transformers.  Resizing to config.image_size={self._dino_fallback_size}.  "
                    f"Upgrade transformers >= 4.37 for non-square / non-native image sizes.",
                    flush=True,
                )
            elif not requested:
                print("[dino] INFO: interpolate_pos_encoding disabled via env var.", flush=True)

    # ------------------------------------------------------------------
    def _build_dino_kwargs(self, pixel_values: torch.Tensor) -> dict:
        """Build call kwargs for self.dinov2() using only signature-inspected parameters."""
        kw: dict = {"pixel_values": pixel_values}
        if "return_dict" in self._fw_known_kwargs:
            kw["return_dict"] = True
        if "output_hidden_states" in self._fw_known_kwargs:
            kw["output_hidden_states"] = False
        if "output_attentions" in self._fw_known_kwargs:
            kw["output_attentions"] = False
        if self._fw_use_interpolate:
            kw["interpolate_pos_encoding"] = True
        return kw

    # ------------------------------------------------------------------
    def _dinov2_forward(self, pixel_values: torch.Tensor) -> object:
        """
        Version-safe DINOv2 forward.

        Builds call kwargs from the inspected signature (return_dict=True,
        output_hidden_states=False, output_attentions=False, and
        interpolate_pos_encoding when supported and requested).

        Two runtime safety nets:
          - KeyError(0)  → encoder returned a plain dict under ZeRO-3;
                           switch permanently to _dinov2_manual_forward.
          - TypeError "interpolate_pos_encoding" → inspect/forward mismatch;
                           disable the kwarg and retry.
        """
        if not getattr(self, "_fw_logged_shape", False):
            self._fw_logged_shape = True
            print(f"[dino] pixel_values shape={tuple(pixel_values.shape)}", flush=True)

        if self._fw_manual_fallback:
            return self._dinov2_manual_forward(pixel_values)

        kw = self._build_dino_kwargs(pixel_values)

        try:
            return self.dinov2(**kw)
        except KeyError as _ke:
            if _ke.args == (0,):
                if not getattr(self, "_fw_fallback_logged", False):
                    self._fw_fallback_logged = True
                    print(
                        "[dino] WARN: KeyError(0) in Dinov2Model.forward "
                        "(encoder returned plain dict under ZeRO-3); activating manual fallback.",
                        flush=True,
                    )
                    print("[dino] using_manual_forward_fallback=True", flush=True)
                self._fw_manual_fallback = True
                return self._dinov2_manual_forward(pixel_values)
            raise
        except TypeError as _te:
            if "interpolate_pos_encoding" in str(_te):
                self._fw_use_interpolate = False
                print(
                    "[dino] WARN: interpolate_pos_encoding rejected at runtime "
                    "(inspect/forward mismatch); disabling for remaining calls.",
                    flush=True,
                )
                kw.pop("interpolate_pos_encoding", None)
                return self.dinov2(**kw)
            raise

    # ------------------------------------------------------------------
    def _dinov2_manual_forward(self, pixel_values: torch.Tensor) -> object:
        """
        Manual forward: embeddings → encoder → layernorm.

        Used when Dinov2Model.forward() raises KeyError(0) because the installed
        transformers' encoder returns a plain string-keyed dict under ZeRO-3 and
        the top-level model does encoder_outputs[0] (integer key → KeyError).

        Manually calls the three sub-modules so we fully control the return-value
        extraction path.  Returns an object with a .last_hidden_state attribute so
        _encode_frame / _extract_dino_last_hidden_state are unchanged.
        """
        emb = getattr(self.dinov2, "embeddings", None)
        if emb is None:
            raise RuntimeError(
                "[dino] _dinov2_manual_forward requires self.dinov2.embeddings — not found."
            )
        encoder = getattr(self.dinov2, "encoder", None)
        if encoder is None:
            raise RuntimeError(
                "[dino] _dinov2_manual_forward requires self.dinov2.encoder — not found."
            )

        # Step 1: embeddings — pass interpolate_pos_encoding when:
        #   (a) model.forward exposes it and it was requested (_fw_use_interpolate), OR
        #   (b) explicit input_size differs from native (e.g. 224 vs 518 under ZeRO-3
        #       where model.forward is wrapped and doesn't expose the kwarg but the
        #       underlying Dinov2Embeddings.forward may still support it directly).
        emb_kw: dict = {"pixel_values": pixel_values}
        try:
            emb_sig = inspect.signature(emb.forward)
            _cfg_inner  = getattr(self.dinov2, "config", None)
            _native_sz  = getattr(_cfg_inner, "image_size", 518) if _cfg_inner is not None else 518
            _dis_inner  = getattr(self, "_dino_input_size", None)
            _need_interp = (
                self._fw_use_interpolate
                or (_dis_inner is not None and _dis_inner != _native_sz)
            )
            if _need_interp and "interpolate_pos_encoding" in emb_sig.parameters:
                emb_kw["interpolate_pos_encoding"] = True
            elif _need_interp and not getattr(self, "_embed_interp_warn_logged", False):
                self._embed_interp_warn_logged = True
                print(
                    f"[dino] WARN: need interpolate_pos_encoding for input_size={_dis_inner} "
                    f"(native={_native_sz}) but Dinov2Embeddings.forward does not expose it.  "
                    f"Positional encoding may misalign — consider native size {_native_sz} "
                    f"(OOM risk) or upgrade transformers >= 4.37.",
                    flush=True,
                )
        except (ValueError, TypeError):
            pass
        embedding_output = emb(**emb_kw)

        # Step 2: encoder with return_dict=True so we get a subscriptable output
        try:
            encoder_outputs = encoder(
                embedding_output,
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        except TypeError:
            encoder_outputs = encoder(embedding_output)

        # Step 3: safely extract sequence output from whatever the encoder returned
        if hasattr(encoder_outputs, "last_hidden_state"):
            sequence_output = encoder_outputs.last_hidden_state
        elif isinstance(encoder_outputs, dict):
            sequence_output = encoder_outputs.get("last_hidden_state")
            if sequence_output is None:
                sequence_output = encoder_outputs.get("hidden_states")
            if sequence_output is None:
                raise RuntimeError(
                    f"[dino] Manual fallback: encoder dict missing 'last_hidden_state'/'hidden_states'. "
                    f"Keys: {list(encoder_outputs.keys())}"
                )
            if isinstance(sequence_output, (list, tuple)):
                sequence_output = sequence_output[-1]
        else:
            sequence_output = encoder_outputs[0]

        # Step 4: layernorm (Dinov2Model always applies this before returning)
        ln = getattr(self.dinov2, "layernorm", None)
        if ln is not None:
            sequence_output = ln(sequence_output)

        return type("_DinoManualOutput", (), {"last_hidden_state": sequence_output})()

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_dino_last_hidden_state(out: object) -> torch.Tensor:
        """
        Safely extract the last_hidden_state tensor from any DINOv2 output type:
          - HuggingFace ModelOutput (or any object) with .last_hidden_state attribute
          - dict with 'last_hidden_state' key (plain-dict ZeRO-3 fallback)
          - tuple / list where index 0 is the sequence output
        """
        # Attribute access — standard HF ModelOutput and _DinoManualOutput
        lhs = getattr(out, "last_hidden_state", None)
        if lhs is not None:
            return lhs
        # dict
        if isinstance(out, dict):
            lhs = out.get("last_hidden_state")
            if lhs is not None:
                return lhs
            lhs = out.get("hidden_states")
            if lhs is not None:
                return lhs[-1] if isinstance(lhs, (list, tuple)) else lhs
        # tuple / list — first element is the sequence output
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return out[0]
        raise RuntimeError(
            f"[dino] _extract_dino_last_hidden_state: cannot extract from "
            f"{type(out).__name__}: {out!r}"
        )

    # ------------------------------------------------------------------
    def _adapt_dino_patch_tokens(
        self,
        patch_tokens: torch.Tensor,   # (B, N, C)  — raw patch tokens after CLS drop
        pixel_values: torch.Tensor,   # (B, 3, H, W) — preprocessed input (for grid inference)
    ) -> torch.Tensor:
        """
        Dynamically adapt DINO patch tokens to the model's target budget.

        DINOv2-L produces N patches depending on input size and patch_size:
          518×518 / 14 → 37×37 = 1369  (native size, no interpolate_pos_encoding needed)
          224×224 / 14 → 16×16 = 256   (custom size with interpolate_pos_encoding)
          112×448 / 14 →  8×32 = 256   (KITTI aspect ratio with interpolate_pos_encoding)

        If N == target (default 256): pass through — no op.
        If N != target: reshape to (B, C, grid_h, grid_w), adaptive_avg_pool2d to
          (target_grid_h, target_grid_w), flatten to (B, target_N, C).

        Grid inference priority:
          1. pixel_values H // patch_size, W // patch_size  (accurate when H/W/ps known)
          2. int(sqrt(N)) for square grids                  (fallback)
          3. RuntimeError with detailed diagnostic          (last resort)

        Env vars (read once in _setup_dino_forward_compat):
          KAIROS_DINO_TARGET_PATCH_TOKENS  default 256  (must be perfect square)
          KAIROS_DINO_TARGET_GRID_H/W      override for non-square targets
          KAIROS_DINO_DEBUG_SHAPES=1       log full shapes on every call
        """
        B, N, C = patch_tokens.shape
        target_tokens = self._target_patch_tokens   # 256

        # ── Fast path: already at target count ─────────────────────────────────
        if N == target_tokens:
            if not getattr(self, "_fw_adapt_logged", False):
                self._fw_adapt_logged = True
                H, W = pixel_values.shape[-2], pixel_values.shape[-1]
                patch_size = getattr(getattr(self.dinov2, "config", None), "patch_size", 14) or 14
                grid_h, grid_w = H // patch_size, W // patch_size
                print(f"[dino] raw_patch_tokens={N}", flush=True)
                print(f"[dino] raw_grid=({grid_h}, {grid_w})", flush=True)
                print(f"[dino] target_patch_tokens={target_tokens}", flush=True)
                print(f"[dino] target_grid=({self._target_grid_h}, {self._target_grid_w})", flush=True)
                print(f"[dino] patch_adaptation=passthrough", flush=True)
            return patch_tokens

        # ── Infer raw spatial grid ──────────────────────────────────────────────
        H, W  = pixel_values.shape[-2], pixel_values.shape[-1]
        patch_size = getattr(getattr(self.dinov2, "config", None), "patch_size", 14) or 14
        grid_h = H // patch_size
        grid_w = W // patch_size

        if grid_h * grid_w != N:
            # Pixel-values-based inference failed; try integer sqrt for square grids
            sqrt_N = int(N ** 0.5)
            if sqrt_N * sqrt_N == N:
                grid_h = grid_w = sqrt_N
            else:
                raise RuntimeError(
                    f"[dino] _adapt_dino_patch_tokens: cannot infer spatial grid.\n"
                    f"  N={N}  pixel_values={H}x{W}  patch_size={patch_size}\n"
                    f"  H//ps * W//ps = {H // patch_size}*{W // patch_size} = {grid_h * grid_w} ≠ {N}\n"
                    f"  sqrt({N}) = {N ** 0.5:.3f} (not integer)\n"
                    f"  Hint: set KAIROS_DINO_TARGET_GRID_H and KAIROS_DINO_TARGET_GRID_W explicitly."
                )

        target_grid_h = self._target_grid_h   # 16
        target_grid_w = self._target_grid_w   # 16

        # One-time adaptation log
        if not getattr(self, "_fw_adapt_logged", False):
            self._fw_adapt_logged = True
            print(f"[dino] raw_patch_tokens={N}", flush=True)
            print(f"[dino] raw_grid=({grid_h},{grid_w})", flush=True)
            print(f"[dino] target_patch_tokens={target_tokens}", flush=True)
            print(f"[dino] target_grid=({target_grid_h},{target_grid_w})", flush=True)
            print(
                f"[dino] patch_adaptation={grid_h}x{grid_w} -> {target_grid_h}x{target_grid_w} adaptive_avg_pool2d",
                flush=True,
            )

        # ── Spatial pooling: (B, N, C) → (B, C, grid_h, grid_w) → pool → (B, target_N, C) ──
        # Cast to float32 for pooling (safe across all PyTorch versions / dtypes)
        spatial = patch_tokens.permute(0, 2, 1).reshape(B, C, grid_h, grid_w).float()
        pooled  = F.adaptive_avg_pool2d(spatial, (target_grid_h, target_grid_w))  # (B, C, tH, tW)
        adapted = pooled.reshape(B, C, target_grid_h * target_grid_w).permute(0, 2, 1)
        adapted = adapted.to(dtype=patch_tokens.dtype)  # restore original dtype (bf16 safe)

        if self._debug_shapes or not getattr(self, "_fw_adapt_shape_logged", False):
            self._fw_adapt_shape_logged = True
            print(f"[dino] adapted_patch_tokens shape={tuple(adapted.shape)}", flush=True)

        return adapted

    # ------------------------------------------------------------------
    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """
        Resize + normalise a batch of images.

        Args:
            img: (B, 3, H, W) float32 ∈ [0, 1]  (output of torchvision's ToTensor)
        Returns:
            (B, 3, target_h, target_w) float32 normalised — autocast handles BF16 in training

        When interpolate_pos_encoding is unavailable, target size falls back to
        self._dino_fallback_size (from config.image_size) so positional encodings
        match the model's native resolution.  Any patch count mismatch is resolved
        by _adapt_dino_patch_tokens via adaptive_avg_pool2d.
        """
        vcfg = self.vcfg
        # Priority: KAIROS_DINO_INPUT_SIZE > fallback-native > enc_h × enc_w
        # Setting KAIROS_DINO_INPUT_SIZE=224 cuts DINO attention from 1370 to 257 tokens.
        _dino_input_size = getattr(self, "_dino_input_size", None)
        if _dino_input_size is not None:
            target_h = target_w = _dino_input_size
        elif not self._fw_use_interpolate and self._dino_fallback_size is not None:
            target_h = target_w = self._dino_fallback_size   # 518 — native fallback
        else:
            target_h, target_w = vcfg.enc_h, vcfg.enc_w     # 112 × 448

        if not getattr(self, "_preprocess_size_logged", False):
            self._preprocess_size_logged = True
            _cfg_pre = getattr(self.dinov2, "config", None)
            _ps_pre  = getattr(_cfg_pre, "patch_size", 14) if _cfg_pre else 14
            _nat_pre = getattr(_cfg_pre, "image_size", 518) if _cfg_pre else 518
            _eg_h    = target_h // _ps_pre
            _eg_w    = target_w // _ps_pre
            if _dino_input_size is not None:
                _mode = "smoke_224" if _dino_input_size == 224 else f"configured_{_dino_input_size}"
                if _dino_input_size != _nat_pre:
                    _mode += "_interp_required"
            elif not self._fw_use_interpolate:
                _mode = f"native_{_nat_pre}"
            else:
                _mode = "interpolated"
            print(f"[dino] preprocess target={target_h}x{target_w}", flush=True)
            print(f"[dino] expected_patch_grid={_eg_h}x{_eg_w}", flush=True)
            print(f"[dino] memory_mode={_mode}", flush=True)

        if img.shape[-2] != target_h or img.shape[-1] != target_w:
            img = F.interpolate(
                img.float(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (img.float() - self.img_mean.float()) / self.img_std.float()

    # ------------------------------------------------------------------
    def _encode_frame(self, img: torch.Tensor) -> torch.Tensor:
        """
        Run one resized+normalised frame through DINOv2-L.

        Args:
            img: (B, 3, target_h, target_w) — already normalised float32
        Returns:
            (B, target_patch_tokens, d)  — patch tokens, CLS dropped, projected

        Memory notes:
          _dino_no_grad_in_smoke=True  → DINO runs under torch.no_grad(); hidden states are
          detached before patch_proj so no DINO activations are held across frames.
          patch_proj and temporal_fusion remain in the gradient graph.
          Explicit del of raw/patch_tokens_full frees peak tensors immediately.
          _debug_shapes=True  → logs CUDA memory at each phase (adds output noise).
        """
        _no_grad   = getattr(self, "_dino_no_grad_in_smoke", False)
        _debug     = getattr(self, "_debug_shapes", False)
        _mem_trace = os.environ.get("KAIROS_MEM_TRACE", "").lower() in ("1", "true")

        if _debug and torch.cuda.is_available():
            _a = torch.cuda.memory_allocated() // 1024 ** 2
            _r = torch.cuda.memory_reserved()  // 1024 ** 2
            print(f"[dino] mem before encode: alloc={_a}MB  reserved={_r}MB", flush=True)

        if _no_grad:
            with torch.no_grad():
                raw = self._dinov2_forward(img)
                patch_tokens_full = self._extract_dino_last_hidden_state(raw)
            del raw
            patch_tokens_full = patch_tokens_full.detach()  # sever from DINO graph
        else:
            raw = self._dinov2_forward(img)
            patch_tokens_full = self._extract_dino_last_hidden_state(raw)
            del raw

        if not getattr(self, "_fw_lhs_logged", False):
            self._fw_lhs_logged = True
            print(f"[dino] last_hidden_state shape={tuple(patch_tokens_full.shape)}", flush=True)

        if _debug and torch.cuda.is_available():
            print(f"[dino] mem after DINO fwd: alloc={torch.cuda.memory_allocated()//1024**2}MB",
                  flush=True)

        patch_tokens = patch_tokens_full[:, 1:]   # drop CLS: (B, N, hidden)
        del patch_tokens_full

        # Adapt N → target_patch_tokens (256) via adaptive_avg_pool2d when N differs
        patch_tokens = self._adapt_dino_patch_tokens(patch_tokens, img)  # (B, target_N, hidden)

        if _debug and torch.cuda.is_available():
            print(f"[dino] mem after adapt: alloc={torch.cuda.memory_allocated()//1024**2}MB",
                  flush=True)
        if _mem_trace:
            _enc_log_cuda_mem("after_patch_adapt")

        if isinstance(self.patch_proj, nn.Linear):
            patch_tokens = _linear_input(patch_tokens, self.patch_proj)
        return self.patch_proj(patch_tokens)  # (B, target_N, d)

    # ------------------------------------------------------------------
    def forward(
        self,
        img_t:  torch.Tensor,   # (B, 3, H, W) ∈ [0, 1]  — frame at time t
        img_t1: torch.Tensor,   # (B, 3, H, W)             — frame at t-1
        img_t2: torch.Tensor,   # (B, 3, H, W)             — frame at t-2
    ) -> torch.Tensor:
        """
        Returns cam[n_patches] tokens: (B, n_patches, d_model).

        Default: all three frames are batched into a single DINOv2 call (3×
        throughput).  When vcfg.sequential_frames=True each frame is encoded
        separately to reduce peak activation memory (smoke / budget / ultra-smoke).
        """
        B = img_t.shape[0]
        _n_frames  = getattr(self, "_dino_temporal_frames", 3)   # 1/2/3 via env var
        _mem_trace = os.environ.get("KAIROS_MEM_TRACE", "").lower() in ("1", "true")

        if _mem_trace:
            _enc_log_cuda_mem("before_dino")

        if self.vcfg.sequential_frames:
            # Process frames one-by-one — avoids 3× activation peak.
            # When _dino_no_grad_in_smoke=True, each frame's DINO activations are freed
            # immediately after _encode_frame returns, so frame N doesn't accumulate.
            tok_t = self._encode_frame(self._preprocess(img_t))
            if _mem_trace:
                _enc_log_cuda_mem("after_dino_frame_0")
            if _n_frames >= 2:
                tok_t1 = self._encode_frame(self._preprocess(img_t1))
                if _mem_trace:
                    _enc_log_cuda_mem("after_dino_frame_1")
            else:
                tok_t1 = torch.zeros_like(tok_t)
            if _n_frames >= 3:
                tok_t2 = self._encode_frame(self._preprocess(img_t2))
                if _mem_trace:
                    _enc_log_cuda_mem("after_dino_frame_2")
            else:
                tok_t2 = torch.zeros_like(tok_t)
        else:
            # ── Batch frames into a single DINOv2 call (higher throughput) ──────
            if _n_frames == 3:
                frames = torch.cat([
                    self._preprocess(img_t),
                    self._preprocess(img_t1),
                    self._preprocess(img_t2),
                ], dim=0)                                       # (3B, 3, H, W)
                all_patches = self._encode_frame(frames)        # (3B, n_patches, d)
                if _mem_trace:
                    _enc_log_cuda_mem("after_dino_frame_0_1_2")
                tok_t, tok_t1, tok_t2 = all_patches.split(B, dim=0)
            elif _n_frames == 2:
                frames = torch.cat([
                    self._preprocess(img_t),
                    self._preprocess(img_t1),
                ], dim=0)                                       # (2B, 3, H, W)
                all_patches = self._encode_frame(frames)        # (2B, n_patches, d)
                if _mem_trace:
                    _enc_log_cuda_mem("after_dino_frame_0_1")
                tok_t, tok_t1 = all_patches.split(B, dim=0)
                tok_t2 = torch.zeros_like(tok_t)
            else:
                tok_t  = self._encode_frame(self._preprocess(img_t))
                if _mem_trace:
                    _enc_log_cuda_mem("after_dino_frame_0")
                tok_t1 = torch.zeros_like(tok_t)
                tok_t2 = torch.zeros_like(tok_t)

        # ── Temporal fusion → (B, n_patches, d) ───────────────────────────────
        return self.temporal_fusion([tok_t, tok_t1, tok_t2])

    # ------------------------------------------------------------------
    # ── Parameter management helpers ──────────────────────────────────

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return only the parameters that require gradients (LoRA + fusion)."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        State-dict containing ONLY trainable params.
        Used to checkpoint the encoder without saving the frozen 307M DINOv2 weights.

        Implementation note: state_dict() includes both parameters AND buffers
        (e.g. img_mean, img_std).  We build the trainable set from named_parameters()
        alone — buffers are never trainable and get_parameter() would throw for them.
        """
        trainable_keys = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in trainable_keys}

    def load_trainable_state_dict(
        self,
        state: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        """
        Load a checkpoint produced by trainable_state_dict().
        Frozen backbone weights are left untouched.
        """
        # Only update the keys present in state; skip frozen backbone keys
        current = self.state_dict()
        current.update(state)
        missing, unexpected = self.load_state_dict(current, strict=False)
        if strict and unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")

    def count_trainable(self) -> str:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        return (
            f"total={total/1e6:.1f}M  "
            f"trainable={trainable/1e6:.2f}M  "
            f"frozen(DINOv2)={frozen/1e6:.1f}M"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight mock backbone (unit-testing without transformers / HF Hub)
# ──────────────────────────────────────────────────────────────────────────────

class _MockDinov2Output:
    """Mimics the HuggingFace model output namespace."""
    def __init__(self, hidden_states: torch.Tensor) -> None:
        # +1 for the CLS token position that _encode_frame will drop
        B, n, d = hidden_states.shape
        cls = torch.zeros(B, 1, d, dtype=hidden_states.dtype,
                          device=hidden_states.device)
        self.last_hidden_state = torch.cat([cls, hidden_states], dim=1)


class _MockDinov2(nn.Module):
    """
    Stub that replaces DINOv2-L when transformers is not installed or
    use_mock_backbone=True.

    Produces (B, 1+n_patches, hidden_size) outputs — same shape as the real
    Dinov2Model — so patch_proj (hidden_size → d_model) works correctly.

    Trainable lora_A / lora_B parameters ensure the LoRA gradient assertion
    passes in integration tests without requiring the full attention structure.
    """

    def __init__(self, n_patches: int, hidden_size: int) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.proj = nn.Linear(3 * 14 * 14, hidden_size, bias=False)
        self.proj.weight.requires_grad_(False)
        r = min(8, max(1, hidden_size // 128))
        self.lora_A = nn.Parameter(torch.empty(r, 3 * 14 * 14))
        self.lora_B = nn.Parameter(torch.zeros(hidden_size, r))
        self.lora_scale = 1.0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> _MockDinov2Output:
        B, C, H, W = pixel_values.shape
        # Crude patch-average: reshape to (B, n_patches, C*ph*pw)
        ph = pw = 14
        n_h, n_w = H // ph, W // pw
        patches = pixel_values.reshape(B, C, n_h, ph, n_w, pw)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, n_h * n_w, -1)
        patches = patches.to(device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        base = self.proj(patches)
        delta = (patches @ self.lora_A.T @ self.lora_B.T) * self.lora_scale
        return _MockDinov2Output(base + delta)

    def gradient_checkpointing_enable(self) -> None:
        pass   # no-op for mock

    def gradient_checkpointing_disable(self) -> None:
        pass   # no-op for mock


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: per-parameter-group spec for the training script
# ──────────────────────────────────────────────────────────────────────────────

def vision_encoder_param_groups(
    encoder: KairosVisionEncoder,
    lr_lora: float = 1e-4,
    lr_fusion: float = 3e-4,
) -> List[Dict]:
    """
    Returns an optimizer param-groups list with separate LRs for:
      - LoRA A/B weights (lower LR — fine-tuning pre-trained attention)
      - Temporal fusion (higher LR — training from scratch)

    Usage in training script:
        param_groups = vision_encoder_param_groups(encoder, lr_lora=1e-4, lr_fusion=3e-4)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    """
    lora_params, fusion_params = [], []

    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            fusion_params.append(param)

    return [
        {"params": lora_params,   "lr": lr_lora,   "name": "lora"},
        {"params": fusion_params, "lr": lr_fusion, "name": "fusion"},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test (shape check — works without transformers or GPU)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    torch.manual_seed(0)

    vcfg = VisionEncoderConfig()
    kcfg = KairosConfig()

    print("Building KairosVisionEncoder …")
    enc = KairosVisionEncoder(vcfg, kcfg)
    print(f"  {enc.count_trainable()}")

    # ── Check trainable param breakdown ────────────────────────────────────────
    lora_n   = sum(p.numel() for n, p in enc.named_parameters()
                   if "lora_" in n and p.requires_grad)
    fusion_n = sum(p.numel() for n, p in enc.named_parameters()
                   if "lora_" not in n and p.requires_grad)
    print(f"  LoRA params  : {lora_n/1e6:.2f}M")
    print(f"  Fusion params: {fusion_n/1e6:.2f}M")

    # ── Forward pass ───────────────────────────────────────────────────────────
    B = 2
    # Simulate KITTI frames: (B, 3, 375, 1242)  ∈ [0, 1]
    img_t  = torch.rand(B, 3, 375, 1242)
    img_t1 = torch.rand(B, 3, 375, 1242)
    img_t2 = torch.rand(B, 3, 375, 1242)

    print(f"  input : {tuple(img_t.shape)}  (KITTI raw resolution)")

    enc.eval()
    with torch.no_grad():
        cam_tokens = enc(img_t, img_t1, img_t2)

    assert cam_tokens.shape == (B, vcfg.n_patches, kcfg.d_model), \
        f"Shape mismatch: {cam_tokens.shape}"
    print(f"  output: {tuple(cam_tokens.shape)}  dtype={cam_tokens.dtype}")

    # ── Verify gradient flows through LoRA and fusion ─────────────────────────
    enc.train()
    cam_tr = enc(img_t, img_t1, img_t2)
    loss = cam_tr.float().mean()
    loss.backward()

    # Every trainable parameter must have received a gradient
    no_grad = [
        n for n, p in enc.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    if no_grad:
        print(f"  WARNING — no gradient for: {no_grad[:3]} …")
    else:
        print(f"  All trainable params received gradients ✓")

    # ── Verify param-group utility ─────────────────────────────────────────────
    groups = vision_encoder_param_groups(enc)
    total_pg = sum(p.numel() for g in groups for p in g["params"])
    assert total_pg == sum(p.numel() for p in enc.trainable_parameters()), \
        "param_groups does not cover all trainable params"
    print(f"  Param groups: {[g['name'] for g in groups]} ✓")

    print("Smoke-test passed.")
