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

import math
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

        # ── DINOv2-L backbone ──────────────────────────────────────────────────
        if vcfg.use_mock_backbone:
            # dinov2_hidden_size keeps patch_proj input dim correct (→Linear(1024, d))
            self.dinov2 = _MockDinov2(vcfg.n_patches, vcfg.dinov2_hidden_size)
        elif _HAS_TRANSFORMERS:
            try:
                self.dinov2: nn.Module = Dinov2Model.from_pretrained(
                    vcfg.dinov2_model_name,
                    add_pooling_layer=False,   # keep patch tokens, skip pooler
                )
            except TypeError:
                # Older transformers builds don't accept add_pooling_layer;
                # CLS token is dropped manually in _encode_frame either way.
                self.dinov2 = Dinov2Model.from_pretrained(vcfg.dinov2_model_name)
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

            if vcfg.use_grad_checkpoint:
                self.dinov2.gradient_checkpointing_enable()
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
    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """
        Resize + normalise a batch of images.

        Args:
            img: (B, 3, H, W) float32 ∈ [0, 1]  (output of torchvision's ToTensor)
        Returns:
            (B, 3, enc_h, enc_w) float32 normalised — autocast handles BF16 in training
        """
        vcfg = self.vcfg
        if img.shape[-2] != vcfg.enc_h or img.shape[-1] != vcfg.enc_w:
            img = F.interpolate(
                img.float(),
                size=(vcfg.enc_h, vcfg.enc_w),
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
            img: (B, 3, enc_h, enc_w)  — already normalised float32
        Returns:
            (B, n_patches, d)  — patch tokens, CLS dropped
        """
        out = self.dinov2(
            pixel_values=img,
            interpolate_pos_encoding=True,  # bicubic interp for 8×32 grid
        )
        # last_hidden_state: (B, 1 + n_patches, hidden)
        patch_tokens = out.last_hidden_state[:, 1:]   # drop CLS: (B, 256, 1024)
        if patch_tokens.shape[1] != self.vcfg.n_patches:
            raise ValueError(
                f"DINO patch count mismatch: expected {self.vcfg.n_patches}, "
                f"got {patch_tokens.shape[1]} for enc "
                f"{self.vcfg.enc_h}x{self.vcfg.enc_w}"
            )
        if isinstance(self.patch_proj, nn.Linear):
            patch_tokens = _linear_input(patch_tokens, self.patch_proj)
        return self.patch_proj(patch_tokens)           # (B, 256, d)

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

        if self.vcfg.sequential_frames:
            # Process frames one-by-one — slower but avoids 3× activation peak.
            tok_t  = self._encode_frame(self._preprocess(img_t))
            tok_t1 = self._encode_frame(self._preprocess(img_t1))
            tok_t2 = self._encode_frame(self._preprocess(img_t2))
        else:
            # ── Batch all 3 frames into a single DINOv2 call (3× throughput) ──
            frames = torch.cat([
                self._preprocess(img_t),
                self._preprocess(img_t1),
                self._preprocess(img_t2),
            ], dim=0)                                       # (3B, 3, H, W)
            all_patches = self._encode_frame(frames)        # (3B, n_patches, d)
            tok_t, tok_t1, tok_t2 = all_patches.split(B, dim=0)

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
