"""
kairos_train.py  —  Kairos-4B SageMaker training script.

Reads the Gold Delta table (S2FT triplets) from S3, runs curriculum
learning (easy → medium → hard), and trains with DeepSpeed ZeRO-3 + BF16.

Launch from SageMaker:

    from sagemaker.pytorch import PyTorch
    estimator = PyTorch(
        entry_point="kairos_train.py",
        role=os.environ["SAGEMAKER_ROLE_ARN"],
        instance_type="ml.g5.48xlarge",
        instance_count=1,
        framework_version="2.3",
        py_version="py311",
        distribution={"torch_distributed": {"enabled": True}},
        use_spot_instances=True,
        max_wait=86400,
        hyperparameters={
            "total_steps": 10000,
            "micro_batch": 2,
            "grad_accum": 4,
            "warmup_frac": 0.03,
            "stable_frac": 0.77,
        },
        checkpoint_s3_uri="s3://kairos-emr-assets-use1-195231312992/checkpoints/kairos-4b/",
        output_path="s3://kairos-emr-assets-use1-195231312992/checkpoints/kairos-4b/",
        region_name="us-east-1",
    )

Optimisations vs v1:
  - Fork-safe S3 cache via threading.local (boto3 per-thread, not per-process)
  - Parallel S3 downloads per sample (ThreadPoolExecutor inside __getitem__)
  - Single S3 read for all training tiers; curriculum phases filtered in Python
  - _to_device() defined once outside training loop (no per-step closure alloc)
  - Micro-step counter instead of is_gradient_accumulation_boundary() ordering
  - DistributedSampler.set_epoch() called correctly on each epoch cycle
  - NaN / Inf loss guard (zeroes loss, logs warning, continues cleanly)
  - Gradient norm logged via DeepSpeed engine
  - pandas NaN check for optional columns (oxts_path, lidar_path_t_minus_1)

Accuracy vs v1:
  - z-loss now rolled into output.total_loss in kairos_model.py — no risk of
    forgetting to add it in the training script
  - label_smoothing=0.1 in CE loss (kairos_model.py)
  - z-loss accumulation bug fixed in kairos_hybrid_block.py (was picking up
    stale tensors from blocks not yet called this forward pass)
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    import deepspeed  # type: ignore
    _HAS_DS = True
except ImportError:
    _HAS_DS = False

try:
    from rich.console import Console
    from rich.live import Live
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TaskProgressColumn, TextColumn, TimeElapsedColumn,
        TimeRemainingColumn,
    )
    _HAS_RICH = True
    _console = Console(highlight=False)
except ImportError:
    _HAS_RICH = False
    _console = None  # type: ignore[assignment]

sys.path.insert(0, "/opt/ml/code")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kairos_model as _kairos_model_mod
from kairos_model import (
    KairoBatch, KairosModel, KairosModelConfig, KairosOutput,
    sync_moe_expert_bias,
)
from kairos_fusion import CalibMatrices

try:
    from botocore.exceptions import ClientError
    _CLIENT_ERROR_TYPES = (ClientError,)
except ImportError:
    ClientError = None  # type: ignore[assignment]
    _CLIENT_ERROR_TYPES = ()


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

AWS_REGION    = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
# DATA_REGION: region of the S3 bucket holding Gold parquet + raw KITTI assets.
# Set this to eu-north-1 (Stockholm) when SageMaker runs in eu-central-1 (Frankfurt).
# Defaults to AWS_REGION so single-region setups need no change.
DATA_REGION   = os.getenv("DATA_REGION", AWS_REGION)
GOLD_S3       = os.getenv("GOLD_S3", "")   # must be set via env — no stale default
CKPT_S3       = os.getenv("CKPT_S3", "")   # must be set via env — no stale default
CKPT_LOCAL    = Path("/tmp/kairos_ckpt")
CACHE_DIR     = Path("/tmp/kairos_cache")
MAX_PTS       = 30_000
MAX_PROMPT    = 512
MAX_TARGET    = 512
IMU_TOKENS    = 8
BOS, EOS, PAD = 256, 257, 0
bad_oxts_seen = 0
oxts_fallbacks_used = 0

# Curriculum phase boundaries as fraction of total_steps
PHASE_EASY_END   = 0.15   # easy only for first 15 %
PHASE_MEDIUM_END = 0.45   # easy + medium for next 30 %
# Remaining 55 %: full data


# ─────────────────────────────────────────────────────────────────────────────
# Env helpers — safe reads for plain-python SageMaker launches (no MPI)
# ─────────────────────────────────────────────────────────────────────────────

def _str2bool(v) -> bool:
    """Argparse bool parser that accepts SageMaker-style string values."""
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {v!r}")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total_steps",   type=int,   default=10_000)
    p.add_argument("--micro_batch",   type=int,   default=2,
                   help="Samples per GPU per forward pass")
    p.add_argument("--grad_accum",    type=int,   default=4)
    p.add_argument("--warmup_frac",   type=float, default=0.03)
    p.add_argument("--stable_frac",   type=float, default=0.77)
    p.add_argument("--lr_lora",       type=float, default=5e-5)
    p.add_argument("--lr_encoders",   type=float, default=5e-5)
    p.add_argument("--lr_core",       type=float, default=1e-5)
    p.add_argument("--lr_decoder",    type=float, default=5e-5)
    p.add_argument("--freeze_core_steps", type=int, default=3000)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--z_loss_coeff",  type=float, default=1e-3)
    p.add_argument("--gold_s3",       type=str,   default=GOLD_S3)
    p.add_argument("--ckpt_s3",       type=str,   default=CKPT_S3)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--ckpt_every_min",type=int,   default=30)
    p.add_argument("--log_every",     type=int,   default=10)
    p.add_argument("--val_every",     type=int,   default=None,
                   help="Validation interval; defaults to 500 in full mode and 0 in smoke/budget")
    p.add_argument("--skip_bad_rows", nargs="?", const=True, default=None,
                   type=_str2bool,
                   help="Retry past rows with missing S3 assets. Default True in "
                        "smoke/ultra_smoke/budget, False in full training.")
    p.add_argument("--preflight_rows", type=int, default=None,
                   help="HeadObject-check N sampled training rows before DataLoader creation. "
                        "Default 20 in smoke/ultra_smoke, 0 otherwise.")
    p.add_argument("--allow_oxts_fallback", nargs="?", const=True, default=None,
                   type=_str2bool,
                   help="Use zero/small-noise IMU tokens when OXTS is malformed. "
                        "Default True in smoke/ultra_smoke, False in budget/full.")
    p.add_argument("--allow_single_rank", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Allow WORLD_SIZE=1 even when multiple GPUs are visible. "
                        "Use only for intentional single-rank debugging.")
    # DeepSpeed absorb
    p.add_argument("--deepspeed",        default=None)
    p.add_argument("--deepspeed_config", default=None)
    p.add_argument("--local_rank",    type=int,   default=-1)
    # Smoke / debug mode — reduces memory pressure for a quick sanity run
    p.add_argument("--smoke_mode",    nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Reduce lidar/prompt/target size, disable val+ckpt, "
                        "limit workers for low-VRAM debug runs")
    p.add_argument("--budget_mode",   nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Memory-safe real training mode for g5.12xlarge/g5.24xlarge")
    p.add_argument("--zero_offload",  nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Emergency fallback: CPU offload for ZeRO-3 params/optimizer")
    p.add_argument("--save_ckpt_in_smoke", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Explicitly allow checkpoint saving in smoke_mode")
    p.add_argument("--mem_trace", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Enable KAIROS_MEM_TRACE=1 for direct, non-SageMaker launches. "
                        "SageMaker launches should set this through estimator.environment.")
    # ── Ultra-smoke mode (implies smoke_mode; designed for g5.12xlarge 24 GB/GPU) ──
    p.add_argument("--ultra_smoke_mode", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Ultra-light smoke path: aggressively shrinks every component "
                        "while preserving real code paths.  Implies smoke_mode. "
                        "Designed to fit on g5.12xlarge (24 GB/GPU).")
    p.add_argument("--ultra_smoke_mock_vision", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="In ultra_smoke: swap DINO for mock backbone to validate "
                        "DeepSpeed/data/training loop without real DINO memory cost.")
    # ── Per-component freeze flags (default=None → mode-driven default applies) ──
    p.add_argument("--freeze_lidar_in_smoke", nargs="?", const=True, default=None,
                   type=_str2bool,
                   help="Freeze LiDAR encoder during smoke/ultra_smoke. "
                        "Default True for ultra_smoke, False for smoke/budget.")
    p.add_argument("--freeze_imu_in_smoke",   nargs="?", const=True, default=None,
                   type=_str2bool,
                   help="Freeze IMU encoder during smoke/ultra_smoke. "
                        "Default True for ultra_smoke, False for smoke/budget.")
    p.add_argument("--freeze_core_in_smoke",  nargs="?", const=True, default=None,
                   type=_str2bool,
                   help="Freeze hybrid core during smoke/ultra_smoke. "
                        "Default False for all modes.")
    p.add_argument("--instance_type", type=str, default="unknown",
                   help="SageMaker instance type string; used only for VRAM warnings.")
    # ── ultra_smoke debug binary-search flags ─────────────────────────────────
    p.add_argument("--ultra_smoke_dino_no_grad", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="In ultra_smoke: run entire vision_encoder (incl. LoRA) under "
                        "torch.no_grad(). Freezes LoRA — use only to validate "
                        "DeepSpeed/data plumbing without DINO activation overhead.")
    p.add_argument("--ultra_smoke_skip_lidar", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Replace LiDAR encoder output with zero tensors of correct shape. "
                        "Shape-preserving; ultra_smoke debug only.")
    p.add_argument("--ultra_smoke_skip_imu", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Replace IMU encoder output with zero tensors of correct shape. "
                        "Shape-preserving; ultra_smoke debug only.")
    p.add_argument("--ultra_smoke_skip_decoder_loss", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Skip S2FT decoder CE and use an anchor-only dummy loss. "
                        "Ultra_smoke debug only.")
    p.add_argument("--ultra_smoke_skip_core", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Bypass hybrid core (Mamba+CfC+SWA+MoE) entirely in forward. "
                        "Validates SageMaker launch/data/DeepSpeed without core backward. "
                        "Use with --ultra_smoke_skip_decoder_loss. Ultra_smoke debug only.")
    p.add_argument("--force_core_with_anchor_loss", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="When --ultra_smoke_skip_decoder_loss=True, force the hybrid core "
                        "to still run (default: auto skip_core for safety). "
                        "moe_z_loss is always excluded from total_loss in this mode. "
                        "Ultra_smoke debug only.")
    p.add_argument("--ultra_smoke_core_loss", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Replace total_loss with x.pow(2).mean()*1e-4 from hybrid core output. "
                        "Tests the core backward path end-to-end. "
                        "Use with --dense_moe_fallback True for ZeRO-3 safety. "
                        "Ultra_smoke debug only.")
    p.add_argument("--dense_moe_fallback", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Use dense weighted-sum MoE instead of sparse top-k dispatch. "
                        "Avoids ZeRO-3 shape mismatches in the expert GEMM backward. "
                        "Auto-enabled when --ultra_smoke_core_loss=True. "
                        "Ultra_smoke debug only.")
    p.add_argument("--zero_stage", type=int, default=None, choices=[2, 3],
                   help="Override DeepSpeed ZeRO stage (2 or 3). Default: 3. "
                        "Stage 2 is recommended for ultra_smoke plumbing runs because "
                        "it is less fragile with sparse/custom backward graphs.")
    p.add_argument("--disable_curriculum", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Use full training dataset for all curriculum phases "
                        "(easy / easy+medium / full → all map to df_train). "
                        "Automatically True in ultra_smoke_mode. "
                        "Prevents empty DataLoaders when curriculum subsets contain "
                        "fewer rows than world_size * micro_batch.")
    # ── ZeRO-3 backward isolation / module debug ─────────────────────────────
    p.add_argument("--core_debug_bypass_mamba", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="ZeRO-3 debug: replace Mamba-2 sub-block with identity pass. "
                        "Requires --ultra_smoke_mode True.")
    p.add_argument("--core_debug_bypass_cfc", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="ZeRO-3 debug: replace CfC (IMU) sub-block with identity pass.")
    p.add_argument("--core_debug_bypass_swa", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="ZeRO-3 debug: replace Sliding-Window Attention with identity pass.")
    p.add_argument("--core_debug_bypass_moe", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="ZeRO-3 debug: replace MoE FFN with identity pass.")
    p.add_argument("--core_debug_layers", type=int, default=0,
                   help="ZeRO-3 debug: limit hybrid core to first N loop iterations. "
                        "0 = run all loops (default). Requires --ultra_smoke_mode True.")
    p.add_argument("--ultra_smoke_core_loss_scope", type=str, default="post_core",
                   choices=["post_core", "post_mamba", "post_cfc", "post_swa", "post_moe"],
                   help="When --ultra_smoke_core_loss=True, scope which sub-block output "
                        "is used as the backward target. post_mamba/cfc/swa auto-sets "
                        "corresponding bypass flags for sub-blocks beyond that scope.")
    p.add_argument("--debug_grad_shapes", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Register backward hooks to print gradient shapes and catch "
                        "zero-sized gradients. Prints for first 3 backward passes. "
                        "Useful for diagnosing ZeRO-3 shape mismatches.")
    p.add_argument("--debug_dtype_shapes", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="On first batch, rank-0 prints dtype/device of all major "
                        "modality tensors at LiDAR/IMU/core boundaries. "
                        "Sets KAIROS_DEBUG_DTYPE=1. Helps diagnose BF16/float32 "
                        "mismatches when LiDAR or IMU encoders are enabled.")
    p.add_argument("--zero3_debug_safe", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Use conservative ZeRO-3 settings for backward debug: "
                        "overlap_comm=False, contiguous_gradients=False, "
                        "small bucket/prefetch sizes. Only active with --zero_stage 3.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DeepSpeed config
# ─────────────────────────────────────────────────────────────────────────────

def _ds_config(args: argparse.Namespace) -> dict:
    _is_ultra = getattr(args, "ultra_smoke_mode", False)
    _is_smoke = args.smoke_mode or args.budget_mode
    _bucket   = 5e6 if _is_ultra else (2e7 if _is_smoke else 5e7)
    # ZeRO stage selection:
    # - Default to ZeRO-2 for ultra_smoke and all smoke/budget runs.
    #   Stage 2 is significantly more stable with sparse/custom backward graphs
    #   (no parameter sharding means no cross-rank shape coordination).
    #   Stage 1d failure: ZeRO-3 + dense MoE core backward raised shape 0 vs 8.
    # - Full training defaults to ZeRO-3.
    # - Explicit --zero_stage always takes precedence.
    _zero_stage = getattr(args, "zero_stage", None)
    if _zero_stage is None:
        if _is_ultra or _is_smoke:
            _zero_stage = 2
        else:
            _zero_stage = 3
    zero_cfg: dict = {
        "stage": _zero_stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": _bucket,
    }
    if _zero_stage == 3:
        # stage3_* keys are only valid for ZeRO stage 3; passing them with
        # stage 2 causes DeepSpeed to raise a ConfigurationError.
        zero_cfg.update({
            "stage3_prefetch_bucket_size": _bucket,
            "stage3_param_persistence_threshold": 1e4 if _is_ultra else 1e6,
            "stage3_max_live_parameters": 5e8 if _is_ultra else 1e9,
            "stage3_max_reuse_distance": 5e8 if _is_ultra else 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        })
        if getattr(args, 'zero3_debug_safe', False):
            # Conservative ZeRO-3: disable async comm overlap and use small buckets
            # to reduce race conditions in the backward graph under ZeRO-3.
            zero_cfg["overlap_comm"] = False
            zero_cfg["contiguous_gradients"] = False
            zero_cfg["reduce_bucket_size"] = 5e5
            zero_cfg["stage3_prefetch_bucket_size"] = 5e5
            zero_cfg["stage3_param_persistence_threshold"] = 1e3
    if args.zero_offload:
        zero_cfg["offload_param"] = {"device": "cpu", "pin_memory": True}
        zero_cfg["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    ds_cfg = {
        "bf16": {"enabled": True},
        "zero_optimization": zero_cfg,
        "gradient_accumulation_steps": args.grad_accum,
        "gradient_clipping": args.max_grad_norm,
        "train_micro_batch_size_per_gpu": args.micro_batch,
        "steps_per_print": args.log_every,
        "wall_clock_breakdown": False,
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 4,
        },
    }
    if _is_ultra:
        ds_cfg["activation_checkpointing"] = {
            "partition_activations": False,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "number_checkpoints": 0,
        }
    return ds_cfg


# ─────────────────────────────────────────────────────────────────────────────
# DeepSpeed 0.14.4 NameError workaround
# ─────────────────────────────────────────────────────────────────────────────

def _patch_ds_warned() -> None:
    """
    DeepSpeed 0.14.4 NameError fix for apply_to_tensors_only.

    Bug: `warned` is referenced before being defined (either never initialised,
    or assigned only after use, making Python treat it as local).

    Correct DS signature is apply_to_tensors_only(function, value, ...) —
    function first, value second.  The previous patch had them reversed, which
    caused the hook itself to be returned instead of the processed batch, leading
    to AttributeError: 'function' has no attribute 'img_t'.

    This replacement:
      - Fixes argument order: (function, value)
      - Recurses into list / tuple (preserving type)
      - Reconstructs namedtuples via positional args (not iterable constructor)
      - Recurses into @dataclass instances field-by-field (KairoBatch, CalibMatrices)
      - Recurses into dicts
      - Returns all other objects unchanged (no warning-message logic needed for DS 0.14)
      - Patches deepspeed.runtime.zero.utils AND all already-imported caller namespaces
    """
    if not _HAS_DS:
        print("[patch] DeepSpeed warned fix not needed", flush=True)
        return
    try:
        import dataclasses as _dc
        import inspect as _inspect
        import sys as _sys

        # `import deepspeed` at module load already pulls in the full chain.
        _ds_zero_utils = _sys.modules.get("deepspeed.runtime.zero.utils")
        if _ds_zero_utils is None:
            print("[patch] DeepSpeed warned fix not needed", flush=True)
            return

        fn = getattr(_ds_zero_utils, "apply_to_tensors_only", None)
        if fn is None:
            print("[patch] DeepSpeed warned fix not needed", flush=True)
            return

        try:
            src = _inspect.getsource(fn)
        except (OSError, TypeError):
            print("[patch] DeepSpeed warned fix not needed", flush=True)
            return

        # Bug present: uses `not warned` with no prior `warned = False`.
        has_check = "not warned" in src
        has_init  = "warned = False" in src or "warned=False" in src
        if not has_check or has_init:
            print("[patch] DeepSpeed warned fix not needed", flush=True)
            return

        import torch as _torch

        _call_logged = [False]   # print once on first invocation (SageMaker runtime)

        def _safe_apply(function, value, warning_msg_fn=None, **_kw):
            if not _call_logged[0]:
                print("[patch] DeepSpeed apply_to_tensors_only warned fix active",
                      flush=True)
                _call_logged[0] = True

            def _recurse(obj):
                if isinstance(obj, _torch.Tensor):
                    return function(obj)
                # namedtuple — subclass of tuple but needs positional constructor
                if isinstance(obj, tuple) and hasattr(type(obj), "_fields"):
                    return type(obj)(*[_recurse(v) for v in obj])
                if isinstance(obj, (list, tuple)):
                    return type(obj)(_recurse(v) for v in obj)
                if isinstance(obj, dict):
                    return {k: _recurse(v) for k, v in obj.items()}
                if _dc.is_dataclass(obj) and not isinstance(obj, type):
                    new_fields = {
                        f.name: _recurse(getattr(obj, f.name))
                        for f in _dc.fields(obj)
                    }
                    return type(obj)(**new_fields)
                return obj   # scalar, string, None, etc. — unchanged

            return _recurse(value)

        _ds_zero_utils.apply_to_tensors_only = _safe_apply
        patched = 1

        # Also patch caller namespaces in deepspeed.runtime.zero.* that imported
        # the function by name at load time (stage3, stage_1_and_2, etc.).
        for _mod_name, _mod in list(_sys.modules.items()):
            if _mod is None or _mod is _ds_zero_utils:
                continue
            if not _mod_name.startswith("deepspeed.runtime.zero"):
                continue
            if getattr(_mod, "apply_to_tensors_only", None) is fn:
                setattr(_mod, "apply_to_tensors_only", _safe_apply)
                patched += 1

        print(
            f"[patch] applied DeepSpeed apply_to_tensors_only warned fix "
            f"({patched} location{'s' if patched != 1 else ''})",
            flush=True,
        )
    except Exception as _exc:
        print(f"[patch] DeepSpeed warned fix check failed: {_exc}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dependency version guard — fails fast if botocore was downgraded
# ─────────────────────────────────────────────────────────────────────────────

def _check_botocore_version() -> None:
    """
    Fail fast if requirements.txt accidentally downgraded the DLC botocore.

    s3fs/aiobotocore are the known culprits — they pull botocore 1.31.x which
    breaks SageMaker imports (is_s3express_bucket missing from botocore.utils).
    Only enforced inside SageMaker (SM_TRAINING_ENV present) so local dev works.
    """
    if not os.environ.get("SM_TRAINING_ENV"):
        return
    try:
        import botocore as _bc
        parts = tuple(int(x) for x in _bc.__version__.split(".")[:3])
        if parts < (1, 34, 112):
            raise RuntimeError(
                f"[kairos] FATAL: botocore {_bc.__version__} is too old — "
                "the SageMaker DLC requires botocore>=1.34.112. "
                "Remove s3fs, fsspec, and aiobotocore from requirements.txt; "
                "they downgrade botocore and break SageMaker S3/import paths. "
                "Use pyarrow.fs.S3FileSystem for parquet reads instead of s3fs."
            )
        print(f"[dep_guard] botocore OK: {_bc.__version__}", flush=True)
    except ImportError:
        pass  # botocore may not be importable in unusual envs; skip silently


def _log_dependency_versions(rank: int) -> None:
    """Print key package versions on rank-0 for post-hoc CloudWatch debugging."""
    if rank != 0:
        return
    import importlib
    pkgs = [
        "torch", "deepspeed", "transformers",
        "boto3", "botocore",
        "pyarrow", "pandas",
    ]
    versions = {}
    for pkg in pkgs:
        try:
            mod = importlib.import_module(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "NOT_INSTALLED"
    print(
        "[dep_versions] " + "  ".join(f"{k}={v}" for k, v in versions.items()),
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# WSD LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def _wsd_lambda(step: int, warmup: int, stable: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    if step < warmup + stable:
        return 1.0
    decay_steps = total - warmup - stable
    decay_pos   = step - warmup - stable
    return 0.5 * (1.0 + math.cos(math.pi * decay_pos / max(1, decay_steps)))


# ─────────────────────────────────────────────────────────────────────────────
# S3 cache — fork-safe via threading.local
# ─────────────────────────────────────────────────────────────────────────────

class _S3Cache:
    """
    Thread- and fork-safe S3 file cache.

    boto3 connection pools are NOT fork-safe.  Using threading.local() means
    each thread (= each DataLoader worker process after fork) gets its own
    boto3 client on first use, avoiding shared-socket corruption.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._tls         = threading.local()  # per-thread boto3 client
        self._file_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._resolved_calib_uris: Dict[str, str] = {}

    @property
    def _s3(self):
        if not hasattr(self._tls, "client"):
            self._tls.client = boto3.client("s3", region_name=DATA_REGION)
        return self._tls.client

    def local(self, s3_uri: str) -> Path:
        bucket, key = self._split(s3_uri)
        local = self._dir / key.replace("/", "_")
        if local.exists():
            return local
        with self._lock_for(s3_uri):
            if local.exists():
                return local
            try:
                self._s3.download_file(bucket, key, str(local))
            except _CLIENT_ERROR_TYPES as exc:
                if _is_missing_s3_error(exc):
                    raise FileNotFoundError(
                        f"S3 object not found during download: {s3_uri}"
                    ) from exc
                raise
        return local

    def _split(self, uri: str) -> Tuple[str, str]:
        parts = uri.removeprefix("s3://").split("/", 1)
        return parts[0], parts[1]

    def _lock_for(self, key: str) -> threading.Lock:
        with self._global_lock:
            if key not in self._file_locks:
                self._file_locks[key] = threading.Lock()
            return self._file_locks[key]


def _is_missing_s3_error(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    err = response.get("Error", {})
    code = str(err.get("Code", "")).strip()
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def _nested_kitti_calib_uri(uri: str) -> Optional[str]:
    filename = uri.rsplit("/", 1)[-1]
    if filename not in {"calib_cam_to_cam.txt", "calib_velo_to_cam.txt"}:
        return None
    parent, _ = uri.rsplit("/", 1)
    calib_dir = parent.rsplit("/", 1)[-1]
    if not calib_dir.endswith("_calib"):
        return None
    date = calib_dir.removesuffix("_calib")
    if len(date) != 10 or date.count("_") != 2:
        return None
    return f"{parent}/{date}/{filename}"


def _resolve_calib_uri(uri: str, cache: _S3Cache) -> str:
    with cache._global_lock:
        resolved = cache._resolved_calib_uris.get(uri)
    if resolved is not None:
        return resolved

    try:
        cache.local(uri)
        resolved = uri
    except FileNotFoundError as original_exc:
        alt = _nested_kitti_calib_uri(uri)
        if alt is None:
            raise
        try:
            cache.local(alt)
        except FileNotFoundError as alt_exc:
            raise FileNotFoundError(
                f"Calibration file missing: original={uri}; "
                f"tried_nested={alt}; nested_error={alt_exc}"
            ) from original_exc
        resolved = alt
        print(f"[WARN][calib_path_fix] original={uri} resolved={resolved}",
              flush=True)

    with cache._global_lock:
        cache._resolved_calib_uris[uri] = resolved
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# KITTI calibration parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_calib(
    cc_path: str,
    vc_path: str,
    cache: _S3Cache,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (P2 [3,4], R0_rect [3,3], Tr_velo_to_cam [3,4]) float32."""

    def _read(uri: str, required_keys: frozenset) -> Dict[str, np.ndarray]:
        try:
            resolved_uri = _resolve_calib_uri(uri, cache)
            local = cache.local(resolved_uri)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Calibration file missing: {uri}; detail={exc}"
            ) from exc
        except _CLIENT_ERROR_TYPES as exc:
            if _is_missing_s3_error(exc):
                raise FileNotFoundError(f"Calibration file missing: {uri}") from exc
            raise
        data: Dict[str, np.ndarray] = {}
        with open(local) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                tag, vals_str = line.split(":", 1)
                key = tag.strip()
                nums: list = []
                all_numeric = True
                for tok in vals_str.split():
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        all_numeric = False
                        break
                if not all_numeric:
                    if key in required_keys:
                        raise ValueError(
                            f"Calibration key {key!r} in {uri!r} has non-numeric "
                            f"value: {vals_str.strip()!r}"
                        )
                    continue  # silently skip metadata lines (e.g. calib_time)
                if nums:
                    data[key] = np.array(nums, dtype=np.float32)
        return data

    _CC_REQUIRED = frozenset({"P2", "P_rect_02", "R_rect_00", "R0_rect"})
    _VC_REQUIRED = frozenset({"R", "T"})
    cc = _read(cc_path, _CC_REQUIRED)
    vc = _read(vc_path, _VC_REQUIRED)

    P2_key  = next(k for k in cc if k in ("P2", "P_rect_02"))
    R0_key  = next(k for k in cc if k in ("R_rect_00", "R0_rect"))
    P2      = cc[P2_key].reshape(3, 4)
    R0_rect = cc[R0_key].reshape(3, 3)

    R_v2c = vc["R"].reshape(3, 3)
    T_v2c = vc["T"].reshape(3, 1)
    Tr    = np.hstack([R_v2c, T_v2c])   # (3, 4)

    return P2, R0_rect, Tr


# ─────────────────────────────────────────────────────────────────────────────
# Low-level loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(s3_uri: str, cache: _S3Cache) -> np.ndarray:
    """Load PNG → (3, H, W) float32 ∈ [0, 1]."""
    img = np.array(
        Image.open(cache.local(s3_uri)).convert("RGB"), dtype=np.float32
    ) / 255.0
    return img.transpose(2, 0, 1)   # CHW


def _load_lidar(s3_uri: str, cache: _S3Cache, max_pts: int = MAX_PTS) -> np.ndarray:
    """Load velodyne .bin → (max_pts, 4) float32 XYZR."""
    pts = np.fromfile(str(cache.local(s3_uri)), dtype=np.float32).reshape(-1, 4)
    if pts.shape[0] >= max_pts:
        rng = np.random.default_rng(abs(hash(s3_uri)) % (2**32))
        pts = pts[rng.choice(pts.shape[0], max_pts, replace=False)]
    else:
        pts = np.vstack([pts, np.zeros((max_pts - pts.shape[0], 4), dtype=np.float32)])
    return pts


_NUMERIC_TOKEN_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)

# Matches the image_02/data/<frame>.png suffix so we can derive the OXTS path.
_IMAGE02_RE = re.compile(r"(/image_02/data/)(\d{10})\.png$")


def _derive_oxts_from_image_path(image_path: str) -> Optional[str]:
    """Derive OXTS S3 URI from image_path when oxts_path is absent from the Gold table.

    Transforms:  .../image_02/data/0000000042.png
    Into:        .../oxts/data/0000000042.txt
    Returns None if image_path doesn't match the expected pattern.
    """
    m = _IMAGE02_RE.search(image_path)
    if m is None:
        return None
    return image_path[: m.start()] + "/oxts/data/" + m.group(2) + ".txt"


def _oxts_fallback(s3_uri: str, n_tokens: int = IMU_TOKENS) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(abs(hash(s3_uri)) % (2**32))
    imu = rng.normal(0, 1e-4, (n_tokens, 7)).astype(np.float32)
    ts = np.arange(n_tokens, dtype=np.float32) / 30.0
    return imu, ts


def _warn_bad_oxts(s3_uri: str, reason: str) -> None:
    global bad_oxts_seen
    bad_oxts_seen += 1
    print(f"[WARN][bad_oxts] uri={s3_uri} reason={reason}", flush=True)


def _warn_oxts_fallback(s3_uri: str) -> None:
    global oxts_fallbacks_used
    oxts_fallbacks_used += 1
    print(f"[WARN][oxts_fallback] uri={s3_uri}", flush=True)


def _looks_like_oxts_data_path(s3_uri: str) -> bool:
    norm = s3_uri.replace("\\", "/")
    return "/oxts/data/" in norm and norm.endswith(".txt") and not norm.endswith("timestamps.txt")


def _extract_first_numeric_oxts_row(text: str) -> Optional[np.ndarray]:
    for line in text.splitlines():
        tokens = line.strip().split()
        if not tokens:
            continue
        numeric = [float(tok) for tok in tokens if _NUMERIC_TOKEN_RE.match(tok)]
        if len(numeric) >= 9:
            return np.asarray(numeric, dtype=np.float32)
    return None


def _load_oxts(
    s3_uri: str,
    cache: _S3Cache,
    n_tokens: int = IMU_TOKENS,
    allow_fallback: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one KITTI OXTS file, replicate to n_tokens rows with tiny noise.
    Returns imu_data (n_tokens, 7) and timestamps (n_tokens,).
    """
    if not _looks_like_oxts_data_path(s3_uri):
        reason = "path does not match /oxts/data/*.txt"
        print(f"[WARN][bad_oxts_path] uri={s3_uri} reason={reason}", flush=True)
        _warn_bad_oxts(s3_uri, reason)
        if allow_fallback:
            _warn_oxts_fallback(s3_uri)
            return _oxts_fallback(s3_uri, n_tokens)
        raise ValueError(f"Bad OXTS path: uri={s3_uri} reason={reason}")

    local = cache.local(s3_uri)
    text = local.read_text(encoding="utf-8", errors="replace")
    row = _extract_first_numeric_oxts_row(text)
    if row is None:
        preview = text[:200].replace("\n", "\\n")
        reason = "no line with at least 9 numeric tokens"
        _warn_bad_oxts(s3_uri, reason)
        if allow_fallback:
            _warn_oxts_fallback(s3_uri)
            return _oxts_fallback(s3_uri, n_tokens)
        raise ValueError(
            f"Bad OXTS file: uri={s3_uri} reason={reason} preview={preview!r}"
        )

    # KITTI OXTS indices: 0=lat,1=lon,2=alt,5=yaw,8=vf,14=af
    def _get(i: int) -> float:
        return float(row[i]) if len(row) > i else 0.0
    base = np.array([_get(8), _get(14), 0.0, _get(0), _get(1), _get(2), _get(5)],
                    dtype=np.float32)
    rng  = np.random.default_rng(abs(hash(s3_uri)) % (2**32))
    imu  = base[None] + rng.normal(0, 1e-3, (n_tokens, 7)).astype(np.float32)
    ts   = np.arange(n_tokens, dtype=np.float32) / 30.0
    return imu, ts


def _encode_text(
    system_prompt: str,
    user_prompt: str,
    reasoning_chain: str,
    answer: str,
    max_prompt: int = MAX_PROMPT,
    max_target: int = MAX_TARGET,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    UTF-8 byte-encode text fields.
    Returns prompt_ids (max_prompt,), target_ids (max_target,), loss_mask (max_target,).
    """
    sep = b"\n\n"
    prompt_bytes = system_prompt.encode() + sep + user_prompt.encode()
    target_bytes = reasoning_chain.encode() + sep + answer.encode()

    def _pack(raw: bytes, length: int, eos: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        ids  = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
        if eos:
            ids = np.append(ids[: max(length - 1, 0)], EOS)
        else:
            ids = ids[:length]
        mask = np.ones(len(ids), dtype=bool)
        pad  = length - len(ids)
        return np.pad(ids, (0, pad), constant_values=PAD), \
               np.pad(mask, (0, pad), constant_values=False)

    prompt_ids, _   = _pack(prompt_bytes,  max_prompt)
    target_ids, lm  = _pack(target_bytes,  max_target, eos=True)
    target_ids = np.where(lm, target_ids, -1).astype(np.int16)
    return prompt_ids, target_ids, lm


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — lazy S3Cache init (fork-safe) + parallel downloads
# ─────────────────────────────────────────────────────────────────────────────

def _nan_str(val) -> Optional[str]:
    """Return val if it's a non-NaN string, else None."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return str(val) if val else None


_ROW_CONTEXT_KEYS = (
    "dataset_split", "split", "date", "drive", "drive_id", "sequence",
    "frame", "frame_idx", "frame_id", "sample_id",
)


def _row_context(row: pd.Series) -> str:
    parts = []
    for key in _ROW_CONTEXT_KEYS:
        val = _nan_str(row.get(key))
        if val is not None:
            parts.append(f"{key}={val}")
    return " ".join(parts) if parts else "<no drive/frame columns>"


def _row_paths_for_log(row: pd.Series) -> Dict[str, Optional[str]]:
    return {
        "calib_cam_to_cam_path": _nan_str(row.get("calib_cam_to_cam_path")),
        "calib_velo_to_cam_path": _nan_str(row.get("calib_velo_to_cam_path")),
        "image_path": _nan_str(row.get("image_path")),
        "lidar_path": _nan_str(row.get("lidar_path")),
    }


def _warn_bad_row(idx: int, row: pd.Series, exc: BaseException, attempt: int) -> None:
    paths = _row_paths_for_log(row)
    print(
        "[WARN][bad_row] "
        f"idx={idx} attempt={attempt} context={_row_context(row)} "
        f"error={type(exc).__name__}: {exc} "
        f"calib_cam_to_cam_path={paths['calib_cam_to_cam_path']} "
        f"calib_velo_to_cam_path={paths['calib_velo_to_cam_path']} "
        f"image_path={paths['image_path']} "
        f"lidar_path={paths['lidar_path']}",
        flush=True,
    )


class KairosDataset(Dataset):
    """
    Reads one curriculum tier of the Gold Delta table.

    _S3Cache is created LAZILY inside __getitem__ (not in __init__) so the
    boto3 client is always created AFTER the DataLoader fork — fork-safe.
    """
    # One-time compatibility warning: printed on the first sample where oxts_path
    # is absent and must be derived from image_path.  Class-level so it fires once
    # per worker process regardless of how many dataset instances exist.
    _oxts_compat_warned: bool = False

    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: Path,
        max_pts: int = MAX_PTS,
        max_prompt: int = MAX_PROMPT,
        max_target: int = MAX_TARGET,
        skip_bad_rows: bool = False,
        allow_oxts_fallback: bool = False,
    ) -> None:
        self._df         = df.reset_index(drop=True)
        self._cache_dir  = cache_dir
        self._cache: Optional[_S3Cache] = None   # created in worker process
        self._max_pts    = max_pts
        self._max_prompt = max_prompt
        self._max_target = max_target
        self._skip_bad_rows = skip_bad_rows
        self._allow_oxts_fallback = allow_oxts_fallback

    def _get_cache(self) -> _S3Cache:
        if self._cache is None:
            self._cache = _S3Cache(self._cache_dir)
        return self._cache

    def __len__(self) -> int:
        return len(self._df)

    @staticmethod
    def _path(row: pd.Series, key: str, fallback_key: Optional[str] = None) -> str:
        value = _nan_str(row.get(key))
        if value is not None:
            return value
        if fallback_key is not None:
            fallback = _nan_str(row.get(fallback_key))
            if fallback is not None:
                return fallback
            raise KeyError(f"Missing required '{key}' and fallback '{fallback_key}'")
        raise KeyError(f"Missing required path column '{key}'")

    def __getitem__(self, idx: int) -> dict:
        last_exc: Optional[BaseException] = None
        max_attempts = min(5, max(len(self), 1)) if self._skip_bad_rows else 1
        for attempt in range(max_attempts):
            cur_idx = (idx + attempt) % len(self)
            try:
                return self._load_one(cur_idx)
            except (FileNotFoundError, ValueError) as exc:
                last_exc = exc
                row = self._df.iloc[cur_idx]
                _warn_bad_row(cur_idx, row, exc, attempt + 1)
                if not self._skip_bad_rows:
                    raise
        raise RuntimeError(
            f"Unable to load a valid sample after {max_attempts} attempts "
            f"starting at idx={idx}; last_error={last_exc}"
        ) from last_exc

    def _load_one(self, idx: int) -> dict:
        row   = self._df.iloc[idx]
        cache = self._get_cache()

        image_t_path  = self._path(row, "image_path")
        image_t1_path = self._path(row, "image_path_t_minus_1", "image_path")
        image_t2_path = _nan_str(row.get("image_path_t_minus_2")) or image_t1_path
        lidar_t_path  = self._path(row, "lidar_path")
        lidar_t1_path = self._path(row, "lidar_path_t_minus_1", "lidar_path")
        calib_cam_path = self._path(row, "calib_cam_to_cam_path")
        calib_velo_path = self._path(row, "calib_velo_to_cam_path")

        # Parallel S3 downloads — I/O bound, GIL released during network I/O.
        # 5 concurrent downloads: 3 images + 2 LiDAR frames.
        # Calib submitted separately to reuse the same executor.
        _mpts = self._max_pts
        with ThreadPoolExecutor(max_workers=6) as ex:
            fut_img_t  = ex.submit(_load_image,  image_t_path,                cache)
            fut_img_t1 = ex.submit(_load_image,  image_t1_path,               cache)
            fut_img_t2 = ex.submit(_load_image,  image_t2_path,               cache)
            fut_lid_t  = ex.submit(_load_lidar,  lidar_t_path,  cache, _mpts)
            fut_lid_t1 = ex.submit(_load_lidar,  lidar_t1_path, cache, _mpts)
            fut_calib  = ex.submit(_parse_calib,
                                   calib_cam_path,
                                   calib_velo_path,
                                   cache)

        img_t  = fut_img_t.result()
        img_t1 = fut_img_t1.result()
        img_t2 = fut_img_t2.result()
        lidar_t  = fut_lid_t.result()
        lidar_t1 = fut_lid_t1.result()
        P2, R0_rect, Tr = fut_calib.result()

        oxts_uri = _nan_str(row.get("oxts_path"))
        _oxts_derived = False
        if not oxts_uri:
            # oxts_path column absent from this Gold table — derive from image_path.
            derived = _derive_oxts_from_image_path(image_t_path)
            if derived:
                oxts_uri = derived
                _oxts_derived = True
                if (not KairosDataset._oxts_compat_warned
                        and int(os.environ.get("RANK", "0")) == 0):
                    print(
                        "[compat] oxts_path column missing in Gold table; "
                        "deriving OXTS path from image_path  "
                        f"(e.g. {derived})",
                        flush=True,
                    )
                    KairosDataset._oxts_compat_warned = True

        if oxts_uri:
            imu_data, imu_ts = _load_oxts(
                oxts_uri, cache,
                allow_fallback=self._allow_oxts_fallback,
            )
        else:
            # Neither stored nor derivable — zero IMU tokens.
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"[WARN][oxts_missing] idx={idx} image_path={image_t_path} "
                    "— could not derive OXTS path; using zero IMU tokens",
                    flush=True,
                )
            imu_data = np.zeros((IMU_TOKENS, 7), dtype=np.float32)
            imu_ts   = np.arange(IMU_TOKENS, dtype=np.float32) / 30.0

        prompt_ids, target_ids, loss_mask = _encode_text(
            row["system_prompt"], row["user_prompt"],
            row["reasoning_chain"], row["answer"],
            max_prompt=self._max_prompt,
            max_target=self._max_target,
        )

        return {
            "img_t": img_t, "img_t1": img_t1, "img_t2": img_t2,
            "lidar_t": lidar_t, "lidar_t1": lidar_t1,
            "imu_data": imu_data, "imu_ts": imu_ts,
            "P2": P2, "R0_rect": R0_rect, "Tr": Tr,
            "prompt_ids": prompt_ids, "target_ids": target_ids, "loss_mask": loss_mask,
        }


def collate_fn(batch: List[dict]) -> KairoBatch:
    def _t(key: str, dtype=torch.float32) -> torch.Tensor:
        return torch.from_numpy(np.stack([s[key] for s in batch])).to(dtype)

    return KairoBatch(
        img_t          = _t("img_t"),
        img_t1         = _t("img_t1"),
        img_t2         = _t("img_t2"),
        lidar_t        = _t("lidar_t"),
        lidar_t1       = _t("lidar_t1"),
        imu_data       = _t("imu_data"),
        imu_timestamps = _t("imu_ts"),
        calib          = CalibMatrices(
            P2             = _t("P2"),
            R0_rect        = _t("R0_rect"),
            Tr_velo_to_cam = _t("Tr"),
        ),
        text_bytes   = _t("prompt_ids", dtype=torch.long),
        target_bytes = _t("target_ids", dtype=torch.long),
        loss_mask    = torch.from_numpy(np.stack([s["loss_mask"] for s in batch])),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gold table loader — reads directly from Hive partition path
# ─────────────────────────────────────────────────────────────────────────────

def _join_s3(base: str, child: str) -> str:
    """Join an S3 base path with a child segment, ensuring exactly one slash."""
    return base.rstrip("/") + "/" + child.strip("/") + "/"


REQUIRED_GOLD_COLUMNS = [
    "image_path",
    "image_path_t_minus_1",
    "image_path_t_minus_2",
    "lidar_path",
    "lidar_path_t_minus_1",
    "calib_cam_to_cam_path",
    "calib_velo_to_cam_path",
    "system_prompt",
    "user_prompt",
    "reasoning_chain",
    "answer",
    "dataset_split",
    "complexity_tier",
    "curriculum_order",
]


def _hive_partitions_from_path(path: str) -> Dict[str, str]:
    parts: Dict[str, str] = {}
    for segment in path.removeprefix("s3://").split("/"):
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        if key and value:
            parts[key] = value
    return parts


def _derive_complexity_tier(order) -> str:
    try:
        value = int(order)
    except (TypeError, ValueError):
        return "unknown"
    if value <= 1:
        return "easy"
    if value <= 2:
        return "medium"
    return "hard"


def _add_gold_compat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve existing Gold columns and add effective partition/helper columns
    needed by training validation when they are not stored inside parquet rows.
    """
    if "complexity_tier" not in df.columns and "curriculum_order" in df.columns:
        df = df.copy()
        df["complexity_tier"] = df["curriculum_order"].map(_derive_complexity_tier)
    return df


def _validate_gold_columns(df: pd.DataFrame, label: str) -> None:
    missing = [col for col in REQUIRED_GOLD_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Gold table partition {label!r} is missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )


def _load_parquet_partition(path: str) -> pd.DataFrame:
    """
    Read all parquet files under an S3 Hive partition path.

    Uses pyarrow.fs.S3FileSystem — does NOT require s3fs or aiobotocore.
    s3fs downgrades the SageMaker DLC botocore and breaks S3/SageMaker imports;
    pyarrow.fs.S3FileSystem is already available in every PyTorch DLC.

    The Gold table is partitioned by dataset_split= / complexity_tier= in S3
    and may also store those columns physically inside each parquet file.
    """
    bucket, key_prefix = _split_s3_uri(path)
    base_path = f"{bucket}/{key_prefix.strip('/')}"

    fs = pafs.S3FileSystem(region=DATA_REGION)
    print(f"[data] loading partition: {path}  (DATA_REGION={DATA_REGION})", flush=True)

    try:
        selector = pafs.FileSelector(base_path, recursive=True)
        file_infos = fs.get_file_info(selector)
    except Exception as exc:
        raise FileNotFoundError(
            f"Cannot list S3 path {path}: {exc}\n"
            f"  Searched: s3://{base_path} (recursive)"
        ) from exc

    parquet_paths = [
        fi.path for fi in file_infos
        if fi.type == pafs.FileType.File and fi.path.endswith(".parquet")
    ]

    if not parquet_paths:
        raise FileNotFoundError(
            f"No parquet files found under {path}\n"
            f"  Searched: s3://{base_path}/**/*.parquet (recursive)"
        )

    tables = []
    for fp in parquet_paths:
        # Read via pyarrow NativeFile — no s3fs dependency.
        with fs.open_input_file(fp) as f:
            part_df = pq.read_table(f).to_pandas()

        # Inject Hive partition key=value columns missing from physical rows.
        for key, value in _hive_partitions_from_path(fp).items():
            part_df[key] = value

        for col in ("dataset_split", "complexity_tier"):
            if col in part_df.columns:
                part_df[col] = part_df[col].astype(str)

        tables.append(part_df)
    df = _add_gold_compat_columns(pd.concat(tables, ignore_index=True))
    _validate_gold_columns(df, path)

    if "curriculum_order" in df.columns:
        df = df.sort_values("curriculum_order").reset_index(drop=True)

    print(f"[data] loaded  rows={len(df):,}  columns={list(df.columns)}", flush=True)
    if len(df) > 0:
        print(f"[data] sample  image_path={df.iloc[0].get('image_path', 'N/A')}",
              flush=True)
        if "split" in df.columns:
            print(f"[data] scene counts={df['split'].value_counts(dropna=False).to_dict()}",
                  flush=True)
        if "curriculum_order" in df.columns:
            print(f"[data] curriculum counts="
                  f"{df['curriculum_order'].value_counts(dropna=False).sort_index().to_dict()}",
                  flush=True)
    return df


def _split_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return bucket, key


def _s3_head_exists(s3_client, uri: str) -> bool:
    bucket, key = _split_s3_uri(uri)
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except _CLIENT_ERROR_TYPES as exc:
        if _is_missing_s3_error(exc):
            return False
        raise


def _preflight_s3_paths(df: pd.DataFrame, n_rows: int, rank: int) -> None:
    if n_rows <= 0 or rank != 0:
        return
    if len(df) == 0:
        print("[preflight] skipped: empty DataFrame", flush=True)
        return

    path_cols = [
        c for c in (
            "image_path", "image_path_t_minus_1", "image_path_t_minus_2",
            "lidar_path", "lidar_path_t_minus_1",
            "calib_cam_to_cam_path", "calib_velo_to_cam_path",
        )
        if c in df.columns
    ]
    sample_n = min(n_rows, len(df))
    sample = df.sample(n=sample_n, random_state=17) if sample_n < len(df) else df
    s3_client = boto3.client("s3", region_name=DATA_REGION)

    bad_counts = {c: 0 for c in path_cols}
    checked_counts = {c: 0 for c in path_cols}
    examples: List[str] = []
    for idx, row in sample.iterrows():
        for col in path_cols:
            uri = _nan_str(row.get(col))
            if uri is None:
                bad_counts[col] += 1
                if len(examples) < 10:
                    examples.append(f"idx={idx} col={col} uri=<missing>")
                continue
            checked_counts[col] += 1
            try:
                ok = _s3_head_exists(s3_client, uri)
            except Exception as exc:
                bad_counts[col] += 1
                if len(examples) < 10:
                    examples.append(
                        f"idx={idx} col={col} uri={uri} error={type(exc).__name__}: {exc}"
                    )
                continue
            if not ok:
                bad_counts[col] += 1
                if len(examples) < 10:
                    examples.append(f"idx={idx} col={col} uri={uri} error=missing")

    print(
        f"[preflight] sampled_rows={sample_n} checked_columns={path_cols}",
        flush=True,
    )
    print(f"[preflight] checked_path_count_by_column={checked_counts}", flush=True)
    print(f"[preflight] bad_path_count_by_column={bad_counts}", flush=True)
    for example in examples:
        print(f"[preflight][bad_path] {example}", flush=True)


def _curriculum_df(
    df_all: pd.DataFrame,
    max_order: int,
    min_rows: int = 1,
    rank: int = 0,
) -> pd.DataFrame:
    """Return rows where curriculum_order <= max_order.

    Falls back to df_all when:
      - column is absent
      - filtered result is empty
      - filtered result has fewer rows than min_rows (avoids empty DataLoaders
        when the subset is smaller than world_size * micro_batch)
    """
    if "curriculum_order" not in df_all.columns:
        return df_all
    filtered = df_all[df_all["curriculum_order"] <= max_order].reset_index(drop=True)
    if len(filtered) == 0:
        if rank == 0:
            print(
                f"[data] WARNING: curriculum_order <= {max_order} produced 0 rows — "
                "falling back to full dataset",
                flush=True,
            )
        return df_all
    if len(filtered) < min_rows:
        if rank == 0:
            print(
                f"[data] WARNING: curriculum_order <= {max_order} produced only "
                f"{len(filtered)} rows < min_rows={min_rows}; "
                "falling back to full dataset",
                flush=True,
            )
        return df_all
    return filtered


def _min_rows_for_distributed(world_size: int, batch_size: int) -> int:
    """Minimum dataset rows to guarantee at least one batch per rank with drop_last=True."""
    return max(world_size * batch_size, world_size)


def _make_dataloader(
    df: pd.DataFrame,
    cache_dir: Path,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int,
    shuffle: bool = True,
    max_pts: int = MAX_PTS,
    max_prompt: int = MAX_PROMPT,
    max_target: int = MAX_TARGET,
    skip_bad_rows: bool = False,
    allow_oxts_fallback: bool = False,
    drop_last: bool = True,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    if len(df) == 0:
        raise RuntimeError(
            "_make_dataloader received an empty DataFrame (0 rows). "
            "Check partition path and curriculum filtering logic."
        )
    if rank == 0:
        print(
            f"[dataloader] rows={len(df):,}  world_size={world_size}  "
            f"batch_size={batch_size}  drop_last={drop_last}  "
            f"num_workers={num_workers}",
            flush=True,
        )
    dataset = KairosDataset(df, cache_dir, max_pts=max_pts,
                            max_prompt=max_prompt, max_target=max_target,
                            skip_bad_rows=skip_bad_rows,
                            allow_oxts_fallback=allow_oxts_fallback)
    pf = 2 if num_workers > 0 else None
    pw = num_workers > 0
    if world_size > 1:
        # drop_last=False: DistributedSampler pads the dataset to ceil(N/world_size)
        # per rank rather than dropping the last incomplete batch.  This prevents
        # empty DataLoaders when curriculum subsets have fewer rows than world_size.
        sampler: Optional[DistributedSampler] = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=shuffle, drop_last=drop_last,
        )
        loader = DataLoader(
            dataset,
            batch_size        = batch_size,
            sampler           = sampler,
            num_workers       = num_workers,
            collate_fn        = collate_fn,
            pin_memory        = True,
            prefetch_factor   = pf,
            persistent_workers= pw,
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset,
            batch_size        = batch_size,
            shuffle           = shuffle,
            num_workers       = num_workers,
            collate_fn        = collate_fn,
            pin_memory        = True,
            prefetch_factor   = pf,
            persistent_workers= pw,
        )
    return loader, sampler


# ─────────────────────────────────────────────────────────────────────────────
# Progress display
# ─────────────────────────────────────────────────────────────────────────────

class ProgressDisplay:
    """Rich live display (rank-0 only). Falls back to \r bar if rich absent."""

    def __init__(self, total_steps: int, rank: int) -> None:
        self._total   = total_steps
        self._main    = rank == 0
        self._t0      = time.monotonic()
        self._live    = None
        self._prog    = None
        self._task    = None

        if not self._main:
            return

        if _HAS_RICH:
            self._prog = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Kairos-4B"),
                BarColumn(bar_width=32),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=_console,
                refresh_per_second=2,
            )
            self._task = self._prog.add_task("train", total=total_steps)
            self._live = Live(self._prog, console=_console, refresh_per_second=2)
            self._live.start()

    def update(
        self,
        step: int,
        losses: Dict[str, float],
        lr: float,
        phase: str,
        samp_per_s: float,
        vram_gb: float,
        grad_norm: float,
    ) -> None:
        if not self._main:
            return

        elapsed = time.monotonic() - self._t0
        pct     = 100.0 * step / max(1, self._total)
        eta_s   = elapsed / max(1, step) * max(0, self._total - step)

        def _fmt(s: float) -> str:
            return f"{int(s//3600)}h {int((s%3600)//60):02d}m"

        loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
        lr_phase = ("warmup" if step / self._total < 0.03
                    else "stable" if step / self._total < 0.80 else "decay")

        if _HAS_RICH and self._prog is not None:
            self._prog.update(self._task, completed=step)
            _console.log(
                f"[cyan]step[/cyan] {step:>6}/{self._total}  "
                f"[magenta]{pct:5.1f}%[/magenta]  ETA {_fmt(eta_s)}  "
                f"phase=[yellow]{phase}[/yellow]  "
                f"{loss_str}  lr={lr:.2e}({lr_phase})  "
                f"gnorm={grad_norm:.2f}  {samp_per_s:.1f}samp/s  "
                f"VRAM={vram_gb:.0f}GB"
            )
        else:
            bar = "█" * int(32 * step / self._total) + "░" * (32 - int(32 * step / self._total))
            print(
                f"\r[{bar}] {pct:5.1f}%  step {step}/{self._total}  "
                f"ETA {_fmt(eta_s)}  phase={phase}  {loss_str}  "
                f"lr={lr:.2e}  gnorm={grad_norm:.2f}  "
                f"{samp_per_s:.1f}samp/s  VRAM={vram_gb:.0f}GB",
                end="", flush=True,
            )

    def close(self) -> None:
        if not self._main:
            return
        if _HAS_RICH and self._live is not None:
            self._live.stop()
        else:
            print()


# ─────────────────────────────────────────────────────────────────────────────
# GPU helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_mem_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _log_cuda_mem(tag: str, rank: int = 0) -> None:
    """Print GPU memory stats at key training milestones (rank-0 only)."""
    if rank != 0 or not torch.cuda.is_available():
        return
    print(
        f"[cuda_mem] {tag}  "
        f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB  "
        f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB  "
        f"max={torch.cuda.max_memory_allocated()/1e9:.2f}GB",
        flush=True,
    )


def _get_lr(optimizer) -> float:
    for g in optimizer.param_groups:
        return float(g["lr"])
    return 0.0


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _set_core_trainable(model, trainable: bool) -> None:
    core = _unwrap_model(model).hybrid_core
    for param in core.parameters():
        param.requires_grad_(trainable)


def _vision_expert_util(model) -> float:
    core = _unwrap_model(model).hybrid_core
    total = None
    for block in core.blocks:
        load = block.moe_ffn.last_expert_load.detach().float()
        total = load if total is None else total + load
    if total is None or total.sum().item() <= 0:
        return 0.0
    return float(total[:40].sum() / total.sum().clamp(min=1.0))


def _param_count(module: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _log_model_param_counts(model: KairosModel) -> None:
    """Print concise parameter counts, including DINO/LoRA trainability."""
    total_p, trainable_p = _param_count(model)
    print(
        f"[params] total={total_p/1e6:.1f}M  "
        f"trainable={trainable_p/1e6:.1f}M  "
        f"frozen={(total_p-trainable_p)/1e6:.1f}M",
        flush=True,
    )
    components = {
        "vision": model.vision_encoder,
        "lidar": model.lidar_encoder,
        "imu": model.imu_encoder,
        "fusion": model.calib_gate,
        "hybrid_core": model.hybrid_core,
        "decoder": model.s2ft_decoder,
        "detection": model.det_head,
    }
    for name, module in components.items():
        total, trainable = _param_count(module)
        print(
            f"[params] {name:<11} total={total/1e6:.1f}M  "
            f"trainable={trainable/1e6:.1f}M",
            flush=True,
        )

    dino = getattr(model.vision_encoder, "dinov2", None)
    if dino is not None:
        dino_total, dino_trainable = _param_count(dino)
        lora_trainable = sum(
            p.numel()
            for n, p in dino.named_parameters()
            if "lora_" in n and p.requires_grad
        )
        print(
            f"[params] DINO total={dino_total/1e6:.1f}M  "
            f"trainable={dino_trainable/1e6:.1f}M  "
            f"LoRA_trainable={lora_trainable/1e6:.2f}M",
            flush=True,
        )


def _to_device(batch: KairoBatch, device: torch.device) -> KairoBatch:
    """Move all batch tensors to device. Defined once, called every step."""
    def _t(x: torch.Tensor) -> torch.Tensor:
        return x.to(device, non_blocking=True)
    return KairoBatch(
        img_t          = _t(batch.img_t),
        img_t1         = _t(batch.img_t1),
        img_t2         = _t(batch.img_t2),
        lidar_t        = _t(batch.lidar_t),
        lidar_t1       = _t(batch.lidar_t1),
        imu_data       = _t(batch.imu_data),
        imu_timestamps = _t(batch.imu_timestamps),
        calib          = CalibMatrices(
            P2             = _t(batch.calib.P2),
            R0_rect        = _t(batch.calib.R0_rect),
            Tr_velo_to_cam = _t(batch.calib.Tr_velo_to_cam),
        ),
        text_bytes   = _t(batch.text_bytes),
        target_bytes = _t(batch.target_bytes),
        loss_mask    = _t(batch.loss_mask),
    )


def _dist_backend() -> str:
    """Return current dist backend name, or 'none' if not initialized."""
    if not dist.is_available() or not dist.is_initialized():
        return "none"
    return dist.get_backend()


def _dist_barrier(device: torch.device) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index or torch.cuda.current_device()])
    else:
        dist.barrier()


def _broadcast_string(
    value: Optional[str], src: int, device: torch.device
) -> Optional[str]:
    """
    Broadcast an optional string from rank `src` to all ranks.
    All tensors are allocated on `device` so NCCL collectives work correctly.
    Non-src ranks ignore their local `value` — the src value wins.
    """
    if not dist.is_initialized():
        return value
    cur_rank = dist.get_rank()
    tag_bytes = (value.encode() if value is not None else b"") if cur_rank == src else b""
    length = torch.tensor(
        len(tag_bytes) if cur_rank == src else 0, dtype=torch.int32, device=device
    )
    dist.broadcast(length, src=src)
    if length.item() == 0:
        return None
    buf = torch.zeros(length.item(), dtype=torch.uint8, device=device)
    if cur_rank == src:
        buf.copy_(torch.tensor(list(tag_bytes), dtype=torch.uint8, device=device))
    dist.broadcast(buf, src=src)
    return bytes(buf.cpu().tolist()).decode()


def _all_reduce_scalar(value: float, device: torch.device) -> float:
    """Sum a float scalar across all ranks. Returns the global sum."""
    if not dist.is_initialized():
        return value
    t = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt_tag(step: int) -> str:
    return f"step_{step:07d}"


def _save_checkpoint(engine, step: int, phase: str, ckpt_s3: str, rank: int) -> None:
    """DeepSpeed save (all ranks) then S3 upload (rank-0 only)."""
    tag = _ckpt_tag(step)
    CKPT_LOCAL.mkdir(parents=True, exist_ok=True)
    engine.save_checkpoint(str(CKPT_LOCAL), tag=tag,
                           client_state={"step": step, "phase": phase})
    if rank != 0:
        return

    bucket, prefix = ckpt_s3.removeprefix("s3://").split("/", 1)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    local_tag = CKPT_LOCAL / tag
    if local_tag.exists():
        for fp in local_tag.rglob("*"):
            if fp.is_file():
                key = f"{prefix.rstrip('/')}/{tag}/{fp.relative_to(local_tag)}"
                s3.upload_file(str(fp), bucket, key)
    print(f"[ckpt] saved step {step} → s3://{bucket}/{prefix.rstrip('/')}/{tag}/")


def _load_latest_checkpoint(
    engine, ckpt_s3: str, rank: int, device: torch.device
) -> Tuple[int, str]:
    """Download latest checkpoint from S3, restore into engine. Returns (step, phase)."""
    bucket, prefix = ckpt_s3.removeprefix("s3://").split("/", 1)
    s3 = boto3.client("s3", region_name=AWS_REGION)

    latest_tag = None
    if rank == 0:
        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
            tags = sorted(
                p["Prefix"].split("/")[-2]
                for p in resp.get("CommonPrefixes", [])
                if "step_" in p["Prefix"]
            )
            if tags:
                latest_tag = tags[-1]
        except Exception as e:
            print(f"[ckpt] S3 list failed: {e}")

    latest_tag = _broadcast_string(latest_tag, src=0, device=device)

    if latest_tag is None:
        return 0, "easy"

    local_tag = CKPT_LOCAL / latest_tag
    local_tag.mkdir(parents=True, exist_ok=True)
    try:
        resp = s3.list_objects_v2(
            Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/{latest_tag}/"
        )
        for obj in resp.get("Contents", []):
            key   = obj["Key"]
            rel   = key.removeprefix(f"{prefix.rstrip('/')}/{latest_tag}/")
            lf    = local_tag / rel
            lf.parent.mkdir(parents=True, exist_ok=True)
            if not lf.exists():
                s3.download_file(bucket, key, str(lf))
    except Exception as e:
        print(f"[ckpt] download error: {e}")
        return 0, "easy"

    _, client = engine.load_checkpoint(str(CKPT_LOCAL), tag=latest_tag)
    step  = client.get("step",  0)
    phase = client.get("phase", "easy")
    if rank == 0:
        print(f"[ckpt] resumed from step {step}, phase={phase}")
    return step, phase


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _validate(
    engine,
    dl_val: DataLoader,
    device: torch.device,
    rank: int,
    max_batches: int = 50,
) -> Dict[str, float]:
    engine.eval()
    total = s2ft = n = 0.0

    for batch in dl_val:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            out = engine(_to_device(batch, device))
        total += float(out.total_loss)
        s2ft  += float(out.s2ft_loss) if out.s2ft_loss is not None else 0.0
        n     += 1.0
        if n >= max_batches:
            break

    t = torch.tensor([total, s2ft, n], device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total, s2ft, n = t.tolist()

    engine.train()
    return {"val_total": total / max(n, 1), "val_s2ft": s2ft / max(n, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _register_grad_hooks(engine, rank: int, max_steps: int = 3) -> None:
    """
    Register backward gradient shape hooks for ZeRO-3 backward debugging.

    Prints gradient shapes and zero-size warnings on rank-0 for the first
    max_steps backward passes.  Helps pinpoint which parameter's gradient has
    shape 0 under ZeRO-3, which is the root cause of the Stage 1d failure:
      RuntimeError: The size of tensor a (0) must match the size of tensor b (8)
    """
    model = _unwrap_model(engine)
    backward_count = [0]

    def _hook_factory(name):
        def hook(grad):
            if backward_count[0] >= max_steps or rank != 0:
                return grad
            if grad is None:
                print(
                    f"[grad/debug] {name}: grad=None (step={backward_count[0]+1})",
                    flush=True,
                )
            elif grad.numel() == 0:
                print(
                    f"[grad/debug] ZERO_GRAD_SHAPE name={name} "
                    f"rank={rank} step={backward_count[0]+1}",
                    flush=True,
                )
            else:
                print(
                    f"[grad/debug] {name}: shape={tuple(grad.shape)} "
                    f"dtype={grad.dtype} numel={grad.numel()} "
                    f"rank={rank} step={backward_count[0]+1}",
                    flush=True,
                )
            return grad
        return hook

    def _counter_hook(grad):
        backward_count[0] += 1
        return grad

    core = model.hybrid_core
    watched: List[str] = []

    # One probe param per sub-block (first trainable param found)
    _probe_sources: List[tuple] = [
        ("mamba", core.blocks[0].mamba2),
        ("cfc",   core.blocks[0].cfc),
        ("swa",   core.blocks[0].swa),
    ]
    for prefix, mod in _probe_sources:
        for n, p in mod.named_parameters():
            if p.requires_grad:
                full = f"hybrid_core.blocks[0].{prefix}.{n}"
                p.register_hook(_hook_factory(full))
                watched.append(full)
                break

    # Router and expert weight probes
    for _name, _p in [
        ("hybrid_core.blocks[0].moe_ffn.router_proj.weight",
         core.blocks[0].moe_ffn.router_proj.weight),
        ("hybrid_core.blocks[0].moe_ffn.W1",
         core.blocks[0].moe_ffn.W1),
    ]:
        if _p.requires_grad:
            _p.register_hook(_hook_factory(_name))
            watched.append(_name)

    # Use smoke_loss_anchor as the step counter hook (always receives gradient)
    model.smoke_loss_anchor.register_hook(_counter_hook)

    if rank == 0:
        print(
            f"[grad/debug] Registered hooks on {len(watched)} params: {watched}\n"
            f"[grad/debug] Printing gradient shapes for first {max_steps} backward passes.",
            flush=True,
        )


def train(args: argparse.Namespace) -> None:
    # Fail fast if requirements.txt downgraded the DLC botocore (only in SM env).
    _check_botocore_version()

    if getattr(args, "mem_trace", False):
        os.environ["KAIROS_MEM_TRACE"] = "1"
        _kairos_model_mod._MEM_TRACE = True

    if getattr(args, "debug_dtype_shapes", False):
        os.environ["KAIROS_DEBUG_DTYPE"] = "1"
        # Also enable shape logging for maximum visibility
        os.environ["KAIROS_DEBUG_SHAPES"] = "1"

    ultra_only_flags = {
        "ultra_smoke_dino_no_grad": getattr(args, "ultra_smoke_dino_no_grad", False),
        "ultra_smoke_skip_lidar": getattr(args, "ultra_smoke_skip_lidar", False),
        "ultra_smoke_skip_imu": getattr(args, "ultra_smoke_skip_imu", False),
        "ultra_smoke_skip_decoder_loss": getattr(args, "ultra_smoke_skip_decoder_loss", False),
        "ultra_smoke_skip_core": getattr(args, "ultra_smoke_skip_core", False),
        "force_core_with_anchor_loss": getattr(args, "force_core_with_anchor_loss", False),
        "ultra_smoke_core_loss": getattr(args, "ultra_smoke_core_loss", False),
        "dense_moe_fallback": getattr(args, "dense_moe_fallback", False),
        # ZeRO-3 module isolation flags
        "core_debug_bypass_mamba": getattr(args, "core_debug_bypass_mamba", False),
        "core_debug_bypass_cfc":   getattr(args, "core_debug_bypass_cfc",   False),
        "core_debug_bypass_swa":   getattr(args, "core_debug_bypass_swa",   False),
        "core_debug_bypass_moe":   getattr(args, "core_debug_bypass_moe",   False),
        "core_debug_layers_nonzero": bool(getattr(args, "core_debug_layers", 0)),
        "ultra_smoke_core_loss_scope_nondefault": (
            getattr(args, "ultra_smoke_core_loss_scope", "post_core") != "post_core"
        ),
    }
    if any(ultra_only_flags.values()) and not getattr(args, "ultra_smoke_mode", False):
        enabled = ", ".join(k for k, v in ultra_only_flags.items() if v)
        raise ValueError(
            f"{enabled} are ultra_smoke debug flags and require --ultra_smoke_mode True"
        )

    # ── Rank detection — safe fallbacks for plain-python SageMaker launch ─────
    rank       = _env_int("RANK",       _env_int("OMPI_COMM_WORLD_RANK",       0))
    local_rank = _env_int("LOCAL_RANK", _env_int("OMPI_COMM_WORLD_LOCAL_RANK",
                          args.local_rank if args.local_rank >= 0 else 0))
    world_size = _env_int("WORLD_SIZE", _env_int("OMPI_COMM_WORLD_SIZE",       1))

    # Publish safe defaults so every library (DeepSpeed, NCCL, torch.distributed)
    # finds the vars it needs regardless of how SageMaker launched this process.
    os.environ.setdefault("RANK",        str(rank))
    os.environ.setdefault("LOCAL_RANK",  str(local_rank))
    os.environ.setdefault("WORLD_SIZE",  str(world_size))
    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))

    cuda_available = torch.cuda.is_available()
    cuda_dev_count = torch.cuda.device_count() if cuda_available else 0

    if not cuda_available:
        raise RuntimeError(
            "CUDA unavailable — training requires a GPU instance. "
            "Verify the PyTorch GPU DLC image is selected in the estimator."
        )

    if not _HAS_DS:
        raise RuntimeError("deepspeed not installed")

    # Clamp local_rank to valid device range
    local_rank = min(local_rank, max(cuda_dev_count - 1, 0))
    os.environ["LOCAL_RANK"] = str(local_rank)   # keep env in sync after clamp
    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", local_rank)

    print(
        f"[kairos_train] BUILD_ID=layout-backward-g6-g5-opt-v20  "
        f"rank={rank}  local_rank={local_rank}  world_size={world_size}  "
        f"cuda_device_count={cuda_dev_count}  "
        f"MASTER_ADDR={os.environ['MASTER_ADDR']}  "
        f"MASTER_PORT={os.environ['MASTER_PORT']}  "
        f"cuda_available={cuda_available}  "
        f"current_device={torch.cuda.current_device()}  "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}  "
        f"deepspeed={deepspeed.__version__}  "
        f"instance_type={getattr(args, 'instance_type', 'unknown')}  "
        f"smoke_mode={args.smoke_mode}  "
        f"ultra_smoke_mode={getattr(args, 'ultra_smoke_mode', False)}  "
        f"budget_mode={args.budget_mode}  "
        f"zero_offload={args.zero_offload}",
        flush=True,
    )

    # Always initialize torch.distributed BEFORE deepspeed.initialize().
    # Both single- and multi-GPU use env:// — MASTER_ADDR/PORT are now guaranteed
    # to be in the environment (set above) so the rendezvous always succeeds.
    if not dist.is_initialized():
        dist.init_process_group(
            backend     = "nccl",
            init_method = "env://",
            rank        = rank,
            world_size  = world_size,
        )

    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    # Log all key dependency versions on rank-0 for CloudWatch debugging.
    _log_dependency_versions(rank)

    print(
        f"[dist] BUILD_ID=layout-backward-g6-g5-opt-v20  "
        f"rank={rank}  local_rank={local_rank}  world_size={world_size}  "
        f"MASTER_ADDR={os.environ['MASTER_ADDR']}  "
        f"MASTER_PORT={os.environ['MASTER_PORT']}  "
        f"cuda_device_count={cuda_dev_count}  device={device}  "
        f"dist_initialized={dist.is_initialized()}  "
        f"backend={_dist_backend()}",
        flush=True,
    )
    _log_cuda_mem("after_dist_init", rank)

    eff_batch = world_size * args.micro_batch * args.grad_accum

    # ── WORLD_SIZE / GPU count consistency check ──────────────────────────────
    if world_size == 1 and cuda_dev_count > 1:
        msg = (
            f"WORLD_SIZE=1 but {cuda_dev_count} CUDA devices are visible. "
            "ZeRO-3 parameter sharding across all GPUs is INACTIVE. "
            "SageMaker should launch one process per GPU with "
            "distribution={'torch_distributed': {'enabled': True}}. "
            "Set --allow_single_rank True only for intentional single-rank debugging."
        )
        if not getattr(args, "allow_single_rank", False):
            raise RuntimeError(msg)
        print(f"\n[WARN] {msg}\n", flush=True)
    if rank == 0:
        print(
            f"[dist_info] BUILD_ID=layout-backward-g6-g5-opt-v20  "
            f"RANK={rank}  LOCAL_RANK={local_rank}  WORLD_SIZE={world_size}  "
            f"cuda_device_count={cuda_dev_count}  deepspeed={deepspeed.__version__}  "
            f"micro_batch_per_gpu={args.micro_batch}  grad_accum={args.grad_accum}  "
            f"eff_batch={eff_batch}  instance_type={getattr(args, 'instance_type', 'unknown')}",
            flush=True,
        )

    # ── Mode-specific runtime limits. Full-training defaults are preserved. ─────
    if args.ultra_smoke_mode:
        args.smoke_mode = True   # ultra_smoke implies smoke; _ds_config sees it
        mode_name = "ultra_smoke"
        run_max_pts = 512
        run_max_prompt = 64
        run_max_target = 64
        run_max_gen_len = 64
        run_workers = 0
        run_val_every = 0
        run_ckpt_enabled = bool(args.save_ckpt_in_smoke)
    elif args.smoke_mode:
        mode_name = "smoke"
        run_max_pts = 4_096
        run_max_prompt = 256
        run_max_target = 256
        run_max_gen_len = 256
        run_workers = 0
        run_val_every = 0 if args.val_every is None else args.val_every
        run_ckpt_enabled = bool(args.save_ckpt_in_smoke)
    elif args.budget_mode:
        mode_name = "budget"
        run_max_pts = 8_192
        run_max_prompt = 384
        run_max_target = 384
        run_max_gen_len = 384
        run_workers = args.num_workers
        run_val_every = 0 if args.val_every is None else args.val_every
        run_ckpt_enabled = True
    else:
        mode_name = "full"
        run_max_pts = MAX_PTS
        run_max_prompt = MAX_PROMPT
        run_max_target = MAX_TARGET
        run_max_gen_len = KairosModelConfig().max_gen_len
        run_workers = args.num_workers
        run_val_every = 500 if args.val_every is None else args.val_every
        run_ckpt_enabled = True

    # ── Instance-type-aware budget_mode overrides ──────────────────────────────
    # g5.48xlarge has 8x A10G (192 GB total) — can afford larger max_pts.
    # g6.12xlarge has 4x L4 (96 GB total, 24 GB/GPU) — keep conservative.
    _instance_type = getattr(args, "instance_type", "unknown")
    if args.budget_mode and not args.smoke_mode and not args.ultra_smoke_mode:
        if "g5.48xlarge" in _instance_type:
            run_max_pts    = 16_384
            run_max_prompt = 512
            run_max_target = 512
            run_max_gen_len = 512
            if rank == 0:
                print(
                    f"[mode] g5.48xlarge budget_mode override: "
                    f"max_pts={run_max_pts}  max_prompt={run_max_prompt}  "
                    f"max_target={run_max_target}  max_gen_len={run_max_gen_len}",
                    flush=True,
                )
        elif "g6" in _instance_type or "g5.12xlarge" in _instance_type:
            # 24 GB/GPU — stay at 8192 (already set by budget_mode)
            if rank == 0:
                print(
                    f"[mode] {_instance_type} budget_mode: "
                    f"max_pts={run_max_pts}  (conservative for 24 GB/GPU)",
                    flush=True,
                )

    run_skip_bad_rows = (
        args.skip_bad_rows
        if args.skip_bad_rows is not None
        else bool(args.smoke_mode or args.ultra_smoke_mode or args.budget_mode)
    )
    run_preflight_rows = (
        args.preflight_rows
        if args.preflight_rows is not None
        else (20 if (args.smoke_mode or args.ultra_smoke_mode) else 0)
    )
    run_allow_oxts_fallback = (
        args.allow_oxts_fallback
        if args.allow_oxts_fallback is not None
        else bool(args.smoke_mode or args.ultra_smoke_mode)
    )

    if rank == 0:
        print(f"[init] ranks={world_size}  local_rank={local_rank}")
        print(f"       total_steps={args.total_steps}  micro_batch={args.micro_batch}"
              f"  grad_accum={args.grad_accum}  eff_batch={eff_batch}")
        print(
            f"[mode] mode={mode_name}  smoke_mode={args.smoke_mode}  "
            f"ultra_smoke_mode={args.ultra_smoke_mode}  "
            f"budget_mode={args.budget_mode}  zero_offload={args.zero_offload}",
            flush=True,
        )
        print(
            f"[mode] max_pts={run_max_pts}  max_prompt={run_max_prompt}  "
            f"max_target={run_max_target}  max_gen_len={run_max_gen_len}  "
            f"num_workers={run_workers}  val_every={run_val_every}  "
            f"checkpointing={'enabled' if run_ckpt_enabled else 'disabled'}  "
            f"skip_bad_rows={run_skip_bad_rows}  preflight_rows={run_preflight_rows}  "
            f"allow_oxts_fallback={run_allow_oxts_fallback}",
            flush=True,
        )
        if args.smoke_mode or args.budget_mode:
            print("[mode] DINO unfreeze_last_n=0; LoRA remains trainable", flush=True)
        if args.ultra_smoke_mode:
            print(
                f"[mode] ultra_smoke: mock_vision={args.ultra_smoke_mock_vision}  "
                f"freeze_lidar={args.freeze_lidar_in_smoke}  "
                f"freeze_imu={args.freeze_imu_in_smoke}  "
                f"freeze_core={args.freeze_core_in_smoke}  "
                "(None → mode default)",
                flush=True,
            )

    # ── Data: load from Hive partition paths ─────────────────────────────────
    if rank == 0:
        print("[data] reading Gold Delta table …")
        print(f"[data] AWS_REGION={AWS_REGION}  DATA_REGION={DATA_REGION}  "
              f"gold_s3={args.gold_s3}", flush=True)

    if not args.gold_s3:
        raise ValueError(
            "GOLD_S3 is not set. Export the env var before launching:\n"
            "  $env:GOLD_S3 = 's3://YOUR-BUCKET/delta/gold/kitti_s2ft_triplets/'"
        )
    if not args.ckpt_s3:
        raise ValueError(
            "CKPT_S3 is not set. Export the env var before launching:\n"
            "  $env:CKPT_S3 = 's3://YOUR-CKPT-BUCKET/checkpoints/kairos-4b/'"
        )

    train_path = _join_s3(args.gold_s3, "dataset_split=train")
    val_path   = _join_s3(args.gold_s3, "dataset_split=val")

    df_train = _load_parquet_partition(train_path)
    df_val   = _load_parquet_partition(val_path)

    if rank == 0:
        split_counts: Dict[str, int] = {}
        for _df in (df_train, df_val):
            if "dataset_split" not in _df.columns:
                continue
            for _split, _count in _df["dataset_split"].value_counts(dropna=False).items():
                split_counts[str(_split)] = split_counts.get(str(_split), 0) + int(_count)

        print(f"[startup] resolved gold_s3={args.gold_s3}", flush=True)
        print(f"[startup] resolved ckpt_s3={args.ckpt_s3}", flush=True)
        print(f"[startup] loaded_rows={len(df_train) + len(df_val):,}", flush=True)
        if split_counts:
            print(f"[startup] dataset_split counts={split_counts}", flush=True)

    if len(df_train) == 0:
        raise RuntimeError(
            f"Train partition is empty — no rows loaded from {train_path}\n"
            "Check that the Gold Delta table has been written and the S3 path is correct."
        )
    if len(df_val) == 0:
        print("[data] WARNING: val partition empty; validation will be skipped",
              flush=True)

    # Upweight text-heavy samples if the flag is present
    if "has_reasoning_chain" in df_train.columns:
        text_heavy = df_train[df_train["has_reasoning_chain"] == True]
        if not text_heavy.empty:
            df_train = pd.concat([df_train] + [text_heavy] * 3, ignore_index=True)

    # Disable curriculum by default in ultra_smoke_mode: tiny curriculum subsets
    # (e.g. only 2 "easy" rows) yield 0 batches per rank with drop_last=True and
    # world_size=4, causing StopIteration to escape the training loop → exit code 1.
    run_disable_curriculum = (
        bool(getattr(args, "disable_curriculum", False)) or bool(args.ultra_smoke_mode)
    )

    # Curriculum tiers: curriculum_order 1=easy, 1-2=easy+medium, all=full
    if run_disable_curriculum:
        df_easy = df_train
        df_medium = df_train
        df_full = df_train
        if rank == 0:
            print(
                "[data] curriculum disabled: all phases use full train dataset"
                f"  (disable_curriculum={getattr(args, 'disable_curriculum', False)}"
                f"  ultra_smoke_mode={args.ultra_smoke_mode})",
                flush=True,
            )
    else:
        min_rows = _min_rows_for_distributed(world_size, args.micro_batch)
        df_easy   = _curriculum_df(df_train, 1, min_rows=min_rows, rank=rank)
        df_medium = _curriculum_df(df_train, 2, min_rows=min_rows, rank=rank)
        df_full   = df_train

    if rank == 0:
        print(f"[data] train easy={len(df_easy):,}  easy+medium={len(df_medium):,}"
              f"  full={len(df_full):,}  val={len(df_val):,}", flush=True)
        if "split" in df_train.columns:
            print(f"[data] train scene counts="
                  f"{df_train['split'].value_counts(dropna=False).to_dict()}", flush=True)

    _preflight_s3_paths(df_train, run_preflight_rows, rank)
    _dist_barrier(device)

    cache_dir = CACHE_DIR / f"rank{rank}"

    # drop_last=False in smoke/ultra_smoke: DistributedSampler pads the dataset to
    # ceil(N/world_size) per rank so every rank gets at least 1 batch even when
    # the dataset has fewer rows than world_size.  Full/budget training keeps
    # drop_last=True for clean epoch boundaries and no data duplication.
    _drop_last = not (args.smoke_mode or args.ultra_smoke_mode)
    if not _drop_last and rank == 0:
        print(
            "[data] drop_last=False (smoke/ultra_smoke): DataLoader pads last batch "
            "so all ranks receive at least one sample",
            flush=True,
        )

    def _dl(df: pd.DataFrame, shuffle: bool = True):
        return _make_dataloader(
            df, cache_dir,
            batch_size  = args.micro_batch,
            rank        = rank,
            world_size  = world_size,
            num_workers = run_workers,
            shuffle     = shuffle,
            max_pts     = run_max_pts,
            max_prompt  = run_max_prompt,
            max_target  = run_max_target,
            skip_bad_rows = run_skip_bad_rows,
            allow_oxts_fallback = run_allow_oxts_fallback,
            drop_last   = _drop_last,
        )

    dl_easy,   samp_easy   = _dl(df_easy)
    dl_medium, samp_medium = _dl(df_medium)
    dl_full,   samp_full   = _dl(df_full)
    dl_val,    _           = _dl(df_val, shuffle=False)
    _log_cuda_mem("after_data_load", rank)

    samplers = {"easy": samp_easy, "easy+medium": samp_medium, "full": samp_full}
    loaders  = {"easy": dl_easy,   "easy+medium": dl_medium,   "full": dl_full}

    # ── Model ─────────────────────────────────────────────────────────────────
    if rank == 0:
        print("[model] building KairosModel …")

    cfg              = KairosModelConfig()
    cfg.z_loss_coeff = args.z_loss_coeff
    cfg.max_gen_len  = run_max_gen_len
    cfg.lidar_cfg.n_points = min(cfg.lidar_cfg.n_points, run_max_pts)
    if args.smoke_mode or args.budget_mode:
        cfg.vcfg.unfreeze_last_n = 0

    # ── Ultra-smoke component overrides (applied before model construction) ────
    if args.ultra_smoke_mode:
        # enc 56×56 at 14 px patch stride → 4 rows × 4 cols = 16 cam patches
        _us_n_cam = 16
        _us_n_lid = 8
        cfg.use_grad_checkpoint       = False
        cfg.kcfg.use_grad_checkpoint  = False
        cfg.vcfg.use_grad_checkpoint  = False
        cfg.vcfg.unfreeze_last_n  = 0
        cfg.vcfg.enc_h            = 56
        cfg.vcfg.enc_w            = 56
        cfg.vcfg.n_patches        = _us_n_cam
        cfg.n_cam                 = _us_n_cam
        cfg.fcfg.n_cam_tokens     = _us_n_cam
        cfg.fcfg.patch_rows       = 4       # 56 / 14
        cfg.fcfg.patch_cols       = 4       # 56 / 14
        cfg.fcfg.enc_h            = 56
        cfg.fcfg.enc_w            = 56
        cfg.lidar_cfg.n_tokens    = _us_n_lid
        cfg.lidar_cfg.n_points    = run_max_pts
        cfg.n_lidar               = _us_n_lid
        cfg.fcfg.n_lidar_tokens   = _us_n_lid
        cfg.fcfg.n_imu_tokens     = cfg.n_imu
        cfg.lidar_cfg.n_mamba_layers = 1
        cfg.lidar_cfg.moe_experts    = 1
        cfg.lidar_cfg.moe_d_ff       = 128
        cfg.imu_cfg.cfc_hidden       = 256
        cfg.imu_cfg.n_cfc_layers     = 1
        cfg.imu_cfg.n_tokens         = cfg.n_imu
        cfg.decoder_layers           = 1   # 1 decoder layer for ultra_smoke
        cfg.return_debug_tensors     = False
        cfg.w_det                    = 0
        # ── CRITICAL: reduce hybrid core MoE — the primary OOM source.
        # Full-train defaults: num_experts=64, moe_d_ff=5460, top_k=2 (→ 1.07B/block).
        # Ultra-smoke: 4 experts × 256 d_ff × top_k=1 (→ ~3.2M/block, -99.7%).
        # This drops total trainable params from ~3.44B to ~131M, reducing the
        # ZeRO-3 fp32 optimizer-state shard from ~3.4 GB to ~131 MB per GPU.
        cfg.kcfg.num_experts      = 2
        cfg.kcfg.moe_d_ff         = 128
        cfg.kcfg.top_k            = 1
        # ── Text encoder: 8 layers → 2 layers (~106M → ~26M)
        cfg.n_text_enc_layers     = 2
        # ── Vision no_grad flag
        if getattr(args, 'ultra_smoke_dino_no_grad', False):
            cfg.no_grad_vision    = True
        if args.ultra_smoke_mock_vision:
            cfg.vcfg.use_mock_backbone = True
        if rank == 0:
            print(
                f"[ultra_smoke] cfg overrides applied: "
                f"enc({cfg.vcfg.enc_h}x{cfg.vcfg.enc_w})  n_patches={cfg.vcfg.n_patches}  "
                f"n_cam={cfg.n_cam}  n_lidar={cfg.n_lidar}  "
                f"lidar_pts={cfg.lidar_cfg.n_points}  lidar_tok={cfg.lidar_cfg.n_tokens}  "
                f"lidar_mamba={cfg.lidar_cfg.n_mamba_layers}  lidar_moe_e={cfg.lidar_cfg.moe_experts}  "
                f"imu_cfc_h={cfg.imu_cfg.cfc_hidden}  imu_layers={cfg.imu_cfg.n_cfc_layers}  "
                f"dec_layers={cfg.decoder_layers}  max_gen={cfg.max_gen_len}  "
                f"w_det={cfg.w_det}  mock_vis={cfg.vcfg.use_mock_backbone}  "
                f"core_experts={cfg.kcfg.num_experts}  core_moe_d_ff={cfg.kcfg.moe_d_ff}  "
                f"core_top_k={cfg.kcfg.top_k}  text_enc_layers={cfg.n_text_enc_layers}  "
                f"no_grad_vision={cfg.no_grad_vision}  "
                f"grad_checkpoint={cfg.use_grad_checkpoint}",
                flush=True,
            )

    # sequential_frames: encode DINO frames one-by-one to cut peak activation memory.
    # ultra_smoke sets smoke_mode=True above, so this condition catches all three modes.
    if args.smoke_mode or args.budget_mode:
        cfg.vcfg.sequential_frames = True
        if rank == 0:
            print("[mode] DINO sequential_frames=True  (lower peak activation VRAM)",
                  flush=True)

    # ── Dense MoE fallback: set in cfg before model construction ──────────────
    # Auto-enabled when ultra_smoke_core_loss=True (required for ZeRO-3 safety).
    _use_dense_moe = getattr(args, 'dense_moe_fallback', False) or \
                     getattr(args, 'ultra_smoke_core_loss', False)
    if _use_dense_moe:
        cfg.kcfg.dense_moe_fallback = True
        if rank == 0:
            print(
                "[ultra_smoke] dense_moe_fallback=True -> MoeSwiGLUFFN uses dense "
                "weighted-sum dispatch; no sparse expert slices under ZeRO-3.",
                flush=True,
            )

    # ── Core debug bypass flags: set in cfg before model construction ──────────
    # --core_debug_bypass_* replaces individual sub-blocks with identity passes.
    # --core_debug_layers N limits hybrid core to first N loop iterations.
    # These are set before KairosModel() so KairosHybridBlock.__init__ picks them up.
    for _dflag in ('core_debug_bypass_mamba', 'core_debug_bypass_cfc',
                   'core_debug_bypass_swa',   'core_debug_bypass_moe'):
        if getattr(args, _dflag, False):
            setattr(cfg.kcfg, _dflag, True)
            if rank == 0:
                print(f"[core_debug] {_dflag}=True → identity pass in HybridBlock",
                      flush=True)

    _n_debug_layers = getattr(args, 'core_debug_layers', 0)
    if _n_debug_layers > 0:
        cfg.kcfg.core_debug_layers = _n_debug_layers
        if rank == 0:
            print(
                f"[core_debug] core_debug_layers={_n_debug_layers}: "
                f"limiting hybrid core to first {_n_debug_layers} loop iterations",
                flush=True,
            )

    # ── ultra_smoke_core_loss_scope: auto-bypass sub-blocks after scope point ──
    # Enables ZeRO-3 backward isolation: run full forward but compute loss from
    # the output at the specified scope, bypassing later sub-blocks.
    # post_mamba → bypass cfc/swa/moe; post_cfc → bypass swa/moe; etc.
    _scope = getattr(args, 'ultra_smoke_core_loss_scope', 'post_core')
    _core_loss_active = getattr(args, 'ultra_smoke_core_loss', False)
    if _scope != 'post_core' and _core_loss_active:
        _scope_bypass_map = {
            'post_mamba': ['cfc', 'swa', 'moe'],
            'post_cfc':   ['swa', 'moe'],
            'post_swa':   ['moe'],
            'post_moe':   [],
        }
        for _comp in _scope_bypass_map.get(_scope, []):
            setattr(cfg.kcfg, f'core_debug_bypass_{_comp}', True)
        if rank == 0 and _scope_bypass_map.get(_scope):
            print(
                f"[core_debug] ultra_smoke_core_loss_scope={_scope}: "
                f"auto-bypassing {_scope_bypass_map[_scope]} sub-blocks for "
                "scoped backward test",
                flush=True,
            )

    model            = KairosModel(cfg)

    if rank == 0:
        print(f"[model] {model.count_params()}")
        _log_model_param_counts(model)
    _log_cuda_mem("after_model_creation", rank)

    # ── Per-component freeze (smoke / ultra_smoke only) ────────────────────────
    # Freeze flags: None → apply mode default (ultra_smoke freezes lidar+imu).
    _is_smoke_run = args.smoke_mode or args.ultra_smoke_mode
    run_freeze_lidar = (args.freeze_lidar_in_smoke
                        if args.freeze_lidar_in_smoke is not None
                        else args.ultra_smoke_mode)
    run_freeze_imu   = (args.freeze_imu_in_smoke
                        if args.freeze_imu_in_smoke   is not None
                        else args.ultra_smoke_mode)
    run_freeze_core  = (args.freeze_core_in_smoke
                        if args.freeze_core_in_smoke  is not None
                        else False)
    if _is_smoke_run:
        if run_freeze_lidar:
            for p in model.lidar_encoder.parameters():
                p.requires_grad_(False)
            if rank == 0:
                print("[freeze] lidar_encoder → all params frozen", flush=True)
        if run_freeze_imu:
            for p in model.imu_encoder.parameters():
                p.requires_grad_(False)
            if rank == 0:
                print("[freeze] imu_encoder → all params frozen", flush=True)
        if run_freeze_core:
            # Loud warning: freezing the hybrid core in ultra_smoke is non-default
            # and was explicitly requested (default is False).
            if rank == 0 and args.ultra_smoke_mode:
                print(
                    "\n[WARN] freeze_core_in_smoke=True was EXPLICITLY requested in "
                    "ultra_smoke_mode. This is NOT the default (default=False). "
                    "Gradients will NOT flow to the hybrid core. "
                    "Proceed only if you intentionally want to validate the "
                    "data/DeepSpeed pipeline with a fully frozen core.\n",
                    flush=True,
                )
            for p in model.hybrid_core.parameters():
                p.requires_grad_(False)
            if rank == 0:
                print("[freeze] hybrid_core → all params frozen", flush=True)
        if rank == 0:
            print(
                f"[freeze] effective: lidar={run_freeze_lidar}  "
                f"imu={run_freeze_imu}  core={run_freeze_core}",
                flush=True,
            )

        # ── Wrap frozen encoders in torch.no_grad() during forward ────────────
        # Setting these instance flags on the model before deepspeed.initialize()
        # ensures forward() skips activation-graph construction for frozen modules,
        # saving activation memory even though requires_grad is already False.
        if run_freeze_lidar:
            model._smoke_no_grad_lidar = True
            if rank == 0:
                print("[no_grad] lidar_encoder forward wrapped in torch.no_grad()",
                      flush=True)
        if run_freeze_imu:
            model._smoke_no_grad_imu = True
            if rank == 0:
                print("[no_grad] imu_encoder forward wrapped in torch.no_grad()",
                      flush=True)

        # ── Debug binary-search skip flags (shape-preserving zero stubs) ─────
        if getattr(args, 'ultra_smoke_skip_lidar', False):
            model._smoke_skip_lidar = True
            if rank == 0:
                print("[WARN][ultra_smoke_debug] ultra_smoke_skip_lidar=True: "
                      "LiDAR tokens/centroids are zero stubs; LiDAR encoder is bypassed.",
                      flush=True)
        if getattr(args, 'ultra_smoke_skip_imu', False):
            model._smoke_skip_imu = True
            if rank == 0:
                print("[WARN][ultra_smoke_debug] ultra_smoke_skip_imu=True: "
                      "IMU tokens/delta_t are zero stubs; IMU encoder is bypassed.",
                      flush=True)
        if getattr(args, 'ultra_smoke_skip_decoder_loss', False):
            model._smoke_skip_decoder_loss = True
            if rank == 0:
                print("[WARN][ultra_smoke_debug] ultra_smoke_skip_decoder_loss=True: "
                      "decoder CE is bypassed; using smoke_loss_anchor (disconnected from core).",
                      flush=True)
        if getattr(args, 'ultra_smoke_skip_core', False):
            model._smoke_skip_core = True
            if rank == 0:
                print("[WARN][ultra_smoke_debug] ultra_smoke_skip_core=True: "
                      "hybrid core (Mamba+CfC+SWA+MoE) bypassed in forward. "
                      "Loss flows only through smoke_loss_anchor. "
                      "Use with --ultra_smoke_skip_decoder_loss for plumbing-only validation.",
                      flush=True)

        # ── Auto skip_core when skip_decoder_loss=True (unless force_core) ────
        # When anchor-only loss is used and force_core_with_anchor_loss is False,
        # automatically bypass hybrid core to prevent accidental moe_z backward.
        # kairos_model.py already excludes moe_z from total_loss in this mode, but
        # auto-skip adds a belt-and-suspenders guard for ZeRO-3 backward safety.
        _skip_dec_flag = getattr(args, 'ultra_smoke_skip_decoder_loss', False)
        _force_core    = getattr(args, 'force_core_with_anchor_loss', False)
        _core_loss_flag = getattr(args, 'ultra_smoke_core_loss', False)
        if _skip_dec_flag and not _force_core and not _core_loss_flag:
            if not getattr(model, '_smoke_skip_core', False):
                model._smoke_skip_core = True
                if rank == 0:
                    print(
                        "[ultra_smoke] skip_decoder_loss=True + force_core_with_anchor_loss=False "
                        "-> auto skip_core=True. Use --force_core_with_anchor_loss True "
                        "to run core with anchor loss (moe_z always excluded from backward).",
                        flush=True,
                    )
        elif _skip_dec_flag and _force_core and rank == 0:
            print(
                "[ultra_smoke] skip_decoder_loss=True + force_core_with_anchor_loss=True "
                "-> hybrid core runs; moe_z excluded from total_loss (core graph detached).",
                flush=True,
            )

        # ── ultra_smoke_core_loss: safe core backward test ────────────────────
        if _core_loss_flag:
            model._ultra_smoke_core_loss = True
            if rank == 0:
                print(
                    "[ultra_smoke] ultra_smoke_core_loss=True -> total_loss = "
                    "x.pow(2).mean()*1e-4 (safe core backward test). "
                    "Use with --dense_moe_fallback True for ZeRO-3 safety.",
                    flush=True,
                )

        # ── dino_no_grad: freeze all vision_encoder params (incl. LoRA) ──────
        if getattr(args, 'ultra_smoke_dino_no_grad', False):
            for p in model.vision_encoder.parameters():
                p.requires_grad_(False)
            if rank == 0:
                print(
                    "[WARN][ultra_smoke_debug] ultra_smoke_dino_no_grad=True: "
                    "ALL vision_encoder params (incl. LoRA) frozen. "
                    "Running under torch.no_grad() in forward(). "
                    "NO LoRA gradients — plumbing-validation only.",
                    flush=True,
                )

    # ── Optimizer + WSD schedule ──────────────────────────────────────────────
    optimizer = AdamW(
        model.param_groups(
            lr_backbone_lora = args.lr_lora,
            lr_encoders      = args.lr_encoders,
            lr_core          = args.lr_core,
            lr_decoder       = args.lr_decoder,
        ),
        weight_decay = args.weight_decay,
    )

    warmup_steps = int(args.total_steps * args.warmup_frac)
    stable_steps = int(args.total_steps * args.stable_frac)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _wsd_lambda(s, warmup_steps, stable_steps, args.total_steps),
    )

    # ── DeepSpeed ─────────────────────────────────────────────────────────────
    # dist_init_required=False: we always init torch.distributed above ourselves.
    # Do NOT call model.cuda() before this — ZeRO-3 shards params across ranks;
    # letting DeepSpeed place shards avoids materialising the full model on one GPU.
    _patch_ds_warned()
    _ds_cfg_dict = _ds_config(args)
    _eff_zero = _ds_cfg_dict.get("zero_optimization", {}).get("stage", "?")
    if rank == 0:
        import json as _json
        _zero_note = ""
        if args.zero_stage is None and (args.ultra_smoke_mode or args.smoke_mode or args.budget_mode):
            _zero_note = " (stable default for smoke/budget — use --zero_stage 3 to override)"
        print(
            f"[deepspeed_config] ZeRO stage : {_eff_zero}{_zero_note}",
            flush=True,
        )
        print(
            f"[deepspeed_config] full config:\n{_json.dumps(_ds_cfg_dict, indent=2)}",
            flush=True,
        )
        if _eff_zero == 3 and getattr(args, 'ultra_smoke_core_loss', False):
            print(
                "\n[WARN] ZeRO-3 core backward is experimental. "
                "Stage 1d previously failed with:\n"
                "  RuntimeError: The size of tensor a (0) must match the size of "
                "tensor b (8) at non-singleton dimension 1\n"
                "Consider --zero_stage 2 for stability, or add --zero3_debug_safe True "
                "for conservative ZeRO-3 settings.\n",
                flush=True,
            )
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model              = model,
        optimizer          = optimizer,
        lr_scheduler       = scheduler,
        config             = _ds_cfg_dict,
        dist_init_required = False,
    )
    _log_cuda_mem("after_deepspeed_init", rank)

    if getattr(args, 'debug_grad_shapes', False):
        os.environ["KAIROS_DEBUG_GRAD_SHAPES"] = "1"
        _register_grad_hooks(engine, rank)
        if rank == 0:
            print(
                "[grad/debug] debug_grad_shapes=True: backward gradient shape hooks registered.",
                flush=True,
            )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step, _start_phase = _load_latest_checkpoint(engine, args.ckpt_s3, rank, device)
    _set_core_trainable(engine, start_step >= args.freeze_core_steps)
    _dist_barrier(device)

    # ── Curriculum phase boundaries ───────────────────────────────────────────
    phase_easy_end   = int(args.total_steps * PHASE_EASY_END)
    phase_medium_end = int(args.total_steps * PHASE_MEDIUM_END)

    def _active_phase(step: int) -> str:
        if step < phase_easy_end:   return "easy"
        if step < phase_medium_end: return "easy+medium"
        return "full"

    # ── Progress ──────────────────────────────────────────────────────────────
    display = ProgressDisplay(args.total_steps, rank)

    # ── Training state ────────────────────────────────────────────────────────
    step          = start_step
    micro_step    = 0          # raw forward/backward counter (reset never)
    last_ckpt_t   = time.monotonic()
    t_log_start   = time.monotonic()
    _diag_done    = False      # one-shot first-step memory diagnostics

    # Per-phase epoch counters for DistributedSampler.set_epoch()
    epoch_ctr: Dict[str, int] = {"easy": 0, "easy+medium": 0, "full": 0}

    cur_phase = _active_phase(step)
    data_iter = iter(loaders[cur_phase])

    def _next_batch(phase: str) -> KairoBatch:
        """Fetch next batch, cycling epochs. Raises explicit RuntimeError on empty DataLoader."""
        nonlocal data_iter
        for _attempt in range(2):
            try:
                return next(data_iter)
            except StopIteration:
                epoch_ctr[phase] += 1
                s = samplers[phase]
                if s is not None:
                    s.set_epoch(epoch_ctr[phase])
                data_iter = iter(loaders[phase])
        n_rows = len(loaders[phase].dataset)
        raise RuntimeError(
            f"[data] DataLoader for phase={phase!r} yielded no batches after reset. "
            f"dataset_rows={n_rows}  world_size={world_size}  "
            f"micro_batch={args.micro_batch}  drop_last={_drop_last}. "
            "Ensure the curriculum subset has >= world_size * micro_batch rows, "
            "or use --disable_curriculum True."
        )

    acc_total = acc_s2ft = acc_z = acc_vutil = acc_text_ratio = 0.0
    acc_count = 0
    nan_streak = 0   # consecutive NaN batches — abort if too many

    while step < args.total_steps:
        # ── Curriculum switch ─────────────────────────────────────────────────
        new_phase = _active_phase(step)
        if new_phase != cur_phase:
            cur_phase = new_phase
            data_iter = iter(loaders[cur_phase])
            if rank == 0:
                print(f"\n[curriculum] → {cur_phase}  step={step}")

        # ── Fetch batch (epoch cycling with set_epoch) ────────────────────────
        batch = _next_batch(cur_phase)

        if not _diag_done:
            _log_cuda_mem("after_first_batch_cpu_load", rank)
        batch = _to_device(batch, device)
        if not _diag_done:
            _log_cuda_mem("after_first_batch_to_gpu", rank)
        _set_core_trainable(engine, step >= args.freeze_core_steps)

        # ── Forward ───────────────────────────────────────────────────────────
        autocast_enabled = torch.cuda.is_available()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output: KairosOutput = engine(batch)
        if not _diag_done:
            _log_cuda_mem("after_first_forward", rank)
        loss = output.total_loss   # includes z-loss (kairos_model.py handles it)
        if loss is None:
            raise RuntimeError("model returned total_loss=None for a training batch")

        # ── NaN / Inf guard ───────────────────────────────────────────────────
        if not torch.isfinite(loss):
            nan_streak += 1
            if rank == 0:
                print(f"\n[warn] non-finite loss at micro_step {micro_step}"
                      f" (streak={nan_streak}) — zeroed out")
            loss = loss * 0.0   # DeepSpeed still needs a graph-connected loss
            if nan_streak >= 20:
                raise RuntimeError("20 consecutive NaN/Inf losses — aborting")
        else:
            nan_streak = 0

        # ── Backward + optimizer step ─────────────────────────────────────────
        engine.backward(loss)
        if not _diag_done:
            _log_cuda_mem("after_first_backward", rank)
        engine.step()           # advances micro_steps; calls scheduler.step()
                                # automatically at gradient_accumulation boundary
        if not _diag_done:
            _log_cuda_mem("after_first_optim_step", rank)
            _diag_done = True
        micro_step += 1

        # ── Post-optimizer actions (once per ACTUAL optimizer step) ───────────
        if micro_step % args.grad_accum == 0:
            sync_moe_expert_bias(engine)
            step += 1

            # Gradient norm (best-effort — not all DS versions expose this)
            try:
                grad_norm = float(engine.get_global_grad_norm())
            except Exception:
                grad_norm = float("nan")

            total_val = float(output.total_loss.detach().item())
            s2ft_val  = float(output.s2ft_loss.detach().item())  if output.s2ft_loss  is not None else 0.0
            z_val     = float(output.moe_z_loss.detach().item()) if output.moe_z_loss is not None else 0.0
            acc_total += total_val
            acc_s2ft  += s2ft_val
            acc_z     += z_val
            acc_vutil += _vision_expert_util(engine)
            acc_text_ratio += s2ft_val / max(abs(total_val), 1e-12)
            acc_count += 1

            # ── Logging ───────────────────────────────────────────────────────
            if step % args.log_every == 0 and rank == 0:
                t_now      = time.monotonic()
                elapsed    = t_now - t_log_start
                t_log_start = t_now
                samp_per_s = (eff_batch * args.log_every) / max(elapsed, 1e-3)

                avg = max(acc_count, 1)
                display.update(
                    step       = step,
                    losses     = {
                        "total": acc_total/avg,
                        "s2ft": acc_s2ft/avg,
                        "z": acc_z/avg,
                        "vision_expert_util": acc_vutil/avg,
                        "text_loss_ratio": acc_text_ratio/avg,
                    },
                    lr         = _get_lr(optimizer),
                    phase      = cur_phase,
                    samp_per_s = samp_per_s,
                    vram_gb    = _gpu_mem_gb(),
                    grad_norm  = grad_norm,
                )
                print(
                    f"[data_quality] bad_oxts_seen={bad_oxts_seen} "
                    f"oxts_fallbacks_used={oxts_fallbacks_used}",
                    flush=True,
                )
                acc_total = acc_s2ft = acc_z = acc_vutil = acc_text_ratio = 0.0
                acc_count = 0

            # ── Validation ────────────────────────────────────────────────────
            if run_val_every > 0 and step % run_val_every == 0:
                val_losses = _validate(engine, dl_val, device, rank)
                if rank == 0:
                    vl = "  ".join(f"{k}={v:.4f}" for k, v in val_losses.items())
                    if _HAS_RICH:
                        _console.log(f"[green][val][/green] step {step}  {vl}")
                    else:
                        print(f"\n[val] step {step}  {vl}")

            # ── Checkpoint ────────────────────────────────────────────────────
            if (run_ckpt_enabled
                    and time.monotonic() - last_ckpt_t >= args.ckpt_every_min * 60):
                _save_checkpoint(engine, step, cur_phase, args.ckpt_s3, rank)
                last_ckpt_t = time.monotonic()

    # ── Final checkpoint ───────────────────────────────────────────────────────
    if run_ckpt_enabled:
        _save_checkpoint(engine, step, "done", args.ckpt_s3, rank)
    display.close()

    if rank == 0:
        print(
            f"\n[done] training complete at step {step}  "
            f"bad_oxts_seen={bad_oxts_seen}  "
            f"oxts_fallbacks_used={oxts_fallbacks_used}"
        )

    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(_args())
