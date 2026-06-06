"""
sagemaker_launch.py — Launch Kairos-4B training on SageMaker.

Examples:
    python sagemaker_launch.py --dry_run
    python sagemaker_launch.py --role_arn "arn:aws:iam::195231312992:role/SageMakerKairosRole"
    python sagemaker_launch.py --instance ml.g5.12xlarge --steps 20 --micro_batch 1 --grad_accum 1 --smoke_mode
    python sagemaker_launch.py --instance ml.g5.12xlarge --steps 100 --micro_batch 1 --grad_accum 4 --budget_mode
"""

import argparse
from datetime import datetime, timezone
import fnmatch
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
# DATA_REGION: region of the S3 bucket that holds Gold parquet + KITTI raw assets.
# Set to eu-north-1 when SageMaker job runs in eu-central-1 but data is in Stockholm.
DATA_REGION = os.getenv("DATA_REGION", os.getenv("DATA_AWS_REGION", AWS_REGION))
SM_ROLE = os.getenv("SAGEMAKER_ROLE_ARN")

GOLD_S3 = os.getenv("GOLD_S3", "")    # must be set — no stale default
CKPT_S3 = os.getenv("CKPT_S3", "")    # must be set — no stale default
OUTPUT_S3 = os.getenv("OUTPUT_S3", "") # must be set — no stale default
SOURCE_DIR = "."

FRAMEWORK_VER = "2.3.0"
PY_VER = "py311"


# (gpu_count, total_vram_gb)
INSTANCE_INFO: Dict[str, Tuple[int, int]] = {
    "ml.g5.xlarge":   (1, 24),
    "ml.g5.2xlarge":  (1, 24),
    "ml.g5.4xlarge":  (1, 24),
    "ml.g5.8xlarge":  (1, 24),
    "ml.g5.12xlarge": (4, 96),
    "ml.g5.16xlarge": (1, 24),
    "ml.g5.24xlarge": (4, 96),
    "ml.g5.48xlarge": (8, 192),
    # NVIDIA L4 instances (g6 family)
    "ml.g6.2xlarge":  (1, 24),
    "ml.g6.4xlarge":  (1, 24),
    "ml.g6.8xlarge":  (1, 24),
    "ml.g6.12xlarge": (4, 96),
    "ml.g6.16xlarge": (1, 24),
    "ml.g6.24xlarge": (4, 96),
    "ml.g6.48xlarge": (8, 192),
}

# Per-instance GPU name for the launch summary
_INSTANCE_GPU_NAME: Dict[str, str] = {
    "ml.g5.xlarge":   "NVIDIA A10G",
    "ml.g5.2xlarge":  "NVIDIA A10G",
    "ml.g5.4xlarge":  "NVIDIA A10G",
    "ml.g5.8xlarge":  "NVIDIA A10G",
    "ml.g5.12xlarge": "NVIDIA A10G",
    "ml.g5.16xlarge": "NVIDIA A10G",
    "ml.g5.24xlarge": "NVIDIA A10G",
    "ml.g5.48xlarge": "NVIDIA A10G",
    "ml.g6.2xlarge":  "NVIDIA L4",
    "ml.g6.4xlarge":  "NVIDIA L4",
    "ml.g6.8xlarge":  "NVIDIA L4",
    "ml.g6.12xlarge": "NVIDIA L4",
    "ml.g6.16xlarge": "NVIDIA L4",
    "ml.g6.24xlarge": "NVIDIA L4",
    "ml.g6.48xlarge": "NVIDIA L4",
}

# Approximate on-demand SageMaker training prices in us-east-1 (USD/hr).
# Used only for warm-pool idle-cost estimates — verify on aws.amazon.com/sagemaker/pricing.
_INSTANCE_ON_DEMAND_COST: Dict[str, float] = {
    "ml.g5.xlarge":    1.006,
    "ml.g5.2xlarge":   1.212,
    "ml.g5.4xlarge":   1.624,
    "ml.g5.8xlarge":   2.448,
    "ml.g5.12xlarge":  5.672,
    "ml.g5.16xlarge":  4.096,
    "ml.g5.24xlarge":  8.144,
    "ml.g5.48xlarge": 16.288,
    "ml.g6.2xlarge":   1.323,
    "ml.g6.4xlarge":   1.938,
    "ml.g6.8xlarge":   3.167,
    "ml.g6.12xlarge":  7.093,
    "ml.g6.16xlarge":  4.624,
    "ml.g6.24xlarge": 11.033,
    "ml.g6.48xlarge": 22.065,
}


# ─────────────────────────────────────────────────────────────────────────────
# IAM role helper
# ─────────────────────────────────────────────────────────────────────────────

def _str2bool(v) -> bool:
    """Argparse bool parser that accepts explicit True/False values."""
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


def _ensure_sagemaker_role() -> str:
    iam = boto3.client("iam", region_name=AWS_REGION)
    role_name = "SageMakerKairosRole"

    try:
        resp = iam.get_role(RoleName=role_name)
        arn = resp["Role"]["Arn"]
        print(f"[iam] Role exists: {arn}")
        return arn
    except iam.exceptions.NoSuchEntityException:
        pass

    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })

    role = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=trust,
        Description="Kairos-4B SageMaker training role",
    )

    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    )

    arn = role["Role"]["Arn"]
    print(f"[iam] Created role: {arn}")
    print("[iam] Attach scoped S3 permissions or AmazonS3FullAccess before training.")
    return arn


def _sagemaker_safe_job_name(base: str, instance_type: str) -> str:
    """Build a SageMaker-safe unique training job name."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = f"-{timestamp}"
    max_prefix_len = 63 - len(suffix)

    instance_safe = re.sub(r"[^A-Za-z0-9-]+", "-", instance_type)
    instance_safe = re.sub(r"-+", "-", instance_safe).strip("-")
    required_prefix = f"kairos-4b-{instance_safe or 'instance'}"
    custom = ""
    if base and base != "kairos-4b":
        custom = re.sub(r"[^A-Za-z0-9-]+", "-", base)
        custom = re.sub(r"-+", "-", custom).strip("-")

    prefix = required_prefix[:max_prefix_len].rstrip("-")
    if custom and len(required_prefix) + 1 < max_prefix_len:
        custom_len = max_prefix_len - len(required_prefix) - 1
        custom = custom[:custom_len].rstrip("-")
        if custom:
            prefix = f"{required_prefix}-{custom}"

    if not prefix:
        prefix = "kairos"
    return f"{prefix}{suffix}"


def _bundle_stats(source_dir: str) -> Tuple[int, int, int]:
    """Return (included_files, included_bytes, excluded_bytes).

    Walks source_dir, applying .sourceignore patterns so the logged numbers
    match what SageMaker actually uploads.  Falls back gracefully on OS errors.
    Directory patterns (trailing /) exclude the whole subtree; filename globs
    match only the final path component; patterns containing / match the full
    relative path.
    """
    root = Path(source_dir).resolve()
    patterns: List[str] = []
    si = root / ".sourceignore"
    if si.exists():
        for raw in si.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                patterns.append(line)

    n_files = n_included = n_excluded = 0
    try:
        for fp in root.rglob("*"):
            if not fp.is_file():
                continue
            try:
                sz = fp.stat().st_size
            except OSError:
                continue

            parts = fp.relative_to(root).parts
            rel_fwd = "/".join(parts)
            excluded = False
            for pat in patterns:
                is_dir_pat = pat.endswith("/")
                bare = pat.rstrip("/")
                if is_dir_pat:
                    if any(fnmatch.fnmatch(p, bare) for p in parts):
                        excluded = True
                        break
                    continue
                if "/" in pat:
                    if fnmatch.fnmatch(rel_fwd, pat):
                        excluded = True
                        break
                    continue
                if fnmatch.fnmatch(parts[-1], pat):
                    excluded = True
                    break

            if excluded:
                n_excluded += sz
            else:
                n_files += 1
                n_included += sz
    except Exception:
        pass
    return n_files, n_included, n_excluded


def _watch_job_startup(
    job_name: str,
    region: str,
    max_poll_sec: int = 2700,
    poll_interval: int = 15,
) -> None:
    """Block until the job reaches Training, a terminal state, or timeout.

    Prints every SecondaryStatus transition with a UTC timestamp and elapsed
    minutes.  For Spot jobs, also reports when capacity is acquired and the
    total startup time (submit → Training).
    """
    sm = boto3.client("sagemaker", region_name=region)
    t0 = time.time()
    last_key: tuple = ("", "")
    capacity_acquired_at: Optional[float] = None

    print(f"[watch] Monitoring {job_name} (poll={poll_interval}s, max={max_poll_sec}s) …",
          flush=True)

    while True:
        elapsed = time.time() - t0
        if elapsed > max_poll_sec:
            print(f"[watch] Polling timeout after {elapsed/60:.1f} min. "
                  "Run the monitor command above to continue.", flush=True)
            break

        try:
            resp = sm.describe_training_job(TrainingJobName=job_name)
        except Exception as exc:
            print(f"[watch] describe error: {exc}", flush=True)
            time.sleep(poll_interval)
            continue

        status    = resp.get("TrainingJobStatus", "")
        secondary = resp.get("SecondaryStatus", "")
        key = (status, secondary)

        if key != last_key:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
            em = elapsed / 60
            print(f"[watch] {ts}  +{em:.1f}min  {status}/{secondary}", flush=True)
            last_key = key

            # Spot capacity is considered acquired once the instance is launching
            if (capacity_acquired_at is None
                    and secondary in ("LaunchingMLInstances", "PreparingTrainingStack",
                                      "Downloading", "DownloadingTrainingImage")):
                capacity_acquired_at = elapsed
                print(f"[watch] Spot capacity acquired  capacity_wait={elapsed:.0f}s "
                      f"({elapsed/60:.1f}min)", flush=True)

        if status in ("Completed", "Failed", "Stopped"):
            reason = resp.get("FailureReason") or ""
            suffix = f"  reason={reason}" if reason else ""
            print(f"[watch] Terminal: {status}{suffix}", flush=True)
            break

        if secondary == "Training":
            cap_wait = capacity_acquired_at or elapsed
            provision = elapsed - cap_wait
            print(
                f"[watch] Training STARTED  "
                f"total_startup={elapsed:.0f}s ({elapsed/60:.1f}min)  "
                f"capacity_wait={cap_wait:.0f}s ({cap_wait/60:.1f}min)  "
                f"provision_time={provision:.0f}s ({provision/60:.1f}min)",
                flush=True,
            )
            break

        time.sleep(poll_interval)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch Kairos-4B on SageMaker")

    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--micro_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--stable_frac", type=float, default=0.77)
    p.add_argument("--z_loss", type=float, default=1e-3)

    p.add_argument("--smoke_mode", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Enable memory-safe smoke/debug mode in kairos_train.py")
    p.add_argument("--budget_mode", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Enable memory-safe budget mode in kairos_train.py")
    p.add_argument("--ultra_smoke_mode", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Ultra-light smoke mode for g5.12xlarge 24 GB/GPU. "
                        "Implies smoke_mode. Aggressively shrinks all components.")
    p.add_argument("--ultra_smoke_mock_vision", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="In ultra_smoke: replace DINO with mock backbone to verify "
                        "DeepSpeed/data/training loop without real DINO OOM risk. "
                        "Does NOT validate real DINO memory — use dino_no_grad for that.")
    p.add_argument("--ultra_smoke_dino_no_grad", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="In ultra_smoke: run real DINO backbone under torch.no_grad() "
                        "(disables LoRA gradient flow). Use for Stage 2 smoke validation "
                        "before enabling full LoRA training in Stage 3.")
    p.add_argument("--ultra_smoke_skip_lidar", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Ultra_smoke debug only: replace LiDAR tokens/centroids with zeros.")
    p.add_argument("--ultra_smoke_skip_imu", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Ultra_smoke debug only: replace IMU tokens/delta_t with zeros.")
    p.add_argument("--ultra_smoke_skip_decoder_loss", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Ultra_smoke debug only: skip decoder CE and use anchor-only dummy loss.")
    p.add_argument("--ultra_smoke_skip_core", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Ultra_smoke debug only: bypass hybrid core (Mamba+CfC+SWA+MoE) "
                        "entirely. Use with --ultra_smoke_skip_decoder_loss for "
                        "plumbing-only validation (no MoE sparse backward).")
    p.add_argument("--force_core_with_anchor_loss", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="When --ultra_smoke_skip_decoder_loss=True, force the hybrid core "
                        "to still run (default: auto skip_core for safety). "
                        "moe_z_loss is always excluded from total_loss in this mode.")
    p.add_argument("--ultra_smoke_core_loss", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Replace total_loss with x.pow(2).mean()*1e-4 from hybrid core "
                        "output to test the core backward path. "
                        "Use with --dense_moe_fallback True for ZeRO-3 safety.")
    p.add_argument("--dense_moe_fallback", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Use dense weighted-sum MoE instead of sparse top-k dispatch. "
                        "Avoids ZeRO-3 shape mismatches. "
                        "Auto-enabled when --ultra_smoke_core_loss=True.")
    p.add_argument("--zero_stage", type=int, default=None, choices=[2, 3],
                   help="Override DeepSpeed ZeRO stage (2 or 3). "
                        "Default: 2 for ultra_smoke/budget runs (stable), "
                        "3 for full training. Use --zero_stage 3 with caution in "
                        "ultra_smoke: Stage 1d failed with shape 0 vs 8 error.")
    p.add_argument("--zero3_debug_safe", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Conservative ZeRO-3 settings for backward debug: "
                        "overlap_comm=False, contiguous_gradients=False, small buckets. "
                        "Only active when --zero_stage 3 is explicit.")
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
                        "0 = all loops (default). Requires --ultra_smoke_mode True.")
    p.add_argument("--ultra_smoke_core_loss_scope", type=str, default="post_core",
                   choices=["post_core", "post_mamba", "post_cfc", "post_swa", "post_moe"],
                   help="Scope for --ultra_smoke_core_loss backward test. "
                        "post_mamba/cfc/swa auto-bypasses later sub-blocks.")
    p.add_argument("--debug_grad_shapes", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Register backward hooks to detect zero-sized gradients under ZeRO-3.")
    p.add_argument("--debug_dtype_shapes", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Print dtype/device of modality tensors at LiDAR/IMU/core "
                        "boundaries on first batch. Diagnoses BF16/float32 mismatches.")
    p.add_argument("--disable_curriculum", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Use full training dataset for all curriculum phases. "
                        "Automatically True in ultra_smoke_mode. "
                        "Prevents empty DataLoaders when curriculum subsets are "
                        "smaller than world_size * micro_batch.")
    p.add_argument("--mem_trace", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Inject KAIROS_MEM_TRACE=1 into the SageMaker container "
                        "environment for per-component CUDA memory logging in CloudWatch.")
    p.add_argument("--freeze_lidar_in_smoke", nargs="?", const=True, default=None,
                   type=_str2bool,
                   help="Freeze LiDAR encoder in smoke/ultra_smoke. "
                        "Default True for ultra_smoke, False for smoke/budget.")
    p.add_argument("--freeze_imu_in_smoke",   nargs="?", const=True, default=None,
                   type=_str2bool,
                   help="Freeze IMU encoder in smoke/ultra_smoke. "
                        "Default True for ultra_smoke, False for smoke/budget.")
    p.add_argument("--freeze_core_in_smoke",  nargs="?", const=True, default=None,
                   type=_str2bool,
                   help="Freeze hybrid core in smoke/ultra_smoke. Default False.")
    p.add_argument("--zero_offload", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Emergency ZeRO-3 CPU offload fallback; disabled by default")
    p.add_argument("--save_ckpt_in_smoke", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Allow checkpoint saving during smoke_mode")
    p.add_argument("--val_every", type=int, default=None,
                   help="Override validation interval; budget/smoke default to 0")
    p.add_argument("--skip_bad_rows", nargs="?", const=True, default=None,
                   type=_str2bool,
                   help="Retry past rows with missing S3 assets. Default True in "
                        "smoke/ultra_smoke/budget, False in full training.")
    p.add_argument("--preflight_rows", type=int, default=None,
                   help="HeadObject-check N sampled training rows before training. "
                        "Default 20 in smoke/ultra_smoke.")
    p.add_argument("--allow_oxts_fallback", nargs="?", const=True, default=None,
                   type=_str2bool,
                   help="Use zero/small-noise IMU tokens when OXTS is malformed. "
                        "Default True in smoke/ultra_smoke, False in budget/full.")
    p.add_argument("--allow_single_rank", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Allow single-rank training on a multi-GPU instance.")

    p.add_argument("--no_spot", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Use on-demand instance instead of spot")
    p.add_argument("--keep_alive", type=int, default=0,
                   help="Managed Warm Pool keep-alive seconds after job ends "
                        "(0 = disabled, default). 3600 = keep instance warm 1 hr. "
                        "Requires --no_spot True — warm pools do not support spot. "
                        "After the first run, repeat runs skip image pull and "
                        "re-use installed pip packages, cutting startup to ~2-4 min.")
    p.add_argument("--instance", type=str, default="ml.g5.12xlarge")
    p.add_argument("--dry_run", nargs="?", const=True, default=False, type=_str2bool)
    p.add_argument("--job_name", type=str, default=None)

    p.add_argument("--role_arn", type=str, default=SM_ROLE,
                   help="Existing SageMaker execution role ARN")
    p.add_argument("--create_role", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="Create SageMakerKairosRole instead of requiring --role_arn")

    p.add_argument("--max_run", type=int, default=57_600,
                   help="Max training runtime seconds")
    p.add_argument("--max_wait", type=int, default=86_400,
                   help="Max spot wait seconds")
    p.add_argument("--watch_startup", nargs="?", const=True, default=False,
                   type=_str2bool,
                   help="After submitting, poll describe-training-job every 15s "
                        "until Training state or failure. Logs Spot capacity wait "
                        "and all SecondaryStatus transitions with timestamps. "
                        "Blocks the terminal. Useful for timing Spot cold starts.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

def launch(args: argparse.Namespace) -> None:
    if not GOLD_S3:
        raise ValueError(
            "GOLD_S3 env var is not set.\n"
            "Export it before running sagemaker_launch.py, e.g.:\n"
            "  $env:GOLD_S3 = 's3://YOUR-BUCKET/delta/gold/kitti_s2ft_triplets/'"
        )
    if not CKPT_S3:
        raise ValueError(
            "CKPT_S3 env var is not set.\n"
            "Export it before running sagemaker_launch.py, e.g.:\n"
            "  $env:CKPT_S3 = 's3://YOUR-CKPT-BUCKET/checkpoints/kairos-4b/'"
        )
    use_spot = not args.no_spot
    keep_alive = max(0, getattr(args, "keep_alive", 0))
    if keep_alive > 0 and use_spot:
        print(
            "[warn] Warm pools are on-demand only and are not used for Spot. "
            "--keep_alive ignored; Spot remains enabled."
        )
        keep_alive = 0  # keep_alive_period_in_seconds=0 on estimator; use_spot stays True
    bool_s = lambda v: "True" if v else "False"
    job_name = _sagemaker_safe_job_name(args.job_name or "kairos-4b", args.instance)

    if args.create_role:
        role = _ensure_sagemaker_role()
    else:
        if not args.role_arn:
            raise ValueError("--role_arn is required unless --create_role is set")
        role = args.role_arn

    sess = sagemaker.Session(
        boto_session=boto3.Session(region_name=AWS_REGION)
    )

    num_gpus, total_vram_gb = INSTANCE_INFO.get(args.instance, (1, 24))
    gpu_name = _INSTANCE_GPU_NAME.get(args.instance, "unknown GPU")

    # ── Effective ZeRO stage: default 2 for smoke/budget, 3 for full training ──
    # Stage 2 is stable for sparse/custom backward graphs (no param sharding).
    # Stage 1d failed with ZeRO-3 + dense MoE core backward (shape 0 vs 8).
    effective_zero_stage = args.zero_stage
    if effective_zero_stage is None:
        if args.ultra_smoke_mode or args.budget_mode:
            effective_zero_stage = 2

    hyperparams = {
        "total_steps": args.steps,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "warmup_frac": args.warmup_frac,
        "stable_frac": args.stable_frac,
        "z_loss_coeff": args.z_loss,
        "smoke_mode": bool_s(args.smoke_mode),
        "budget_mode": bool_s(args.budget_mode),
        "ultra_smoke_mode": bool_s(args.ultra_smoke_mode),
        "ultra_smoke_mock_vision": bool_s(args.ultra_smoke_mock_vision),
        "ultra_smoke_dino_no_grad": bool_s(args.ultra_smoke_dino_no_grad),
        "ultra_smoke_skip_lidar": bool_s(args.ultra_smoke_skip_lidar),
        "ultra_smoke_skip_imu": bool_s(args.ultra_smoke_skip_imu),
        "ultra_smoke_skip_decoder_loss": bool_s(args.ultra_smoke_skip_decoder_loss),
        "ultra_smoke_skip_core": bool_s(args.ultra_smoke_skip_core),
        "force_core_with_anchor_loss": bool_s(args.force_core_with_anchor_loss),
        "ultra_smoke_core_loss": bool_s(args.ultra_smoke_core_loss),
        "dense_moe_fallback": bool_s(args.dense_moe_fallback),
        "disable_curriculum": bool_s(args.disable_curriculum),
        "mem_trace": bool_s(args.mem_trace),
        # freeze_lidar/imu/core_in_smoke are intentionally OMITTED here when None.
        # Passing a None value causes the SageMaker SDK to serialise the key as a
        # bare flag (--freeze_core_in_smoke) which nargs="?" const=True turns ON.
        # kairos_train.py computes mode-specific defaults when these are absent.
        "zero_offload": bool_s(args.zero_offload),
        "save_ckpt_in_smoke": bool_s(args.save_ckpt_in_smoke),
        "allow_single_rank": bool_s(args.allow_single_rank),
        "freeze_core_steps": 3000,
        "lr_lora": 5e-5,
        "lr_encoders": 5e-5,
        "lr_core": 1e-5,
        "lr_decoder": 5e-5,
        "gold_s3": GOLD_S3,
        "ckpt_s3": CKPT_S3,
        "num_workers": 0 if (args.smoke_mode or args.budget_mode
                              or args.ultra_smoke_mode) else 4,
        "ckpt_every_min": 30,
        "log_every": 10,
        "instance_type": args.instance,
        # ZeRO-3 debug flags
        "core_debug_bypass_mamba": bool_s(args.core_debug_bypass_mamba),
        "core_debug_bypass_cfc":   bool_s(args.core_debug_bypass_cfc),
        "core_debug_bypass_swa":   bool_s(args.core_debug_bypass_swa),
        "core_debug_bypass_moe":   bool_s(args.core_debug_bypass_moe),
        "debug_grad_shapes":  bool_s(args.debug_grad_shapes),
        "debug_dtype_shapes": bool_s(args.debug_dtype_shapes),
        "zero3_debug_safe":   bool_s(args.zero3_debug_safe),
    }
    if args.core_debug_layers > 0:
        hyperparams["core_debug_layers"] = args.core_debug_layers
    if args.ultra_smoke_core_loss_scope != "post_core":
        hyperparams["ultra_smoke_core_loss_scope"] = args.ultra_smoke_core_loss_scope
    if args.val_every is not None:
        hyperparams["val_every"] = args.val_every
    if args.skip_bad_rows is not None:
        hyperparams["skip_bad_rows"] = bool_s(args.skip_bad_rows)
    if args.preflight_rows is not None:
        hyperparams["preflight_rows"] = args.preflight_rows
    if args.allow_oxts_fallback is not None:
        hyperparams["allow_oxts_fallback"] = bool_s(args.allow_oxts_fallback)
    if effective_zero_stage is not None:
        hyperparams["zero_stage"] = effective_zero_stage

    # Only inject freeze flags when the user explicitly provided a value.
    # Passing explicit "True"/"False" strings avoids the bare-flag/const=True trap.
    for _flag, _val in [
        ("freeze_lidar_in_smoke", args.freeze_lidar_in_smoke),
        ("freeze_imu_in_smoke",   args.freeze_imu_in_smoke),
        ("freeze_core_in_smoke",  args.freeze_core_in_smoke),
    ]:
        if _val is not None:
            hyperparams[_flag] = bool_s(_val)

    # SageMaker's torch_distributed launcher runs one torchrun worker per GPU.
    # DeepSpeed is still initialized inside kairos_train.py; the launcher only
    # provides the correct RANK/WORLD_SIZE/LOCAL_RANK environment.
    distribution = {
        "torch_distributed": {
            "enabled": True,
        },
    }

    spot_kwargs = {}
    if use_spot:
        spot_kwargs = {
            "use_spot_instances": True,
            "max_wait": args.max_wait,
            "max_run": args.max_run,
            "checkpoint_s3_uri": CKPT_S3,
            "checkpoint_local_path": "/home/ec2-user/kairos_ckpt",
        }

    # KAIROS_MEM_TRACE must be set in the container env; a local export does NOT
    # propagate into SageMaker training containers automatically.
    sagemaker_env: dict = {}
    if args.mem_trace:
        sagemaker_env["KAIROS_MEM_TRACE"] = "1"
    # Always inject DATA_REGION so the training container uses the correct S3
    # endpoint for parquet reads, KITTI downloads, and preflight checks —
    # independent of which region SageMaker runs the compute in.
    sagemaker_env["DATA_REGION"] = DATA_REGION

    estimator = PyTorch(
        entry_point="kairos_train.py",
        source_dir=SOURCE_DIR,
        role=role,
        instance_type=args.instance,
        instance_count=1,
        framework_version=FRAMEWORK_VER,
        py_version=PY_VER,
        hyperparameters=hyperparams,
        distribution=distribution,
        environment=sagemaker_env,
        output_path=OUTPUT_S3,
        base_job_name=job_name,
        sagemaker_session=sess,
        keep_alive_period_in_seconds=keep_alive,
        **spot_kwargs,
    )

    eff_batch = num_gpus * args.micro_batch * args.grad_accum
    est_epoch_fraction = (args.steps * eff_batch) / 154_606

    _bundle_files, _bundle_bytes, _bundle_excl = _bundle_stats(SOURCE_DIR)

    print("\n" + "-" * 60)
    print("  KAIROS-4B SAGEMAKER LAUNCH CONFIG")
    print("-" * 60)
    print(f"  SM region   : {AWS_REGION}  (SageMaker job)")
    print(f"  Data region : {DATA_REGION}  (S3 Gold + KITTI assets)")
    print(f"  Instance    : {args.instance}  ({num_gpus}x {gpu_name}, ~{total_vram_gb} GB VRAM)")
    print(f"  GPU count   : {num_gpus}")
    print(f"  Spot        : {use_spot}")
    if use_spot:
        print("  Warm pool  : not available for Spot (on-demand only)")
        print(f"  Max wait   : {args.max_wait}s  ({args.max_wait / 3600:.1f}h)")
    else:
        _wp_note = (f"{keep_alive}s  (<5 min repeats after first run)"
                    if keep_alive > 0 else "disabled")
        print(f"  Warm pool  : {_wp_note}")
    print(f"  Bundle     : {_bundle_files} files  {_bundle_bytes / 1024:.0f} KB uploaded  "
          f"(excluded {_bundle_excl / 1024 / 1024:.0f} MB via .sourceignore)")
    print(f"  Smoke mode  : {args.smoke_mode}")
    print(f"  Ultra-smoke : {args.ultra_smoke_mode}")
    print(f"  Mock vision : {args.ultra_smoke_mock_vision}")
    print(f"  DINO no_grad: {args.ultra_smoke_dino_no_grad}")
    print(f"  Skip LiDAR  : {args.ultra_smoke_skip_lidar}")
    print(f"  Skip IMU    : {args.ultra_smoke_skip_imu}")
    print(f"  Skip dec CE : {args.ultra_smoke_skip_decoder_loss}")
    print(f"  Skip core   : {args.ultra_smoke_skip_core}")
    print(f"  Force core  : {args.force_core_with_anchor_loss}")
    print(f"  Core loss   : {args.ultra_smoke_core_loss}")
    print(f"  Dense MoE   : {args.dense_moe_fallback}")
    print(f"  Dis. curricl: {args.disable_curriculum} (auto-True in ultra_smoke)")
    _zero_note = ""
    if args.zero_stage is None and (args.ultra_smoke_mode or args.budget_mode):
        _zero_note = " (stable default for G6 ultra_smoke/budget)"
    elif args.zero_stage is None:
        _zero_note = " (default for full training)"
    print(f"  ZeRO stage  : {effective_zero_stage if effective_zero_stage is not None else 3}{_zero_note}")
    print(f"  ZeRO3 safe  : {args.zero3_debug_safe}")
    print(f"  Core bypass : mamba={args.core_debug_bypass_mamba}  cfc={args.core_debug_bypass_cfc}  swa={args.core_debug_bypass_swa}  moe={args.core_debug_bypass_moe}")
    print(f"  Core layers : {args.core_debug_layers} (0=all)")
    print(f"  Core scope  : {args.ultra_smoke_core_loss_scope}")
    print(f"  Grad shapes : {args.debug_grad_shapes}")
    print(f"  Dtype debug : {args.debug_dtype_shapes}")
    print(f"  Mem trace   : {args.mem_trace}")
    print(f"  Budget mode : {args.budget_mode}")
    print(f"  ZeRO offload: {args.zero_offload}")
    print(f"  Skip bad row: {args.skip_bad_rows}  (None->mode default)")
    print(f"  Preflight   : {args.preflight_rows}  (None->mode default)")
    print(f"  OXTS fallback: {args.allow_oxts_fallback}  (None->mode default)")
    print(f"  Allow 1 rank: {args.allow_single_rank}")
    print(f"  Freeze lidar: {args.freeze_lidar_in_smoke}  (None->mode default)")
    print(f"  Freeze IMU  : {args.freeze_imu_in_smoke}  (None->mode default)")
    print(f"  Freeze core : {args.freeze_core_in_smoke}  (None->False)")
    print(f"  Steps       : {args.steps:,}  (eff_batch={eff_batch})")
    print(f"  Max run     : {args.max_run}")
    print(f"  Max wait    : {args.max_wait if use_spot else 'n/a'}")
    print(f"  ~Epoch frac : {est_epoch_fraction:.4f}")
    print(f"  Job name    : {job_name}")
    print(f"  Checkpoint  : {CKPT_S3}")
    print(f"  Gold data   : {GOLD_S3}")
    print(f"  Output      : {OUTPUT_S3}")
    print("-" * 60 + "\n")
    print(f"[distribution] {distribution!r}")
    print("[hyperparams] exact dict passed to training script:")
    for _k, _v in sorted(hyperparams.items()):
        print(f"    {_k} = {_v!r}")
    print()
    print("[environment] exact dict passed to SageMaker container:")
    for _k, _v in sorted(sagemaker_env.items()):
        print(f"    {_k} = {_v!r}")
    if not sagemaker_env:
        print("    <empty>")
    print()

    if num_gpus < 4 and not (args.smoke_mode or args.ultra_smoke_mode):
        print("[warn] Real budget/full runs are intended for ml.g5.12xlarge or larger.")

    # ── Instance-type VRAM awareness warnings ─────────────────────────────────
    _is_small_instance = args.instance in (
        "ml.g5.12xlarge", "ml.g5.24xlarge",
        "ml.g6.12xlarge", "ml.g6.24xlarge",
    )
    _is_g6 = args.instance.startswith("ml.g6.")
    if _is_small_instance and not (args.smoke_mode or args.budget_mode or args.ultra_smoke_mode):
        print(
            f"\n[WARN] {args.instance} has {total_vram_gb} GB VRAM split across "
            f"{num_gpus} GPUs ({total_vram_gb // num_gpus} GB/GPU, {gpu_name}). "
            "Full training mode may OOM. Consider --budget_mode or --ultra_smoke_mode.\n"
        )
    if _is_small_instance and args.smoke_mode and not args.ultra_smoke_mode:
        print(
            f"[WARN] smoke_mode on {args.instance} may CUDA OOM "
            f"({gpu_name}, {total_vram_gb // num_gpus} GB/GPU). If this run OOMs, "
            "re-launch with --ultra_smoke_mode True.\n"
        )
    if _is_g6 and args.budget_mode:
        print(
            f"[INFO] g6 budget_mode: {gpu_name} ({total_vram_gb // num_gpus} GB/GPU). "
            "kairos_train.py will cap max_pts=8192. "
            "Consider --freeze_lidar_in_smoke True --freeze_imu_in_smoke True "
            "for first 1000 budget steps to reduce VRAM.\n"
        )
    if args.instance == "ml.g6.12xlarge":
        print(
            "[INFO] g6.12xlarge smoke preset: recommended "
            "--micro_batch 1 --grad_accum 1 --ultra_smoke_mode True "
            "--ultra_smoke_mock_vision True.\n"
        )
    if args.instance == "ml.g6.48xlarge":
        print(
            "[INFO] g6.48xlarge larger test: micro_batch 1 or 2 and "
            "grad_accum 1 or 2 are the intended first settings.\n"
        )
    if args.instance == "ml.g5.48xlarge" and args.budget_mode:
        print(
            "[INFO] g5.48xlarge budget_mode: kairos_train.py will use max_pts=16384 "
            "and max_prompt/target=512.\n"
        )

    # Loud warning: ZeRO-3 + core backward is experimental
    if effective_zero_stage == 3 and args.ultra_smoke_core_loss:
        print(
            "\n[WARN] ZeRO-3 core backward is experimental; "
            "Stage 1d has previously failed with:\n"
            "  RuntimeError: The size of tensor a (0) must match the size of "
            "tensor b (8) at non-singleton dimension 1\n"
            "Consider --zero_stage 2 for stability, or add --zero3_debug_safe True "
            "for conservative ZeRO-3 settings (overlap_comm=False, small buckets).\n",
            flush=True,
        )

    # ── Startup latency analysis ──────────────────────────────────────────────
    if use_spot:
        print(
            f"\n[spot] Spot ENABLED  max_wait={args.max_wait}s ({args.max_wait / 3600:.1f}h)  "
            f"source_bundle={_bundle_bytes / 1024:.0f} KB  warm_pool=N/A\n"
            "[spot] Startup breakdown (after capacity acquired):\n"
            "[spot]   Spot capacity wait : 0-30+ min  (non-deterministic)\n"
            "[spot]   Image pull + init  : ~5-8 min   (cold DLC container)\n"
            f"[spot]   Source download   : <5s         ({_bundle_bytes / 1024:.0f} KB bundle)\n"
            "[spot]   pip install        : ~1-3 min   (requirements.txt)\n"
            "[spot]   Script init        : ~1-2 min   (dist init + data load)\n"
            "[spot] <5 min total startup is NOT achievable for Spot "
            "(warm pools are on-demand only).\n"
            "[spot] Add --watch_startup True to log capacity wait and all transitions.\n",
            flush=True,
        )
    elif keep_alive > 0:
        _od_cost = _INSTANCE_ON_DEMAND_COST.get(args.instance, 0.0)
        _idle_usd = _od_cost * keep_alive / 3600
        print(
            f"\n[warm_pool] Managed Warm Pool ENABLED  keep_alive={keep_alive}s  "
            f"on_demand=True"
        )
        if _od_cost > 0:
            print(
                f"[warm_pool] idle cost: ~${_idle_usd:.2f} per run interval "
                f"({keep_alive}s at ~${_od_cost:.3f}/hr for {args.instance})"
            )
        print(
            "[warm_pool] <5 min startup is achievable on SECOND+ run when: "
            "same instance_type, same framework version, source unchanged.\n"
            "[warm_pool] Cold first run still requires image pull (~5-10 min).\n"
        )
    else:
        print(
            "\n[info] On-demand + no warm pool: cold-start is ~8-15 min "
            "(image pull + container init + pip install). "
            "Add --keep_alive 3600 to enable warm pool for repeat runs.\n"
        )

    if _bundle_bytes > 50 * 1024 * 1024:
        print(
            f"\n[warn] Source bundle is {_bundle_bytes / 1024 / 1024:.0f} MB — "
            "large bundles slow down every launch. "
            "Ensure .sourceignore is present in source_dir.\n"
        )

    if args.dry_run:
        print("[dry_run] Config printed - not submitted.")
        return

    _t_submit = time.time()
    estimator.fit(job_name=job_name, wait=False)
    _submit_elapsed = time.time() - _t_submit
    job = job_name
    submit_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"[launch] Job submitted: {job}")
    print(
        f"[timing] submitted_at={submit_ts}  sdk_call={_submit_elapsed:.1f}s  "
        f"spot={use_spot}  keep_alive={keep_alive}s  "
        f"bundle={_bundle_bytes / 1024:.0f} KB"
    )
    print(
        "[monitor] aws sagemaker describe-training-job "
        f"--training-job-name {job} "
        f"--region {AWS_REGION} "
        "--query '{Status:TrainingJobStatus,Secondary:SecondaryStatus,Failure:FailureReason}'"
    )

    if getattr(args, "watch_startup", False):
        _watch_job_startup(job, AWS_REGION)


if __name__ == "__main__":
    launch(_args())
