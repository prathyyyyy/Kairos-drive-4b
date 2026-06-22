"""
kairos_infer.py — Conservative inference/evaluation for the trained Kairos-4B
DeepSpeed ZeRO-3 checkpoint.  Training code is NOT modified or imported for
training — only the model definition and the dataset/dataloader helpers are
reused from kairos_train.py / kairos_model.py.

Why consolidation is required
-----------------------------
The final checkpoint is a DeepSpeed ZeRO-3 save with 8 model shards
(zero_pp_rank_<r>_mp_rank_00_model_states.pt) and 8 optimizer shards
((bf16_)zero_pp_rank_<r>_mp_rank_00_optim_states.pt).  Under ZeRO-3 the model
shards contain per-rank PARTITIONS / placeholders, not full tensors: the full
fp32 weights of TRAINABLE params live in the optimizer shards'
`fp32_flat_groups`, and FROZEN params live in `frozen_param_fragments` inside
the model shards.  It cannot be torch.load()-ed like a normal checkpoint.

This script therefore:
  1. Downloads the checkpoint tag dir from S3 (skips files already present).
  2. Consolidates ZeRO-3 shards -> a single fp32 state dict.  Uses DeepSpeed's
     official `get_fp32_state_dict_from_zero_checkpoint` when deepspeed is
     importable; otherwise falls back to a built-in consolidator implementing
     the same reconstruction protocol (works on Windows without deepspeed).
     The consolidated state dict is cached as consolidated_fp32.pt so this
     happens once.
  3. Rebuilds KairosModel with the default full-training config, auto-detecting
     `max_gen_len` from the checkpoint's s2ft_decoder.pos_emb shape, and loads
     the weights with an explicit missing/unexpected/shape-mismatch report.
  4. Runs torch.no_grad() evaluation over the requested gold split:
       - teacher-forced total_loss / s2ft_loss (directly comparable to the
         training run's val_total=0.9158 / val_s2ft=0.9158)
       - byte-level token accuracy (argmax of teacher-forced logits)
       - optional greedy generation (temperature=0 -> deterministic)
  5. Writes predictions.jsonl, inference_summary.json, inference_report.md.

ADE/FDE: NOT computed.  Kairos-4B outputs a byte-level reasoning chain +
answer text and an (untrained, w_det=0) detection head; the gold table has no
trajectory/waypoint targets, so ADE/FDE cannot be derived without a confirmed
waypoint decoding schema.  They are reported as "unavailable", not faked.

Windows PowerShell usage (first conservative pass, 16 val samples):
  python scripts\kairos_infer.py --split val --max_samples 16 --device cpu

  # On a CUDA machine:
  python scripts\kairos_infer.py --split val --max_samples 32 --device cuda

  # If consolidation was done elsewhere (e.g. on SageMaker/Linux):
  python scripts\kairos_infer.py --consolidated_ckpt kairos_fp32.pt

Notes
-----
- RAM: consolidation + fp32 model needs roughly 2x the fp32 parameter size
  (~16 GB state dict + ~16 GB module) — run on a machine with >= 48 GB RAM,
  or on a GPU instance with --dtype bfloat16.
- transformers must be installed locally (real DINOv2 backbone, ~1.2 GB
  download on first run; cached afterwards).
- KAIROS_DECODER_PREPEND_BOS must match training (default: unset/legacy).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Force UTF-8 output on Windows.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# training/ subfolders on sys.path so kairos_model / kairos_train import as in training.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (REPO_ROOT / "training" / "training-files", REPO_ROOT / "training" / "train-support-files"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_CHECKPOINT_S3 = (
    "s3://kairos-emr-assets-use1-195231312992/checkpoints/"
    "kairos-motion/main-100k-lidar-kitti68-zod32-8000/step_0008000/"
)
DEFAULT_GOLD_S3 = (
    "s3://project-kairos-raw-use1-s3-195231312992/delta/gold/"
    "kitti_zod_mixed_100k_lidar_kitti68_zod32_walkforward_s2ft_triplets/"
)
DEFAULT_REGION = "us-east-1"

ADE_FDE_STATUS = (
    "unavailable — Kairos-4B emits a byte-level reasoning chain + answer text "
    "and an untrained detection head (w_det=0); the gold table has no "
    "trajectory/waypoint targets, so ADE/FDE cannot be computed until a "
    "waypoint decoding schema is defined and validated."
)


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kairos-4B ZeRO-3 checkpoint inference/evaluation"
    )
    p.add_argument("--checkpoint_s3", default=DEFAULT_CHECKPOINT_S3,
                   help="S3 prefix of the ZeRO-3 checkpoint tag dir")
    p.add_argument("--gold_s3", default=DEFAULT_GOLD_S3,
                   help="S3 prefix of the gold S2FT triplet table")
    p.add_argument("--split", default="val", choices=["val", "test", "train"],
                   help="dataset_split partition to evaluate (default val)")
    p.add_argument("--max_samples", type=int, default=16,
                   help="Max samples to evaluate (default 16, conservative)")
    p.add_argument("--output_dir", default="./inference_out",
                   help="Directory for predictions/summary/report")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Inference device (auto -> cuda if available)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--save_predictions", type=str2bool,
                   default=True, help="Write predictions.jsonl (default True)")
    # ── Output upload (optional) ────────────────────────────────────────────
    p.add_argument("--output_s3", default=None,
                   help="If set, recursively upload --output_dir to "
                        "<output_s3>/<job_name>/ after inference completes "
                        "(skips the ckpt/ and s3_cache/ subdirs — see "
                        "upload_directory_to_s3)")
    p.add_argument("--job_name", default=None,
                   help="Subfolder name under --output_s3 (default: the "
                        "TRAINING_JOB_NAME / SAGEMAKER_JOB_NAME env var, else "
                        "a UTC timestamp)")
    # ── Checkpoint handling ────────────────────────────────────────────────
    p.add_argument("--consolidated_ckpt", default=None,
                   help="Path to an already-consolidated fp32 .pt state dict "
                        "(skips S3 download + ZeRO-3 consolidation)")
    p.add_argument("--ckpt_local_dir", default=None,
                   help="Where to download/cache the checkpoint "
                        "(default <output_dir>/ckpt)")
    p.add_argument("--region", default=DEFAULT_REGION)
    # ── Generation ─────────────────────────────────────────────────────────
    p.add_argument("--generate", type=str2bool,
                   default=True,
                   help="Run autoregressive generation per sample (default True)")
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="Cap on generated byte tokens (default 128)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy/deterministic (default)")
    p.add_argument("--top_p", type=float, default=0.9)
    # ── Data caps (match training full-mode defaults) ──────────────────────
    p.add_argument("--max_prompt", type=int, default=512)
    p.add_argument("--max_target", type=int, default=512)
    p.add_argument("--max_pts", type=int, default=30_000,
                   help="LiDAR point cap per frame (default 30000 = full mode)")
    p.add_argument("--num_workers", type=int, default=0)
    # ── Misc ───────────────────────────────────────────────────────────────
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float32", "bfloat16"],
                   help="auto -> bfloat16 on cuda, float32 on cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict_data", action="store_true",
                   help="Fail on bad rows instead of counting them as failed")
    p.add_argument("--allow_mock_vision", action="store_true",
                   help="Permit running without transformers (mock DINOv2 — "
                        "results are NOT meaningful; for dry runs only)")
    p.add_argument("--multi_gpu_eval", type=str2bool, default=False,
                   help="Spawn one inference worker per GPU (default False)")
    p.add_argument("--num_gpu_workers", type=int, default=0,
                   help="Number of GPU workers; 0 auto-detects CUDA GPUs")
    p.add_argument("--rank", type=int, default=None,
                   help="Internal worker rank for multi-GPU eval")
    p.add_argument("--world_size", type=int, default=None,
                   help="Internal worker world size for multi-GPU eval")
    p.add_argument("--sample_start", type=int, default=None,
                   help="Internal inclusive sample start for worker sharding")
    p.add_argument("--sample_end", type=int, default=None,
                   help="Internal exclusive sample end for worker sharding")
    p.add_argument("--tier2_metrics", type=str2bool, default=True,
                   help="Write Tier-2 validation diagnostics (default True)")
    p.add_argument("--group_metrics", type=str2bool, default=True,
                   help="Write grouped slice metrics (default True)")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers — unit-testable without S3 / GPU
# ─────────────────────────────────────────────────────────────────────────────

def parse_tag_from_checkpoint_uri(uri: str) -> Tuple[str, Optional[str]]:
    """
    's3://b/p/step_0008000/' -> ('s3://b/p/', 'step_0008000').
    If the last segment is not a step_* tag, returns (uri, None).
    """
    trimmed = uri.rstrip("/")
    root, _, last = trimmed.rpartition("/")
    if re.fullmatch(r"step_\d+", last):
        return root + "/", last
    return uri if uri.endswith("/") else uri + "/", None


def detect_max_gen_len(sd: Dict[str, Any], default: int = 512) -> int:
    """The S2FT decoder pos_emb table is sized by max_gen_len at train time."""
    t = sd.get("s2ft_decoder.pos_emb.weight")
    if t is not None and hasattr(t, "shape") and len(t.shape) == 2:
        return int(t.shape[0])
    return default


def filtered_load_state_dict(model, sd: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Load `sd` into `model` with strict=False semantics PLUS shape-mismatch
    protection (plain strict=False still raises on size mismatch).

    Returns {"missing": [...], "unexpected": [...], "shape_mismatch": [...]}.
    """
    model_sd = model.state_dict()
    loadable: "OrderedDict[str, Any]" = OrderedDict()
    shape_mismatch: List[str] = []
    unexpected: List[str] = []

    for key, value in sd.items():
        if key not in model_sd:
            unexpected.append(key)
            continue
        if tuple(model_sd[key].shape) != tuple(value.shape):
            shape_mismatch.append(
                f"{key}: ckpt{tuple(value.shape)} != model{tuple(model_sd[key].shape)}"
            )
            continue
        loadable[key] = value

    result = model.load_state_dict(loadable, strict=False)
    missing = [k for k in result.missing_keys]
    return {
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
    }


def classify_missing_keys(missing: List[str]) -> Dict[str, List[str]]:
    """
    Split missing keys into benign vs concerning.

    Benign:
      - vision_encoder.dinov2.* : frozen pretrained DINOv2 backbone — weights
        come from HuggingFace at model construction time.
      - s2ft_decoder.lm_head.*  : weight-tied to s2ft_decoder.embedding.
    Everything else missing means those modules run with random init — flagged.
    """
    benign, concerning = [], []
    for k in missing:
        if k.startswith("vision_encoder.dinov2.") or "lm_head" in k:
            benign.append(k)
        else:
            concerning.append(k)
    return {"benign": benign, "concerning": concerning}


def token_accuracy_from_logits(logits, dec_label) -> Tuple[float, int]:
    """
    Byte-level next-token accuracy under teacher forcing.
    `dec_label` uses -1 for ignored positions (same convention as the model).
    Returns (accuracy, n_valid_positions); accuracy is nan when no positions.
    """
    import torch

    with torch.no_grad():
        valid = dec_label != -1
        n = int(valid.sum().item())
        if n == 0:
            return float("nan"), 0
        pred = logits.argmax(dim=-1)
        correct = ((pred == dec_label) & valid).sum().item()
        return float(correct) / n, n


def strip_module_prefix(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a state dict that was saved with a 'module.' wrapper prefix."""
    keys = list(sd.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return OrderedDict((k[len("module."):], v) for k, v in sd.items())
    return sd


def build_decoder_labels(target_bytes, loss_mask):
    """
    Reconstruct the teacher-forcing label tensor with the SAME convention as
    KairosModel.forward (including the KAIROS_DECODER_PREPEND_BOS env gate),
    so token accuracy aligns exactly with the loss the model computed.
    """
    prepend_bos = os.environ.get(
        "KAIROS_DECODER_PREPEND_BOS", ""
    ).strip().lower() in ("1", "true", "yes")
    if prepend_bos:
        dec_label = target_bytes.clone()
        label_mask = loss_mask
    else:
        dec_label = target_bytes[:, 1:].clone()
        label_mask = loss_mask[:, 1:] if loss_mask is not None else None
    if label_mask is not None:
        dec_label = dec_label.masked_fill(~label_mask.bool(), -1)
    return dec_label


def decode_bytes(token_ids) -> str:
    """Byte token ids -> text (drops BOS/EOS/PAD; tolerant utf-8 decode)."""
    out = bytearray()
    for tok in token_ids.tolist():
        if 0 < tok < 256:
            out.append(tok)
    return out.decode("utf-8", errors="replace")


def split_s3_uri(uri: str) -> Tuple[str, str]:
    """'s3://bucket/key/prefix/' -> ('bucket', 'key/prefix')"""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    return bucket, prefix.rstrip("/")


def resolve_job_name(job_name: Optional[str]) -> str:
    """--job_name, else SageMaker's job-name env vars, else a UTC timestamp."""
    if job_name:
        return job_name
    for env_key in ("TRAINING_JOB_NAME", "SAGEMAKER_JOB_NAME"):
        v = os.environ.get(env_key, "").strip()
        if v:
            return v
    return "infer-" + _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def render_consolidated_checkpoint_note(
    method: str, tag: str, cache_path: Optional[Path]
) -> str:
    """Small text note describing where the consolidated fp32 state dict
    lives (or would live) — written alongside predictions/summary/report so
    a re-run can pass --consolidated_ckpt and skip re-consolidation."""
    lines = [
        "# Consolidated checkpoint cache note",
        "",
        f"- consolidation_method: {method}",
        f"- tag: {tag}",
    ]
    if cache_path is not None:
        lines.append(f"- local_path: {cache_path}")
        if cache_path.exists():
            lines.append(f"- size_gb: {cache_path.stat().st_size / 1e9:.2f}")
    else:
        lines.append("- local_path: n/a")
    lines += [
        "",
        "The consolidated fp32 state dict is NOT uploaded to S3 (10-20 GB). "
        "Re-run with --consolidated_ckpt pointing at this local path to skip "
        "re-downloading and re-consolidating the ZeRO-3 shards.",
        "",
    ]
    return "\n".join(lines)


def upload_directory_to_s3(
    local_dir: Path,
    s3_prefix: str,
    region: str,
    exclude_dirnames: Optional[set] = None,
) -> List[str]:
    """
    Recursively upload local_dir to s3_prefix (an 's3://bucket/key/.../' URI),
    preserving relative paths. Top-level subdirectories named in
    exclude_dirnames are skipped (default: {"ckpt", "s3_cache"}) — these hold
    the downloaded ZeRO-3 shards / consolidated fp32 checkpoint and the S3
    asset cache, both far too large and pointless to round-trip back to S3.
    Returns the list of uploaded s3:// URIs.
    """
    import boto3

    exclude_dirnames = exclude_dirnames or {"ckpt", "s3_cache"}
    bucket, prefix = split_s3_uri(s3_prefix)
    s3 = boto3.client("s3", region_name=region)

    uploaded: List[str] = []
    for fp in sorted(local_dir.rglob("*")):
        if not fp.is_file():
            continue
        rel = fp.relative_to(local_dir)
        if rel.parts and rel.parts[0] in exclude_dirnames:
            continue
        rel_posix = rel.as_posix()
        key = f"{prefix}/{rel_posix}" if prefix else rel_posix
        s3.upload_file(str(fp), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def deterministic_shards(n_samples: int, n_workers: int) -> List[Tuple[int, int]]:
    """Contiguous deterministic [start, end) sample ranges."""
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")
    n = max(0, int(n_samples))
    return [
        ((n * rank) // n_workers, (n * (rank + 1)) // n_workers)
        for rank in range(n_workers)
    ]


def _is_ok_record(r: Dict[str, Any]) -> bool:
    if "failed" in r:
        return not bool(r.get("failed"))
    return r.get("status") == "ok"


def _sample_index(r: Dict[str, Any]) -> int:
    return int(r.get("sample_index", r.get("index", 0)))


def _metric_values(records: Iterable[Dict[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for r in records:
        if not _is_ok_record(r):
            continue
        v = r.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(fv):
            vals.append(fv)
    return vals


def _mean_records(records: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    vals = _metric_values(records, key)
    return (sum(vals) / len(vals)) if vals else None


def quantiles(values: Iterable[Any],
              probs: Tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99)
              ) -> Dict[str, Optional[float]]:
    vals: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(fv):
            vals.append(fv)
    if not vals:
        return {f"p{int(p * 100)}": None for p in probs}
    vals.sort()
    out: Dict[str, Optional[float]] = {}
    for p in probs:
        pos = (len(vals) - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            q = vals[lo]
        else:
            q = vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
        out[f"p{int(p * 100)}"] = q
    return out


def _present(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    return bool(s) and s.lower() not in ("none", "nan", "null", "na")


def _curriculum_bucket(value: Any, width: int = 1000) -> Optional[str]:
    if value is None or not _present(value):
        return None
    try:
        iv = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    start = (iv // width) * width
    return f"{start}-{start + width - 1}"


def _group_stats(group_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_samples = len(group_records)
    n_failed = sum(1 for r in group_records if not _is_ok_record(r))
    row: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_failed": n_failed,
        "total_loss_mean": _mean_records(group_records, "total_loss"),
        "s2ft_loss_mean": _mean_records(group_records, "s2ft_loss"),
        "per_sample_ce_mean": _mean_records(group_records, "per_sample_ce"),
        "token_accuracy_mean": _mean_records(group_records, "token_accuracy"),
    }
    if any(r.get("generated_length") is not None for r in group_records):
        row["generated_length_mean"] = _mean_records(group_records, "generated_length")
    if any(r.get("target_length") is not None for r in group_records):
        row["target_length_mean"] = _mean_records(group_records, "target_length")
    return row


def build_group_metrics(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Slice metrics from existing prediction metadata; no trajectory metrics."""
    group_specs: List[Tuple[str, Any]] = [
        ("dataset_type", lambda r: r.get("dataset_type")),
        ("source_table", lambda r: r.get("source_table")),
        ("camera_id", lambda r: r.get("camera_id")),
        ("complexity_tier", lambda r: r.get("complexity_tier")),
        ("curriculum_order_bucket", lambda r: _curriculum_bucket(
            r.get("curriculum_order")
        )),
        ("lidar_presence", lambda r: "lidar_present" if _present(
            r.get("lidar_path")) else "lidar_missing"),
        ("oxts_presence", lambda r: "oxts_present" if _present(
            r.get("oxts_path")) else "oxts_missing"),
        ("image_temporal_completeness", lambda r: (
            "t_minus_1_and_2_present"
            if _present(r.get("image_path_t_minus_1")) and
            _present(r.get("image_path_t_minus_2"))
            else "temporal_images_incomplete"
        )),
    ]
    rows: List[Dict[str, Any]] = []
    for group_by, getter in group_specs:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            value = getter(r)
            if not _present(value):
                continue
            buckets.setdefault(str(value), []).append(r)
        for group, group_records in sorted(buckets.items()):
            rows.append({
                "group_by": group_by,
                "group": group,
                **_group_stats(group_records),
            })
    return rows


def select_extreme_samples(records: List[Dict[str, Any]], key: str,
                           n: int = 10, reverse: bool = True
                           ) -> List[Dict[str, Any]]:
    candidates = []
    for r in records:
        if not _is_ok_record(r) or r.get(key) is None:
            continue
        try:
            score = float(r[key])
        except (TypeError, ValueError):
            continue
        if not math.isnan(score):
            candidates.append((score, _sample_index(r), r))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=reverse)
    keep_keys = [
        "sample_index", "index", "dataset_type", "source_table", "camera_id",
        "complexity_tier", "curriculum_order", "drive_id", "sequence_id",
        "frame_index", "total_loss", "s2ft_loss", "per_sample_ce",
        "token_accuracy", "target_length", "generated_length", "failed",
        "failure_reason", "status", "error",
    ]
    out = []
    for _, _, r in candidates[:n]:
        out.append({k: r.get(k) for k in keep_keys if k in r})
    return out


def build_tier2_metrics(
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    include_groups: bool = True,
) -> Dict[str, Any]:
    ok = [r for r in records if _is_ok_record(r)]
    failed = [r for r in records if not _is_ok_record(r)]
    n_ok = len(ok)
    s2ft_vals = _metric_values(records, "s2ft_loss")
    acc_vals = _metric_values(records, "token_accuracy")
    gen_vals = _metric_values(records, "generated_length")

    def _rate(pred) -> Optional[float]:
        return (sum(1 for r in ok if pred(r)) / n_ok) if n_ok else None

    training_ref = summary.get("training_reference", {})
    eval_s2ft = summary.get("metrics", {}).get("s2ft_loss_mean")
    ref_s2ft = training_ref.get("val_s2ft", 0.9158)
    try:
        delta = None if eval_s2ft is None else float(eval_s2ft) - float(ref_s2ft)
    except (TypeError, ValueError):
        delta = None
    full_n = summary.get("full_validation_samples")
    requested = summary.get("max_samples_requested")
    sample_dependent = False
    if full_n is not None and requested is not None:
        try:
            sample_dependent = int(requested) < int(full_n)
        except (TypeError, ValueError):
            sample_dependent = False

    metrics: Dict[str, Any] = {
        "note": (
            "Tier-2 metrics are slice-based validation diagnostics computed "
            "from held-out validation samples. They are not trajectory ADE/FDE metrics."
        ),
        "overall": summary.get("metrics", {}),
        "training_reference_comparison": {
            "training_reference": {
                "val_total": training_ref.get("val_total", 0.9158),
                "val_s2ft": ref_s2ft,
            },
            "eval_s2ft_loss_mean": eval_s2ft,
            "eval_minus_training_val_s2ft": delta,
            "sample_dependent": sample_dependent,
        },
        "robustness": {
            "failed_samples": [
                {
                    "sample_index": _sample_index(r),
                    "reason": r.get("failure_reason", r.get("error", "")),
                }
                for r in failed
            ],
            "high_loss_rate_s2ft_gt_2": _rate(
                lambda r: r.get("s2ft_loss") is not None and
                float(r["s2ft_loss"]) > 2.0
            ),
            "low_accuracy_rate_lt_0_90": _rate(
                lambda r: r.get("token_accuracy") is not None and
                float(r["token_accuracy"]) < 0.90
            ),
            "exact_or_near_exact_rate_ge_0_99": _rate(
                lambda r: r.get("token_accuracy") is not None and
                float(r["token_accuracy"]) >= 0.99
            ),
            "worst_10_samples_by_s2ft_loss": select_extreme_samples(
                records, "s2ft_loss", n=10, reverse=True
            ),
            "best_10_samples_by_s2ft_loss": select_extreme_samples(
                records, "s2ft_loss", n=10, reverse=False
            ),
        },
        "calibration_style_summary": {
            "loss_quantiles": quantiles(s2ft_vals),
            "token_accuracy_quantiles": quantiles(acc_vals),
            "generated_length_quantiles": quantiles(gen_vals) if gen_vals else {},
        },
        "ade": None,
        "fde": None,
        "ade_fde_status": ADE_FDE_STATUS,
    }
    if include_groups:
        metrics["groups"] = build_group_metrics(records)
    return metrics


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_tier2_for_csv(tier2: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, list):
            rows.append({"metric": prefix, "value": json.dumps(value, default=str)})
        else:
            rows.append({"metric": prefix, "value": value})

    walk("", {k: v for k, v in tier2.items() if k != "groups"})
    return rows


def aggregate_worker_summaries(
    summaries: List[Dict[str, Any]]
) -> Dict[str, Optional[float]]:
    """Weighted metric aggregation by n_valid_samples for parent eval."""
    metric_keys = [
        "total_loss_mean", "s2ft_loss_mean", "per_sample_ce_mean",
        "token_accuracy_mean",
    ]
    out: Dict[str, Optional[float]] = {}
    total_valid = sum(int(s.get("n_valid_samples", 0)) for s in summaries)
    total_failed = sum(int(s.get("n_failed_samples", 0)) for s in summaries)
    for key in metric_keys:
        num = 0.0
        den = 0
        for s in summaries:
            n = int(s.get("n_valid_samples", 0))
            v = s.get("metrics", {}).get(key)
            if n and v is not None:
                num += float(v) * n
                den += n
        out[key] = (num / den) if den else None
    out["n_valid_samples"] = total_valid
    out["n_failed_samples"] = total_failed
    return out


def build_summary(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    load_report: Dict[str, Any],
    timing: Dict[str, float],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate per-sample records into inference_summary.json content."""
    ok = [r for r in records if _is_ok_record(r)]
    failed = [r for r in records if not _is_ok_record(r)]

    def _mean(key: str) -> Optional[float]:
        return _mean_records(ok, key)

    summary: Dict[str, Any] = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "checkpoint_s3": args.checkpoint_s3,
        "consolidated_ckpt": args.consolidated_ckpt,
        "gold_s3": args.gold_s3,
        "split": args.split,
        "device": timing.get("device"),
        "dtype": timing.get("dtype"),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_samples_requested": args.max_samples,
        "n_valid_samples": len(ok),
        "n_failed_samples": len(failed),
        "metrics": {
            "total_loss_mean": _mean("total_loss"),
            "s2ft_loss_mean": _mean("s2ft_loss"),
            "per_sample_ce_mean": _mean("per_sample_ce"),
            "token_accuracy_mean": _mean("token_accuracy"),
            "moe_z_loss_mean": _mean("moe_z_loss"),
            "ade": None,
            "fde": None,
            "ade_fde_status": ADE_FDE_STATUS,
        },
        "training_reference": {
            "val_total": 0.9158,
            "val_s2ft": 0.9158,
            "note": "final training-run validation losses for comparison",
        },
        "checkpoint_load": load_report,
        "timing": timing,
        "generation": {
            "enabled": bool(args.generate),
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "deterministic": args.temperature == 0.0,
        },
    }
    if extra:
        summary.update(extra)
    return summary


def render_report_md(summary: Dict[str, Any],
                     records: List[Dict[str, Any]]) -> str:
    """Human-readable inference_report.md content."""
    m = summary["metrics"]

    def _f(v, prec=4):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.{prec}f}"
        return str(v)

    lines = [
        "# Kairos-4B Inference Report",
        "",
        f"Generated (UTC): {summary['generated_utc']}",
        "",
        "## Run configuration",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Checkpoint | {summary['checkpoint_s3']} |",
        f"| Gold table | {summary['gold_s3']} |",
        f"| Split | {summary['split']} |",
        f"| Device / dtype | {summary['device']} / {summary['dtype']} |",
        f"| Batch size | {summary['batch_size']} |",
        f"| Seed | {summary['seed']} |",
        f"| Samples requested | {summary['max_samples_requested']} |",
        f"| Valid samples | {summary['n_valid_samples']} |",
        f"| Failed samples | {summary['n_failed_samples']} |",
        "",
        "## Metrics (teacher-forced, torch.no_grad, model.eval)",
        "",
        "| Metric | Value | Training-run reference |",
        "|---|---|---|",
        f"| total_loss (mean) | {_f(m['total_loss_mean'])} | "
        f"val_total = {summary['training_reference']['val_total']} |",
        f"| s2ft_loss (mean) | {_f(m['s2ft_loss_mean'])} | "
        f"val_s2ft = {summary['training_reference']['val_s2ft']} |",
        f"| per-sample CE (mean) | {_f(m['per_sample_ce_mean'])} | — |",
        f"| byte token accuracy | {_f(m['token_accuracy_mean'])} | — |",
        f"| moe_z_loss (mean) | {_f(m['moe_z_loss_mean'])} | — |",
        f"| ADE | unavailable | — |",
        f"| FDE | unavailable | — |",
        "",
        f"**ADE/FDE status:** {m['ade_fde_status']}",
        "",
    ]
    tier2 = summary.get("tier2_metrics", {})
    lines += [
        "## Tier-2 Validation Diagnostics",
        "",
        "Tier-2 metrics are slice-based validation diagnostics computed from "
        "held-out validation samples. They are not trajectory ADE/FDE metrics.",
        "",
    ]
    trc = tier2.get("training_reference_comparison", {})
    if trc:
        lines += [
            "### Training-reference comparison",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| eval_s2ft_loss_mean | {_f(trc.get('eval_s2ft_loss_mean'))} |",
            f"| training val_s2ft | {_f(trc.get('training_reference', {}).get('val_s2ft'))} |",
            f"| eval - training val_s2ft | {_f(trc.get('eval_minus_training_val_s2ft'))} |",
            f"| sample-dependent comparison | {trc.get('sample_dependent')} |",
            "",
        ]

    groups = tier2.get("groups", [])

    def _group_table(title: str, names: set, limit: int = 12) -> None:
        rows = [g for g in groups if g.get("group_by") in names]
        lines.extend([f"### {title}", ""])
        if not rows:
            lines.extend(["n/a", ""])
            return
        lines.extend([
            "| Group by | Group | n | failed | s2ft | token acc |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for g in rows[:limit]:
            lines.append(
                f"| {g.get('group_by')} | {g.get('group')} | "
                f"{g.get('n_samples')} | {g.get('n_failed')} | "
                f"{_f(g.get('s2ft_loss_mean'))} | "
                f"{_f(g.get('token_accuracy_mean'))} |"
            )
        lines.append("")

    _group_table(
        "Dataset/source breakdown",
        {"dataset_type", "source_table", "camera_id", "curriculum_order_bucket"},
    )
    _group_table("Complexity breakdown", {"complexity_tier"})
    _group_table(
        "Data-quality/modality breakdown",
        {"lidar_presence", "oxts_presence", "image_temporal_completeness"},
    )

    rob = tier2.get("robustness", {})
    if rob:
        lines += [
            "### Robustness summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| high_loss_rate_s2ft_gt_2 | {_f(rob.get('high_loss_rate_s2ft_gt_2'))} |",
            f"| low_accuracy_rate_lt_0_90 | {_f(rob.get('low_accuracy_rate_lt_0_90'))} |",
            f"| exact_or_near_exact_rate_ge_0_99 | {_f(rob.get('exact_or_near_exact_rate_ge_0_99'))} |",
            f"| failed_samples | {len(rob.get('failed_samples', []))} |",
            "",
            "### Worst-sample summary",
            "",
            "| sample_index | s2ft_loss | token_accuracy | slice |",
            "|---:|---:|---:|---|",
        ]
        for r in rob.get("worst_10_samples_by_s2ft_loss", [])[:10]:
            lines.append(
                f"| {r.get('sample_index', r.get('index'))} | "
                f"{_f(r.get('s2ft_loss'))} | "
                f"{_f(r.get('token_accuracy'))} | "
                f"{r.get('dataset_type', '')}/{r.get('complexity_tier', '')} |"
            )
        lines.append("")

    lines += [
        "### ADE/FDE unavailable note",
        "",
        ADE_FDE_STATUS,
        "",
        "## Checkpoint load",
        "",
    ]
    cl = summary.get("checkpoint_load", {})
    lines += [
        f"- Consolidation: {cl.get('consolidation', 'n/a')}",
        f"- Tag: {cl.get('tag', 'n/a')}",
        f"- Params loaded: {cl.get('n_loaded', 'n/a')}",
        f"- Missing keys (benign): {cl.get('n_missing_benign', 'n/a')}",
        f"- Missing keys (CONCERNING): {cl.get('n_missing_concerning', 'n/a')}",
        f"- Unexpected keys: {cl.get('n_unexpected', 'n/a')}",
        f"- Shape mismatches: {cl.get('n_shape_mismatch', 'n/a')}",
        f"- Detected max_gen_len: {cl.get('detected_max_gen_len', 'n/a')}",
        "",
    ]
    if cl.get("missing_concerning"):
        lines.append("### Concerning missing keys (running with random init!)")
        lines.append("")
        for k in cl["missing_concerning"][:20]:
            lines.append(f"- `{k}`")
        lines.append("")

    gen = summary.get("generation", {})
    if gen.get("enabled"):
        lines += [
            "## Sample generations",
            "",
            f"(temperature={gen['temperature']}, "
            f"max_new_tokens={gen['max_new_tokens']}, "
            f"{'deterministic greedy' if gen.get('deterministic') else 'sampled'})",
            "",
        ]
        shown = 0
        for r in records:
            if r.get("status") != "ok" or not r.get("generated_text"):
                continue
            lines += [
                f"### Sample {r['index']} "
                f"({r.get('dataset_type', '?')}/{r.get('complexity_tier', '?')})",
                "",
                f"- per-sample CE: {_f(r.get('per_sample_ce'))}  "
                f"token acc: {_f(r.get('token_accuracy'))}",
                "",
                "**Generated:**",
                "",
                "```",
                (r["generated_text"][:600] or "<empty>"),
                "```",
                "",
                "**Gold (target excerpt):**",
                "",
                "```",
                (r.get("gold_text", "")[:600] or "<empty>"),
                "```",
                "",
            ]
            shown += 1
            if shown >= 5:
                break

    timing = summary.get("timing", {})
    lines += [
        "## Timing",
        "",
        f"- Consolidation: {_f(timing.get('consolidate_s'), 1)} s",
        f"- Model build + load: {_f(timing.get('model_load_s'), 1)} s",
        f"- Evaluation: {_f(timing.get('eval_s'), 1)} s "
        f"({_f(timing.get('s_per_sample'), 1)} s/sample)",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    save_predictions: bool = True,
    rank: Optional[int] = None,
    tier2_enabled: bool = True,
    group_metrics_enabled: bool = True,
) -> List[Path]:
    """Write predictions, summary, report, and optional Tier-2 diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    suffix = f"_rank{rank}" if rank is not None else ""

    tier2: Optional[Dict[str, Any]] = None
    if tier2_enabled:
        tier2 = build_tier2_metrics(
            records, summary, include_groups=group_metrics_enabled
        )
        summary["tier2_metrics"] = tier2

    if save_predictions:
        pred_path = output_dir / f"predictions{suffix}.jsonl"
        write_jsonl(pred_path, records)
        written.append(pred_path)

    summary_path = output_dir / f"inference_summary{suffix}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    written.append(summary_path)

    report_path = output_dir / f"inference_report{suffix}.md"
    report_path.write_text(render_report_md(summary, records), encoding="utf-8")
    written.append(report_path)

    if tier2 is not None:
        tier2_path = output_dir / f"tier2_metrics{suffix}.json"
        tier2_path.write_text(
            json.dumps(tier2, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        written.append(tier2_path)

        tier2_csv = output_dir / f"tier2_metrics{suffix}.csv"
        write_csv(tier2_csv, flatten_tier2_for_csv(tier2))
        written.append(tier2_csv)

        if group_metrics_enabled:
            group_csv = output_dir / f"group_metrics{suffix}.csv"
            write_csv(group_csv, tier2.get("groups", []))
            written.append(group_csv)

        worst_path = output_dir / f"worst_samples{suffix}.jsonl"
        write_jsonl(
            worst_path,
            tier2["robustness"]["worst_10_samples_by_s2ft_loss"],
        )
        written.append(worst_path)

        best_path = output_dir / f"best_samples{suffix}.jsonl"
        write_jsonl(
            best_path,
            tier2["robustness"]["best_10_samples_by_s2ft_loss"],
        )
        written.append(best_path)
    return written


# ─────────────────────────────────────────────────────────────────────────────
# ZeRO-3 consolidation
# ─────────────────────────────────────────────────────────────────────────────

def fallback_zero3_consolidate(tag_dir: Path) -> "OrderedDict[str, Any]":
    """
    Minimal ZeRO-3 -> fp32 consolidation, mirroring DeepSpeed's
    zero_to_fp32.py reconstruction protocol.  Used only when deepspeed is not
    importable (e.g. Windows).  Validates element counts and fails loudly on
    anything unexpected rather than producing silently wrong weights.

    Protocol (per DeepSpeed zero_to_fp32.py, stable across 0.9-0.16):
      - trainable params: each rank's optim shard holds fp32_flat_groups
        (one flat fp32 tensor per optimizer group; concatenated per rank).
        Each param occupies ceil(numel/world) elements per rank, in
        param_shapes order.  full = cat(rank slices)[:numel].view(shape)
      - frozen params: each rank's model shard holds frozen_param_fragments;
        full = cat(rank fragments)[:numel].view(shape)
      - buffers: taken whole from rank-0 model shard's module dict.
      - shared params: [tied_key, source_key] pairs copied at the end.
    """
    import torch

    model_files = sorted(
        tag_dir.glob("*_model_states.pt"),
        key=lambda p: int(re.search(r"zero_pp_rank_(\d+)", p.name).group(1)),
    )
    optim_files = sorted(
        tag_dir.glob("*_optim_states.pt"),
        key=lambda p: int(re.search(r"zero_pp_rank_(\d+)", p.name).group(1)),
    )
    if not model_files or not optim_files:
        raise FileNotFoundError(
            f"No ZeRO shard files found in {tag_dir} "
            "(expected *_model_states.pt and *_optim_states.pt)"
        )
    if len(model_files) != len(optim_files):
        raise RuntimeError(
            f"Shard count mismatch: {len(model_files)} model vs "
            f"{len(optim_files)} optimizer shards"
        )
    world_size = len(model_files)
    print(f"[consolidate] fallback ZeRO-3 consolidator: world_size={world_size}")

    # ── Parse model shards ────────────────────────────────────────────────
    model_states = []
    for fp in model_files:
        st = torch.load(str(fp), map_location="cpu", weights_only=False)
        model_states.append(st)

    rank0 = model_states[0]
    if "param_shapes" not in rank0:
        raise RuntimeError(
            "model shard has no 'param_shapes' — not a ZeRO checkpoint or an "
            "unsupported DeepSpeed version. Use deepspeed's zero_to_fp32 instead."
        )
    # param_shapes: list of OrderedDict(name -> shape), one per optimizer group.
    raw_shapes = rank0["param_shapes"]
    if isinstance(raw_shapes, dict):
        raw_shapes = [raw_shapes]
    param_shapes: "OrderedDict[str, Any]" = OrderedDict()
    for group in raw_shapes:
        for name, shape in group.items():
            param_shapes[name] = shape

    zero_stage = None

    # ── Parse optimizer shards: fp32 flat groups per rank ─────────────────
    flat_per_rank: List[Any] = []
    for fp in optim_files:
        st = torch.load(str(fp), map_location="cpu", weights_only=False)
        osd = st.get("optimizer_state_dict", st)
        zero_stage = osd.get("zero_stage", zero_stage)
        groups = None
        for key in ("fp32_flat_groups", "single_partition_of_fp32_groups"):
            if key in osd:
                groups = osd[key]
                break
        if groups is None:
            raise RuntimeError(
                f"{fp.name}: no fp32_flat_groups in optimizer shard — "
                "unsupported DeepSpeed version for the fallback consolidator."
            )
        if isinstance(groups, (list, tuple)):
            flat = torch.cat([g.float() for g in groups], dim=0)
        else:
            flat = groups.float()
        flat_per_rank.append(flat)

    if zero_stage is not None and int(zero_stage) != 3:
        raise RuntimeError(
            f"Checkpoint zero_stage={zero_stage}, fallback consolidator only "
            "supports ZeRO-3. Use deepspeed's zero_to_fp32 for stage <= 2."
        )

    state_dict: "OrderedDict[str, Any]" = OrderedDict()

    # ── Buffers (whole tensors on every rank; take rank 0) ────────────────
    buffer_names = set(rank0.get("buffer_names", []))
    module_sd = rank0.get("module", {})
    for name in buffer_names:
        if name in module_sd:
            state_dict[name] = module_sd[name].clone()

    # ── Frozen params from fragments ──────────────────────────────────────
    frozen_shapes = rank0.get("frozen_param_shapes") or {}
    if frozen_shapes:
        print(f"[consolidate] merging {len(frozen_shapes)} frozen params")
        for name, shape in frozen_shapes.items():
            numel = 1
            for s in shape:
                numel *= int(s)
            frags = [ms["frozen_param_fragments"][name] for ms in model_states]
            full = torch.cat([f.float().flatten() for f in frags], dim=0)
            if full.numel() < numel:
                raise RuntimeError(
                    f"frozen param {name}: fragments give {full.numel()} elems, "
                    f"need {numel}"
                )
            state_dict[name] = full.narrow(0, 0, numel).view(shape).clone()

    # ── Trainable params from fp32 flat groups ────────────────────────────
    offset = 0
    avail = flat_per_rank[0].numel()
    for name, shape in param_shapes.items():
        numel = 1
        for s in shape:
            numel *= int(s)
        part = int(math.ceil(numel / world_size))
        if offset + part > avail:
            raise RuntimeError(
                f"flat group exhausted at {name}: need offset {offset}+{part}, "
                f"have {avail} per rank — reconstruction protocol mismatch."
            )
        full = torch.cat(
            [flat_per_rank[r].narrow(0, offset, part) for r in range(world_size)],
            dim=0,
        )
        state_dict[name] = full.narrow(0, 0, numel).view(shape).clone()
        offset += part

    # Sanity: leftover per-rank elements must be < world_size alignment pad.
    leftover = avail - offset
    if leftover > 2 * world_size * max(1, len(raw_shapes)):
        print(
            f"[consolidate][WARN] {leftover} unconsumed elements per rank "
            f"(alignment padding expected < {2 * world_size * len(raw_shapes)}). "
            "Verify losses against training val metrics."
        )

    # ── Shared (tied) params ──────────────────────────────────────────────
    for pair in rank0.get("shared_params", []) or []:
        try:
            tied, src = pair[0], pair[1]
            if src in state_dict and tied not in state_dict:
                state_dict[tied] = state_dict[src]
        except Exception:
            pass

    print(f"[consolidate] reconstructed {len(state_dict)} tensors "
          f"({sum(v.numel() for v in state_dict.values())/1e6:.1f}M elements)")
    del model_states, flat_per_rank
    gc.collect()
    return state_dict


def consolidate_checkpoint(ckpt_root: Path, tag: str) -> Dict[str, Any]:
    """
    ZeRO-3 shards -> {"state_dict": ..., "method": "deepspeed"|"fallback"}.
    Prefers DeepSpeed's official utility; falls back to the built-in
    consolidator above.
    """
    try:
        from deepspeed.utils.zero_to_fp32 import (   # type: ignore
            get_fp32_state_dict_from_zero_checkpoint,
        )
        print("[consolidate] using deepspeed.utils.zero_to_fp32")
        sd = get_fp32_state_dict_from_zero_checkpoint(str(ckpt_root), tag=tag)
        return {"state_dict": sd, "method": "deepspeed"}
    except ImportError:
        print("[consolidate] deepspeed not importable — using built-in "
              "fallback consolidator")
        sd = fallback_zero3_consolidate(ckpt_root / tag)
        return {"state_dict": sd, "method": "fallback"}


# ─────────────────────────────────────────────────────────────────────────────
# S3 download
# ─────────────────────────────────────────────────────────────────────────────

def download_checkpoint(checkpoint_s3: str, local_root: Path,
                        region: str) -> Tuple[Path, str]:
    """
    Download the checkpoint tag dir into local_root/<tag>/.
    Returns (local_root, tag). Files already present with matching size skip.
    """
    import boto3

    root_uri, tag = parse_tag_from_checkpoint_uri(checkpoint_s3)
    bucket, prefix = checkpoint_s3.removeprefix("s3://").split("/", 1)
    s3 = boto3.client("s3", region_name=region)

    if tag is None:
        # Prefix is the checkpoint root: find the latest step_* tag.
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
        tags = sorted(
            p["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            for p in resp.get("CommonPrefixes", [])
            if "step_" in p["Prefix"]
        )
        if not tags:
            raise FileNotFoundError(f"No step_* tags under {checkpoint_s3}")
        tag = tags[-1]
        prefix = f"{prefix.rstrip('/')}/{tag}/"
        print(f"[download] auto-selected latest tag: {tag}")

    dest = local_root / tag
    dest.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    n_new = n_skip = 0
    total_bytes = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key, size = obj["Key"], int(obj["Size"])
            rel = key.removeprefix(prefix).lstrip("/")
            if not rel:
                continue
            lf = dest / rel
            total_bytes += size
            if lf.exists() and lf.stat().st_size == size:
                n_skip += 1
                continue
            lf.parent.mkdir(parents=True, exist_ok=True)
            print(f"[download] {rel}  ({size/1e9:.2f} GB)", flush=True)
            s3.download_file(bucket, key, str(lf))
            n_new += 1
    if n_new + n_skip == 0:
        raise FileNotFoundError(f"No objects under s3://{bucket}/{prefix}")
    print(f"[download] done: {n_new} downloaded, {n_skip} already present "
          f"({total_bytes/1e9:.2f} GB total)")
    return local_root, tag


def ensure_consolidated_checkpoint(
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[Path, str, str, float]:
    """
    Parent-safe consolidation path: creates/reuses consolidated_fp32.pt without
    building the model. Returns (path, method, tag, seconds).
    """
    t0 = time.time()
    if args.consolidated_ckpt:
        path = Path(args.consolidated_ckpt)
        return path, "preconsolidated", path.name, round(time.time() - t0, 1)

    ckpt_local = Path(args.ckpt_local_dir or (output_dir / "ckpt"))
    ckpt_root, tag = download_checkpoint(args.checkpoint_s3, ckpt_local,
                                         args.region)
    cache_file = ckpt_root / tag / "consolidated_fp32.pt"
    if cache_file.exists():
        print(f"[infer] using cached consolidation: {cache_file}")
        return cache_file, "cached", tag, round(time.time() - t0, 1)

    result = consolidate_checkpoint(ckpt_root, tag)
    sd, method = result["state_dict"], result["method"]
    print(f"[infer] caching consolidated state dict -> {cache_file}")
    import torch
    torch.save(sd, str(cache_file))
    del sd, result
    gc.collect()
    return cache_file, method, tag, round(time.time() - t0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Model build + load
# ─────────────────────────────────────────────────────────────────────────────

def build_and_load_model(sd: Dict[str, Any], device, dtype, args,
                         consolidation_method: str,
                         tag: str) -> Tuple[Any, Dict[str, Any]]:
    """Build KairosModel with checkpoint-matched config and load weights."""
    import torch
    from kairos_model import KairosModel, KairosModelConfig

    detected_gen_len = detect_max_gen_len(sd, default=512)
    cfg = KairosModelConfig()
    cfg.max_gen_len = detected_gen_len
    cfg.lidar_cfg.n_points = min(cfg.lidar_cfg.n_points, args.max_pts)
    # Inference never needs activation checkpointing.
    cfg.use_grad_checkpoint = False
    cfg.kcfg.use_grad_checkpoint = False
    cfg.vcfg.use_grad_checkpoint = False

    print(f"[model] building KairosModel (max_gen_len={detected_gen_len}) …")
    model = KairosModel(cfg)
    print(f"[model] {model.count_params()}")

    report_raw = filtered_load_state_dict(model, sd)
    n_loaded = len(sd) - len(report_raw["unexpected"]) \
        - len(report_raw["shape_mismatch"])
    missing_cls = classify_missing_keys(report_raw["missing"])

    load_report = {
        "consolidation": consolidation_method,
        "tag": tag,
        "detected_max_gen_len": detected_gen_len,
        "n_ckpt_tensors": len(sd),
        "n_loaded": n_loaded,
        "n_missing_benign": len(missing_cls["benign"]),
        "n_missing_concerning": len(missing_cls["concerning"]),
        "missing_concerning": missing_cls["concerning"][:50],
        "n_unexpected": len(report_raw["unexpected"]),
        "unexpected": report_raw["unexpected"][:50],
        "n_shape_mismatch": len(report_raw["shape_mismatch"]),
        "shape_mismatch": report_raw["shape_mismatch"][:50],
    }

    print(f"[model] loaded {n_loaded}/{len(sd)} checkpoint tensors")
    if missing_cls["benign"]:
        print(f"[model] missing-but-benign keys: {len(missing_cls['benign'])} "
              "(frozen DINOv2 from HF / tied lm_head)")
    if missing_cls["concerning"]:
        print(f"[model][WARN] {len(missing_cls['concerning'])} model keys NOT "
              "in checkpoint — these run with RANDOM INIT:")
        for k in missing_cls["concerning"][:10]:
            print(f"          - {k}")
    if report_raw["shape_mismatch"]:
        print(f"[model][WARN] {len(report_raw['shape_mismatch'])} shape "
              "mismatches (skipped):")
        for k in report_raw["shape_mismatch"][:10]:
            print(f"          - {k}")

    del sd
    gc.collect()

    model.to(device)
    if dtype == torch.bfloat16:
        model.to(torch.bfloat16)
    model.eval()
    return model, load_report


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, df, args, device, dtype) -> List[Dict[str, Any]]:
    """
    Sample-by-sample evaluation (manual batching for precise failed-sample
    accounting). Returns one record per requested sample.
    """
    import torch
    import torch.nn.functional as F
    from kairos_train import KairosDataset, collate_fn, _to_device
    from kairos_model import KairoBatch

    cache_dir = Path(args.output_dir) / "s3_cache"
    dataset = KairosDataset(
        df, cache_dir,
        max_pts=args.max_pts,
        max_prompt=args.max_prompt,
        max_target=args.max_target,
        skip_bad_rows=False,           # we do our own failure accounting
        allow_oxts_fallback=not args.strict_data,
    )

    autocast_enabled = (dtype == torch.bfloat16)
    device_type = device.type

    records: List[Dict[str, Any]] = []
    pending: List[Tuple[int, dict]] = []
    n = min(args.max_samples, len(dataset))
    base_sample_index = int(args.sample_start or 0)

    def _row_meta(i: int) -> Dict[str, Any]:
        row = df.iloc[i]
        sample_index = base_sample_index + int(i)
        return {
            "index": sample_index,
            "sample_index": sample_index,
            "dataset_type": str(row.get("dataset_type", "")),
            "source_table": str(row.get("source_table", "")),
            "camera_id": str(row.get("camera_id", "")),
            "complexity_tier": str(row.get("complexity_tier", "")),
            "curriculum_order": row.get("curriculum_order", None),
            "drive_id": str(row.get("drive_id", "")),
            "sequence_id": str(row.get("sequence_id", "")),
            "frame_index": row.get("frame_index", None),
            "image_path": str(row.get("image_path", "")),
            "image_path_t_minus_1": str(row.get("image_path_t_minus_1", "")),
            "image_path_t_minus_2": str(row.get("image_path_t_minus_2", "")),
            "lidar_path": str(row.get("lidar_path", "")),
            "oxts_path": str(row.get("oxts_path", "")),
        }

    def _flush_batch(items: List[Tuple[int, dict]]) -> None:
        idxs = [i for i, _ in items]
        batch = collate_fn([s for _, s in items])
        batch = _to_device(batch, device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            enabled=autocast_enabled):
            out = model(batch)

        total = float(out.total_loss) if out.total_loss is not None else None
        s2ft = float(out.s2ft_loss) if out.s2ft_loss is not None else None
        moe_z = float(out.moe_z_loss) if out.moe_z_loss is not None else None

        # Per-sample CE + token accuracy from the eval-mode logits.
        per_ce: List[Optional[float]] = [None] * len(items)
        per_acc: List[Optional[float]] = [None] * len(items)
        target_lengths: List[Optional[int]] = [None] * len(items)
        if out.logits is not None and batch.target_bytes is not None:
            dec_label = build_decoder_labels(
                batch.target_bytes, batch.loss_mask
            )
            logits = out.logits.float()
            loss_flat = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                dec_label.reshape(-1),
                ignore_index=-1,
                label_smoothing=0.1,
                reduction="none",
            ).view(dec_label.shape)
            for b in range(len(items)):
                valid = dec_label[b] != -1
                nv = int(valid.sum().item())
                target_lengths[b] = nv
                if nv > 0:
                    per_ce[b] = float(loss_flat[b][valid].mean().item())
                    acc, _ = token_accuracy_from_logits(
                        logits[b], dec_label[b]
                    )
                    per_acc[b] = acc

        gen_texts: List[Optional[str]] = [None] * len(items)
        generated_lengths: List[Optional[int]] = [None] * len(items)
        if args.generate:
            # Cap generation length without touching training code: max_len
            # is a plain attribute on the decoder.
            orig_max_len = model.s2ft_decoder.max_len
            model.s2ft_decoder.max_len = min(orig_max_len, args.max_new_tokens)
            try:
                inf_batch = KairoBatch(
                    img_t=batch.img_t, img_t1=batch.img_t1, img_t2=batch.img_t2,
                    lidar_t=batch.lidar_t, lidar_t1=batch.lidar_t1,
                    imu_data=batch.imu_data,
                    imu_timestamps=batch.imu_timestamps,
                    calib=batch.calib, text_bytes=batch.text_bytes,
                )
                with torch.autocast(device_type=device_type,
                                    dtype=torch.bfloat16,
                                    enabled=autocast_enabled):
                    gen_out = model.generate(
                        inf_batch,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                if gen_out.generated is not None:
                    for b in range(len(items)):
                        gen_tokens = gen_out.generated[b]
                        gen_texts[b] = decode_bytes(gen_tokens)
                        generated_lengths[b] = sum(
                            1 for tok in gen_tokens.tolist() if 0 < tok < 256
                        )
            finally:
                model.s2ft_decoder.max_len = orig_max_len

        for b, (i, _) in enumerate(items):
            gold_text = decode_bytes(batch.target_bytes[b]) \
                if batch.target_bytes is not None else ""
            rec = {
                **_row_meta(i),
                "status": "ok",
                "total_loss": total,
                "s2ft_loss": s2ft,
                "moe_z_loss": moe_z,
                "per_sample_ce": per_ce[b],
                "token_accuracy": per_acc[b],
                "target_length": target_lengths[b],
                "generated_length": generated_lengths[b],
                "failed": False,
                "failure_reason": None,
                "prediction_text": gen_texts[b] if args.save_predictions else None,
                "target_text": gold_text[:2000],
                "generated_text": gen_texts[b],
                "gold_text": gold_text[:2000],
                "batch_note": (
                    "total_loss/s2ft_loss/moe_z_loss are batch-level "
                    f"(batch_size={len(items)}); per_sample_ce is per-sample"
                ),
            }
            records.append(rec)

    t0 = time.time()
    for i in range(n):
        try:
            sample = dataset[i]
        except Exception as exc:
            if args.strict_data:
                raise
            print(f"[eval][WARN] sample {i} failed to load: {exc}")
            reason = f"{type(exc).__name__}: {exc}"
            records.append({
                **_row_meta(i),
                "status": "failed",
                "failed": True,
                "failure_reason": reason,
                "error": reason,
                "total_loss": None,
                "s2ft_loss": None,
                "per_sample_ce": None,
                "token_accuracy": None,
                "target_length": None,
                "generated_length": None,
                "prediction_text": None,
                "target_text": None,
            })
            continue
        pending.append((i, sample))
        if len(pending) >= args.batch_size:
            _flush_batch(pending)
            pending = []
            done = len(records)
            el = time.time() - t0
            print(f"[eval] {done}/{n} samples  ({el:.0f}s elapsed, "
                  f"{el/max(done,1):.1f}s/sample)", flush=True)
    if pending:
        _flush_batch(pending)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def set_determinism(seed: int) -> None:
    import torch
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_split_dataframe(args: argparse.Namespace):
    from kairos_train import _join_s3, _load_parquet_partition

    split_path = _join_s3(args.gold_s3, f"dataset_split={args.split}")
    df = _load_parquet_partition(split_path)
    return df, split_path


def prepare_eval_dataframe(args: argparse.Namespace):
    df, split_path = load_split_dataframe(args)
    full_n = len(df)
    if full_n == 0:
        return df, split_path, full_n

    capped_n = min(args.max_samples, full_n)
    start = int(args.sample_start or 0)
    end = int(args.sample_end if args.sample_end is not None else capped_n)
    start = max(0, min(start, capped_n))
    end = max(start, min(end, capped_n))
    df = df.head(capped_n).iloc[start:end].reset_index(drop=True)
    return df, split_path, full_n


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bool_cli(value: bool) -> str:
    return "True" if value else "False"


def build_worker_command(
    args: argparse.Namespace,
    consolidated_ckpt: Path,
    rank: int,
    world_size: int,
    sample_start: int,
    sample_end: int,
) -> List[str]:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--checkpoint_s3", args.checkpoint_s3,
        "--gold_s3", args.gold_s3,
        "--split", args.split,
        "--max_samples", str(args.max_samples),
        "--output_dir", args.output_dir,
        "--device", "cuda",
        "--batch_size", str(args.batch_size),
        "--save_predictions", _bool_cli(args.save_predictions),
        "--consolidated_ckpt", str(consolidated_ckpt),
        "--region", args.region,
        "--generate", _bool_cli(args.generate),
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--max_prompt", str(args.max_prompt),
        "--max_target", str(args.max_target),
        "--max_pts", str(args.max_pts),
        "--num_workers", str(args.num_workers),
        "--dtype", args.dtype,
        "--seed", str(args.seed),
        "--multi_gpu_eval", "False",
        "--rank", str(rank),
        "--world_size", str(world_size),
        "--sample_start", str(sample_start),
        "--sample_end", str(sample_end),
        "--tier2_metrics", _bool_cli(args.tier2_metrics),
        "--group_metrics", _bool_cli(args.group_metrics),
    ]
    if args.strict_data:
        cmd.append("--strict_data")
    if args.allow_mock_vision:
        cmd.append("--allow_mock_vision")
    return cmd


def run_multi_gpu_parent(args: argparse.Namespace, torch_mod) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch_mod.cuda.is_available():
        print("[infer] ERROR: --multi_gpu_eval true requires CUDA")
        return 1
    detected = int(torch_mod.cuda.device_count())
    workers = int(args.num_gpu_workers or detected)
    workers = max(1, min(workers, detected))

    print(f"[multi-gpu] detected cuda device count: {detected}")
    print(f"[multi-gpu] launching workers: {workers}")

    consolidated_ckpt, method, tag, consolidate_s = ensure_consolidated_checkpoint(
        args, output_dir
    )
    gc.collect()

    df, split_path = load_split_dataframe(args)
    full_n = len(df)
    if full_n == 0:
        print(f"[infer] ERROR: split partition is empty: {split_path}")
        return 1
    n_eval = min(args.max_samples, full_n)
    ranges = deterministic_shards(n_eval, workers)
    print(f"[multi-gpu] sample ranges: {ranges}")

    procs = []
    for rank, (start, end) in enumerate(ranges):
        cmd = build_worker_command(
            args, consolidated_ckpt, rank, workers, start, end
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        print(
            f"[multi-gpu] rank={rank}/{workers} device=cuda:0 "
            f"global_device=cuda:{rank} range=[{start},{end})"
        )
        procs.append((rank, subprocess.Popen(cmd, env=env)))

    failed_ranks: List[int] = []
    for rank, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failed_ranks.append(rank)
            print(f"[multi-gpu][ERROR] rank {rank} exited with {rc}")
    if failed_ranks:
        return 1

    records: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for rank in range(workers):
        pred_path = output_dir / f"predictions_rank{rank}.jsonl"
        summary_path = output_dir / f"inference_summary_rank{rank}.json"
        records.extend(load_jsonl(pred_path))
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    records.sort(key=_sample_index)
    if args.save_predictions:
        write_jsonl(output_dir / "predictions.jsonl", records)

    timing = {
        "device": f"cuda x {workers}",
        "dtype": "torch.bfloat16" if args.dtype in ("auto", "bfloat16") else "torch.float32",
        "consolidate_s": consolidate_s,
        "model_load_s": None,
        "eval_s": sum(
            float(s.get("timing", {}).get("eval_s", 0.0) or 0.0)
            for s in summaries
        ),
        "s_per_sample": None,
        "worker_count": workers,
        "worker_ranges": ranges,
    }
    n_ok = sum(1 for r in records if _is_ok_record(r))
    if n_ok:
        max_worker_eval = max(
            [float(s.get("timing", {}).get("eval_s", 0.0) or 0.0)
             for s in summaries] or [0.0]
        )
        timing["wall_eval_s_estimate"] = max_worker_eval
        timing["s_per_sample"] = round(max_worker_eval / max(n_ok, 1), 3)

    weighted = aggregate_worker_summaries(summaries)
    print(f"[multi-gpu] weighted metrics: {weighted}")
    load_report = {
        "consolidation": method,
        "tag": tag,
        "consolidated_ckpt": str(consolidated_ckpt),
        "worker_checkpoint_load": [
            s.get("checkpoint_load", {}) for s in summaries
        ],
    }
    resolved_job_name = resolve_job_name(args.job_name) if args.output_s3 else None
    summary = build_summary(
        args, records, load_report, timing,
        extra={
            "output_s3": args.output_s3,
            "job_name": resolved_job_name,
            "full_validation_samples": full_n,
            "multi_gpu_eval": {
                "enabled": True,
                "detected_cuda_device_count": detected,
                "num_gpu_workers": workers,
                "sample_ranges": ranges,
            },
        },
    )
    written = write_outputs(
        output_dir, records, summary,
        save_predictions=False,
        rank=None,
        tier2_enabled=args.tier2_metrics,
        group_metrics_enabled=args.group_metrics,
    )
    if args.save_predictions:
        written.insert(0, output_dir / "predictions.jsonl")

    note_path = output_dir / "consolidated_checkpoint_note.txt"
    note_path.write_text(
        render_consolidated_checkpoint_note(method, tag, consolidated_ckpt),
        encoding="utf-8",
    )
    written.append(note_path)

    m = summary["metrics"]
    print("\n[infer] -- RESULTS ----------------------------------------")
    print(f"[infer] valid samples      : {summary['n_valid_samples']}")
    print(f"[infer] failed samples     : {summary['n_failed_samples']}")
    print(f"[infer] s2ft_loss  (mean)  : {m['s2ft_loss_mean']}")
    print(f"[infer] token accuracy     : {m['token_accuracy_mean']}")
    if args.rank is not None:
        print(
            f"[worker] valid/failed: {summary['n_valid_samples']}/"
            f"{summary['n_failed_samples']}"
        )
        print(
            f"[worker] loss/accuracy: s2ft={m['s2ft_loss_mean']} "
            f"token_accuracy={m['token_accuracy_mean']}"
        )
    if args.rank is not None:
        print(
            f"[worker] valid/failed: {summary['n_valid_samples']}/"
            f"{summary['n_failed_samples']}"
        )
        print(
            f"[worker] loss/accuracy: s2ft={m['s2ft_loss_mean']} "
            f"token_accuracy={m['token_accuracy_mean']}"
        )
    for p in written:
        print(f"[infer] wrote {p}")

    if args.rank is not None:
        print(
            f"[worker] valid/failed: {summary['n_valid_samples']}/"
            f"{summary['n_failed_samples']}"
        )
        print(
            f"[worker] loss/accuracy: s2ft={m['s2ft_loss_mean']} "
            f"token_accuracy={m['token_accuracy_mean']}"
        )

    if args.output_s3:
        dest_prefix = f"{args.output_s3.rstrip('/')}/{resolved_job_name}/"
        print(f"\n[infer] uploading {output_dir} -> {dest_prefix}")
        uploaded = upload_directory_to_s3(output_dir, dest_prefix, args.region)
        for u in uploaded:
            print(f"[infer] uploaded {u}")
        print(f"[infer] uploaded {len(uploaded)} file(s) "
              "(ckpt/ and s3_cache/ excluded)")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    import torch

    # Region env BEFORE importing kairos_train (it reads AWS_REGION at import).
    os.environ.setdefault("AWS_REGION", args.region)
    os.environ.setdefault("AWS_DEFAULT_REGION", args.region)

    set_determinism(args.seed)

    # Without transformers the vision encoder silently builds a MOCK DINOv2
    # backbone — the 307M pretrained weights would be random garbage.
    try:
        import transformers  # noqa: F401
    except ImportError:
        if not args.allow_mock_vision:
            print(
                "[infer] ERROR: transformers is not installed — the real "
                "DINOv2 backbone cannot be built and results would be "
                "meaningless.  pip install transformers, or pass "
                "--allow_mock_vision for a plumbing-only dry run."
            )
            return 1
        print("[infer][WARN] transformers missing — using MOCK vision "
              "backbone; results are NOT meaningful.")

    if args.multi_gpu_eval and args.rank is None:
        return run_multi_gpu_parent(args, torch)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[infer] ERROR: --device cuda requested but CUDA is unavailable")
        return 1

    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_job_name = resolve_job_name(args.job_name) if args.output_s3 else None

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[infer] device={device}  dtype={dtype}  split={args.split}  "
          f"max_samples={args.max_samples}  batch_size={args.batch_size}  "
          f"seed={args.seed}")
    print(f"[infer] detected cuda device count: {cuda_count}")
    if args.rank is not None:
        print(
            f"[worker] rank/world_size: {args.rank}/{args.world_size}  "
            f"assigned_device=cuda:{args.rank}  logical_device={device}"
        )
        print(f"[worker] sample range: [{args.sample_start}, {args.sample_end})")
        print(
            f"[worker] output paths: predictions_rank{args.rank}.jsonl, "
            f"inference_summary_rank{args.rank}.json, "
            f"inference_report_rank{args.rank}.md"
        )
    print(f"[infer] checkpoint: {args.consolidated_ckpt or args.checkpoint_s3}")
    print(f"[infer] gold table: {args.gold_s3}")

    timing: Dict[str, Any] = {"device": str(device), "dtype": str(dtype)}

    # ── 1. Obtain consolidated fp32 state dict ────────────────────────────
    t0 = time.time()
    cache_file_path: Optional[Path] = None
    if args.consolidated_ckpt:
        print(f"[infer] loading consolidated state dict: {args.consolidated_ckpt}")
        sd = torch.load(args.consolidated_ckpt, map_location="cpu",
                        weights_only=False)
        if isinstance(sd, dict) and "module" in sd:
            sd = sd["module"]
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        method, tag = "preconsolidated", Path(args.consolidated_ckpt).name
        cache_file_path = Path(args.consolidated_ckpt)
    else:
        ckpt_local = Path(args.ckpt_local_dir or (output_dir / "ckpt"))
        ckpt_root, tag = download_checkpoint(args.checkpoint_s3, ckpt_local,
                                             args.region)
        cache_file = ckpt_root / tag / "consolidated_fp32.pt"
        cache_file_path = cache_file
        if cache_file.exists():
            print(f"[infer] using cached consolidation: {cache_file}")
            sd = torch.load(str(cache_file), map_location="cpu",
                            weights_only=False)
            method = "cached"
        else:
            result = consolidate_checkpoint(ckpt_root, tag)
            sd, method = result["state_dict"], result["method"]
            print(f"[infer] caching consolidated state dict -> {cache_file}")
            torch.save(sd, str(cache_file))
    sd = strip_module_prefix(sd)
    timing["consolidate_s"] = round(time.time() - t0, 1)

    # ── 2. Build model + load weights ─────────────────────────────────────
    t0 = time.time()
    model, load_report = build_and_load_model(sd, device, dtype, args,
                                              method, tag)
    timing["model_load_s"] = round(time.time() - t0, 1)

    # ── 3. Load gold split dataframe ──────────────────────────────────────
    df, split_path, full_validation_samples = prepare_eval_dataframe(args)
    if len(df) == 0:
        if args.rank is not None:
            print(f"[worker] empty sample range for rank {args.rank}")
        else:
            print(f"[infer] ERROR: split partition is empty: {split_path}")
            return 1
    print(f"[infer] evaluating {len(df)} samples from {args.split} split")

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    t0 = time.time()
    with torch.no_grad():
        records = evaluate(model, df, args, device, dtype)
    eval_s = time.time() - t0
    n_ok = sum(1 for r in records if r["status"] == "ok")
    timing["eval_s"] = round(eval_s, 1)
    timing["s_per_sample"] = round(eval_s / max(n_ok, 1), 1)

    # ── 5. Outputs ────────────────────────────────────────────────────────
    summary = build_summary(
        args, records, load_report, timing,
        extra={
            "output_s3": args.output_s3,
            "job_name": resolved_job_name,
            "full_validation_samples": full_validation_samples,
            "multi_gpu_eval": {
                "enabled": bool(args.rank is not None),
                "rank": args.rank,
                "world_size": args.world_size,
                "sample_start": args.sample_start,
                "sample_end": args.sample_end,
            },
        },
    )
    written = write_outputs(
        output_dir, records, summary,
        save_predictions=args.save_predictions,
        rank=args.rank,
        tier2_enabled=args.tier2_metrics,
        group_metrics_enabled=args.group_metrics,
    )

    note_path = output_dir / "consolidated_checkpoint_note.txt"
    note_path.write_text(
        render_consolidated_checkpoint_note(method, tag, cache_file_path),
        encoding="utf-8",
    )
    written.append(note_path)

    m = summary["metrics"]
    print("\n[infer] ── RESULTS ──────────────────────────────────────────")
    print(f"[infer] valid samples      : {summary['n_valid_samples']}")
    print(f"[infer] failed samples     : {summary['n_failed_samples']}")
    print(f"[infer] total_loss (mean)  : {m['total_loss_mean']}")
    print(f"[infer] s2ft_loss  (mean)  : {m['s2ft_loss_mean']}   "
          f"(training val_s2ft = 0.9158)")
    print(f"[infer] token accuracy     : {m['token_accuracy_mean']}")
    print(f"[infer] ADE/FDE            : unavailable "
          "(no trajectory decoding schema — see report)")
    for p in written:
        print(f"[infer] wrote {p}")

    # ── 6. Optional upload of results to S3 ────────────────────────────────
    if args.output_s3:
        dest_prefix = f"{args.output_s3.rstrip('/')}/{resolved_job_name}/"
        print(f"\n[infer] uploading {output_dir} -> {dest_prefix}")
        uploaded = upload_directory_to_s3(output_dir, dest_prefix, args.region)
        for u in uploaded:
            print(f"[infer] uploaded {u}")
        print(f"[infer] uploaded {len(uploaded)} file(s) "
              "(ckpt/ and s3_cache/ excluded)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
