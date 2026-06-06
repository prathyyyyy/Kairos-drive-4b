"""
kairos_preflight.py — Local pre-launch verification for the 5-step real-data run.

Run this LOCALLY (no GPU, no SageMaker) before launching any paid job.

What it checks:
  1. GOLD_S3 and DATA_REGION env vars are set and non-stale
  2. Gold parquet train/val partitions are readable from S3
  3. All required columns are present (oxts_path is optional — derived if absent)
  4. dataset_split and complexity_tier values look correct
  5. HeadObject on N sampled rows for:
       image_path, image_path_t_minus_1, image_path_t_minus_2
       lidar_path, lidar_path_t_minus_1
       calib_cam_to_cam_path, calib_velo_to_cam_path
       oxts_path (stored) OR derived /oxts/data/<frame>.txt (if column absent)
  6. Stale old-account ID scan on path values

Usage (PowerShell):
    $env:DATA_REGION = "eu-north-1"
    $env:GOLD_S3     = "s3://YOUR-BUCKET/delta/gold/kitti_s2ft_triplets/"
    python kairos_preflight.py [--n_rows 10]
"""

import argparse
import os
import re
import sys

import boto3
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pandas as pd


# Required columns that MUST be present in the parquet files.
# oxts_path is intentionally absent — it is optional and derived from image_path
# using the KITTI path convention when the column is missing.
REQUIRED_COLS = [
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

STALE_PATTERNS = ["195231312992", "253440504432", "use1-s3-", "-use1-"]

_IMAGE02_RE = re.compile(r"(/image_02/data/)(\d{10})\.png$")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _derive_oxts(image_path: str):
    """Derive OXTS S3 URI from image_path when oxts_path column is absent.
    .../image_02/data/0000000042.png  →  .../oxts/data/0000000042.txt
    Returns None if the pattern doesn't match.
    """
    m = _IMAGE02_RE.search(image_path)
    if m is None:
        return None
    return image_path[: m.start()] + "/oxts/data/" + m.group(2) + ".txt"


def fail(msg: str) -> None:
    print(f"\n[FAIL] {msg}", flush=True)
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}", flush=True)


def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


# ─── env check ────────────────────────────────────────────────────────────────

def check_env():
    gold_s3 = os.environ.get("GOLD_S3", "").strip()
    data_region = os.environ.get("DATA_REGION", "").strip()

    if not gold_s3:
        fail(
            "GOLD_S3 is not set. Export it first:\n"
            "  $env:GOLD_S3 = 's3://YOUR-BUCKET/delta/gold/kitti_s2ft_triplets/'"
        )
    if not data_region:
        fail(
            "DATA_REGION is not set. Export it first:\n"
            "  $env:DATA_REGION = 'eu-north-1'   # or eu-west-1"
        )
    for stale in STALE_PATTERNS:
        if stale in gold_s3:
            fail(
                f"GOLD_S3 contains stale pattern '{stale}': {gold_s3}\n"
                "Update it to your current bucket name."
            )
    ok(f"GOLD_S3       = {gold_s3}")
    ok(f"DATA_REGION   = {data_region}")
    return gold_s3, data_region


# ─── parquet loader ───────────────────────────────────────────────────────────

def _list_parquet(bucket: str, prefix: str, fs: pafs.S3FileSystem) -> list:
    base = f"{bucket}/{prefix.strip('/')}"
    try:
        selector = pafs.FileSelector(base, recursive=True)
        infos = fs.get_file_info(selector)
    except Exception as exc:
        fail(
            f"Cannot list S3 path s3://{base}: {exc}\n"
            "Check DATA_REGION, bucket name, and IAM permissions for the role."
        )
    return [fi.path for fi in infos
            if fi.type == pafs.FileType.File and fi.path.endswith(".parquet")]


def load_partition(path: str, fs: pafs.S3FileSystem) -> pd.DataFrame:
    bucket, prefix = path.removeprefix("s3://").split("/", 1)
    files = _list_parquet(bucket, prefix, fs)
    if not files:
        fail(f"No parquet files found under {path}")
    ok(f"Found {len(files)} parquet file(s) under {path}")

    frames = []
    for fp in files:
        with fs.open_input_file(fp) as f:
            df = pq.read_table(f).to_pandas()
        # inject hive partition columns (dataset_split=train etc.) if not physical
        for seg in fp.split("/"):
            if "=" in seg:
                k, v = seg.split("=", 1)
                if k not in df.columns:
                    df[k] = v
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ─── column checks ────────────────────────────────────────────────────────────

def check_columns(df: pd.DataFrame, label: str) -> bool:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        fail(
            f"Partition '{label}' is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    ok(f"All {len(REQUIRED_COLS)} required columns present in '{label}'")

    has_oxts = "oxts_path" in df.columns
    if has_oxts:
        ok(f"oxts_path column is STORED in '{label}'")
    else:
        info(
            f"oxts_path column ABSENT in '{label}' — "
            "will be derived from image_path at load time"
        )
    return has_oxts


def check_stale_paths(df: pd.DataFrame, label: str) -> None:
    path_cols = [c for c in df.columns if "path" in c]
    bad = 0
    for col in path_cols:
        for val in df[col].dropna().astype(str).head(5):
            for stale in STALE_PATTERNS:
                if stale in val:
                    warn(f"Stale pattern '{stale}' in '{label}'.{col}: {val}")
                    bad += 1
    if bad == 0:
        ok(f"No stale account IDs found in '{label}' path columns")


# ─── HeadObject check ─────────────────────────────────────────────────────────

def head_check(df: pd.DataFrame, n_rows: int, region: str, label: str,
               has_stored_oxts: bool) -> None:
    # Stored path columns that are always checked
    stored_cols = [c for c in [
        "image_path",
        "image_path_t_minus_1",
        "image_path_t_minus_2",
        "lidar_path",
        "lidar_path_t_minus_1",
        "calib_cam_to_cam_path",
        "calib_velo_to_cam_path",
    ] if c in df.columns]

    sample = df.sample(n=min(n_rows, len(df)), random_state=42)
    s3 = boto3.client("s3", region_name=region)
    bad = []
    checked = 0

    def _head(uri: str, tag: str) -> None:
        nonlocal checked
        if not isinstance(uri, str) or not uri.startswith("s3://"):
            bad.append(f"{tag} = <not an s3:// URI: {uri!r}>")
            return
        bucket, key = uri.removeprefix("s3://").split("/", 1)
        try:
            s3.head_object(Bucket=bucket, Key=key)
            checked += 1
        except Exception as exc:
            code = ""
            resp = getattr(exc, "response", None)
            if isinstance(resp, dict):
                code = resp.get("Error", {}).get("Code", "")
            bad.append(f"{tag} [{code or type(exc).__name__}]: {uri}")

    for _, row in sample.iterrows():
        # Stored path columns
        for col in stored_cols:
            val = row.get(col)
            if pd.isna(val) or not val:
                bad.append(f"{col} = <null/empty>")
            else:
                _head(str(val), col)

        # OXTS: stored column or derived from image_path
        if has_stored_oxts:
            oxts_val = row.get("oxts_path")
            if pd.isna(oxts_val) or not oxts_val:
                # Try to derive even if column exists but this row is null
                derived = _derive_oxts(str(row.get("image_path", "")))
                if derived:
                    _head(derived, "oxts_path[derived-from-null]")
                else:
                    bad.append("oxts_path = <null and not derivable>")
            else:
                _head(str(oxts_val), "oxts_path[stored]")
        else:
            img = str(row.get("image_path", ""))
            derived = _derive_oxts(img)
            if derived:
                _head(derived, "oxts_path[derived]")
            else:
                bad.append(
                    f"oxts_path[derived] — could not derive from image_path={img!r}"
                )

    oxts_mode = "stored" if has_stored_oxts else "derived"
    total_expected = len(sample) * (len(stored_cols) + 1)  # +1 for oxts

    if bad:
        warn(
            f"[{label}] {len(bad)}/{total_expected} paths missing or inaccessible "
            f"(oxts mode={oxts_mode}):"
        )
        for b in bad[:15]:
            print(f"        {b}")
        if len(bad) > 15:
            print(f"        ... and {len(bad) - 15} more")
    else:
        ok(
            f"[{label}] All {checked}/{total_expected} HeadObject checks passed "
            f"(oxts_path mode={oxts_mode})"
        )


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_rows", type=int, default=10,
                   help="Number of rows to HeadObject-check per partition")
    args = p.parse_args()

    print("\n" + "=" * 62)
    print("  KAIROS PREFLIGHT VERIFICATION")
    print("=" * 62)

    gold_s3, data_region = check_env()

    fs = pafs.S3FileSystem(region=data_region)
    ok(f"PyArrow S3FileSystem: region={data_region}")

    train_path = gold_s3.rstrip("/") + "/dataset_split=train/"
    val_path   = gold_s3.rstrip("/") + "/dataset_split=val/"

    print(f"\n--- Train partition ---")
    df_train = load_partition(train_path, fs)
    ok(f"Loaded {len(df_train):,} training rows")
    has_oxts_train = check_columns(df_train, "train")
    check_stale_paths(df_train, "train")

    splits = df_train.get("dataset_split", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
    tiers  = df_train.get("complexity_tier", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
    ok(f"dataset_split counts: {splits}")
    ok(f"complexity_tier counts: {tiers}")
    if "curriculum_order" in df_train.columns:
        co = df_train["curriculum_order"].value_counts(dropna=False).sort_index().to_dict()
        ok(f"curriculum_order counts: {co}")

    print(f"\n--- Val partition ---")
    df_val = load_partition(val_path, fs)
    ok(f"Loaded {len(df_val):,} validation rows")
    has_oxts_val = check_columns(df_val, "val")
    check_stale_paths(df_val, "val")

    print(f"\n--- HeadObject check: {args.n_rows} train rows (DATA_REGION={data_region}) ---")
    head_check(df_train, args.n_rows, data_region, "train", has_oxts_train)

    print(f"\n--- Sample paths (first train row) ---")
    row0 = df_train.iloc[0]
    img0 = str(row0.get("image_path", ""))
    for col in ["image_path", "lidar_path", "calib_cam_to_cam_path", "calib_velo_to_cam_path"]:
        print(f"  {col}: {row0.get(col, '<missing>')}")
    if has_oxts_train:
        print(f"  oxts_path[stored]: {row0.get('oxts_path', '<missing>')}")
    else:
        derived = _derive_oxts(img0)
        print(f"  oxts_path[derived]: {derived or '<could not derive>'}")

    print("\n" + "=" * 62)
    print("  PREFLIGHT COMPLETE — review any [WARN]/[FAIL] above")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
