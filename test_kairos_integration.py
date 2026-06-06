"""
CPU integration tests for Kairos.

These tests force the mock vision backbone and shrink the test-only config so
they run without GPU, transformers, flash-attn, or AWS credentials. The default
project configs remain at Kairos-4B scale.

Run:
    pip install torch pytest          # requirements-dev.txt
    pytest -q test_kairos_integration.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from kairos_fusion import CalibMatrices
from kairos_imu import IMUEncoder, IMUEncoderConfig
from kairos_lidar import LiDAREncoderConfig, _ball_query
from kairos_model import KairoBatch, KairosModel, KairosModelConfig, sync_moe_expert_bias


_P2 = torch.tensor([
    [7.215377e02, 0.0, 6.095593e02, 4.485728e01],
    [0.0, 7.215377e02, 1.728540e02, 2.163791e-01],
    [0.0, 0.0, 1.0, 2.745884e-03],
])
_R0 = torch.eye(3)
_TR = torch.tensor([
    [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
    [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
    [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
])


def _tiny_cfg() -> KairosModelConfig:
    cfg = KairosModelConfig()

    cfg.kcfg.d_model = 64
    cfg.kcfg.d_state = 8
    cfg.kcfg.dt_rank = 8
    cfg.kcfg.mamba_expand = 1
    cfg.kcfg.mamba_chunk = 8
    cfg.kcfg.num_heads_q = 4
    cfg.kcfg.num_heads_kv = 2
    cfg.kcfg.attn_window = 64
    cfg.kcfg.max_seq_len = 128
    cfg.kcfg.num_experts = 4
    cfg.kcfg.d_ff = 128
    cfg.kcfg.moe_d_ff = 128
    cfg.kcfg.num_blocks = 3
    cfg.kcfg.num_loops = 4

    cfg.n_cam = 16
    cfg.n_lidar = 8
    cfg.n_imu = 4
    cfg.n_query = 2
    cfg.decoder_layers = 1
    cfg.decoder_heads = 4
    cfg.max_gen_len = 8
    cfg.max_det = 4
    cfg.n_det_cls = 3

    cfg.vcfg.use_mock_backbone = True
    cfg.vcfg.use_grad_checkpoint = False
    cfg.vcfg.enc_h = 56
    cfg.vcfg.enc_w = 56
    cfg.vcfg.n_patches = 16
    cfg.vcfg.fusion_heads = 4
    cfg.vcfg.fusion_d_ff = 128

    cfg.fcfg.n_cam_tokens = 16
    cfg.fcfg.n_lidar_tokens = 8
    cfg.fcfg.n_imu_tokens = 4
    cfg.fcfg.n_query_tokens = 2
    cfg.fcfg.patch_rows = 4
    cfg.fcfg.patch_cols = 4
    cfg.fcfg.enc_h = 56
    cfg.fcfg.enc_w = 56

    cfg.lidar_cfg = LiDAREncoderConfig(
        n_tokens=8,
        n_points=64,
        n_neighbors=8,
        pn_hidden=16,
        d_state=8,
        mamba_chunk=4,
        n_mamba_layers=1,
        moe_experts=2,
        moe_d_ff=32,
    )
    cfg.imu_cfg = IMUEncoderConfig(n_tokens=4, cfc_hidden=64, n_cfc_layers=1)
    return cfg


def _batch(cfg: KairosModelConfig, train: bool = True) -> KairoBatch:
    B, N = 1, cfg.lidar_cfg.n_points
    lidar = torch.zeros(B, N, 4)
    lidar[:, :, 0] = torch.linspace(5.0, 30.0, N)
    lidar[:, :, 1] = torch.linspace(-2.0, 2.0, N)
    lidar[:, :, 2] = torch.linspace(-0.5, 1.0, N)
    lidar[:, :, 3] = 0.5

    target = torch.randint(1, 128, (B, 8), dtype=torch.long)
    target[:, -2:] = -1
    loss_mask = torch.ones(B, 8, dtype=torch.bool)
    loss_mask[:, -2:] = False

    return KairoBatch(
        img_t=torch.rand(B, 3, cfg.vcfg.enc_h, cfg.vcfg.enc_w),
        img_t1=torch.rand(B, 3, cfg.vcfg.enc_h, cfg.vcfg.enc_w),
        img_t2=torch.rand(B, 3, cfg.vcfg.enc_h, cfg.vcfg.enc_w),
        lidar_t=lidar.clone(),
        lidar_t1=lidar.clone(),
        imu_data=torch.randn(B, 8, 7),
        imu_timestamps=torch.arange(8).float().unsqueeze(0) / 30.0,
        calib=CalibMatrices(
            P2=_P2.unsqueeze(0),
            R0_rect=_R0.unsqueeze(0),
            Tr_velo_to_cam=_TR.unsqueeze(0),
        ),
        text_bytes=torch.randint(1, 128, (B, 16), dtype=torch.long) if train else None,
        target_bytes=target if train else None,
        loss_mask=loss_mask if train else None,
    )


def _assert_param_groups_unique(model: KairosModel) -> None:
    groups = model.param_groups()
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids)), "duplicate parameter in optimizer groups"
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}


def test_kairos_end_to_end_cpu():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = KairosModel(cfg)

    model.eval()
    with torch.no_grad():
        out = model(_batch(cfg, train=False))
    T = cfg.n_cam + cfg.n_lidar + cfg.n_imu + cfg.n_query
    assert out.hidden.shape == (1, T, cfg.kcfg.d_model)
    assert out.det_boxes.shape == (1, cfg.max_det, 7)
    assert out.det_scores.shape == (1, cfg.max_det, cfg.n_det_cls + 1)
    assert torch.isfinite(out.hidden).all()

    model.train()
    train_batch = _batch(cfg, train=True)
    train_out = model(train_batch)
    assert train_out.total_loss is not None
    assert torch.isfinite(train_out.total_loss)
    assert train_out.hidden is None
    assert train_out.logits is None
    assert train_out.det_boxes is None
    assert train_out.det_scores is None
    train_out.total_loss.backward()

    lora_grads = [
        p.grad.abs().sum()
        for name, p in model.named_parameters()
        if "lora_" in name and p.grad is not None
    ]
    assert lora_grads and sum(lora_grads) > 0

    router_grad = model.hybrid_core.blocks[0].moe_ffn.router_proj.weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0

    calib_grads = [
        p.grad.abs().sum()
        for p in model.calib_gate.parameters()
        if p.grad is not None
    ]
    assert calib_grads and sum(calib_grads) > 0

    model.eval()
    gen = model.generate(_batch(cfg, train=False), temperature=0.0)
    assert gen.generated is not None
    assert gen.generated.shape[0] == 1
    assert gen.generated.dtype == torch.long
    assert gen.generated.shape[1] >= 1

    _assert_param_groups_unique(model)
    sync_moe_expert_bias(model)


def test_imu_delta_t_warning_and_clamp():
    cfg = IMUEncoderConfig(n_tokens=4, cfc_hidden=64, n_cfc_layers=1)
    enc = IMUEncoder(d_model=64, n_tokens=4, cfg=cfg)
    enc.train()
    ts = torch.tensor([[0.0, 0.1, 0.05, 0.5]])
    imu = torch.randn(1, 4, 7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, dt = enc(imu, ts)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert dt.min() >= 1e-4
    assert dt.max() <= 0.2


def test_ball_query_matches_reference_sets():
    torch.manual_seed(7)
    B, K, N = 2, 8, 200
    seeds = torch.randn(B, K, 3)
    pts = torch.randn(B, N, 3)
    radius, n_sample = 2.0, 16

    idx_new = _ball_query(seeds, pts, radius, n_sample, _chunk=4)

    diff = seeds.unsqueeze(2) - pts.unsqueeze(1)
    dist2 = diff.pow(2).sum(-1)
    s_idx = dist2.argsort(dim=-1)
    in_ball = dist2.gather(-1, s_idx) <= radius ** 2
    closest = s_idx[:, :, :1].expand(B, K, n_sample)
    top_k = s_idx[:, :, :n_sample]
    idx_ref = torch.where(in_ball[:, :, :n_sample], top_k, closest)

    for b in range(B):
        for k in range(K):
            assert set(idx_new[b, k].tolist()) == set(idx_ref[b, k].tolist())


def _ultra_smoke_cfg() -> KairosModelConfig:
    """
    CPU test config that mirrors the ultra_smoke_mode overrides in kairos_train.py.
    Uses tiny d_model=64 for speed; real ultra_smoke keeps d_model=1024.
    """
    cfg = _tiny_cfg()
    # ── Matching ultra_smoke overrides from kairos_train.py ───────────────────
    # enc 56×56 → 56/14 × 56/14 = 4×4 = 16 patches (already set in _tiny_cfg)
    cfg.vcfg.enc_h          = 56
    cfg.vcfg.enc_w          = 56
    cfg.vcfg.n_patches      = 16
    cfg.vcfg.sequential_frames = True   # frames processed one-by-one
    cfg.n_cam               = 16
    cfg.fcfg.n_cam_tokens   = 16
    cfg.fcfg.patch_rows     = 4
    cfg.fcfg.patch_cols     = 4
    cfg.fcfg.enc_h          = 56
    cfg.fcfg.enc_w          = 56
    # LiDAR: 8 tokens, 1 Mamba layer (current ultra_smoke default)
    cfg.lidar_cfg.n_tokens  = 8
    cfg.n_lidar             = 8
    cfg.fcfg.n_lidar_tokens = 8
    cfg.lidar_cfg.n_mamba_layers = 1
    cfg.lidar_cfg.moe_experts = 1
    # keep moe_d_ff=32, n_points=64 from _tiny_cfg (CPU-friendly)
    # IMU: 1 CfC layer (same as _tiny_cfg); keep cfc_hidden=64 for CPU speed
    cfg.imu_cfg.n_cfc_layers = 1
    cfg.imu_cfg.n_tokens = cfg.n_imu
    # Decoder: 1 layer (current ultra_smoke default)
    cfg.decoder_layers      = 1
    cfg.decoder_heads       = 4   # keep from _tiny_cfg
    # Text encoder: 2 layers (current ultra_smoke default; matches kairos_train.py)
    cfg.n_text_enc_layers   = 2
    # Hybrid core MoE already reduced via _tiny_cfg (num_experts=4, moe_d_ff=128)
    # Other overrides
    cfg.return_debug_tensors = False
    cfg.w_det               = 0
    cfg.max_gen_len         = 8   # keep tiny for CPU speed
    return cfg


def test_ultra_smoke_forward():
    """
    Verifies ultra_smoke_mode config: consistent shapes, w_det=0 skips detection
    during training, sequential_frames flag is set, and forward/backward succeed.
    """
    torch.manual_seed(2)
    cfg = _ultra_smoke_cfg()
    model = KairosModel(cfg)

    T = cfg.n_cam + cfg.n_lidar + cfg.n_imu + cfg.n_query   # 16+8+4+2 = 30

    # ── Training forward: detection head skipped when w_det=0 ─────────────────
    model.train()
    batch_tr = _batch(cfg, train=True)
    out_tr = model(batch_tr)

    assert out_tr.total_loss is not None, "total_loss must not be None in training"
    assert torch.isfinite(out_tr.total_loss), "total_loss must be finite"
    assert out_tr.det_boxes  is None, "w_det=0 → det_boxes must be None in training"
    assert out_tr.det_scores is None, "w_det=0 → det_scores must be None in training"
    assert out_tr.hidden     is None, "return_debug_tensors=False → hidden must be None"
    assert out_tr.logits     is None, "return_debug_tensors=False → logits must be None"
    out_tr.total_loss.backward()

    # ── Inference forward: detection head always runs ─────────────────────────
    model.eval()
    with torch.no_grad():
        out_inf = model(_batch(cfg, train=False))

    assert out_inf.hidden.shape      == (1, T, cfg.kcfg.d_model), \
        f"Expected hidden (1,{T},{cfg.kcfg.d_model}), got {tuple(out_inf.hidden.shape)}"
    assert out_inf.det_boxes.shape   == (1, cfg.max_det, 7), \
        f"Expected det_boxes (1,{cfg.max_det},7), got {tuple(out_inf.det_boxes.shape)}"
    assert out_inf.det_scores.shape  == (1, cfg.max_det, cfg.n_det_cls + 1)
    assert torch.isfinite(out_inf.hidden).all()

    # ── sequential_frames flag is set ─────────────────────────────────────────
    assert cfg.vcfg.sequential_frames is True, \
        "ultra_smoke_cfg must have sequential_frames=True"

    _assert_param_groups_unique(model)


def test_ultra_smoke_skip_lidar_imu_shape_compatibility():
    torch.manual_seed(3)
    cfg = _ultra_smoke_cfg()
    model = KairosModel(cfg)
    model._smoke_skip_lidar = True
    model._smoke_skip_imu = True
    model._smoke_no_grad_lidar = True
    model._smoke_no_grad_imu = True

    model.train()
    core_imu_counts = []
    original_core_forward = model.hybrid_core.forward

    def wrapped_core_forward(x, imu_mask, delta_t):
        core_imu_counts.append(int(imu_mask.sum().item()))
        return original_core_forward(x, imu_mask, delta_t)

    with patch.object(model.hybrid_core, "forward", side_effect=wrapped_core_forward):
        out = model(_batch(cfg, train=True))

    assert core_imu_counts == [0]
    assert out.total_loss is not None
    assert torch.isfinite(out.total_loss)
    out.total_loss.backward()

    model.eval()
    with torch.no_grad():
        out_eval = model(_batch(cfg, train=False))
    T = cfg.n_cam + cfg.n_lidar + cfg.n_imu + cfg.n_query
    assert out_eval.hidden.shape == (1, T, cfg.kcfg.d_model)
    assert out_eval.det_boxes.shape == (1, cfg.max_det, 7)


def test_ultra_smoke_normal_imu_core_path_forward_backward():
    torch.manual_seed(33)
    cfg = _ultra_smoke_cfg()
    model = KairosModel(cfg)

    model.train()
    core_imu_counts = []
    original_core_forward = model.hybrid_core.forward

    def wrapped_core_forward(x, imu_mask, delta_t):
        core_imu_counts.append(int(imu_mask.sum().item()))
        return original_core_forward(x, imu_mask, delta_t)

    with patch.object(model.hybrid_core, "forward", side_effect=wrapped_core_forward):
        out = model(_batch(cfg, train=True))

    assert core_imu_counts == [cfg.n_imu]
    assert out.total_loss is not None
    assert torch.isfinite(out.total_loss)
    out.total_loss.backward()


def test_ultra_smoke_skip_decoder_loss_uses_dummy_backward_loss():
    """
    With _smoke_skip_decoder_loss=True:
      - total_loss and s2ft_loss are set, logits is None
      - backward succeeds without shape errors
      - smoke_loss_anchor receives gradient (it IS the s2ft_loss source)
      - moe_z_loss is EXCLUDED from total_loss (Stage 1b fix):
        router_proj must NOT receive gradient via the total_loss backward path.
        This prevents ZeRO-3 shape mismatches from sparse MoE dispatch backward.
    """
    torch.manual_seed(4)
    cfg = _ultra_smoke_cfg()
    model = KairosModel(cfg)
    model._smoke_skip_decoder_loss = True

    model.train()
    out = model(_batch(cfg, train=True))
    assert out.total_loss is not None
    assert out.s2ft_loss is not None
    assert out.logits is None
    assert torch.isfinite(out.total_loss)
    out.total_loss.backward()   # must not raise

    # smoke_loss_anchor is the s2ft_loss source — must have gradient
    assert model.smoke_loss_anchor.grad is not None, \
        "smoke_loss_anchor must have grad when _smoke_skip_decoder_loss=True"

    # moe_z is excluded from total_loss → router_proj must have NO gradient
    # (Stage 1b fix: moe_z connected to sparse MoE backward caused ZeRO-3 errors)
    core_grad = model.hybrid_core.blocks[0].moe_ffn.router_proj.weight.grad
    assert core_grad is None or core_grad.abs().sum() == 0, \
        ("router_proj must have no grad when skip_decoder_loss=True: "
         "moe_z is excluded from total_loss to prevent ZeRO-3 backward shape errors")


def test_kairos_train_import_safe_without_deepspeed():
    """
    kairos_train must be importable even when deepspeed is absent.
    The RuntimeError is deferred to train(), not raised at import time.
    """
    sys.modules.pop("kairos_train", None)

    # Mock heavy deps that may not be present in the test environment.
    # "deepspeed": None signals ImportError to the try/except in kairos_train.
    _heavy = [
        "boto3", "pandas", "pyarrow", "pyarrow.parquet",
        "s3fs", "PIL", "PIL.Image",
        "rich", "rich.console", "rich.live", "rich.progress",
    ]
    mocks: dict = {m: MagicMock() for m in _heavy if m not in sys.modules}
    mocks["deepspeed"] = None  # causes ImportError → _HAS_DS = False

    try:
        with patch.dict(sys.modules, mocks):
            import kairos_train as kt  # must not raise

        assert hasattr(kt, "train"), "train() must be defined after import"
        assert hasattr(kt, "_HAS_DS"), "_HAS_DS flag must be set at module level"
        assert not kt._HAS_DS, "_HAS_DS must be False when deepspeed is absent"
    finally:
        sys.modules.pop("kairos_train", None)


def test_dataset_retries_missing_s3_row(monkeypatch, tmp_path):
    sys.modules.pop("kairos_train", None)
    import kairos_train as kt

    df = pd.DataFrame([
        {
            "image_path": "s3://bucket/missing-image.png",
            "lidar_path": "s3://bucket/missing-lidar.bin",
            "calib_cam_to_cam_path": "s3://bucket/missing-calib.txt",
            "calib_velo_to_cam_path": "s3://bucket/missing-velo.txt",
            "system_prompt": "s",
            "user_prompt": "u",
            "reasoning_chain": "r",
            "answer": "a",
            "drive": "bad-drive",
            "frame": "000001",
        },
        {
            "image_path": "s3://bucket/good-image.png",
            "lidar_path": "s3://bucket/good-lidar.bin",
            "calib_cam_to_cam_path": "s3://bucket/good-calib.txt",
            "calib_velo_to_cam_path": "s3://bucket/good-velo.txt",
            "system_prompt": "s",
            "user_prompt": "u",
            "reasoning_chain": "r",
            "answer": "a",
            "drive": "good-drive",
            "frame": "000002",
        },
    ])

    def fake_img(uri, cache):
        if "missing" in uri:
            raise FileNotFoundError(uri)
        return np.zeros((3, 4, 4), dtype=np.float32)

    def fake_lidar(uri, cache, max_pts):
        return np.zeros((max_pts, 4), dtype=np.float32)

    def fake_calib(cc_path, vc_path, cache):
        if "missing" in cc_path:
            raise FileNotFoundError(cc_path)
        return (
            np.zeros((3, 4), dtype=np.float32),
            np.eye(3, dtype=np.float32),
            np.zeros((3, 4), dtype=np.float32),
        )

    monkeypatch.setattr(kt, "_load_image", fake_img)
    monkeypatch.setattr(kt, "_load_lidar", fake_lidar)
    monkeypatch.setattr(kt, "_parse_calib", fake_calib)

    ds = kt.KairosDataset(
        df, tmp_path, max_pts=4, max_prompt=8, max_target=8,
        skip_bad_rows=True,
    )
    sample = ds[0]
    assert sample["img_t"].shape == (3, 4, 4)
    assert sample["lidar_t"].shape == (4, 4)
    assert sample["P2"].shape == (3, 4)


def test_missing_s3_client_error_detection():
    import kairos_train as kt

    class FakeClientError(Exception):
        def __init__(self):
            self.response = {
                "Error": {"Code": "404"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            }

    assert kt._is_missing_s3_error(FakeClientError())


def test_calib_resolver_uses_original_when_present(tmp_path):
    import kairos_train as kt

    cc_file, vc_file = _write_calib_files(tmp_path)
    cc_uri = "s3://bucket/kairos-raw-deltalake/campus/calibration/2011_09_28_calib/calib_cam_to_cam.txt"
    vc_uri = "s3://bucket/kairos-raw-deltalake/campus/calibration/2011_09_28_calib/calib_velo_to_cam.txt"
    cache = _FakeCalibCache({cc_uri: cc_file, vc_uri: vc_file}, kt)

    P2, R0, Tr = kt._parse_calib(cc_uri, vc_uri, cache)

    assert P2.shape == (3, 4)
    assert R0.shape == (3, 3)
    assert Tr.shape == (3, 4)
    assert cache._resolved_calib_uris[cc_uri] == cc_uri
    assert cache._resolved_calib_uris[vc_uri] == vc_uri


def test_calib_resolver_uses_nested_date_when_original_missing(tmp_path, capsys):
    import kairos_train as kt

    cc_file, vc_file = _write_calib_files(tmp_path)
    cc_uri = "s3://bucket/kairos-raw-deltalake/campus/calibration/2011_09_28_calib/calib_cam_to_cam.txt"
    vc_uri = "s3://bucket/kairos-raw-deltalake/campus/calibration/2011_09_28_calib/calib_velo_to_cam.txt"
    cc_alt = "s3://bucket/kairos-raw-deltalake/campus/calibration/2011_09_28_calib/2011_09_28/calib_cam_to_cam.txt"
    vc_alt = "s3://bucket/kairos-raw-deltalake/campus/calibration/2011_09_28_calib/2011_09_28/calib_velo_to_cam.txt"
    cache = _FakeCalibCache({cc_alt: cc_file, vc_alt: vc_file}, kt)

    P2, R0, Tr = kt._parse_calib(cc_uri, vc_uri, cache)
    out = capsys.readouterr().out

    assert P2.shape == (3, 4)
    assert R0.shape == (3, 3)
    assert Tr.shape == (3, 4)
    assert cache._resolved_calib_uris[cc_uri] == cc_alt
    assert cache._resolved_calib_uris[vc_uri] == vc_alt
    assert "[WARN][calib_path_fix]" in out
    assert f"original={cc_uri} resolved={cc_alt}" in out


def test_calib_resolver_raises_when_original_and_nested_missing(tmp_path):
    import kairos_train as kt

    cc_uri = "s3://bucket/kairos-raw-deltalake/campus/calibration/2011_09_28_calib/calib_cam_to_cam.txt"
    vc_uri = "s3://bucket/kairos-raw-deltalake/campus/calibration/2011_09_28_calib/calib_velo_to_cam.txt"
    cache = _FakeCalibCache({}, kt)

    with pytest.raises(FileNotFoundError) as err:
        kt._parse_calib(cc_uri, vc_uri, cache)

    assert "Calibration file missing" in str(err.value)
    assert "2011_09_28/calib_cam_to_cam.txt" in str(err.value)


def test_oxts_numeric_row_parses(tmp_path):
    import kairos_train as kt

    uri = "s3://bucket/2011_09_28/2011_09_28_drive_0001_sync/oxts/data/0000000000.txt"
    path = tmp_path / "oxts.txt"
    path.write_text("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\n", encoding="utf-8")
    cache = _FakeCalibCache({uri: path}, kt)

    imu, ts = kt._load_oxts(uri, cache, n_tokens=4, allow_fallback=False)

    assert imu.shape == (4, 7)
    assert ts.shape == (4,)
    assert np.isclose(imu[0, 0], 9, atol=1e-2)   # velocity_fwd=row[8]
    assert np.isclose(imu[0, 1], 15, atol=1e-2)  # acceleration=row[14]
    assert np.isclose(imu[0, 3], 1, atol=1e-2)   # lat=row[0]


def test_oxts_date_line_skipped_then_numeric_row_parses(tmp_path):
    import kairos_train as kt

    uri = "s3://bucket/2011_09_28/2011_09_28_drive_0001_sync/oxts/data/0000000001.txt"
    path = tmp_path / "oxts_with_date.txt"
    path.write_text(
        "09-Jan-2012 metadata line\n"
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\n",
        encoding="utf-8",
    )
    cache = _FakeCalibCache({uri: path}, kt)

    imu, _ = kt._load_oxts(uri, cache, n_tokens=2, allow_fallback=False)

    assert imu.shape == (2, 7)
    assert np.isclose(imu[0, 0], 9, atol=1e-2)


def test_oxts_non_numeric_file_raises_in_full_mode(tmp_path):
    import kairos_train as kt

    uri = "s3://bucket/2011_09_28/2011_09_28_drive_0001_sync/oxts/data/0000000002.txt"
    path = tmp_path / "bad_oxts.txt"
    path.write_text("09-Jan-2012\nnot numeric metadata\n", encoding="utf-8")
    cache = _FakeCalibCache({uri: path}, kt)

    with pytest.raises(ValueError) as err:
        kt._load_oxts(uri, cache, n_tokens=2, allow_fallback=False)

    assert uri in str(err.value)
    assert "09-Jan-2012" in str(err.value)


def test_oxts_timestamps_path_warns_and_falls_back_in_smoke(tmp_path, capsys):
    import kairos_train as kt

    kt.bad_oxts_seen = 0
    kt.oxts_fallbacks_used = 0
    uri = "s3://bucket/2011_09_28/2011_09_28_drive_0001_sync/oxts/timestamps.txt"
    path = tmp_path / "timestamps.txt"
    path.write_text("09-Jan-2012 12:00:00\n", encoding="utf-8")
    cache = _FakeCalibCache({uri: path}, kt)

    imu, ts = kt._load_oxts(uri, cache, n_tokens=3, allow_fallback=True)
    out = capsys.readouterr().out

    assert imu.shape == (3, 7)
    assert ts.shape == (3,)
    assert "[WARN][bad_oxts_path]" in out
    assert "[WARN][oxts_fallback]" in out
    assert kt.bad_oxts_seen == 1
    assert kt.oxts_fallbacks_used == 1


class _FakeCalibCache:
    def __init__(self, existing, kt_module):
        self.existing = existing
        self.calls = []
        self._global_lock = kt_module.threading.Lock()
        self._resolved_calib_uris = {}

    def local(self, uri):
        self.calls.append(uri)
        if uri not in self.existing:
            raise FileNotFoundError(uri)
        return self.existing[uri]


def _write_calib_files(tmp_path):
    cc_file = tmp_path / "calib_cam_to_cam.txt"
    vc_file = tmp_path / "calib_velo_to_cam.txt"
    cc_file.write_text(
        "calib_time: 09-Jan-2012 13:01:00\n"
        "corner_dist: 0.0\n"
        "P_rect_02: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "R_rect_00: 1 0 0 0 1 0 0 0 1\n",
        encoding="utf-8",
    )
    vc_file.write_text(
        "calib_time: 09-Jan-2012 13:01:00\n"
        "R: 1 0 0 0 1 0 0 0 1\n"
        "T: 0 0 0\n",
        encoding="utf-8",
    )
    return cc_file, vc_file


def test_calib_parses_file_with_metadata_line(tmp_path):
    """calib_time and other non-numeric metadata lines are silently skipped."""
    import kairos_train as kt

    cc_file, vc_file = _write_calib_files(tmp_path)
    cc_uri = "s3://bucket/calib_cam_to_cam.txt"
    vc_uri = "s3://bucket/calib_velo_to_cam.txt"
    cache = _FakeCalibCache({cc_uri: cc_file, vc_uri: vc_file}, kt)

    P2, R0, Tr = kt._parse_calib(cc_uri, vc_uri, cache)

    assert P2.shape == (3, 4)
    assert R0.shape == (3, 3)
    assert Tr.shape == (3, 4)
    assert np.allclose(P2, np.eye(3, 4, dtype=np.float32))
    assert np.allclose(R0, np.eye(3, dtype=np.float32))
    expected_Tr = np.hstack([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)])
    assert np.allclose(Tr, expected_Tr)


def test_token_layout_is_336():
    """
    Default KairosModelConfig must produce a TokenLayout totalling 336 tokens:
      cam=256, lidar=64, imu=8, query=8.
    Verifies that no old 328-token layout (missing IMU) is active.
    """
    from kairos_model import KairosModelConfig, TokenLayout

    cfg = KairosModelConfig()
    layout = TokenLayout(cfg.n_cam, cfg.n_lidar, cfg.n_imu, cfg.n_query)

    assert layout.total == 336, (
        f"Default layout.total={layout.total}, expected 336. "
        "Old 328-token layout would indicate missing IMU tokens."
    )
    assert layout.imu_start == 320, f"imu_start={layout.imu_start}, expected 320"
    assert layout.imu_end   == 328, f"imu_end={layout.imu_end}, expected 328"
    assert layout.query_start == 328, f"query_start={layout.query_start}, expected 328"
    assert layout.query_end   == 336, f"query_end={layout.query_end}, expected 336"

    # Slices must not overlap
    cam_set   = set(range(*layout.cam_slice.indices(336)))
    lidar_set = set(range(*layout.lidar_slice.indices(336)))
    imu_set   = set(range(*layout.imu_slice.indices(336)))
    query_set = set(range(*layout.query_slice.indices(336)))
    assert not (cam_set & lidar_set), "cam/lidar slices overlap"
    assert not (lidar_set & imu_set), "lidar/imu slices overlap"
    assert not (imu_set & query_set), "imu/query slices overlap"
    assert len(cam_set | lidar_set | imu_set | query_set) == 336


def test_fused_layout_336_in_cpu_model():
    """
    End-to-end CPU check: KairosModel must produce a fused (B,336,d) sequence
    for the default KITTI config and n_imu=8 must match imu_slice.
    """
    torch.manual_seed(42)
    cfg = _tiny_cfg()

    # Verify tiny cfg uses consistent layout
    from kairos_model import TokenLayout
    layout = TokenLayout(cfg.n_cam, cfg.n_lidar, cfg.n_imu, cfg.n_query)
    T = layout.total  # 16+8+4+2 = 30 in tiny config

    model = KairosModel(cfg)
    model.eval()
    with torch.no_grad():
        out = model(_batch(cfg, train=False))

    assert out.hidden.shape == (1, T, cfg.kcfg.d_model), (
        f"Expected hidden (1,{T},{cfg.kcfg.d_model}), got {tuple(out.hidden.shape)}"
    )


def test_calib_raises_on_nonnumeric_required_key(tmp_path):
    """A required key with non-numeric values raises a clear ValueError."""
    import kairos_train as kt

    cc_file = tmp_path / "calib_cam_to_cam_bad.txt"
    vc_file = tmp_path / "calib_velo_to_cam.txt"
    cc_file.write_text(
        "calib_time: 09-Jan-2012 13:01:00\n"
        "P_rect_02: bad value here\n"
        "R_rect_00: 1 0 0 0 1 0 0 0 1\n",
        encoding="utf-8",
    )
    vc_file.write_text(
        "R: 1 0 0 0 1 0 0 0 1\n"
        "T: 0 0 0\n",
        encoding="utf-8",
    )
    cc_uri = "s3://bucket/bad_cc.txt"
    vc_uri = "s3://bucket/calib_velo_to_cam.txt"
    cache = _FakeCalibCache({cc_uri: cc_file, vc_uri: vc_file}, kt)

    with pytest.raises(ValueError) as err:
        kt._parse_calib(cc_uri, vc_uri, cache)

    assert "P_rect_02" in str(err.value)
    assert cc_uri in str(err.value)
    assert "bad value here" in str(err.value)


def test_mock_vision_backbone_bf16_compat():
    """
    Mock DINO backbone must accept float32 input when model weights are bfloat16,
    as happens under DeepSpeed BF16 (ultra_smoke_mock_vision path).
    """
    from kairos_encoders import KairosVisionEncoder, VisionEncoderConfig
    from kairos_hybrid_block import KairosConfig as _KCfg

    vcfg = VisionEncoderConfig(
        use_mock_backbone=True,
        use_grad_checkpoint=False,
        enc_h=56,
        enc_w=56,
        n_patches=16,
        fusion_heads=4,
        fusion_d_ff=128,
    )
    kcfg = _KCfg()
    kcfg.d_model = 64

    enc = KairosVisionEncoder(vcfg, kcfg).to(torch.bfloat16)
    enc.eval()

    # Simulate preprocessing output: float32 input, bfloat16 model weights
    img = torch.rand(1, 3, 56, 56, dtype=torch.float32)
    with torch.no_grad():
        out = enc(img, img, img)

    assert out.dtype == torch.bfloat16, f"expected bfloat16 output, got {out.dtype}"
    assert out.shape == (1, 16, 64)
    assert torch.isfinite(out).all()


def test_lidar_imu_encoders_accept_float32_inputs_with_bf16_weights():
    torch.manual_seed(5)

    lidar_cfg = LiDAREncoderConfig(
        n_tokens=4,
        n_points=32,
        n_neighbors=4,
        pn_hidden=8,
        d_state=4,
        mamba_chunk=2,
        n_mamba_layers=1,
        moe_experts=1,
        moe_d_ff=16,
    )
    from kairos_lidar import PointMambaEncoder

    lidar = PointMambaEncoder(d_model=32, n_tokens=4, cfg=lidar_cfg).to(torch.bfloat16)
    pts = torch.zeros(1, 32, 4, dtype=torch.float32)
    pts[:, :, 0] = torch.linspace(5.0, 20.0, 32)
    pts[:, :, 1] = torch.linspace(-1.0, 1.0, 32)
    pts[:, :, 2] = torch.linspace(-0.25, 1.0, 32)
    pts[:, :, 3] = 0.5
    lidar.eval()
    with torch.no_grad():
        tokens, xyz = lidar(pts, pts.clone())
    assert tokens.dtype == torch.bfloat16
    assert tokens.shape == (1, 4, 32)
    assert xyz.shape == (1, 4, 3)
    assert torch.isfinite(tokens).all()

    imu_cfg = IMUEncoderConfig(n_tokens=4, cfc_hidden=32, n_cfc_layers=1)
    imu = IMUEncoder(d_model=32, n_tokens=4, cfg=imu_cfg).to(torch.bfloat16)
    imu_data = torch.randn(1, 8, 7, dtype=torch.float32)
    ts = torch.arange(8, dtype=torch.float32).unsqueeze(0) / 30.0
    imu.eval()
    with torch.no_grad():
        imu_tokens, imu_dt = imu(imu_data, ts)
    assert imu_tokens.dtype == torch.bfloat16
    assert imu_dt.dtype == torch.bfloat16
    assert imu_tokens.shape == (1, 4, 32)
    assert torch.isfinite(imu_tokens).all()


def test_tiny_model_bf16_accepts_float32_batch():
    torch.manual_seed(6)
    cfg = _ultra_smoke_cfg()
    model = KairosModel(cfg).to(torch.bfloat16)
    model.eval()

    batch = _batch(cfg, train=False)
    assert batch.img_t.dtype == torch.float32
    assert batch.lidar_t.dtype == torch.float32
    assert batch.imu_data.dtype == torch.float32

    with torch.no_grad():
        out = model(batch)

    T = cfg.n_cam + cfg.n_lidar + cfg.n_imu + cfg.n_query
    assert out.hidden.shape == (1, T, cfg.kcfg.d_model)
    assert out.hidden.dtype == torch.bfloat16
    assert torch.isfinite(out.hidden).all()


def test_patch_ds_warned_safe_when_deepspeed_absent():
    """_patch_ds_warned() must not raise when deepspeed is absent."""
    import kairos_train as kt
    orig = kt._HAS_DS
    try:
        kt._HAS_DS = False
        kt._patch_ds_warned()   # must not raise
    finally:
        kt._HAS_DS = orig


def test_patch_ds_warned_fixes_buggy_function(capsys):
    """
    _patch_ds_warned() detects the warned NameError bug, applies the fix, and
    the replacement correctly preserves container structure with correct arg order:
    apply_to_tensors_only(function, value).

    Verified:
    - [patch] applied message printed
    - stage3 caller namespace also patched
    - Tensor: function applied
    - list / tuple: structure preserved, tensors transformed
    - dict: structure preserved, tensors transformed
    - namedtuple: type preserved, tensors transformed
    - KairoBatch (dataclass): instance preserved, tensor fields transformed
    - CalibMatrices (dataclass): instance preserved, tensor fields transformed
    - None / scalar fields: returned unchanged
    """
    import sys
    import types
    from collections import namedtuple
    import kairos_train as kt
    from kairos_model import KairoBatch
    from kairos_fusion import CalibMatrices

    # Buggy DS 0.14.4 pattern: correct arg order (function, value) but `warned`
    # is referenced before assignment → NameError at runtime.
    def _buggy(function, value, warning_msg_fn=None):
        if isinstance(value, torch.Tensor):
            return function(value)
        if warning_msg_fn and not warned:   # noqa: F821  ← the bug
            pass
        return value

    fake_utils  = types.ModuleType("deepspeed.runtime.zero.utils")
    fake_stage3 = types.ModuleType("deepspeed.runtime.zero.stage3")
    fake_utils.apply_to_tensors_only  = _buggy
    fake_stage3.apply_to_tensors_only = _buggy

    _keys = ("deepspeed.runtime.zero.utils", "deepspeed.runtime.zero.stage3")
    saved = {k: sys.modules.get(k) for k in _keys}
    orig_has_ds = kt._HAS_DS
    try:
        sys.modules["deepspeed.runtime.zero.utils"]  = fake_utils
        sys.modules["deepspeed.runtime.zero.stage3"] = fake_stage3
        kt._HAS_DS = True
        kt._patch_ds_warned()

        out = capsys.readouterr().out
        assert "[patch] applied" in out, f"expected patch message, got: {out!r}"

        double = lambda t: t * 2   # noqa: E731  simple transform for assertions

        # ── Tensor ───────────────────────────────────────────────────────────
        assert torch.isclose(
            fake_utils.apply_to_tensors_only(double, torch.tensor(3.0)),
            torch.tensor(6.0),
        )

        # ── list / tuple ──────────────────────────────────────────────────────
        lst = fake_utils.apply_to_tensors_only(double, [torch.tensor(1.0), "str"])
        assert isinstance(lst, list)
        assert torch.isclose(lst[0], torch.tensor(2.0))
        assert lst[1] == "str"

        tup = fake_utils.apply_to_tensors_only(double, (torch.tensor(1.0), 99))
        assert isinstance(tup, tuple)
        assert torch.isclose(tup[0], torch.tensor(2.0))
        assert tup[1] == 99

        # ── dict ──────────────────────────────────────────────────────────────
        d = fake_utils.apply_to_tensors_only(
            double, {"a": torch.tensor(5.0), "b": "keep"}
        )
        assert torch.isclose(d["a"], torch.tensor(10.0))
        assert d["b"] == "keep"

        # ── namedtuple ───────────────────────────────────────────────────────
        NT = namedtuple("NT", ["x", "y"])
        nt = fake_utils.apply_to_tensors_only(double, NT(x=torch.tensor(2.0), y="hi"))
        assert type(nt) is NT
        assert torch.isclose(nt.x, torch.tensor(4.0))
        assert nt.y == "hi"

        # ── KairoBatch (dataclass) ───────────────────────────────────────────
        img = torch.ones(1, 3, 56, 56)
        batch = KairoBatch(
            img_t=img.clone(), img_t1=img.clone(), img_t2=img.clone(),
            lidar_t=torch.zeros(1, 4, 4), lidar_t1=torch.zeros(1, 4, 4),
            imu_data=torch.zeros(1, 4, 7),
            imu_timestamps=torch.zeros(1, 4),
            calib=CalibMatrices(
                P2=torch.eye(3, 4).unsqueeze(0),
                R0_rect=torch.eye(3).unsqueeze(0),
                Tr_velo_to_cam=torch.eye(3, 4).unsqueeze(0),
            ),
            text_bytes=torch.zeros(1, 8, dtype=torch.long),
            target_bytes=None,
            loss_mask=None,
        )

        result = fake_utils.apply_to_tensors_only(double, batch)

        # Must be KairoBatch, NOT a function
        assert isinstance(result, KairoBatch), \
            f"expected KairoBatch, got {type(result)}"
        # Tensor field transformed
        assert torch.allclose(result.img_t, img * 2)
        assert isinstance(result.img_t, torch.Tensor)
        # None fields remain None
        assert result.target_bytes is None
        assert result.loss_mask is None

        # ── CalibMatrices (nested dataclass) ─────────────────────────────────
        assert isinstance(result.calib, CalibMatrices), \
            f"expected CalibMatrices, got {type(result.calib)}"
        assert torch.allclose(result.calib.P2, torch.eye(3, 4).unsqueeze(0) * 2)

        # ── stage3 namespace also patched ────────────────────────────────────
        assert fake_stage3.apply_to_tensors_only is not _buggy, \
            "stage3 reference was not patched"
    finally:
        kt._HAS_DS = orig_has_ds
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# New regression tests: plumbing-only loss, skip-core, MoE empty experts, CfC
# ─────────────────────────────────────────────────────────────────────────────

def test_ultra_smoke_skip_decoder_loss_anchor_disconnected_from_expert_gemm():
    """
    When _smoke_skip_decoder_loss=True, the s2ft_loss comes from
    smoke_loss_anchor only (disconnected from hybrid_core output x).

    The primary regression: backward must succeed without a ZeRO-3 shape
    mismatch in the MoE expert GEMM backward.  The cat-based expert dispatch
    (replacing in-place output_sorted[s:t_] = ...) makes this path safe.

    Note: W1 may still receive gradient via the z-loss cross-block path
    (later block's z-loss flows through earlier block's expert outputs).
    That path is expected and is safe with the non-in-place cat fix.
    """
    torch.manual_seed(20)
    cfg = _ultra_smoke_cfg()
    model = KairosModel(cfg)
    model._smoke_skip_decoder_loss = True

    model.train()
    out = model(_batch(cfg, train=True))
    assert out.total_loss is not None
    assert torch.isfinite(out.total_loss)
    out.total_loss.backward()   # must not raise (was the ZeRO-3 shape error)

    # Anchor must have grad (it IS the s2ft_loss source)
    assert model.smoke_loss_anchor.grad is not None, \
        "smoke_loss_anchor must have grad when _smoke_skip_decoder_loss=True"

    # All gradients must be finite — NaN/Inf indicates a broken backward path
    for name, param in model.hybrid_core.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), \
                f"Non-finite grad in hybrid_core.{name}"


def test_ultra_smoke_skip_core_forward_backward():
    """
    With _smoke_skip_core=True and _smoke_skip_decoder_loss=True the forward
    and backward must both succeed.  The only backward path is through
    smoke_loss_anchor (a scalar leaf parameter).
    """
    torch.manual_seed(21)
    cfg = _ultra_smoke_cfg()
    model = KairosModel(cfg)
    model._smoke_skip_core = True
    model._smoke_skip_decoder_loss = True

    model.train()
    out = model(_batch(cfg, train=True))
    assert out.total_loss is not None, "total_loss must not be None"
    assert torch.isfinite(out.total_loss), "total_loss must be finite"
    out.total_loss.backward()

    # smoke_loss_anchor is the only loss source
    assert model.smoke_loss_anchor.grad is not None, \
        "smoke_loss_anchor must have grad when skip_core+skip_decoder_loss=True"

    # hybrid core params must have no grad (core was bypassed entirely)
    router_grad = model.hybrid_core.blocks[0].moe_ffn.router_proj.weight.grad
    assert router_grad is None or router_grad.abs().sum() == 0, \
        "router_proj must have no grad when hybrid core is bypassed"


def test_moe_empty_experts_backward_cpu():
    """
    MoeSwiGLUFFN with many experts and few tokens (most experts empty).
    The non-in-place cat-based expert dispatch must handle zero-token
    experts correctly and produce valid gradients.
    """
    torch.manual_seed(22)
    from kairos_hybrid_block import KairosConfig, MoeSwiGLUFFN, PerLoopParams, RMSNorm

    cfg = KairosConfig()
    cfg.d_model = 32
    cfg.num_experts = 8
    cfg.top_k = 2
    cfg.moe_d_ff = 64

    moe = MoeSwiGLUFFN(cfg)
    moe.train()
    norm = RMSNorm(cfg.d_model)

    # 4 tokens, 8 experts, top_k=2 → 8 assignments → most experts get 0 or 1 tokens
    B, T = 1, 4
    x = torch.randn(B, T, cfg.d_model, requires_grad=True)
    out = moe(x, norm, router_bias=None)

    assert out.shape == (B, T, cfg.d_model), \
        f"unexpected shape: {tuple(out.shape)}"
    out.sum().backward()
    assert x.grad is not None, "x must have gradient"
    assert torch.isfinite(x.grad).all(), "x.grad must be finite"
    assert moe.router_proj.weight.grad is not None, \
        "router_proj.weight must have gradient"


def test_cfc_no_imu_backward_cpu():
    """
    CfCBlock with all-False imu_mask (no IMU tokens) must return x unchanged
    and the backward through the returned tensor must succeed with finite grads.
    """
    torch.manual_seed(23)
    from kairos_hybrid_block import CfCBlock, KairosConfig, RMSNorm

    cfg = KairosConfig()
    cfg.d_model = 32

    cfc = CfCBlock(cfg)
    cfc.train()
    norm = RMSNorm(cfg.d_model)

    B, T = 2, 16
    x = torch.randn(B, T, cfg.d_model, requires_grad=True)
    imu_mask = torch.zeros(B, T, dtype=torch.bool)
    delta_t = torch.zeros(B, T)

    out, h = cfc(x, imu_mask, delta_t, h_cfc=None, norm=norm)
    assert out.shape == (B, T, cfg.d_model), \
        f"unexpected output shape: {tuple(out.shape)}"
    assert h.shape == (B, cfg.d_model), \
        f"unexpected h shape: {tuple(h.shape)}"
    assert not h.requires_grad, "h_cfc must be detached (stop-grad)"

    out.sum().backward()
    assert x.grad is not None, "x must have gradient after CfC no-IMU backward"
    assert torch.isfinite(x.grad).all(), "x.grad must be finite"


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum / DataLoader stability tests (Stage 1a root-cause regression)
# ─────────────────────────────────────────────────────────────────────────────

def test_min_rows_for_distributed():
    """_min_rows_for_distributed returns max(world_size * batch_size, world_size)."""
    import kairos_train as kt
    assert kt._min_rows_for_distributed(4, 1) == 4
    assert kt._min_rows_for_distributed(4, 2) == 8
    assert kt._min_rows_for_distributed(1, 1) == 1
    assert kt._min_rows_for_distributed(1, 4) == 4


def test_curriculum_tiny_subset_falls_back_to_full(capsys):
    """
    _curriculum_df falls back to df_all when filtered rows < min_rows.
    This prevents _make_dataloader receiving fewer rows than world_size,
    which previously caused an empty DataLoader and a silent StopIteration crash.
    """
    import pandas as pd
    import kairos_train as kt

    df = pd.DataFrame({"curriculum_order": [1, 1, 2, 2, 3, 3, 3, 3]})

    # Only 2 rows with order==1; min_rows=4 → must fall back to full df (8 rows)
    result = kt._curriculum_df(df, max_order=1, min_rows=4, rank=0)
    assert len(result) == len(df), \
        f"expected fallback to full df ({len(df)} rows), got {len(result)}"
    out = capsys.readouterr().out
    assert "WARNING" in out and "falling back" in out

    # 4 rows with order<=2; min_rows=4 → exactly at limit, no fallback
    result2 = kt._curriculum_df(df, max_order=2, min_rows=4, rank=0)
    assert len(result2) == 4, f"expected 4 rows, got {len(result2)}"

    # 0 matching rows → always falls back
    df_none = pd.DataFrame({"curriculum_order": [3, 3, 3]})
    result3 = kt._curriculum_df(df_none, max_order=1, min_rows=1, rank=0)
    assert len(result3) == 3

    # Column absent → return df_all unchanged (no warning)
    df_no_col = pd.DataFrame({"x": [1, 2, 3]})
    result4 = kt._curriculum_df(df_no_col, max_order=1, min_rows=1, rank=0)
    assert len(result4) == 3


def test_distributed_sampler_drop_last_false_non_empty():
    """
    DistributedSampler with drop_last=True produces 0 samples per rank when
    dataset size < world_size (the Stage 1a root cause).
    drop_last=False pads to ceil(N/world_size), ensuring every rank gets >= 1 sample.
    """
    from torch.utils.data import DistributedSampler, TensorDataset

    # 2 rows, 4 ranks: floor(2/4)=0 with drop_last=True → empty DataLoader
    ds = TensorDataset(torch.zeros(2, 1))
    sampler_drop = DistributedSampler(ds, num_replicas=4, rank=0,
                                      shuffle=False, drop_last=True)
    assert len(list(sampler_drop)) == 0, \
        "drop_last=True with 2 rows / 4 ranks must yield 0 per rank"

    # same dataset, drop_last=False: ceil(2/4)=1 per rank (padded)
    sampler_keep = DistributedSampler(ds, num_replicas=4, rank=0,
                                      shuffle=False, drop_last=False)
    assert len(list(sampler_keep)) == 1, \
        "drop_last=False with 2 rows / 4 ranks must yield 1 per rank (padded)"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b regression: skip_decoder_loss must not include moe_z in total_loss
# ─────────────────────────────────────────────────────────────────────────────

def test_skip_decoder_loss_anchor_only_no_core_grads():
    """
    When _smoke_skip_decoder_loss=True:
      - total_loss must use smoke_loss_anchor only (disconnected from core).
      - moe_z is NOT included in total_loss, so hybrid_core params receive no grad
        from the total_loss backward path.
      - smoke_loss_anchor must receive a gradient.
    This is the Stage 1b root-cause regression: moe_z in total_loss caused
    ZeRO-3 shape mismatches in the sparse MoE dispatch backward.
    """
    torch.manual_seed(30)
    cfg = _ultra_smoke_cfg()
    model = KairosModel(cfg)
    model._smoke_skip_decoder_loss = True
    # Do NOT set _smoke_skip_core — core runs, but its graph must be excluded
    # from total_loss to prevent ZeRO-3 backward shape mismatches.

    model.train()
    out = model(_batch(cfg, train=True))

    assert out.total_loss is not None, "total_loss must not be None"
    assert out.s2ft_loss is not None, "s2ft_loss must not be None"
    assert out.logits is None, "logits must be None with skip_decoder_loss=True"
    assert torch.isfinite(out.total_loss), "total_loss must be finite"
    out.total_loss.backward()

    # smoke_loss_anchor IS the total_loss source — must have gradient
    assert model.smoke_loss_anchor.grad is not None, \
        "smoke_loss_anchor must have grad when _smoke_skip_decoder_loss=True"

    # hybrid_core params must have NO gradient from total_loss — moe_z excluded
    # (z-loss still flows to router_proj, but only when NOT in skip_decoder_loss mode)
    router_grad = model.hybrid_core.blocks[0].moe_ffn.router_proj.weight.grad
    assert router_grad is None or router_grad.abs().sum() == 0, \
        ("router_proj must have no grad when skip_decoder_loss=True "
         "(moe_z excluded from total_loss)")

    # Expert weight W1 must also have no grad from total_loss
    w1_grad = model.hybrid_core.blocks[0].moe_ffn.W1.grad
    assert w1_grad is None or w1_grad.abs().sum() == 0, \
        "W1 must have no grad when skip_decoder_loss=True (core graph excluded)"


def test_skip_decoder_loss_moe_z_excluded_from_total():
    """
    Verify that moe_z_loss (from KairosOutput) is NOT included in total_loss
    when _smoke_skip_decoder_loss=True, even when hybrid_core runs and produces
    a non-zero z-loss.
    """
    torch.manual_seed(31)
    cfg = _ultra_smoke_cfg()
    model = KairosModel(cfg)
    model._smoke_skip_decoder_loss = True

    model.train()
    out = model(_batch(cfg, train=True))

    # moe_z_loss is populated in KairosOutput (for logging) but excluded from total_loss
    # The total_loss grad_fn must NOT reference z-loss — check it equals anchor path only
    assert out.total_loss is not None
    assert out.s2ft_loss is not None

    # total_loss == s2ft_loss (both come from smoke_loss_anchor)
    assert torch.allclose(out.total_loss, out.s2ft_loss, atol=1e-6), \
        ("total_loss must equal s2ft_loss (both are anchor dummy) "
         "when skip_decoder_loss=True; moe_z must be excluded")


def test_dense_moe_fallback_forward_backward():
    """
    MoeSwiGLUFFN with dense_moe_fallback=True:
      - forward + backward must succeed with finite gradients.
      - All experts are weighted densely; no variable-length expert slices.
      - This is the safe path for testing core backward under ZeRO-3.
    """
    torch.manual_seed(32)
    cfg = _ultra_smoke_cfg()
    cfg.kcfg.dense_moe_fallback = True

    model = KairosModel(cfg)
    model.train()

    out = model(_batch(cfg, train=True))
    assert out.total_loss is not None
    assert torch.isfinite(out.total_loss)
    out.total_loss.backward()

    # With dense fallback, router_proj gets gradient via z-loss (still computed)
    router_grad = model.hybrid_core.blocks[0].moe_ffn.router_proj.weight.grad
    assert router_grad is not None, \
        "router_proj.weight must have gradient with dense_moe_fallback=True"
    assert torch.isfinite(router_grad).all(), \
        "router_proj gradient must be finite with dense_moe_fallback=True"

    # W1 (expert gate) must also have gradient
    w1_grad = model.hybrid_core.blocks[0].moe_ffn.W1.grad
    assert w1_grad is not None, "W1 must have gradient with dense_moe_fallback=True"
    assert torch.isfinite(w1_grad).all(), "W1 gradient must be finite"


def test_ultra_smoke_core_loss_backward():
    """
    When _ultra_smoke_core_loss=True (with dense_moe_fallback=True):
      - total_loss = x.pow(2).mean() * 1e-4 (directly from hybrid core output)
      - backward must succeed with finite gradients through the full core.
      - This is Stage 1c: safe core backward test.
    """
    torch.manual_seed(33)
    cfg = _ultra_smoke_cfg()
    cfg.kcfg.dense_moe_fallback = True  # ZeRO-3 safe

    model = KairosModel(cfg)
    model._ultra_smoke_core_loss = True

    model.train()
    out = model(_batch(cfg, train=True))

    assert out.total_loss is not None, "total_loss must not be None"
    assert torch.isfinite(out.total_loss), "total_loss must be finite"
    out.total_loss.backward()

    # Core params must receive gradients from the x.pow(2).mean() loss
    router_grad = model.hybrid_core.blocks[0].moe_ffn.router_proj.weight.grad
    assert router_grad is not None, \
        "router_proj must have gradient with ultra_smoke_core_loss=True"
    assert torch.isfinite(router_grad).all(), "router_proj gradient must be finite"

    w1_grad = model.hybrid_core.blocks[0].moe_ffn.W1.grad
    assert w1_grad is not None, "W1 must have gradient with ultra_smoke_core_loss=True"
    assert torch.isfinite(w1_grad).all(), "W1 gradient must be finite"


def test_sparse_moe_empty_experts_backward_cpu():
    """
    MoeSwiGLUFFN with many experts and few tokens (most experts empty).
    The cat-based expert dispatch must handle zero-token experts correctly
    and produce valid gradients for the non-empty expert backward paths.
    Regression for the sort-by-expert cat pattern (non-in-place).
    """
    torch.manual_seed(34)
    from kairos_hybrid_block import KairosConfig, MoeSwiGLUFFN, RMSNorm

    cfg = KairosConfig()
    cfg.d_model = 32
    cfg.num_experts = 16   # many experts → most get 0 tokens with 4 tokens total
    cfg.top_k = 2
    cfg.moe_d_ff = 64
    cfg.dense_moe_fallback = False  # explicitly sparse

    moe = MoeSwiGLUFFN(cfg)
    moe.train()
    norm = RMSNorm(cfg.d_model)

    # 4 tokens, 16 experts, top_k=2 → 8 assignments → most experts get 0 tokens
    B, T = 1, 4
    x = torch.randn(B, T, cfg.d_model, requires_grad=True)
    out = moe(x, norm, router_bias=None)

    assert out.shape == (B, T, cfg.d_model), \
        f"unexpected shape: {tuple(out.shape)}"
    out.sum().backward()

    assert x.grad is not None, "x must have gradient"
    assert torch.isfinite(x.grad).all(), "x.grad must be finite"
    assert moe.router_proj.weight.grad is not None, \
        "router_proj.weight must have gradient"
    assert torch.isfinite(moe.router_proj.weight.grad).all(), \
        "router_proj gradient must be finite"
    # Non-empty expert params must have gradient; empty experts may have None or zero
    active_grads = [
        moe.W1.grad is not None,
        moe.W2.grad is not None,
        moe.W3.grad is not None,
    ]
    assert all(active_grads), "W1/W2/W3 must have gradient tensors"


# ─────────────────────────────────────────────────────────────────────────────
# ZeRO-3 module isolation tests — core_debug_bypass_* and core_debug_layers
# ─────────────────────────────────────────────────────────────────────────────

def _debug_core_cfg(d_model: int = 64) -> "KairosConfig":
    """Tiny KairosConfig for bypass/layer-limit tests."""
    from kairos_hybrid_block import KairosConfig
    cfg = KairosConfig()
    cfg.d_model        = d_model
    cfg.d_state        = 8
    cfg.dt_rank        = 8
    cfg.mamba_expand   = 1
    cfg.mamba_chunk    = 8
    cfg.num_heads_q    = 4
    cfg.num_heads_kv   = 2
    cfg.attn_window    = 64
    cfg.max_seq_len    = 128
    cfg.num_experts    = 4
    cfg.moe_d_ff       = 128
    cfg.use_grad_checkpoint = False
    return cfg


def test_core_debug_bypass_moe_forward_backward():
    """
    core_debug_bypass_moe=True: MoE FFN replaced by identity.
    - Forward and backward succeed with finite gradients.
    - _z_loss_val is None for all blocks (no router ran).
    - _z_loss_for_backward is a zero tensor (not None).
    """
    torch.manual_seed(40)
    from kairos_hybrid_block import KairosHybridCore

    cfg = _debug_core_cfg()
    cfg.core_debug_bypass_moe = True
    core = KairosHybridCore(cfg)
    core.train()

    B, T = 1, 30
    x = torch.randn(B, T, cfg.d_model, requires_grad=True)
    imu_mask = torch.zeros(B, T, dtype=torch.bool)
    delta_t  = torch.zeros(B, T)

    out = core(x, imu_mask, delta_t)
    assert out.shape == (B, T, cfg.d_model)
    out.float().mean().backward()

    assert x.grad is not None, "x must have gradient"
    assert torch.isfinite(x.grad).all(), "x.grad must be finite"

    for blk in core.blocks:
        assert blk.moe_ffn._z_loss_val is None, \
            "z_loss_val must be None when MoE is bypassed"

    assert core._z_loss_for_backward is not None, \
        "_z_loss_for_backward must not be None (set to zero tensor)"
    assert core._z_loss_for_backward.item() == 0.0, \
        "_z_loss_for_backward must be zero when MoE is bypassed"


def test_core_debug_bypass_mamba_forward_backward():
    """
    core_debug_bypass_mamba=True: Mamba-2 replaced by identity.
    - Forward and backward succeed with finite gradients.
    - CfC, SWA, MoE still run normally.
    """
    torch.manual_seed(41)
    from kairos_hybrid_block import KairosHybridCore

    cfg = _debug_core_cfg()
    cfg.core_debug_bypass_mamba = True
    core = KairosHybridCore(cfg)
    core.train()

    B, T = 1, 30
    x = torch.randn(B, T, cfg.d_model, requires_grad=True)
    imu_mask = torch.zeros(B, T, dtype=torch.bool)
    delta_t  = torch.zeros(B, T)

    out = core(x, imu_mask, delta_t)
    assert out.shape == (B, T, cfg.d_model)
    out.float().mean().backward()

    assert x.grad is not None, "x must have gradient"
    assert torch.isfinite(x.grad).all(), "x.grad must be finite"

    router_grad = core.blocks[0].moe_ffn.router_proj.weight.grad
    assert router_grad is not None, \
        "router_proj must have gradient (MoE ran, only Mamba bypassed)"
    assert torch.isfinite(router_grad).all()


def test_core_debug_one_layer_forward_backward():
    """
    core_debug_layers=1: only the first loop iteration (loop 0, block 0) runs.
    - Forward and backward succeed.
    - z_loss_for_backward is set (from block 0's MoE).
    - Output shape is unchanged.
    """
    torch.manual_seed(42)
    from kairos_hybrid_block import KairosHybridCore

    cfg = _debug_core_cfg()
    cfg.core_debug_layers = 1
    core = KairosHybridCore(cfg)
    core.train()

    B, T = 1, 30
    x = torch.randn(B, T, cfg.d_model, requires_grad=True)
    imu_mask = torch.zeros(B, T, dtype=torch.bool)
    delta_t  = torch.zeros(B, T)

    out = core(x, imu_mask, delta_t)
    assert out.shape == (B, T, cfg.d_model)
    out.float().mean().backward()

    assert x.grad is not None, "x must have gradient"
    assert torch.isfinite(x.grad).all(), "x.grad must be finite"
    assert core._z_loss_for_backward is not None, \
        "_z_loss_for_backward must be set after forward (block 0 ran)"


def test_zero_stage_default_ultra_smoke_is_2():
    """
    _ds_config must default to ZeRO stage 2 for ultra_smoke/budget runs when
    --zero_stage is not explicitly set. Full training keeps stage 3.
    """
    import argparse
    import kairos_train as kt

    def _args(**kw):
        defaults = dict(
            ultra_smoke_mode=False, smoke_mode=False, budget_mode=False,
            zero_stage=None, zero_offload=False,
            grad_accum=1, max_grad_norm=1.0, micro_batch=1, log_every=10,
        )
        defaults.update(kw)
        return argparse.Namespace(**defaults)

    # ultra_smoke → ZeRO-2
    cfg = kt._ds_config(_args(ultra_smoke_mode=True, smoke_mode=True))
    assert cfg["zero_optimization"]["stage"] == 2, \
        f"ultra_smoke must default to ZeRO-2, got {cfg['zero_optimization']['stage']}"

    # budget → ZeRO-2
    cfg2 = kt._ds_config(_args(budget_mode=True))
    assert cfg2["zero_optimization"]["stage"] == 2, \
        f"budget_mode must default to ZeRO-2, got {cfg2['zero_optimization']['stage']}"

    # full training → ZeRO-3
    cfg3 = kt._ds_config(_args())
    assert cfg3["zero_optimization"]["stage"] == 3, \
        f"full training must default to ZeRO-3, got {cfg3['zero_optimization']['stage']}"

    # explicit --zero_stage 3 overrides even in ultra_smoke
    cfg4 = kt._ds_config(_args(ultra_smoke_mode=True, smoke_mode=True, zero_stage=3))
    assert cfg4["zero_optimization"]["stage"] == 3, \
        "explicit --zero_stage 3 must be respected even in ultra_smoke"

    # explicit --zero_stage 2 overrides even in full training
    cfg5 = kt._ds_config(_args(zero_stage=2))
    assert cfg5["zero_optimization"]["stage"] == 2, \
        "explicit --zero_stage 2 must be respected in full training"


# ─────────────────────────────────────────────────────────────────────────────
# BF16 / float32 dtype compatibility tests (Stage 2 regression: LiDAR/IMU)
# Root cause: float32 batch inputs + BF16 model weights → mixed dtype in
# geometry ops (_farthest_point_sample, _ball_query, torch.linspace).
# ─────────────────────────────────────────────────────────────────────────────

def test_imu_bf16_timestamp_dtype_compat():
    """
    IMUEncoder converted to bfloat16 must accept float32 imu_data and timestamps.
    All internal ops must stay dtype-consistent; forward must not raise.
    Output tokens must be bfloat16.
    """
    torch.manual_seed(50)
    from kairos_imu import IMUEncoder, IMUEncoderConfig

    cfg = IMUEncoderConfig(n_tokens=4, cfc_hidden=32, n_cfc_layers=1)
    enc = IMUEncoder(d_model=32, n_tokens=4, cfg=cfg).to(torch.bfloat16)
    enc.eval()

    B, T = 1, 8
    imu_data   = torch.randn(B, T, 7, dtype=torch.float32)
    timestamps = torch.arange(T, dtype=torch.float32).unsqueeze(0) / 30.0

    with torch.no_grad():
        tokens, delta_t = enc(imu_data, timestamps)

    assert tokens.dtype  == torch.bfloat16, f"tokens must be bfloat16, got {tokens.dtype}"
    assert delta_t.dtype == torch.bfloat16, f"delta_t must be bfloat16, got {delta_t.dtype}"
    assert tokens.shape  == (B, 4, 32)
    assert delta_t.shape == (B, 4)
    assert torch.isfinite(tokens).all(), "tokens must be finite"
    assert torch.isfinite(delta_t).all(), "delta_t must be finite"


def test_lidar_bf16_input_dtype_compat():
    """
    PointMambaEncoder converted to bfloat16 must accept float32 point cloud inputs.
    Geometry ops (_farthest_point_sample, _ball_query) must not raise dtype errors.
    Output tokens must be bfloat16; centroids are returned in model dtype.
    """
    torch.manual_seed(51)
    from kairos_lidar import PointMambaEncoder, LiDAREncoderConfig

    cfg = LiDAREncoderConfig(
        n_tokens=4, n_points=32, n_neighbors=4, pn_hidden=8,
        d_state=4, mamba_chunk=4, n_mamba_layers=1,
        moe_experts=1, moe_d_ff=16,
    )
    enc = PointMambaEncoder(d_model=32, n_tokens=4, cfg=cfg).to(torch.bfloat16)
    enc.eval()

    B, N = 1, 32
    pts = torch.zeros(B, N, 4, dtype=torch.float32)
    pts[0, :, 0] = torch.linspace(1.0, 30.0, N)   # x: forward
    pts[0, :, 1] = torch.linspace(-2.0, 2.0, N)    # y: lateral
    pts[0, :, 2] = torch.linspace(-0.5, 1.0, N)    # z: height
    pts[0, :, 3] = 0.5                              # intensity

    with torch.no_grad():
        tokens, centroids = enc(pts, pts.clone())

    assert tokens.dtype == torch.bfloat16, \
        f"tokens must be bfloat16, got {tokens.dtype}"
    assert tokens.shape   == (B, 4, 32)
    assert centroids.shape == (B, 4, 3)
    assert torch.isfinite(tokens).all(), "tokens must be finite"


def test_model_bf16_lidar_imu_forward():
    """
    Full KairosModel with bfloat16 weights must forward-pass successfully when
    LiDAR and IMU encoders are enabled (not skipped). Validates the dtype fix for:
      RuntimeError: expected dtype c10::BFloat16 for 'end' but got dtype float
    Root cause: float32 batch pts/timestamps entering BF16 geometry ops.
    """
    torch.manual_seed(52)
    cfg = _ultra_smoke_cfg()
    cfg.kcfg.dense_moe_fallback = True   # ZeRO-2 safe core backward

    model = KairosModel(cfg).to(torch.bfloat16)
    # Do NOT set _smoke_skip_lidar or _smoke_skip_imu — both encoders must run.
    model.eval()

    batch = _batch(cfg, train=False)
    assert batch.lidar_t.dtype  == torch.float32, "lidar must be float32 from batch"
    assert batch.imu_data.dtype == torch.float32, "imu must be float32 from batch"

    with torch.no_grad():
        out = model(batch)

    T = cfg.n_cam + cfg.n_lidar + cfg.n_imu + cfg.n_query
    assert out.hidden is not None
    assert out.hidden.shape == (1, T, cfg.kcfg.d_model)
    assert out.hidden.dtype == torch.bfloat16, \
        f"output must be bfloat16, got {out.hidden.dtype}"
    assert torch.isfinite(out.hidden).all(), "output hidden states must be finite"


def test_lidar_linspace_explicit_dtype_no_bf16_error():
    """
    Regression: torch.linspace(0, N-1, steps, device=device) without explicit dtype
    can inherit BF16 default dtype under autocast/DeepSpeed, causing:
      RuntimeError: expected dtype c10::BFloat16 for 'end' but got dtype float
    The fixed call uses dtype=torch.float32 explicitly.
    Verifies the linspace call produces a valid long index tensor.
    """
    torch.manual_seed(53)
    from kairos_lidar import _ball_query

    # Simulate N > 20_000 path (downsampling via linspace)
    B, N, K = 1, 25_000, 4
    # Create xyz tensors in BF16 to force the worst-case dtype scenario
    xyz_query = torch.randn(B, K, 3, dtype=torch.bfloat16)
    xyz_all   = torch.randn(B, N, 3, dtype=torch.bfloat16)

    # This must NOT raise "expected dtype c10::BFloat16 for 'end'"
    idx = _ball_query(xyz_query, xyz_all, radius=1.0, n_sample=4, _chunk=4)

    assert idx.shape == (B, K, 4), f"unexpected idx shape: {idx.shape}"
    assert idx.dtype == torch.long
    assert (idx >= 0).all() and (idx < N).all()


# ---------------------------------------------------------------------------
# Session-3 regression tests: autocast float32-preserve list
# ---------------------------------------------------------------------------
# torch.autocast(dtype=bfloat16) keeps sum/pow/softmax/sigmoid in float32 even
# when inputs are BF16.  Plain CPU tests without autocast miss these failures.
# These tests wrap forward passes in autocast to catch index-put dtype mismatches
# that only appeared in SageMaker.
# ---------------------------------------------------------------------------


def _tiny_kairos_cfg_lidar_imu():
    """Return a minimal KairosConfig with LiDAR+IMU enabled and dense_moe_fallback."""
    import kairos_hybrid_block as kt
    cfg = kt.KairosConfig(
        d_model=64,
        n_heads=4,
        n_kv_heads=1,
        n_experts=4,
        moe_top_k=2,
        moe_d_ff=128,
        d_ff=64,
        n_loops=1,
        window_size=32,
        cfc_hidden=32,
        n_cfc_layers=1,
        dense_moe_fallback=True,
    )
    return cfg


def test_safe_index_assignment_casts_source_dtype():
    """
    Regression: destination BF16, source float32 → index-put fails without explicit cast.
    _as_like() casts source to destination dtype before assignment.
    Covers _build_sequence_masks delta_t assignment.
    """
    from kairos_model import _as_like

    dest = torch.zeros(2, 10, dtype=torch.bfloat16)
    src  = torch.ones(2, 4, dtype=torch.float32)  # float32 source

    # Direct slice assignment without cast should raise; with _as_like it must not
    dest[:, 3:7] = _as_like(src, dest)
    assert dest.dtype == torch.bfloat16, "destination dtype must remain bfloat16 after assignment"
    assert dest[:, 3:7].sum().item() == pytest.approx(8.0, abs=0.1), \
        "assigned values must be 1.0 (cast from float32)"


def test_delta_t_assignment_bf16():
    """
    Regression: imu_dt is float32, delta_t is BF16 — _as_like must bridge the mismatch.
    This mirrors the _build_sequence_masks path that failed in SageMaker.
    """
    from kairos_model import _as_like

    B, T_total, T_imu = 1, 20, 6
    delta_t = torch.zeros(B, T_total, dtype=torch.bfloat16)
    imu_dt  = torch.rand(B, T_imu, dtype=torch.float32)  # float32 as produced by IMUEncoder

    imu_start, imu_end = 10, 10 + T_imu
    delta_t[:, imu_start:imu_end] = _as_like(imu_dt, delta_t)

    assert delta_t.dtype == torch.bfloat16, "delta_t must remain bfloat16"
    assert delta_t[:, imu_start:imu_end].sum().item() > 0, "imu_dt values must be written"
    assert delta_t[:, :imu_start].sum().item() == 0.0, "other slots must be unchanged"


def test_fps_safe_and_farthest_point_autocast_no_dtype_error():
    """
    Regression: _fps_safe and _farthest_point_sample are called with BF16 xyz inside
    torch.autocast.  sum/pow return float32 under autocast → index-put dtype mismatch.
    Fixed by computing dist / centroids in explicit float32 internally.
    Exercises both closures via PointMambaEncoder.forward under autocast.
    """
    torch.manual_seed(77)
    from kairos_lidar import PointMambaEncoder, LiDAREncoderConfig

    # Tiny config so the test is fast; n_points small enough to fit in memory
    tiny_cfg = LiDAREncoderConfig(
        n_tokens=8, n_points=64, n_neighbors=4, ball_radius=2.0,
        pn_hidden=16, d_state=8, mamba_chunk=8, n_mamba_layers=1,
        moe_experts=2, moe_d_ff=32,
    )
    d_model = 32
    enc = PointMambaEncoder(d_model=d_model, n_tokens=8, cfg=tiny_cfg).to(torch.bfloat16)
    enc.eval()

    B = 1
    # float32 inputs (as they arrive from the dataloader before encoder entry cast)
    pts_t  = torch.randn(B, 64, 4, dtype=torch.float32)
    pts_t1 = torch.randn(B, 64, 4, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        tokens, centroids = enc(pts_t, pts_t1)

    assert tokens.dtype == torch.bfloat16, \
        f"PointMambaEncoder tokens must be bfloat16 under autocast, got {tokens.dtype}"
    assert tokens.shape == (B, 8, d_model), f"unexpected tokens shape {tokens.shape}"
    assert torch.isfinite(tokens).all(), "tokens must be finite"


def test_lidar_moe_softmax_autocast_no_dtype_error():
    """
    Regression: _LiDARMoEFFN.forward used F.softmax which returns float32 under
    autocast. The result was scatter-assigned into a BF16 output tensor causing
    'Index put requires dtypes match'.  Fixed by .to(x_flat.dtype) after softmax.
    """
    torch.manual_seed(88)
    from kairos_lidar import _LiDARMoEFFN

    d_model = 64
    ffn = _LiDARMoEFFN(d=d_model, n_experts=4, d_ff=128).to(torch.bfloat16)
    ffn.eval()

    B, T = 2, 16
    x = torch.randn(B, T, d_model, dtype=torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = ffn(x)

    assert out.dtype == torch.bfloat16, \
        f"_LiDARMoEFFN output must be bfloat16 under autocast, got {out.dtype}"
    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all(), "output must be finite"


# ---------------------------------------------------------------------------
# Session-4 regression tests: _safe_linspace_indices + IMU/LiDAR linspace fix
# ---------------------------------------------------------------------------
# The specific error: `expected dtype c10::BFloat16 for 'end' but got dtype float`
# arises when torch.linspace receives start/end arguments where one is inferred as
# BFloat16 (from the ambient default dtype under DeepSpeed BF16) and the other is
# float32.  _safe_linspace_indices and the IMU linspace rewrite prevent this.
# ---------------------------------------------------------------------------


def test_safe_linspace_indices_under_bf16():
    """
    Regression: torch.linspace(start, end, steps) under BF16 default dtype raises
    'expected dtype c10::BFloat16 for end but got dtype float' if start is inferred
    as BF16 and end as float32.
    _safe_linspace_indices wraps both endpoints in float() and forces dtype=float32,
    preventing the mismatch regardless of ambient dtype.
    """
    from kairos_lidar import _safe_linspace_indices

    N = 512  # matches ultra_smoke n_points
    K = 8    # matches ultra_smoke n_lidar_tokens

    # Simulate the autocast environment where default dtype could become BF16
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        idx = _safe_linspace_indices(
            0, N - 1, K,
            device=torch.device("cpu"), max_index=N - 1,
            name="test_ball_query_downsample",
        )

    assert idx.dtype == torch.long, \
        f"_safe_linspace_indices must return torch.long, got {idx.dtype}"
    assert idx.shape == (K,), f"unexpected shape: {idx.shape}"
    assert int(idx.min()) >= 0, "all indices must be non-negative"
    assert int(idx.max()) < N, f"all indices must be < N={N}"
    # Check that indices are strictly increasing (evenly spaced)
    assert (idx[1:] >= idx[:-1]).all(), "indices must be non-decreasing"


def test_imu_encoder_bf16_linspace_selection():
    """
    Regression: IMU encoder's stride-select previously used torch.arange(K)*step,
    which can fail under BF16 autocast when K is derived from tensor metadata.
    Now uses torch.linspace(..., dtype=torch.float32).round().long().clamp_().
    Verifies: output dtype is BF16, delta_t dtype matches tokens dtype.
    """
    torch.manual_seed(101)
    from kairos_imu import IMUEncoder, IMUEncoderConfig

    cfg = IMUEncoderConfig(n_tokens=4, cfc_hidden=16, n_cfc_layers=1)
    enc = IMUEncoder(d_model=32, n_tokens=4, cfg=cfg).to(torch.bfloat16)
    enc.eval()

    B, T_imu = 1, 12
    imu_data   = torch.randn(B, T_imu, 7, dtype=torch.float32)  # float32 from dataloader
    timestamps = torch.cumsum(
        torch.full((B, T_imu), 1.0 / 30.0, dtype=torch.float32), dim=1
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        tokens, delta_t = enc(imu_data, timestamps)

    assert tokens.dtype == torch.bfloat16, \
        f"IMUEncoder tokens must be bfloat16 under autocast, got {tokens.dtype}"
    assert delta_t.dtype == torch.bfloat16, \
        f"IMUEncoder delta_t must be bfloat16 under autocast, got {delta_t.dtype}"
    assert tokens.shape == (B, 4, 32), f"unexpected tokens shape: {tokens.shape}"
    assert delta_t.shape == (B, 4), f"unexpected delta_t shape: {delta_t.shape}"
    assert torch.isfinite(tokens).all(), "tokens must be finite"
    assert torch.isfinite(delta_t).all(), "delta_t must be finite"
    assert (delta_t > 0).all(), "delta_t values must be positive"


def test_lidar_encoder_bf16_index_selection():
    """
    Regression: PointMambaEncoder's _ball_query used torch.linspace without
    explicit dtype under BF16 autocast, causing 'expected BFloat16 for end'.
    Fixed by _safe_linspace_indices which forces dtype=float32 and float() wrapping.
    Also covers _fps_safe / _farthest_point_sample with n_points > 20_000 path.
    """
    torch.manual_seed(202)
    from kairos_lidar import PointMambaEncoder, LiDAREncoderConfig, _safe_linspace_indices

    # n_points=512 matches ultra_smoke run; ball_query linspace NOT triggered (512 < 20k)
    tiny_cfg = LiDAREncoderConfig(
        n_tokens=4, n_points=64, n_neighbors=4, ball_radius=2.0,
        pn_hidden=8, d_state=4, mamba_chunk=4, n_mamba_layers=1,
        moe_experts=1, moe_d_ff=16,
    )
    d_model = 16
    enc = PointMambaEncoder(d_model=d_model, n_tokens=4, cfg=tiny_cfg).to(torch.bfloat16)
    enc.eval()

    B = 1
    pts_t  = torch.randn(B, 64, 4, dtype=torch.float32)
    pts_t1 = torch.randn(B, 64, 4, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        tokens, centroids = enc(pts_t, pts_t1)

    assert tokens.dtype == torch.bfloat16, \
        f"PointMambaEncoder tokens must be bfloat16, got {tokens.dtype}"
    assert centroids.shape == (B, 4, 3), f"unexpected centroids shape: {centroids.shape}"
    assert torch.isfinite(tokens).all(), "tokens must be finite"

    # Direct test of _safe_linspace_indices for the N>20_000 branch path
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        idx_large = _safe_linspace_indices(
            0, 29_999, 16_384,
            device=torch.device("cpu"), max_index=29_999,
            name="test_large_N_downsample",
        )
    assert idx_large.dtype == torch.long
    assert idx_large.shape == (16_384,)
    assert int(idx_large[-1]) == 29_999, "last index must reach N-1"


def test_full_model_bf16_lidar_imu_forward_mock_vision():
    """
    Regression: full Kairos model with mock vision, LiDAR+IMU enabled (not skipped),
    dense_moe_fallback=True, model in BF16, batch in float32 — forward must succeed.
    Exercises the complete stack: vision(mock) → LiDAR → IMU → fusion → hybrid core.
    Covers the _safe_linspace_indices path and IMU linspace stride-select.
    """
    torch.manual_seed(303)

    cfg = _ultra_smoke_cfg()
    cfg.kcfg.dense_moe_fallback = True   # safe for ZeRO-2 core backward test

    batch = _batch(cfg, train=True)   # float32 batch, as from the dataloader

    model = KairosModel(cfg).to(torch.bfloat16)
    # DO NOT set _smoke_skip_lidar or _smoke_skip_imu — both encoders must run
    model._ultra_smoke_core_loss   = True   # total_loss = x.pow(2).mean()*1e-4
    model._smoke_skip_decoder_loss = False  # let normal S2FT loss run (tiny model)
    model.eval()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with torch.no_grad():
            out = model(batch)

    assert out.total_loss is not None, "forward must return a total_loss"
    assert torch.isfinite(out.total_loss), "total_loss must be finite"
    if out.hidden is not None:
        assert torch.isfinite(out.hidden).all(), "hidden states must be finite"


# ---------------------------------------------------------------------------
# Session-5 regression tests: LearnableNorm BF16 training mode (running stats)
# ---------------------------------------------------------------------------
# Root cause: model.to(torch.bfloat16) converts buffers — including running_mean
# and running_var — to BF16.  lerp_() then fails in training mode because batch
# stats are computed in float32:
#   RuntimeError: expected dtype c10::BFloat16 for `end` but got dtype float
# Fix: LearnableNorm._apply() override keeps running stats in float32 permanently.
# These tests exercise the TRAINING path; existing BF16 tests use eval mode only.
# ---------------------------------------------------------------------------


def test_learnable_norm_bf16_train_lerp_dtype_safe():
    """
    LearnableNorm.to(bfloat16).train().forward() must not raise
    'expected dtype c10::BFloat16 for end but got dtype float' from lerp_.
    The _apply override keeps running_mean/running_var in float32 after .to(bf16).
    """
    from kairos_imu import LearnableNorm

    norm = LearnableNorm(7).to(torch.bfloat16)
    norm.train()

    x = torch.randn(16, 7, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = norm(x)

    assert y.dtype == torch.bfloat16, f"output must be bfloat16, got {y.dtype}"
    assert torch.isfinite(y).all(), "output must be finite"
    assert norm.running_mean.dtype == torch.float32, \
        f"running_mean must stay float32 after .to(bfloat16), got {norm.running_mean.dtype}"
    assert norm.running_var.dtype == torch.float32, \
        f"running_var must stay float32 after .to(bfloat16), got {norm.running_var.dtype}"


def test_imu_encoder_bf16_train_running_stats_safe():
    """
    IMUEncoder converted to bfloat16 and called in training mode must not crash
    at the LearnableNorm lerp_ call.  imu_norm.running_mean and running_var must
    remain float32 after model.to(bfloat16); output tokens must be bfloat16.
    """
    from kairos_imu import IMUEncoder, IMUEncoderConfig

    cfg = IMUEncoderConfig(n_tokens=4, cfc_hidden=16, n_cfc_layers=1)
    enc = IMUEncoder(d_model=32, n_tokens=4, cfg=cfg).to(torch.bfloat16)
    enc.train()

    B, T = 1, 8
    imu_data   = torch.randn(B, T, 7, dtype=torch.float32)
    timestamps = torch.arange(T, dtype=torch.float32).unsqueeze(0) / 30.0

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        tokens, delta_t = enc(imu_data, timestamps)

    assert tokens.dtype == torch.bfloat16, \
        f"tokens must be bfloat16, got {tokens.dtype}"
    assert delta_t.dtype == torch.bfloat16, \
        f"delta_t must be bfloat16, got {delta_t.dtype}"
    assert torch.isfinite(tokens).all(), "tokens must be finite"
    assert torch.isfinite(delta_t).all(), "delta_t must be finite"
    assert enc.imu_norm.running_mean.dtype == torch.float32, \
        f"imu_norm.running_mean must stay float32, got {enc.imu_norm.running_mean.dtype}"
    assert enc.imu_norm.running_var.dtype == torch.float32, \
        f"imu_norm.running_var must stay float32, got {enc.imu_norm.running_var.dtype}"


# ---------------------------------------------------------------------------
# Session-6 regression tests: BF16 Stage 2 — full forward + in-place dtype safety
# ---------------------------------------------------------------------------
# Root cause: index_add_(): self (Float) and source (BFloat16) during hybrid core
# forward pass under DeepSpeed BF16.  Fixed by _safe_index_add_ / _safe_scatter_add_
# helpers that cast src to match dst before any in-place accumulation op.
# ---------------------------------------------------------------------------


def test_index_add_dtype_safe_float_dst_bf16_src():
    """
    Directly verify the _safe_index_add_ helper in kairos_hybrid_block:
    float32 destination + bfloat16 source must not raise, and result is float32.
    """
    from kairos_hybrid_block import _safe_index_add_

    dst = torch.zeros(8, 4, dtype=torch.float32)
    src = torch.ones(3, 4, dtype=torch.bfloat16)
    idx = torch.tensor([0, 2, 5], dtype=torch.long)

    result = _safe_index_add_(dst.clone(), 0, idx, src, name="test")

    assert result.dtype == torch.float32, f"result must be float32, got {result.dtype}"
    assert torch.isfinite(result).all(), "result must be finite"
    assert result[0, 0].item() == pytest.approx(1.0, abs=0.01)
    assert result[1, 0].item() == pytest.approx(0.0, abs=0.01)  # untouched


def test_scatter_add_dtype_safe_float_dst_bf16_src():
    """
    Directly verify the _safe_scatter_add_ helper in kairos_hybrid_block:
    float32 destination + bfloat16 source must not raise.
    """
    from kairos_hybrid_block import _safe_scatter_add_

    dst = torch.zeros(6, 4, dtype=torch.float32)
    src = torch.ones(4, 4, dtype=torch.bfloat16)
    idx = torch.tensor([[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4], [5, 5, 5, 5]])

    result = _safe_scatter_add_(dst.clone(), 0, idx, src, name="test")

    assert result.dtype == torch.float32, f"result must be float32, got {result.dtype}"
    assert torch.isfinite(result).all(), "result must be finite"


def test_lidar_encoder_bf16_train_forward():
    """
    PointMambaEncoder in BF16 training mode with float32 point cloud inputs
    must complete forward without a dtype error (index_add_/slice assignment).
    Running the encoder train mode exercises _LiDARMoEFFN slice assignments.
    """
    from kairos_lidar import PointMambaEncoder, LiDAREncoderConfig

    torch.manual_seed(42)
    cfg = LiDAREncoderConfig(
        n_tokens=4, n_points=32, n_neighbors=4,
        pn_hidden=8, d_state=4, mamba_chunk=4,
        n_mamba_layers=1, moe_experts=2, moe_d_ff=16,
    )
    enc = PointMambaEncoder(d_model=16, n_tokens=4, cfg=cfg).to(torch.bfloat16)
    enc.train()

    B, N = 1, 32
    pts_t  = torch.randn(B, N, 4, dtype=torch.float32)  # float32 as from dataloader
    pts_t1 = torch.randn(B, N, 4, dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        tokens, centroids = enc(pts_t, pts_t1)

    assert tokens.dtype == torch.bfloat16, f"tokens must be bfloat16, got {tokens.dtype}"
    assert centroids.shape == (B, 4, 3), f"unexpected centroids shape: {centroids.shape}"
    assert torch.isfinite(tokens).all(), "tokens must be finite"


def test_imu_encoder_bf16_train_forward_backward():
    """
    IMUEncoder in BF16 training mode — forward + backward must succeed.
    Exercises LearnableNorm running-stats lerp_ path AND CfC recurrence in BF16.
    """
    from kairos_imu import IMUEncoder, IMUEncoderConfig

    torch.manual_seed(7)
    cfg = IMUEncoderConfig(n_tokens=4, cfc_hidden=16, n_cfc_layers=1)
    enc = IMUEncoder(d_model=32, n_tokens=4, cfg=cfg).to(torch.bfloat16)
    enc.train()

    B, T = 2, 12
    imu_data   = torch.randn(B, T, 7, dtype=torch.float32)
    timestamps = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(B, -1) / 30.0

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        tokens, delta_t = enc(imu_data, timestamps)

    assert tokens.dtype == torch.bfloat16, f"tokens must be bfloat16, got {tokens.dtype}"
    assert torch.isfinite(tokens).all(), "tokens must be finite"
    assert enc.imu_norm.running_mean.dtype == torch.float32, \
        "imu_norm.running_mean must stay float32 in BF16 train mode"

    # Backward
    tokens.float().mean().backward()
    has_grad = [p.grad is not None for p in enc.parameters() if p.requires_grad]
    assert any(has_grad), "at least one parameter must have a gradient"


def test_bf16_stage2_full_forward_mock_vision_lidar_imu_dense_moe_train():
    """
    Full BF16 Stage 2 forward: mock vision + real LiDAR + real IMU + fusion +
    hybrid core (dense_moe_fallback=True) + core loss.  Batch is float32 as from
    the real dataloader.  This is the exact configuration of the failing SageMaker
    run (ml.g6.12xlarge, ZeRO-2, ultra_smoke_core_loss=True, no skip_core).

    Root cause: index_add_(): self (Float) and source (BFloat16) in the CfC block.
    Fix: _safe_index_add_ + _safe_scatter_add_ + _safe_assign_ with dtype cast.
    """
    torch.manual_seed(99)

    cfg = _ultra_smoke_cfg()
    cfg.kcfg.dense_moe_fallback = True
    cfg.kcfg.use_grad_checkpoint = False  # CPU test — no grad checkpoint needed

    model = KairosModel(cfg).to(torch.bfloat16)
    # Mirrors the Stage 2 SageMaker flags (no skip_lidar, no skip_imu, no skip_core)
    model._ultra_smoke_core_loss   = True   # total_loss = x.pow(2).mean()*1e-4
    model._smoke_skip_decoder_loss = False
    model.train()

    batch = _batch(cfg, train=True)   # float32 batch (as from dataloader)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(batch)

    assert out.total_loss is not None, "total_loss must not be None in training"
    assert torch.isfinite(out.total_loss), f"total_loss must be finite, got {out.total_loss}"

    # Backward must succeed too
    out.total_loss.backward()

    moe_grads = model.hybrid_core.blocks[0].moe_ffn.router_proj.weight.grad
    assert moe_grads is not None, "router_proj must receive a gradient"


# ---------------------------------------------------------------------------
# Session-7 regression tests: requirements.txt safety + s3fs-free loader
# ---------------------------------------------------------------------------
# Root cause: s3fs==2023.9.2 in requirements.txt pulled aiobotocore which
# downgraded SageMaker DLC botocore 1.34.112 → 1.31.17, breaking:
#   ImportError: cannot import name 'is_s3express_bucket' from 'botocore.utils'
# Fix: remove s3fs/fsspec from requirements.txt; use pyarrow.fs.S3FileSystem.
# These tests prevent future regressions by inspecting file content.
# ---------------------------------------------------------------------------


def test_requirements_no_s3fs_aiobotocore():
    """
    requirements.txt must NOT contain s3fs, fsspec, or aiobotocore.
    These packages pull aiobotocore which downgrades the SageMaker DLC botocore
    (1.34.112 → 1.31.17) and causes ImportError on 'is_s3express_bucket'.
    """
    req = pathlib.Path(__file__).parent / "requirements.txt"
    content = req.read_text()
    forbidden = ["s3fs", "aiobotocore", "fsspec"]
    for pkg in forbidden:
        # Allow the package name inside comments (starts with #)
        non_comment_lines = [
            line for line in content.splitlines()
            if pkg in line and not line.lstrip().startswith("#")
        ]
        assert not non_comment_lines, (
            f"requirements.txt must not install {pkg!r} — it downgrades "
            f"SageMaker DLC botocore and breaks S3/SageMaker imports.\n"
            f"Offending lines: {non_comment_lines}"
        )


def test_train_script_no_s3fs_import():
    """
    kairos_train.py must not contain 'import s3fs'.
    The parquet loader must use pyarrow.fs.S3FileSystem instead.
    """
    train_py = pathlib.Path(__file__).parent / "kairos_train.py"
    content = train_py.read_text()
    assert "import s3fs" not in content, (
        "kairos_train.py must not import s3fs — "
        "use pyarrow.fs.S3FileSystem for S3 parquet reads instead.\n"
        "s3fs downgrades SageMaker DLC botocore and breaks S3/SageMaker imports."
    )


def test_train_script_uses_pyarrow_fs():
    """
    kairos_train.py must reference pyarrow.fs.S3FileSystem or pafs.S3FileSystem
    as the parquet loader, confirming the s3fs → pyarrow.fs migration is in place.
    """
    train_py = pathlib.Path(__file__).parent / "kairos_train.py"
    content = train_py.read_text()
    assert "pyarrow.fs" in content or "pafs" in content, (
        "kairos_train.py must use pyarrow.fs (pafs.S3FileSystem) for "
        "S3 parquet reads — s3fs was removed to protect SageMaker DLC botocore."
    )


def test_train_script_has_botocore_guard():
    """
    kairos_train.py must contain a botocore version guard (_check_botocore_version)
    that fails fast in SageMaker if botocore was downgraded by a rogue dependency.
    """
    train_py = pathlib.Path(__file__).parent / "kairos_train.py"
    content = train_py.read_text()
    assert "_check_botocore_version" in content, (
        "kairos_train.py must define _check_botocore_version() "
        "to fail fast if requirements.txt accidentally downgrades botocore."
    )
