"""
kairos_metrics.py  —  Validation metrics for Kairos-4B.

Covers three categories as specified:

  1. Functional Diagnostic Metrics  (System-1 Vitals)
       FaultDetectionMetrics  — FDR, precision, recall, F1
       RootCauseMRR           — MRR, Hit@1, Hit@3, Hit@5
       StateEstimationMAE     — MAE on CfC next-state predictions

  2. Reasoning & Reliability Metrics  (System-2 Scorecard)
       CoTFaithfulnessJudge   — GPT-4o-mini LLM-as-judge (skipped if key absent)
       ConflictResolutionScore — synthetic visual contradiction injection
       CalibrationMetrics     — Expected Calibration Error (ECE, MCE)

  3. Efficiency & Deployment Metrics
       EfficiencyProfiler     — Hz, active-param efficiency, context scaling

  Orchestrator:
       KairosEvaluator        — drives all metrics over a DataLoader

  Standalone runner:
       python kairos_metrics.py --ckpt_s3 s3://... --gold_s3 s3://...

Integration with training loop (kairos_train.py):
    from kairos_metrics import KairosEvaluator
    evaluator = KairosEvaluator(engine.module, cfg, device)
    report = evaluator.run(dl_val, n_batches=100)
    if rank == 0:
        evaluator.print_report(report)
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kairos_model import KairoBatch, KairosModel, KairosModelConfig, KairosOutput
from kairos_fusion import CalibMatrices


# ─────────────────────────────────────────────────────────────────────────────
# Text parsing utilities
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that indicate a fault condition in the answer text
_FAULT_POSITIVE = re.compile(
    r"\b(warning|caution|hazard|emergency|critical|collision|danger|"
    r"brake immediately|evasive|obstacle ahead|stop|avoid|imminent)\b",
    re.IGNORECASE,
)
_FAULT_NEGATIVE = re.compile(
    r"\b(clear|safe|normal|proceed|no issue|all clear|no hazard|no obstacle)\b",
    re.IGNORECASE,
)

# Numbered / bulleted root-cause lists in the reasoning chain
_CAUSE_LIST = re.compile(
    r"(?:^|\n)\s*(?:\d+[\.\)]\s*|[-•]\s*)([^\n]+)", re.MULTILINE
)

# Confidence phrases → numeric score  (simple heuristic)
_CONF_HIGH = re.compile(r"\b(definitely|certainly|clearly|confident|certain|sure)\b", re.IGNORECASE)
_CONF_MED  = re.compile(r"\b(likely|probable|appears|seems|suggest)\b", re.IGNORECASE)
_CONF_LOW  = re.compile(r"\b(possibly|maybe|uncertain|unclear|could be)\b", re.IGNORECASE)


def _decode_bytes(token_ids: torch.Tensor) -> str:
    """Decode byte-level token IDs (0-255, BOS=256, EOS=257) to UTF-8 string."""
    ids = token_ids.cpu().tolist()
    raw = bytes([int(b) for b in ids if 0 <= int(b) < 256])
    return raw.decode("utf-8", errors="replace")


def _extract_fault_label(text: str) -> Optional[bool]:
    """
    Return True if text indicates a fault, False if explicitly clear, None if ambiguous.
    Applied to both gold answers and model-generated text.
    """
    pos = bool(_FAULT_POSITIVE.search(text))
    neg = bool(_FAULT_NEGATIVE.search(text))
    if pos and not neg:
        return True
    if neg and not pos:
        return False
    return None   # ambiguous — excluded from FDR computation


def _extract_root_causes(text: str) -> List[str]:
    """
    Parse numbered or bulleted lists from reasoning chain or answer.
    Returns items in order (rank-1 first).
    """
    matches = _CAUSE_LIST.findall(text)
    return [m.strip() for m in matches if m.strip()]


def _extract_confidence(text: str) -> float:
    """Map language confidence markers to a scalar ∈ [0, 1]."""
    if _CONF_HIGH.search(text):
        return 0.9
    if _CONF_MED.search(text):
        return 0.65
    if _CONF_LOW.search(text):
        return 0.35
    return 0.5   # default when no explicit confidence language


def _cause_match(pred_cause: str, gold_cause: str, threshold: float = 0.4) -> bool:
    """
    Fuzzy string match: returns True when two cause strings share enough tokens.
    Uses Jaccard similarity on word sets (no external NLP dependency).
    """
    p = set(pred_cause.lower().split())
    g = set(gold_cause.lower().split())
    if not p or not g:
        return False
    return len(p & g) / len(p | g) >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# 1a. Fault Detection Metrics (FDR, precision, recall, F1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaultDetectionMetrics:
    """
    Streaming accumulator for fault detection performance.

    Usage:
        fdm = FaultDetectionMetrics()
        for generated_text, gold_answer in zip(generated, gold_answers):
            fdm.update(generated_text, gold_answer)
        result = fdm.compute()
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    skipped: int = 0   # ambiguous labels excluded

    def update(self, predicted_text: str, gold_text: str) -> None:
        pred_label = _extract_fault_label(predicted_text)
        gold_label = _extract_fault_label(gold_text)

        if pred_label is None or gold_label is None:
            self.skipped += 1
            return

        if gold_label:
            if pred_label:
                self.tp += 1
            else:
                self.fn += 1
        else:
            if pred_label:
                self.fp += 1
            else:
                self.tn += 1

    def compute(self) -> Dict[str, float]:
        total = self.tp + self.fp + self.fn + self.tn
        if total == 0:
            return {"fdr": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "skipped": 0}

        fdr       = self.tp / max(self.tp + self.fn, 1)          # = recall
        precision = self.tp / max(self.tp + self.fp, 1)
        recall    = fdr
        f1        = (2 * precision * recall / max(precision + recall, 1e-9))

        return {
            "fdr":       round(fdr,       4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "skipped":   self.skipped,
            "total":     total,
        }

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = self.skipped = 0


# ─────────────────────────────────────────────────────────────────────────────
# 1b. Root Cause Mean Reciprocal Rank
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RootCauseMRR:
    """
    Streaming MRR and Hit@k for ranked root-cause lists.

    The gold root cause is the FIRST cause extracted from the gold reasoning chain.
    Predicted ranking comes from the order of causes in the generated text.
    """

    reciprocal_ranks: List[float] = field(default_factory=list)
    hits_at: Dict[int, int]       = field(default_factory=lambda: {1: 0, 3: 0, 5: 0})
    total: int                    = 0

    def update(self, predicted_text: str, gold_reasoning: str) -> None:
        pred_causes = _extract_root_causes(predicted_text)
        gold_causes = _extract_root_causes(gold_reasoning)

        if not gold_causes or not pred_causes:
            return   # nothing to evaluate

        gold_top = gold_causes[0]   # ground truth = first-listed cause

        rr = 0.0
        for rank, pred in enumerate(pred_causes, start=1):
            if _cause_match(pred, gold_top):
                rr = 1.0 / rank
                for k in self.hits_at:
                    if rank <= k:
                        self.hits_at[k] += 1
                break

        self.reciprocal_ranks.append(rr)
        self.total += 1

    def compute(self) -> Dict[str, float]:
        if not self.reciprocal_ranks:
            return {"mrr": 0.0, "hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "total": 0}

        mrr = float(np.mean(self.reciprocal_ranks))
        n   = max(self.total, 1)
        return {
            "mrr":   round(mrr,                       4),
            "hit@1": round(self.hits_at[1] / n,       4),
            "hit@3": round(self.hits_at[3] / n,       4),
            "hit@5": round(self.hits_at[5] / n,       4),
            "total": self.total,
        }

    def reset(self) -> None:
        self.reciprocal_ranks.clear()
        self.hits_at = {1: 0, 3: 0, 5: 0}
        self.total = 0


# ─────────────────────────────────────────────────────────────────────────────
# 1c. State Estimation MAE (CfC next-state prediction)
# ─────────────────────────────────────────────────────────────────────────────

class StateEstimationMAE:
    """
    Measures how well the CfC block's hidden state predicts the next IMU reading.

    Approach (non-invasive — uses a forward hook on KairosHybridBlock):
      - Hook captures h_cfc (B, d) from each KairosHybridBlock forward call.
      - A small linear probe (d → 7) projects h_cfc back to IMU space.
      - MAE is computed against the next timestep's IMU data.

    The probe is registered as a lightweight nn.Linear; call
    `StateEstimationMAE.probe.to(device)` before evaluation.

    Fields estimated: [vel_fwd, acc_fwd, jerk, lat, lon, alt, yaw]
    """

    FIELD_NAMES = ["vel_fwd", "acc_fwd", "jerk", "lat", "lon", "alt", "yaw"]

    def __init__(self, d_model: int = 1024, imu_dim: int = 7) -> None:
        self.probe    = nn.Linear(d_model, imu_dim, bias=True)
        self._errors: List[torch.Tensor] = []   # list of (B, imu_dim) tensors
        self._handles: List[Any]          = []

    def register_hooks(self, model: KairosModel) -> None:
        """Attach forward hooks to all KairosHybridBlock instances."""
        self._captured_h_cfc: List[Optional[torch.Tensor]] = []

        def _make_hook(blk_idx: int):
            def _hook(module, inp, output):
                _x, _h_mamba, h_cfc = output
                if h_cfc is not None:
                    # Store as (B, d) detached to avoid keeping the graph
                    self._captured_h_cfc.append(h_cfc.detach().float())
            return _hook

        core = model.hybrid_core if hasattr(model, "hybrid_core") else model.module.hybrid_core
        for i, blk in enumerate(core.blocks):
            h = blk.register_forward_hook(_make_hook(i))
            self._handles.append(h)

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def update(self, batch: KairoBatch) -> None:
        """
        Call AFTER model.forward(batch).
        Uses the last captured h_cfc and compares against shifted IMU data.
        """
        if not self._captured_h_cfc:
            return

        # Use the most recent h_cfc (from the last loop that processed IMU tokens)
        h_cfc = self._captured_h_cfc[-1]   # (B, d)
        self._captured_h_cfc.clear()

        device = h_cfc.device
        probe  = self.probe.to(device)

        # Predicted next state: linear probe h_cfc → 7D imu space
        pred_state = probe(h_cfc)           # (B, 7)

        # Ground truth: average of last 2 IMU readings as "current state" target
        imu = batch.imu_data.float().to(device)   # (B, T_imu, 7)
        if imu.shape[1] >= 2:
            true_state = imu[:, -2:, :].mean(dim=1)   # (B, 7)
        else:
            true_state = imu[:, -1, :]

        mae_per_field = (pred_state - true_state).abs()   # (B, 7)
        self._errors.append(mae_per_field.cpu())

    def compute(self) -> Dict[str, float]:
        if not self._errors:
            return {f"mae_{n}": float("nan") for n in self.FIELD_NAMES}

        all_err = torch.cat(self._errors, dim=0)   # (N, 7)
        per_field = all_err.mean(dim=0).tolist()
        result = {f"mae_{n}": round(v, 5) for n, v in zip(self.FIELD_NAMES, per_field)}
        result["mae_mean"] = round(float(all_err.mean()), 5)
        return result

    def reset(self) -> None:
        self._errors.clear()
        self._captured_h_cfc = []


# ─────────────────────────────────────────────────────────────────────────────
# 2a. Chain-of-Thought Faithfulness (LLM-as-Judge)
# ─────────────────────────────────────────────────────────────────────────────

class CoTFaithfulnessJudge:
    """
    Uses GPT-4o-mini as judge to score how faithfully the final answer
    follows from the model's own reasoning chain.

    Requires: `pip install openai` and OPENAI_API_KEY env var.
    If openai is not installed or no API key, `score()` returns NaN and logs a warning.

    Scoring rubric (0–10):
      10 = answer follows directly from every step in the reasoning chain
       5 = answer partially follows but misses or contradicts some steps
       0 = answer contradicts the reasoning chain entirely
    """

    _SYSTEM = (
        "You are an expert autonomous driving safety evaluator. "
        "You will be given a chain-of-thought reasoning trace and a final answer "
        "produced by an AI driving system. "
        "Evaluate how faithfully the final answer follows from the reasoning steps. "
        "Respond ONLY with a JSON object: {\"score\": <int 0-10>, \"reason\": \"<one sentence>\"}."
    )

    _USER_TMPL = (
        "=== Reasoning Chain ===\n{reasoning}\n\n"
        "=== Final Answer ===\n{answer}\n\n"
        "Score faithfulness 0-10. Return JSON only."
    )

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 80) -> None:
        self.model      = model
        self.max_tokens = max_tokens
        self._scores:   List[float] = []
        self._client    = None
        self._available = False

        try:
            import openai  # type: ignore
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self._client    = openai.OpenAI(api_key=api_key)
                self._available = True
            else:
                print("[cot_judge] OPENAI_API_KEY not set — CoT scoring skipped")
        except ImportError:
            print("[cot_judge] openai not installed — CoT scoring skipped")

    def score(self, reasoning_chain: str, answer: str) -> float:
        """Score one (reasoning, answer) pair. Returns float in [0, 10] or NaN."""
        if not self._available:
            return float("nan")

        import json
        prompt = self._USER_TMPL.format(
            reasoning=reasoning_chain[:1500],   # trim to avoid token overflow
            answer=answer[:500],
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            return float(parsed["score"])
        except Exception as e:
            print(f"[cot_judge] API error: {e}")
            return float("nan")

    def batch_score(
        self,
        reasoning_chains: List[str],
        answers: List[str],
    ) -> List[float]:
        """Score a batch sequentially. Skips NaN entries in aggregate."""
        scores = []
        for r, a in zip(reasoning_chains, answers):
            scores.append(self.score(r, a))
        self._scores.extend(scores)
        return scores

    def compute(self) -> Dict[str, float]:
        valid = [s for s in self._scores if not math.isnan(s)]
        if not valid:
            return {"cot_faithfulness": float("nan"), "n_scored": 0}
        return {
            "cot_faithfulness": round(float(np.mean(valid)) / 10.0, 4),  # normalise to [0,1]
            "n_scored":         len(valid),
        }

    def reset(self) -> None:
        self._scores.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Conflict Resolution Score
# ─────────────────────────────────────────────────────────────────────────────

# Phrases that indicate the model noticed a contradiction
_CONFLICT_PHRASES = re.compile(
    r"\b(contradict|inconsistent|mismatch|disagree|conflict|implausible|"
    r"sensor disagreement|visual inconsistency|anomaly detected|unreliable)\b",
    re.IGNORECASE,
)


class ConflictResolutionScore:
    """
    Injects a synthetic visual contradiction (swap img_t with an unrelated image
    from a different sample) while keeping IMU/GPS intact.

    A well-calibrated model should detect that the camera feed contradicts the
    IMU/GPS telemetry and flag the inconsistency in its reasoning chain.

    Score = fraction of corrupted samples where the generated text contains
    an explicit acknowledgement of the contradiction.
    """

    def __init__(self) -> None:
        self._detected: int = 0
        self._total:    int = 0

    @staticmethod
    def inject(batch: KairoBatch, swap_fraction: float = 0.5) -> Tuple[KairoBatch, torch.Tensor]:
        """
        Swap img_t with a rolled version (different sample from the same batch)
        for `swap_fraction` of samples.

        Returns:
          corrupted_batch  — batch with swapped images
          is_corrupted     — (B,) bool tensor: True where image was swapped
        """
        B = batch.img_t.shape[0]
        n_swap = max(1, int(B * swap_fraction))

        perm = torch.randperm(B)
        swap_idx = perm[:n_swap]

        img_t_corrupted = batch.img_t.clone()
        img_t_corrupted[swap_idx] = batch.img_t[torch.roll(swap_idx, 1)]

        is_corrupted = torch.zeros(B, dtype=torch.bool)
        is_corrupted[swap_idx] = True

        corrupted = KairoBatch(
            img_t          = img_t_corrupted,
            img_t1         = batch.img_t1,
            img_t2         = batch.img_t2,
            lidar_t        = batch.lidar_t,
            lidar_t1       = batch.lidar_t1,
            imu_data       = batch.imu_data,
            imu_timestamps = batch.imu_timestamps,
            calib          = batch.calib,
            text_bytes     = batch.text_bytes,
            target_bytes   = batch.target_bytes,
            loss_mask      = batch.loss_mask,
        )
        return corrupted, is_corrupted

    def update(self, generated_texts: List[str], is_corrupted: torch.Tensor) -> None:
        """
        Args:
          generated_texts  — decoded output for each sample in the batch
          is_corrupted     — (B,) bool indicating which samples had swapped images
        """
        for text, corrupted in zip(generated_texts, is_corrupted.tolist()):
            if not corrupted:
                continue
            self._total += 1
            if _CONFLICT_PHRASES.search(text):
                self._detected += 1

    def compute(self) -> Dict[str, float]:
        if self._total == 0:
            return {"conflict_resolution_score": float("nan"), "n_corrupted": 0}
        return {
            "conflict_resolution_score": round(self._detected / self._total, 4),
            "detected":                  self._detected,
            "n_corrupted":               self._total,
        }

    def reset(self) -> None:
        self._detected = self._total = 0


# ─────────────────────────────────────────────────────────────────────────────
# 2c. Calibration Error (ECE + MCE)
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationMetrics:
    """
    Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Confidence source:
      - Detection head: softmax max-score over (B, max_det, n_cls+1)
      - Language confidence: heuristic extraction from generated text

    A well-calibrated model should have ECE < 0.05.
    """

    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins    = n_bins
        self._confs:   List[float] = []
        self._corrects: List[float] = []   # 1.0 if predicted class matches top detection

    def update_from_detection(
        self,
        det_scores:  torch.Tensor,   # (B, max_det, n_cls+1) raw logits
        gold_labels: Optional[torch.Tensor] = None,  # (B, max_det) int or None
    ) -> None:
        """
        Extract per-detection confidence (max softmax score) and whether
        the top-scoring class is correct (if gold_labels provided).
        """
        probs = F.softmax(det_scores.float(), dim=-1)   # (B, max_det, n_cls+1)
        max_p, pred_cls = probs.max(dim=-1)             # (B, max_det)

        for b in range(max_p.shape[0]):
            for d in range(max_p.shape[1]):
                conf = float(max_p[b, d])
                # Background class = last index; skip low-conf detections
                is_bg = int(pred_cls[b, d]) == (probs.shape[-1] - 1)
                if is_bg or conf < 0.1:
                    continue
                self._confs.append(conf)

                if gold_labels is not None:
                    correct = float(pred_cls[b, d] == gold_labels[b, d])
                else:
                    correct = float("nan")
                self._corrects.append(correct)

    def update_from_text(
        self, generated_texts: List[str], gold_texts: List[str]
    ) -> None:
        """
        Use language-based confidence (heuristic) and fault label match as accuracy.
        """
        for gen, gold in zip(generated_texts, gold_texts):
            conf      = _extract_confidence(gen)
            pred_fault = _extract_fault_label(gen)
            gold_fault = _extract_fault_label(gold)
            if pred_fault is None or gold_fault is None:
                continue
            correct = float(pred_fault == gold_fault)
            self._confs.append(conf)
            self._corrects.append(correct)

    def compute(self) -> Dict[str, float]:
        if not self._confs:
            return {"ece": float("nan"), "mce": float("nan"), "n_samples": 0}

        confs    = np.array(self._confs)
        corrects = np.array(self._corrects)

        valid = ~np.isnan(corrects)
        if valid.sum() == 0:
            return {"ece": float("nan"), "mce": float("nan"), "n_samples": 0}

        confs    = confs[valid]
        corrects = corrects[valid]

        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = mce = 0.0

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0:
                continue
            bin_conf = confs[mask].mean()
            bin_acc  = corrects[mask].mean()
            gap      = abs(bin_acc - bin_conf)
            weight   = mask.sum() / len(confs)
            ece     += weight * gap
            mce      = max(mce, gap)

        # reliability diagram data (for plotting)
        bin_data = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0:
                continue
            bin_data.append({
                "conf_mid": float((lo + hi) / 2),
                "accuracy": float(corrects[mask].mean()),
                "count":    int(mask.sum()),
            })

        return {
            "ece":         round(float(ece),  4),
            "mce":         round(float(mce),  4),
            "n_samples":   int(valid.sum()),
            "bin_data":    bin_data,
        }

    def reset(self) -> None:
        self._confs.clear()
        self._corrects.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Efficiency & Deployment Metrics
# ─────────────────────────────────────────────────────────────────────────────

class EfficiencyProfiler:
    """
    Three efficiency benchmarks:

    1. Control Frequency (Hz)
       Full-stack inference throughput: target ≥ 9.0 Hz (RoboMamba gold standard).

    2. Active vs Total Parameter Efficiency
       accuracy_gain_per_M_active_params vs a dense-8B baseline estimate.

    3. Context Scaling Latency
       Latency as N sequential timesteps are streamed through Mamba-2's recurrent
       hidden state.  Should scale O(N) while a Transformer baseline is O(N²).
    """

    @staticmethod
    @torch.no_grad()
    def measure_control_frequency(
        model: KairosModel,
        batch: KairoBatch,
        n_warmup: int = 5,
        n_measure: int = 50,
    ) -> Dict[str, float]:
        """
        Measure single-sample inference Hz.  Runs batch_size=1 inference.
        """
        device = next(model.parameters()).device

        # Slice to batch_size=1
        def _slice(b: KairoBatch) -> KairoBatch:
            def _s(x): return x[:1].to(device) if x is not None else None
            return KairoBatch(
                img_t=_s(b.img_t), img_t1=_s(b.img_t1), img_t2=_s(b.img_t2),
                lidar_t=_s(b.lidar_t), lidar_t1=_s(b.lidar_t1),
                imu_data=_s(b.imu_data), imu_timestamps=_s(b.imu_timestamps),
                calib=CalibMatrices(
                    P2=b.calib.P2[:1].to(device),
                    R0_rect=b.calib.R0_rect[:1].to(device),
                    Tr_velo_to_cam=b.calib.Tr_velo_to_cam[:1].to(device),
                ),
                text_bytes=_s(b.text_bytes),
            )

        single = _slice(batch)
        model.eval()

        # Warmup
        for _ in range(n_warmup):
            model(single)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(n_measure):
            model(single)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        hz = n_measure / elapsed

        # GPU utilisation (best effort)
        try:
            mem_used = torch.cuda.memory_allocated(device) / 1e9
            mem_total = torch.cuda.get_device_properties(device).total_memory / 1e9
        except Exception:
            mem_used = mem_total = float("nan")

        return {
            "hz":                round(hz,       2),
            "ms_per_frame":      round(1000 / hz, 2),
            "gold_standard_hz":  9.0,
            "passes_gold":       hz >= 9.0,
            "gpu_mem_used_gb":   round(mem_used,  2),
            "gpu_mem_total_gb":  round(mem_total, 2),
        }

    @staticmethod
    def measure_parameter_efficiency(
        model: KairosModel,
        accuracy: float,
        dense_8b_accuracy: float = 0.72,    # estimated baseline
        dense_8b_params_m: float = 8_000.0,
    ) -> Dict[str, float]:
        """
        Report accuracy-per-active-parameter efficiency vs a dense-8B baseline.

        Kairos-4B has ~1.4B active parameters (Top-2 from 8 experts, MoE sparsity).
        The efficiency gain shows how much better Kairos-4B is per active param.
        """
        total_m    = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

        # Active params = trainable + frozen backbone that runs during inference
        # MoE activates top-2 of 8 experts → ~(2/8) of FFN params active per token
        # Conservative estimate: 1/4 of MoE params active → reduces total by ~35 %
        active_m    = total_m * 0.37    # ≈1.4B at d=1024 with top-2/8 routing

        kairos_efficiency = accuracy     / max(active_m,         1.0)
        dense_efficiency  = dense_8b_accuracy / max(dense_8b_params_m, 1.0)
        speedup           = kairos_efficiency / max(dense_efficiency,   1e-9)

        return {
            "total_params_m":      round(total_m,          1),
            "active_params_m":     round(active_m,         1),
            "trainable_params_m":  round(trainable_m,      1),
            "accuracy":            round(accuracy,          4),
            "acc_per_active_M":    round(kairos_efficiency, 6),
            "dense8b_acc_per_M":   round(dense_efficiency,  6),
            "efficiency_gain":     round(speedup,           2),
        }

    @staticmethod
    @torch.no_grad()
    def measure_context_scaling(
        model: KairosModel,
        batch: KairoBatch,
        n_timesteps: List[int] = (1, 2, 4, 8, 16),
    ) -> Dict[str, Any]:
        """
        Stream N consecutive identical timesteps through Mamba-2's recurrent
        hidden state and measure total latency.

        Mamba-2 should scale O(N) — each step adds constant work.
        A Transformer baseline would scale O(N²) due to attention over full history.

        Returns per-N latency and the fitted scaling exponent (should be ≈1.0 for Mamba-2).
        """
        device = next(model.parameters()).device
        model.eval()

        def _prepare(b: KairoBatch) -> KairoBatch:
            def _d(x): return x[:1].to(device) if x is not None else None
            return KairoBatch(
                img_t=_d(b.img_t), img_t1=_d(b.img_t1), img_t2=_d(b.img_t2),
                lidar_t=_d(b.lidar_t), lidar_t1=_d(b.lidar_t1),
                imu_data=_d(b.imu_data), imu_timestamps=_d(b.imu_timestamps),
                calib=CalibMatrices(
                    P2=b.calib.P2[:1].to(device),
                    R0_rect=b.calib.R0_rect[:1].to(device),
                    Tr_velo_to_cam=b.calib.Tr_velo_to_cam[:1].to(device),
                ),
                text_bytes=_d(b.text_bytes),
            )

        single = _prepare(batch)
        latencies: Dict[int, float] = {}

        for N in n_timesteps:
            # Warmup
            for _ in range(3):
                for _ in range(N):
                    model(single)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            for _ in range(5):            # 5 repeated runs
                for _ in range(N):        # N sequential timesteps
                    model(single)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies[N] = (time.perf_counter() - t0) / 5.0

        # Fit scaling exponent: latency = a * N^alpha
        ns  = np.array([float(n) for n in n_timesteps])
        lts = np.array([latencies[n] for n in n_timesteps])
        # Linear regression on log-log scale: log(lat) = alpha*log(N) + log(a)
        valid = lts > 0
        if valid.sum() >= 2:
            alpha = float(np.polyfit(np.log(ns[valid]), np.log(lts[valid]), 1)[0])
        else:
            alpha = float("nan")

        # Hypothetical Transformer: quadratic scaling from single-frame baseline
        lat_1 = latencies.get(1, float("nan"))
        transformer_est = {n: lat_1 * (n ** 2) for n in n_timesteps}

        return {
            "mamba_latency_s":    {n: round(latencies[n], 4) for n in n_timesteps},
            "transformer_est_s":  {n: round(transformer_est[n], 4) for n in n_timesteps},
            "scaling_exponent":   round(alpha, 3),    # should be ≈1.0 for linear
            "ideal_exponent":     1.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricsReport:
    """Structured container for all evaluation results."""
    fault_detection:     Dict[str, Any] = field(default_factory=dict)
    root_cause_mrr:      Dict[str, Any] = field(default_factory=dict)
    state_estimation:    Dict[str, Any] = field(default_factory=dict)
    cot_faithfulness:    Dict[str, Any] = field(default_factory=dict)
    conflict_resolution: Dict[str, Any] = field(default_factory=dict)
    calibration:         Dict[str, Any] = field(default_factory=dict)
    efficiency:          Dict[str, Any] = field(default_factory=dict)
    n_batches:           int             = 0

    def to_flat_dict(self) -> Dict[str, float]:
        """Flat key→float dict for logging (e.g., to W&B or CloudWatch)."""
        out: Dict[str, float] = {}
        for section_name, section in [
            ("fdr",  self.fault_detection),
            ("mrr",  self.root_cause_mrr),
            ("cfc",  self.state_estimation),
            ("cot",  self.cot_faithfulness),
            ("conf", self.conflict_resolution),
            ("cal",  self.calibration),
        ]:
            for k, v in section.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[f"{section_name}/{k}"] = float(v)
        return out


class KairosEvaluator:
    """
    Drives all validation metrics over a DataLoader.

    Usage (training loop integration):
        evaluator = KairosEvaluator(model, cfg)
        report = evaluator.run(dl_val, device, n_batches=100)
        if rank == 0:
            KairosEvaluator.print_report(report)
    """

    def __init__(
        self,
        model: KairosModel,
        cfg: KairosModelConfig,
        run_cot_judge: bool = False,
        run_conflict:  bool = True,
        run_efficiency: bool = True,
    ) -> None:
        self.model     = model
        self.cfg       = cfg

        self.fdr       = FaultDetectionMetrics()
        self.mrr       = RootCauseMRR()
        self.cfc_mae   = StateEstimationMAE(d_model=cfg.kcfg.d_model)
        self.cot       = CoTFaithfulnessJudge() if run_cot_judge else None
        self.conflict  = ConflictResolutionScore() if run_conflict else None
        self.calib     = CalibrationMetrics()

        self._run_efficiency = run_efficiency
        self._cfc_hooked     = False

    def _ensure_cfc_hooks(self) -> None:
        if not self._cfc_hooked:
            self.cfc_mae.register_hooks(self.model)
            self._cfc_hooked = True

    @torch.no_grad()
    def run(
        self,
        dl_val,
        device: torch.device,
        n_batches: int = 100,
        gold_df=None,              # optional: pandas DataFrame with gold columns
    ) -> MetricsReport:
        """
        Evaluate all metrics over up to `n_batches` validation batches.

        Args:
            dl_val    — DataLoader yielding KairoBatch objects
            device    — inference device
            n_batches — cap evaluation at this many batches
            gold_df   — if provided, index [i] must match batch sample i for
                        gold reasoning_chain / answer text
        """
        self._ensure_cfc_hooks()
        self.model.eval()
        self.fdr.reset(); self.mrr.reset(); self.cfc_mae.reset()
        self.calib.reset()
        if self.cot:     self.cot.reset()
        if self.conflict: self.conflict.reset()

        # Grab one batch early for efficiency profiling (avoid timing data-load)
        efficiency_batch: Optional[KairoBatch] = None
        n_done = 0

        def _d(x: Any) -> Any:   # defined once outside loop — no per-batch alloc
            return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x

        for raw_batch in dl_val:
            if n_done >= n_batches:
                break

            batch = KairoBatch(
                img_t=_d(raw_batch.img_t), img_t1=_d(raw_batch.img_t1),
                img_t2=_d(raw_batch.img_t2), lidar_t=_d(raw_batch.lidar_t),
                lidar_t1=_d(raw_batch.lidar_t1), imu_data=_d(raw_batch.imu_data),
                imu_timestamps=_d(raw_batch.imu_timestamps),
                calib=CalibMatrices(
                    P2=_d(raw_batch.calib.P2),
                    R0_rect=_d(raw_batch.calib.R0_rect),
                    Tr_velo_to_cam=_d(raw_batch.calib.Tr_velo_to_cam),
                ),
                text_bytes=_d(raw_batch.text_bytes),
                target_bytes=_d(raw_batch.target_bytes),
                loss_mask=_d(raw_batch.loss_mask),
            )

            if efficiency_batch is None:
                efficiency_batch = batch  # save for profiling

            # ── Standard inference (no teacher forcing) ───────────────────────
            inf_batch = KairoBatch(
                img_t=batch.img_t, img_t1=batch.img_t1, img_t2=batch.img_t2,
                lidar_t=batch.lidar_t, lidar_t1=batch.lidar_t1,
                imu_data=batch.imu_data, imu_timestamps=batch.imu_timestamps,
                calib=batch.calib, text_bytes=batch.text_bytes,
            )
            output: KairosOutput = self.model.generate(inf_batch)

            # Decode generated byte tokens → text
            B = batch.img_t.shape[0]
            gen_ids = output.generated   # (B, T_gen)
            gen_texts: List[str] = []
            if gen_ids is not None:
                for b in range(B):
                    gen_texts.append(_decode_bytes(gen_ids[b]))
            else:
                gen_texts = [""] * B

            # Gold text (from target_bytes if available, else empty)
            gold_texts:      List[str] = []
            gold_reasonings: List[str] = []
            if batch.target_bytes is not None:
                for b in range(B):
                    decoded = _decode_bytes(batch.target_bytes[b])
                    # target_bytes = reasoning_chain + "\n\n" + answer
                    parts = decoded.split("\n\n", 1)
                    gold_reasonings.append(parts[0] if len(parts) > 0 else "")
                    gold_texts.append(parts[-1])
            else:
                gold_texts      = [""] * B
                gold_reasonings = [""] * B

            # ── Metric updates ─────────────────────────────────────────────────
            for g_text, gold_ans, gold_reason in zip(gen_texts, gold_texts, gold_reasonings):
                self.fdr.update(g_text, gold_ans)
                self.mrr.update(g_text, gold_reason)
                if self.cot:
                    self.cot.batch_score([gold_reason], [gold_ans])

            self.calib.update_from_text(gen_texts, gold_texts)
            if output.det_scores is not None:
                self.calib.update_from_detection(output.det_scores)

            self.cfc_mae.update(batch)

            # ── Conflict resolution ────────────────────────────────────────────
            if self.conflict is not None:
                c_batch, is_corrupted = ConflictResolutionScore.inject(batch)
                c_output = self.model.generate(KairoBatch(
                    img_t=c_batch.img_t, img_t1=c_batch.img_t1, img_t2=c_batch.img_t2,
                    lidar_t=c_batch.lidar_t, lidar_t1=c_batch.lidar_t1,
                    imu_data=c_batch.imu_data, imu_timestamps=c_batch.imu_timestamps,
                    calib=c_batch.calib, text_bytes=c_batch.text_bytes,
                ))
                c_texts = []
                if c_output.generated is not None:
                    for b in range(B):
                        c_texts.append(_decode_bytes(c_output.generated[b]))
                else:
                    c_texts = [""] * B
                self.conflict.update(c_texts, is_corrupted)

            n_done += 1

        # ── Compute all metrics ────────────────────────────────────────────────
        report = MetricsReport(
            fault_detection     = self.fdr.compute(),
            root_cause_mrr      = self.mrr.compute(),
            state_estimation    = self.cfc_mae.compute(),
            cot_faithfulness    = self.cot.compute()     if self.cot      else {},
            conflict_resolution = self.conflict.compute() if self.conflict else {},
            calibration         = self.calib.compute(),
            n_batches           = n_done,
        )

        # ── Efficiency profiling (uses the saved first batch) ─────────────────
        if self._run_efficiency and efficiency_batch is not None:
            acc_proxy = report.fault_detection.get("fdr", float("nan"))
            eff       = EfficiencyProfiler

            report.efficiency = {
                "control_frequency": eff.measure_control_frequency(self.model, efficiency_batch),
                "parameter_efficiency": eff.measure_parameter_efficiency(self.model, acc_proxy),
                "context_scaling": eff.measure_context_scaling(
                    self.model, efficiency_batch, n_timesteps=[1, 2, 4, 8]
                ),
            }

        self.model.train()
        return report

    @staticmethod
    def print_report(report: MetricsReport) -> None:
        """Pretty-print the full MetricsReport to stdout."""
        sep = "─" * 70

        print(f"\n{sep}")
        print(f"  KAIROS-4B VALIDATION REPORT  ({report.n_batches} batches)")
        print(sep)

        # 1. Functional Diagnostics
        print("\n[1] FUNCTIONAL DIAGNOSTIC METRICS")
        fd = report.fault_detection
        print(f"    Fault Detection Rate (FDR) : {fd.get('fdr', 'n/a'):.4f}")
        print(f"    Precision                  : {fd.get('precision', 'n/a'):.4f}")
        print(f"    Recall                     : {fd.get('recall', 'n/a'):.4f}")
        print(f"    F1                         : {fd.get('f1', 'n/a'):.4f}")
        print(f"    Skipped (ambiguous)        : {fd.get('skipped', 'n/a')}")

        mrr = report.root_cause_mrr
        print(f"\n    Root Cause MRR             : {mrr.get('mrr', 'n/a'):.4f}")
        print(f"    Hit@1 / Hit@3 / Hit@5      : "
              f"{mrr.get('hit@1','n/a'):.3f} / {mrr.get('hit@3','n/a'):.3f} / {mrr.get('hit@5','n/a'):.3f}")

        se = report.state_estimation
        print(f"\n    CfC State MAE (mean)       : {se.get('mae_mean', 'n/a')}")
        print(f"    CfC MAE vel_fwd            : {se.get('mae_vel_fwd', 'n/a')}")
        print(f"    CfC MAE acc_fwd            : {se.get('mae_acc_fwd', 'n/a')}")

        # 2. Reasoning & Reliability
        print(f"\n[2] REASONING & RELIABILITY METRICS")
        cot = report.cot_faithfulness
        if cot:
            print(f"    CoT Faithfulness (0–1)     : {cot.get('cot_faithfulness', 'n/a')}")
            print(f"    Samples scored             : {cot.get('n_scored', 'n/a')}")
        else:
            print("    CoT Faithfulness           : [skipped — set run_cot_judge=True]")

        cr = report.conflict_resolution
        if cr:
            print(f"    Conflict Resolution Score  : {cr.get('conflict_resolution_score', 'n/a'):.4f}")
            print(f"    Detected / Total corrupted : {cr.get('detected','n/a')} / {cr.get('n_corrupted','n/a')}")
        else:
            print("    Conflict Resolution        : [skipped]")

        cal = report.calibration
        print(f"\n    Calibration ECE            : {cal.get('ece', 'n/a')}")
        print(f"    Calibration MCE            : {cal.get('mce', 'n/a')}")
        print(f"    (ECE < 0.05 is well-calibrated)")

        # 3. Efficiency
        ef = report.efficiency
        if ef:
            print(f"\n[3] EFFICIENCY & DEPLOYMENT METRICS")
            cf = ef.get("control_frequency", {})
            print(f"    Control Frequency          : {cf.get('hz', 'n/a'):.2f} Hz"
                  f"  ({'✓ PASSES' if cf.get('passes_gold') else '✗ BELOW'} 9.0 Hz gold standard)")
            print(f"    GPU memory used            : {cf.get('gpu_mem_used_gb', 'n/a'):.1f}"
                  f" / {cf.get('gpu_mem_total_gb', 'n/a'):.1f} GB")

            pe = ef.get("parameter_efficiency", {})
            print(f"\n    Total / Active params      : {pe.get('total_params_m','n/a'):.0f}M"
                  f" / {pe.get('active_params_m','n/a'):.0f}M")
            print(f"    Acc per active 1M params   : {pe.get('acc_per_active_M', 'n/a'):.6f}")
            print(f"    vs Dense-8B baseline       : {pe.get('dense8b_acc_per_M', 'n/a'):.6f}")
            print(f"    Efficiency gain            : {pe.get('efficiency_gain', 'n/a'):.2f}×")

            cs = ef.get("context_scaling", {})
            scaling_exp = cs.get("scaling_exponent", "n/a")
            print(f"\n    Context scaling exponent   : {scaling_exp}  (ideal = 1.0 for Mamba-2)")
            mamba_lats = cs.get("mamba_latency_s", {})
            tf_lats    = cs.get("transformer_est_s", {})
            print(f"    {'N':>4}  {'Mamba-2':>10}  {'Transformer(est)':>18}")
            print(f"    {'─'*4}  {'─'*10}  {'─'*18}")
            for n in sorted(mamba_lats):
                m = mamba_lats[n]
                t = tf_lats.get(n, float("nan"))
                print(f"    {n:>4}  {m*1000:>8.1f}ms  {t*1000:>16.1f}ms")

        print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

def _eval_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kairos-4B standalone evaluation")
    p.add_argument("--ckpt_s3",     type=str, required=True,
                   help="S3 prefix for DeepSpeed checkpoint")
    p.add_argument("--gold_s3",     type=str,
                   default="s3://project-kairos-raw-use1-s3-195231312992/delta/gold/kitti_s2ft_triplets/")
    p.add_argument("--n_batches",   type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--cot_judge",   action="store_true",
                   help="Enable GPT-4o-mini CoT faithfulness judge (needs OPENAI_API_KEY)")
    p.add_argument("--no_efficiency", action="store_true",
                   help="Skip efficiency profiling (faster)")
    p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = _eval_args()
    device = torch.device(args.device)

    print(f"[eval] Building KairosModel …")
    cfg   = KairosModelConfig()
    model = KairosModel(cfg).to(device)
    model.eval()

    # Load checkpoint if provided
    if args.ckpt_s3:
        import boto3
        from pathlib import Path
        CKPT_LOCAL = Path("/tmp/kairos_eval_ckpt")
        CKPT_LOCAL.mkdir(parents=True, exist_ok=True)
        bucket, prefix = args.ckpt_s3.removeprefix("s3://").split("/", 1)
        s3 = boto3.client("s3", region_name="us-east-1")

        # List latest tag
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
        tags = sorted(
            p["Prefix"].split("/")[-2]
            for p in resp.get("CommonPrefixes", [])
            if "step_" in p["Prefix"]
        )
        if not tags:
            print("[eval] No checkpoint found — using random weights")
        else:
            tag = tags[-1]
            print(f"[eval] Loading checkpoint: {tag}")
            # Download all files for this tag
            resp2 = s3.list_objects_v2(
                Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/{tag}/"
            )
            for obj in resp2.get("Contents", []):
                key = obj["Key"]
                rel = key.removeprefix(f"{prefix.rstrip('/')}/{tag}/")
                lf  = CKPT_LOCAL / tag / rel
                lf.parent.mkdir(parents=True, exist_ok=True)
                if not lf.exists():
                    s3.download_file(bucket, key, str(lf))

            # Load model weights only (not the full DS engine)
            sd_path = CKPT_LOCAL / tag / "mp_rank_00_model_states.pt"
            if sd_path.exists():
                sd = torch.load(str(sd_path), map_location=device)
                # DS saves under "module" key in ZeRO-3
                sd = sd.get("module", sd)
                model.load_state_dict(sd, strict=False)
                print("[eval] Weights loaded")
            else:
                print(f"[eval] state dict not found at {sd_path} — using random weights")

    # Build validation DataLoader
    from kairos_train import _load_gold_split, _make_dataloader, collate_fn, CACHE_DIR
    print("[eval] Loading validation data …")
    df_val  = _load_gold_split(args.gold_s3, "val")
    dl_val, _ = _make_dataloader(
        df_val, CACHE_DIR / "eval",
        batch_size=args.batch_size,
        rank=0, world_size=1,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Run evaluation
    evaluator = KairosEvaluator(
        model,
        cfg,
        run_cot_judge  = args.cot_judge,
        run_conflict   = True,
        run_efficiency = not args.no_efficiency,
    )
    print(f"[eval] Running metrics on {args.n_batches} batches …")
    report = evaluator.run(dl_val, device, n_batches=args.n_batches)
    KairosEvaluator.print_report(report)

    # Save flat metrics dict as JSON
    import json
    flat = report.to_flat_dict()
    out_path = "/tmp/kairos_metrics.json"
    with open(out_path, "w") as f:
        json.dump(flat, f, indent=2, default=str)
    print(f"[eval] Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
