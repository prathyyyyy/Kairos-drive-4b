# KAIROS

**A 4B-parameter linear-complexity hybrid SSM model for multimodal driving-scene reasoning.**

---

## Intro

KAIROS is a research project that trains a 4-billion-parameter multimodal model to read camera, LiDAR, and IMU inputs from real driving logs and produce structured, step-by-step scene reasoning as byte-level text. The full lifecycle was built end to end: data pipeline, model architecture, distributed training, and a sharded multi-GPU evaluation harness - with every reported number traceable to a logged training or evaluation artifact.

## Why KAIROS is different

Most multimodal models scale quadratic attention over longer contexts. KAIROS takes a different path:

- **Linear-complexity core.** The backbone is built from looped Mamba-2 / selective SSM blocks rather than full attention, with sliding-window attention used only as a residual pathway.
- **Weight-shared looped depth.** Three unique hybrid blocks are reused across four loop iterations, scaling effective depth without scaling parameter count - and a Mixture-of-Experts FFN (top-2 of 64 experts) keeps active compute at roughly 1.4B of the 4B total parameters.
- **Time-aware sensor fusion.** IMU streams flow through closed-form continuous-time (CfC) cells with input-dependent time constants, and a calibration-aware gate projects LiDAR into camera space before learned per-token modality weighting.
- **Structured sparse fine-tuning (S2FT).** The model is trained on reasoning-chain triplets (system prompt, user prompt, chain-of-thought, answer) with a byte-level decoder - no subword tokenizer to maintain.

## Architecture

> 4B-parameter linear-complexity hybrid SSM model with looped Mamba-2 blocks, attention residuals, CfC temporal fusion, and structured sparse fine-tuning.

| Component | Detail |
|---|---|
| Hybrid core | 3 unique blocks x 4 weight-shared loops, d = 1024 |
| Sub-block 1 | Mamba-2 selective SSM (input-dependent A, B, C) |
| Sub-block 2 | CfC / liquid cell on IMU tokens (closed-form, no ODE solver) |
| Sub-block 3 | Sliding-window attention residual (GQA, RoPE, window 1024) |
| Sub-block 4 | MoE SwiGLU FFN, top-2 of 64 experts, aux-loss-free balancing |
| Vision | DINOv2-L + LoRA, 3-frame temporal fusion |
| LiDAR | PointMamba-style SSM encoder |
| Fusion | Calibration-aware gate (LiDAR-to-camera projection + learned modality weighting) |
| Decoder | S2FT byte-level reasoning decoder (vocab 258) |
| Scale | ~4.0B total parameters, ~1.4B active per forward pass |

<img width="1536" height="1024" alt="architecture" src="https://github.com/user-attachments/assets/db160bbf-d594-4ea0-9e9e-37017f105854" />
<img width="1536" height="1024" alt="encoders" src="https://github.com/user-attachments/assets/89c6970a-6484-4be7-b3d8-cd2d11de70cd" />
<img width="1536" height="1024" alt="fusion" src="https://github.com/user-attachments/assets/2b3cc252-74f3-400b-b84a-527348caa90c" />
<img width="1536" height="1024" alt="hybrid-core" src="https://github.com/user-attachments/assets/b8d8cbf4-9412-4c56-9ed8-52795290f2e8" />
<img width="1536" height="1024" alt="outputs" src="https://github.com/user-attachments/assets/d035d225-be72-4018-88dc-e5d027645bf2" />





## Training summary

- **8,000 optimization steps** on a multimodal camera + LiDAR + IMU reasoning dataset (~100k S2FT triplets, walk-forward temporal split), BF16 with DeepSpeed ZeRO-3 on an 8-GPU NVIDIA L4 instance (SageMaker ml.g6.48xlarge, spot).
- Final training loss: **total_loss 0.9357**, **s2ft_loss 0.9042**.
- Final validation: **val_s2ft 0.9158** (val_total 0.9158).
- Zero skipped bad rows; checkpointed to S3 as ZeRO-3 shards (8 model + 8 optimizer).

## Inference and evaluation

A held-out evaluation over **4,096 validation samples** ran on a 4x L4 instance (SageMaker ml.g6.12xlarge):

- **0 failed samples** out of 4,096; all 1,036 checkpoint tensors loaded per worker with no missing, unexpected, or shape-mismatched tensors.
- **S2FT eval loss 0.9167** (teacher-forced, same convention as training validation) - within **+0.0009** of the final training val_s2ft of 0.9158, indicating the deployed checkpoint reproduces training-time validation behavior.
- **98.76% byte-token accuracy**; **65.33%** of samples reach near-exact (>= 99%) byte-token accuracy.
- Optimized SageMaker inference with 4-GPU sharded evaluation, reducing eval latency to **0.542 sec/sample**: the parent process consolidates the ZeRO-3 checkpoint once, four workers evaluate deterministic validation shards, and the parent merges predictions and slice metrics.

<img width="1520" height="715" alt="newplot(1)" src="https://github.com/user-attachments/assets/9cd6ea20-5717-490b-ab95-cfd3d1f91280" />
<img width="1520" height="715" alt="newplot(7)" src="https://github.com/user-attachments/assets/1c18ca54-7ce6-4dfa-9df1-514876e12d5d" />
<img width="1520" height="715" alt="newplot(8)" src="https://github.com/user-attachments/assets/01ba4572-c583-4afb-a166-f24c88b8c1d3" />
<img width="1520" height="715" alt="newplot(9)" src="https://github.com/user-attachments/assets/8ad7fc0b-3c04-4092-b507-255794a92edc" />



## Results

| Metric | Value |
|---|---|
| Training steps completed | 8,000 / 8,000 |
| Final training s2ft_loss | 0.9042 |
| Final validation val_s2ft | 0.9158 |
| Eval samples (held-out val) | 4,096 |
| Failed samples | 0 |
| S2FT eval loss (mean) | 0.9167 |
| Eval minus training val_s2ft | +0.0009 |
| Byte-token accuracy (mean) | 98.76% |
| Near-exact rate (>= 99% byte-token accuracy) | 65.33% |
| High-loss rate (s2ft > 2.0) | 0.0% |
| Eval throughput (4-GPU sharded) | 0.542 s/sample |

## Tech stack

PyTorch - DeepSpeed ZeRO-3 - AWS SageMaker (training: ml.g6.48xlarge, inference/eval: ml.g6.12xlarge, NVIDIA L4) - Plotly - Python / pandas - JSONL + CSV reporting

## Outro

KAIROS demonstrates that a linear-complexity hybrid SSM with looped weight sharing and sparse expert routing can be trained end to end on real multimodal driving logs, evaluated reproducibly at scale, and reported honestly.
