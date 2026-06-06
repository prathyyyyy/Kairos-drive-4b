# KAIROS Motion

> Multimodal autonomous driving research for temporal scene understanding, sensor fusion, motion-aware reasoning, and grounded multimodal learning using camera, LiDAR, OXTS/IMU, vehicle-state, radar, and calibration data.

---

## Overview

**KAIROS Motion** is a multimodal autonomous driving research project focused on understanding dynamic driving scenes over time.

The project combines **camera images**, **LiDAR point clouds**, **OXTS/IMU signals**, **vehicle-state information**, **radar data**, and **calibration metadata** to build a training pipeline for motion-aware scene reasoning.

Unlike single-frame perception systems, KAIROS Motion is designed around **temporal context**. The system is built to reason over multiple time steps, understand ego-motion, interpret moving objects, align different sensor streams, and produce outputs that are grounded in real driving data.

The current data foundation includes:

- **KITTI Raw**
- **Zenseact Open Dataset (ZOD)** selected subsets

The current model direction uses a custom architecture built around:

- Looped sequence processing
- Mamba-style state-space temporal modeling
- Attention-based cross-modal fusion
- Residual feature refinement
- Multimodal temporal reasoning

---

## Purpose

The purpose of KAIROS Motion is to build a research-grade autonomous driving intelligence pipeline that can:

- Understand motion across time
- Fuse multiple sensor modalities
- Learn from real-world autonomous driving data
- Ground outputs in verified sensor evidence
- Reduce hallucinated scene reasoning
- Support temporal driving-scene understanding
- Train efficiently under limited AWS credits
- Compare KITTI-only training against KITTI + ZOD training
- Provide a foundation for future multimodal autonomous-driving research

This project is not a vehicle-control system. It is a research and experimentation pipeline for multimodal perception, temporal reasoning, and autonomous-driving dataset preparation.

---

## Problem Statement

Autonomous driving models often struggle when they rely only on isolated camera frames or a single sensor stream.

Real driving requires understanding:

- How nearby vehicles are moving
- Whether objects are approaching or moving away
- How the ego vehicle is changing position
- How LiDAR geometry aligns with camera images
- How temporal context changes scene interpretation
- How vehicle-state and OXTS/IMU signals influence perception
- How radar, LiDAR, and camera data complement each other
- How to avoid hallucinated outputs when sensor evidence is incomplete

KAIROS Motion addresses these challenges by building a multimodal and temporal training pipeline using verified autonomous-driving datasets.

---

## Project Vision

The long-term vision of KAIROS Motion is to create a multimodal autonomous-driving reasoning system that can interpret dynamic scenes using sensor-grounded temporal context.

The system should eventually be able to:

- Read a sequence of driving frames
- Understand ego-motion and object motion
- Use LiDAR for 3D spatial grounding
- Use OXTS/IMU and vehicle-state data for motion context
- Align camera, LiDAR, radar, and calibration data
- Produce grounded driving-scene reasoning
- Improve generalization across datasets
- Reduce hallucination through multimodal sensor verification
- Support future research in temporal autonomous-driving intelligence

---

## Core Capabilities

### 1. Temporal Scene Understanding

KAIROS Motion is designed to reason over multiple frames instead of a single frame.

Example temporal context:

```text
t-2 -> t-1 -> t
```

This enables the model to understand:

- Motion continuity
- Object persistence
- Ego-vehicle movement
- Scene transitions
- Dynamic object behavior
- Short-term temporal causality

---

### 2. Multimodal Sensor Fusion

The project combines several autonomous-driving sensor streams.

| Modality | Purpose |
|---|---|
| Camera images | Visual scene understanding |
| LiDAR point clouds | 3D spatial structure and depth grounding |
| OXTS/IMU | Ego-motion, pose, and motion context |
| Vehicle data | Speed, control, and vehicle-state context |
| Radar | Range and motion signals where available |
| Calibration files | Sensor alignment and coordinate transformation |
| Temporal frames | Motion reasoning across time |

---

### 3. Motion-Aware Reasoning

The model is designed to understand how the world changes between frames.

This includes:

- Ego-motion reasoning
- Relative object movement
- Temporal consistency
- Dynamic scene interpretation
- Motion-aware prompt/answer supervision

---

### 4. Grounded Outputs

The project emphasizes grounding model outputs in verified sensor paths and real data.

This is important because autonomous-driving models should not hallucinate:

- Objects
- Road structure
- Vehicle motion
- Pedestrian movement
- Scene conditions
- Spatial relationships

---

### 5. Budget-Aware Cloud Training

The project is developed under limited AWS credits, so the workflow prioritizes:

- Preflight checks before training
- Tiny sanity runs before long jobs
- Spot GPU training where available
- S3-native storage
- Checkpoint resume support
- Immediate cleanup of temporary EC2 and EBS resources

---

## System Architecture

High-level architecture:

```text
                  +-----------------------------+
                  |         Raw Datasets         |
                  |    KITTI Raw + ZOD Subsets   |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  |        Data Ingestion        |
                  |  download -> extract -> S3   |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  |         AWS S3 Storage       |
                  |  raw + gold + checkpoints    |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  |     Preflight Validation     |
                  |  schema + path + object chk  |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  |    Multimodal Dataloader     |
                  | camera + lidar + oxts + etc  |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  |      KAIROS Motion Model     |
                  | mamba + attention + residual |
                  +--------------+--------------+
                                 |
                                 v
                  +-----------------------------+
                  |     Evaluation + Outputs     |
                  | checkpoints + predictions    |
                  +-----------------------------+
```

---

## Model Architecture

KAIROS Motion uses a custom multimodal temporal architecture designed for autonomous-driving scene reasoning.

The architecture is built around four main ideas:

1. **Looped temporal processing**
2. **Mamba-style sequence modeling**
3. **Attention-based multimodal fusion**
4. **Residual refinement blocks**

The model is designed to process multimodal driving context across time and refine representations through repeated temporal-fusion loops.

---

## Architecture Diagram

```text
Input Sample
├── Image at t
├── Image at t-1
├── Image at t-2
├── LiDAR at t
├── LiDAR at t-1
├── OXTS/IMU data
├── Vehicle-state data
├── Calibration metadata
└── Prompt / supervision text
        |
        v
+-----------------------------+
|      Modality Encoders      |
|-----------------------------|
| Vision Encoder              |
| LiDAR Encoder               |
| OXTS / IMU Encoder          |
| Vehicle-State Encoder       |
| Calibration Encoder         |
+-------------+---------------+
              |
              v
+-----------------------------+
|   Temporal Token Builder    |
|-----------------------------|
| Builds ordered tokens from  |
| t-2, t-1, and t sensor data |
+-------------+---------------+
              |
              v
+-----------------------------+
|   Looped Mamba Blocks       |
|-----------------------------|
| State-space temporal mixing |
| Efficient long-context flow |
| Motion-aware sequence state |
+-------------+---------------+
              |
              v
+-----------------------------+
|   Cross-Modal Attention     |
|-----------------------------|
| Camera <-> LiDAR            |
| Motion <-> Visual context   |
| Ego-state <-> Scene tokens  |
+-------------+---------------+
              |
              v
+-----------------------------+
|   Residual Refinement       |
|-----------------------------|
| Skip connections            |
| Layer normalization         |
| Stable feature updates      |
+-------------+---------------+
              |
              v
+-----------------------------+
|   Reasoning / Output Head   |
|-----------------------------|
| Grounded scene answer       |
| Motion reasoning output     |
| Training loss computation   |
+-----------------------------+
```
<img width="1536" height="1024" alt="architecture" src="https://github.com/user-attachments/assets/56b05bb3-4bca-4278-979a-e646088822d1" />


---

## Model Components

### 1. Vision Encoder

The vision encoder processes camera frames from the current and previous time steps.

Inputs:

```text
image_t
image_t_minus_1
image_t_minus_2
```

Purpose:

- Extract visual scene features
- Preserve temporal image context
- Support object and lane-level visual grounding
- Provide image tokens for cross-modal fusion

<img width="1536" height="1024" alt="encoders" src="https://github.com/user-attachments/assets/b8299d17-df77-483d-bf3a-48795db04614" />

---

### 2. LiDAR Encoder

The LiDAR encoder processes point-cloud or projected LiDAR-derived features.

Inputs:

```text
lidar_t
lidar_t_minus_1
```

Purpose:

- Add 3D spatial context
- Ground visual objects in geometry
- Support depth-aware reasoning
- Improve robustness over image-only understanding

<img width="1536" height="1024" alt="fusion" src="https://github.com/user-attachments/assets/7089c116-4275-46a8-8b47-fee64afcba19" />

---

### 3. OXTS / IMU Encoder

The OXTS/IMU encoder processes ego-motion and vehicle-pose signals.

Purpose:

- Encode ego-vehicle motion
- Provide pose and movement context
- Help distinguish object motion from ego-motion
- Support temporal scene reasoning

---

### 4. Vehicle-State Encoder

The vehicle-state encoder is designed to process vehicle-level signals such as speed, steering, or motion state when available.

Purpose:

- Condition the model on ego-vehicle state
- Improve motion interpretation
- Add driving-context awareness

---

### 5. Calibration Encoder

Calibration metadata is used to support sensor alignment.

Purpose:

- Preserve camera/LiDAR alignment context
- Support coordinate-aware fusion
- Reduce mismatch between 2D and 3D sensor streams

---

## Looped Mamba Temporal Blocks

A central part of KAIROS Motion is the use of **looped Mamba-style temporal processing**.

Mamba-style state-space modeling is useful for sequential data because it can model temporal dependencies efficiently without relying only on full quadratic attention.

In KAIROS Motion, Mamba-style blocks are used to:

- Process ordered temporal tokens
- Carry motion state across frames
- Model changes between `t-2`, `t-1`, and `t`
- Improve long-range temporal flow
- Support efficient sequence reasoning

The looped design allows repeated refinement:

```text
Temporal tokens
      |
      v
Mamba block
      |
      v
Residual update
      |
      v
Mamba block
      |
      v
Residual update
      |
      v
Refined temporal representation
```

This repeated loop helps the model progressively improve its understanding of motion and scene dynamics.

---

## Attention-Based Fusion

KAIROS Motion also uses attention mechanisms to fuse information across modalities.

Attention is used to connect:

- Camera features with LiDAR features
- Visual context with motion context
- Ego-state tokens with scene tokens
- Temporal tokens with current-frame features

Example fusion flow:

```text
Camera tokens  --------+
                       |
LiDAR tokens   --------+--> Cross-modal attention --> fused scene tokens
                       |
OXTS tokens    --------+
                       |
Vehicle tokens --------+
```

The goal is to allow each modality to contribute useful information without forcing all sensors into a single representation too early.

---

## Residual Refinement

Residual connections are used throughout the model to stabilize training and preserve useful representations.

Residual refinement helps:

- Improve gradient flow
- Prevent feature degradation
- Preserve modality-specific information
- Support repeated looped processing
- Stabilize deeper temporal fusion

General pattern:

```text
x -> block(x) -> residual add -> normalization -> refined x
```

This allows the model to learn incremental updates rather than replacing representations completely at each layer.

---

## Why Mamba + Attention?

The architecture combines Mamba-style temporal modeling and attention-based fusion because each has different strengths.

| Component | Strength |
|---|---|
| Mamba-style blocks | Efficient temporal sequence modeling |
| Attention | Flexible cross-modal interaction |
| Residual layers | Stable deep refinement |
| Looped processing | Iterative motion reasoning |

Together, they support a model that can reason over time while still fusing heterogeneous sensor streams.

<img width="1536" height="1024" alt="hybrid-core" src="https://github.com/user-attachments/assets/4e25a5c7-dccb-4901-ae39-d4bb86ac86ff" />
<img width="1536" height="1024" alt="outputs" src="https://github.com/user-attachments/assets/adf1da64-65d0-4f01-a487-8d1afe167a3b" />

---

## Data Architecture

KAIROS Motion uses a layered data architecture.

```text
S3 Bucket
├── Raw Data
│   ├── KITTI Raw
│   └── ZOD Raw Extracted Subsets
│
├── Gold Data
│   └── KITTI temporal triplet training rows
│
├── Metadata
│   ├── calibration references
│   ├── modality availability
│   └── dataset split information
│
└── Training Artifacts
    ├── checkpoints
    ├── logs
    └── generated outputs
```

---

## Dataset Sources

## KITTI Raw

KITTI Raw is used as the primary base dataset.

Current KITTI components include:

- Camera images
- LiDAR point clouds
- Calibration files
- OXTS/IMU files
- Temporal frame references
- Gold triplet training samples

Gold split structure:

```text
delta/gold/kitti_s2ft_triplets/
├── dataset_split=train/
├── dataset_split=val/
└── dataset_split=test/
```

---

## Zenseact Open Dataset

Selected ZOD subsets were added to improve diversity and multimodal coverage.

ZOD location:

```text
s3://project-kairos-raw-eun1-s3-412906648430/raw/zod/
```

Included ZOD layout:

```text
raw/zod/
├── drives/
│   ├── infos/
│   ├── oxts/
│   ├── vehicle_data/
│   ├── radar_front/
│   ├── mini/
│   ├── front_blur/
│   └── lidar_velodyne/
│
├── minis/
│   ├── sequences_mini/
│   └── frames_mini/
│
└── sequences/
    ├── infos/
    ├── oxts/
    ├── vehicle_data/
    ├── mini/
    ├── images_blur_000000_000490/
    └── lidar_velodyne_000000_000039/
```

Final selected ZOD subset:

```text
Objects: 235,574
Size:    436.2 GiB
```

---

## Current Storage Footprint

Approximate current dataset storage:

```text
KITTI: ~183 GiB
ZOD:   ~436.2 GiB
Total: ~619 GiB
```

Primary data region:

```text
eu-north-1
Stockholm
```

---

## Preflight Verification

Before launching any GPU training job, run a preflight check.

```bash
python kairos_preflight.py --n_rows 10
```

The preflight verifies:

- Environment variables
- S3 connectivity
- Dataset split availability
- Required parquet columns
- Path consistency
- Stale account IDs
- Image paths
- LiDAR paths
- Calibration paths
- OXTS path availability or derivation
- S3 `HeadObject` availability

Expected successful output includes:

```text
[ OK ] GOLD_S3
[ OK ] DATA_REGION
[ OK ] PyArrow S3FileSystem created
[ OK ] Found parquet files
[ OK ] Loaded training rows
[ OK ] Loaded validation rows
[ OK ] Required columns present
[ OK ] HeadObject checks passed
```

---

## OXTS Compatibility

Some gold data rows do not include a stored `oxts_path`.

KAIROS Motion supports deriving OXTS paths from image paths.

Example image path:

```text
.../image_02/data/0000000002.png
```

Derived OXTS path:

```text
.../oxts/data/0000000002.txt
```

This allows older gold data to remain usable without fully rebuilding the dataset.

---

## Training Strategy

Training is intentionally staged to avoid wasting GPU credits.

### Stage 1: Preflight Validation

Before any GPU training, the system verifies:

- S3 bucket access
- Dataset paths
- Parquet availability
- Required columns
- Image object availability
- LiDAR object availability
- Calibration object availability
- OXTS availability or derivability

---

### Stage 2: Tiny Layer Sanity Check

Instead of immediately running hundreds of steps, the project first runs a very small test.

Purpose:

```text
Verify every model layer receives valid tensors
Verify forward pass works
Verify loss is produced
Verify dataloader works
Verify multimodal samples are correctly formed
Verify model outputs can be generated
```

---

### Stage 3: Short Functional Training Run

After the tiny sanity check, a short training job validates:

- Training loop stability
- GPU memory usage
- Checkpoint writing
- Validation loop
- Output generation
- S3 checkpoint storage

---

### Stage 4: Full Training

Full training begins only after data, model, and checkpoint paths are verified.

Planned full training approach:

- Prefer SageMaker managed spot training
- Use checkpoint resume
- Track validation metrics
- Save outputs to S3
- Compare KITTI-only vs KITTI + ZOD
- Maximize accuracy within AWS credit budget

---

## Current Project Status

Completed:

- KITTI Raw downloaded
- KITTI gold data prepared
- KITTI train/validation/test splits created
- KITTI preflight checks passed
- ZOD subset downloaded
- ZOD subset extracted
- ZOD subset uploaded to S3
- ZOD final S3 verification completed
- Large ZOD LiDAR drive archive completed
- Temporary EC2 ingestion disk cleaned
- SageMaker quota checks started

---

## Suggested Repository Structure

```text
project-kairos/
├── README.md
├── CLAUDE.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── data/
│   ├── model/
│   ├── training/
│   └── aws/
│
├── scripts/
│   ├── aws/
│   ├── data_ingest/
│   ├── preflight/
│   ├── training/
│   └── utilities/
│
├── src/
│   ├── data/
│   │   ├── datasets/
│   │   ├── transforms/
│   │   └── loaders/
│   │
│   ├── models/
│   │   ├── encoders/
│   │   ├── fusion/
│   │   └── heads/
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── checkpointing.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── qualitative.py
│   │
│   └── utils/
│       ├── s3.py
│       ├── logging.py
│       └── paths.py
│
├── notebooks/
│   ├── exploration/
│   └── visualization/
│
├── tests/
│   ├── test_dataset.py
│   ├── test_preflight.py
│   └── test_training_step.py
│
└── docs/
    ├── architecture.md
    ├── data_pipeline.md
    └── training_plan.md
```

---

## Key Scripts

Important scripts expected in this project:

```text
kairos_preflight.py
kairos_train.py
check_stockholm_g6_quotas.ps1
request_stockholm_g6_missing.ps1
launch_zod_direct_ec2.ps1
zod_parallel_ingest.sh
turbo_drives_lidar.sh
```

---

## Cost-Aware Development

KAIROS Motion is designed to operate within limited AWS credits.

Primary cost drivers:

- S3 storage
- SageMaker GPU training
- Temporary EC2 ingestion jobs
- Temporary EBS volumes
- S3 request volume

Cost-control practices:

- Use S3 for long-term storage
- Avoid keeping extracted archives on EC2
- Delete temporary EBS volumes
- Terminate EC2 immediately after ingestion
- Use SageMaker spot training when approved
- Run tiny sanity checks before full training
- Use checkpoints to resume interrupted jobs
- Avoid long failed GPU runs

---

## Roadmap

### Phase 1: Data Foundation

- Prepare KITTI Raw
- Build KITTI gold triplets
- Verify gold parquet data
- Add ZOD subset
- Upload datasets to S3
- Confirm storage layout

Status: mostly complete.

---

### Phase 2: Pipeline Validation

- Verify dataloader
- Verify OXTS derivation
- Verify LiDAR loading
- Verify calibration loading
- Run 5-step sanity check
- Inspect first model outputs

Status: next step.

---

### Phase 3: Training Launch

- Confirm SageMaker GPU quota
- Launch small spot training job
- Save checkpoint to S3
- Validate output generation
- Monitor cost

Status: pending quota approval.

---

### Phase 4: Multimodal Improvement

- Add stronger fusion layers
- Add ZOD training support
- Improve temporal modeling
- Evaluate dataset mixing strategies

Status: planned.

---

### Phase 5: Research Evaluation

- Compare model variants
- Evaluate hallucination reduction
- Analyze KITTI-only vs KITTI + ZOD
- Prepare experiment reports

Status: planned.

---

## Future Updates

Planned future work:

### Data Pipeline

- Build a unified KITTI + ZOD metadata index
- Convert ZOD samples into the gold training format
- Add dataset balancing between KITTI and ZOD
- Add modality availability flags
- Add sample-level quality checks
- Add dataset manifest versioning

### Model Architecture

- Add stronger temporal fusion layers
- Add LiDAR feature encoder
- Add OXTS/IMU motion encoder
- Add vehicle-state encoder
- Add calibration-aware sensor alignment
- Add cross-modal attention
- Add trajectory-aware reasoning modules
- Improve looped Mamba temporal blocks
- Experiment with deeper residual fusion stages

### Training

- Run 5-step layer sanity check
- Run short SageMaker spot training job
- Add checkpoint resume support
- Add validation metrics
- Add qualitative output inspection
- Compare KITTI-only vs KITTI + ZOD training
- Tune temporal context window
- Improve mixed-precision training support

### Evaluation

- Add scene-reasoning evaluation
- Add temporal consistency checks
- Add hallucination checks
- Add modality ablation experiments
- Add qualitative visualizations
- Add LiDAR-camera alignment inspection

### Infrastructure

- Automate quota checking
- Add SageMaker launch scripts
- Add cost monitoring scripts
- Add S3 prefix summaries
- Add reproducible training configs
- Add CI checks for code quality and unit tests

---

## Example Workflow

```text
1. Set AWS environment variables
2. Verify S3 paths
3. Run kairos_preflight.py
4. Run tiny sanity training check
5. Confirm model forward pass
6. Confirm outputs are generated
7. Launch short SageMaker training run
8. Save checkpoint to S3
9. Review validation outputs
10. Scale to longer spot training run
```

---

## Known Limitations

Current limitations:

- ZOD data is uploaded but still needs unified training-index conversion
- GPU quota approval is still pending in Stockholm
- Full model training has not yet started
- Storage usage is above the original 600 GB target
- This is a research project, not a production autonomous driving stack

---

## Safety Disclaimer

KAIROS Motion is a research and development project.

It is not a production autonomous driving system. It must not be used for real-world vehicle control, safety-critical decision making, or deployment in vehicles.

The project is intended for experimentation with multimodal perception, temporal reasoning, and autonomous-driving dataset preparation.

---

## Project Name Meaning

**KAIROS** refers to the right or critical moment.

**Motion** reflects the project’s focus on temporal reasoning, movement, ego-motion, and autonomous-driving scene dynamics.

Together, **KAIROS Motion** represents a system designed to understand the right moment in a moving world.

---

## License

License to be decided.

Recommended options:

- MIT License for open research code
- Apache 2.0 for a more industry-friendly open-source license
- Private research license if dataset and training code are not intended for public reuse
