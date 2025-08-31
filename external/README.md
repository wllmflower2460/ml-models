 This repository contains models, datasets, and tools for:
  - **Real-time pose estimation** for dogs and humans
  - **Movement synchrony analysis** between handler-dog pairs
  - **Behavioral pattern recognition** during training sessions
  - **Multi-modal data fusion** (IMU + vision + audio)

  ## 🏗️ Repository Structure

  ml-models/
  ├── datasets/                    # Training and validation data
  │   ├── tartanair/              # TartanAir dataset for visual odometry
  │   ├── custom/                 # Custom dog-handler datasets
  │   └── annotations/            # Manual annotations and labels
  ├── models/                     # Trained model weights and configs
  │   ├── pose/                   # Pose estimation models
  │   ├── tracking/               # Multi-object tracking models
  │   └── synchrony/              # Synchrony analysis models
  ├── external/                   # Third-party repositories (submodules)
  │   ├── castacks/              # CMU AirLab tools
  │   ├── mediapipe/             # Google MediaPipe
  │   ├── mmpose/                # OpenMMLab pose estimation
  │   ├── alphapose/             # AlphaPose framework
  │   └── openpose/              # CMU OpenPose
  ├── scripts/                    # Training and inference scripts
  │   ├── train/                 # Model training scripts
  │   ├── inference/             # Real-time inference
  │   └── evaluation/            # Model evaluation utilities
  ├── configs/                    # Model and pipeline configurations
  ├── docker/                    # Docker containers for different environments
  └── docs/                      # Documentation and research notes
  - **GPU**: NVIDIA RTX 2060 (8GB VRAM)
  - **OS**: PopOS 22.04 LTS
  - **Docker**: NVIDIA Container Toolkit
  - **Storage**: 500GB+ for datasets and models
