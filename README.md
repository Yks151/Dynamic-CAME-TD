# CAME-TD: Performance-Interpretable Fusion of Sound and Vibration Signals for Bearing Fault Diagnosis

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs11071--024--10157--1-green)](https://doi.org/10.1007/s11071-024-10157-1)

Official implementation of **CAME-TD**, an interpretable deep learning framework for bearing fault diagnosis via dynamic fusion of sound and vibration signals, as proposed in the paper ["A Performance-interpretable Intelligent Fusion of Sound and Vibration Signals for Bearing Fault Diagnosis via Dynamic CAME"](#).

---

## 📌 Table of Contents
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset Preparation](#-dataset-preparation)
- [Training & Inference](#-training--inference)
- [Results](#-results)
- [Visual Interpretation](#-visual-interpretation)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Key Features
- **Dynamic Signal Fusion**:  
  - **Strong Correlation**: Complex feature weight fusion strategy for mutually enhanced feature learning
  - **Weak Correlation**: Hybrid input strategy to avoid noise amplification
- **Compressed Attention Mechanism Encoder (CAME)**:  
  - Automatically learns cross-modal correlations without manual feature engineering
  - Real-time correlation sensing via dynamic attention
- **Interpretable Framework**:  
  - Visual analysis of attention weights and feature importance
  - Regularized loss constraints for adaptive threshold updates
- **Robust Performance**:  
  - Superior accuracy under variable operating conditions (98.7% on CWRU dataset)
  - Noise immunity (tested with SNR from -4dB to 20dB)

---

## 🧠 Model Architecture
![CAME-TD Architecture](docs/architecture.png)  
*(Schematic diagram of the CAME-TD framework)*

### Core Components:
1. **Multimodal Encoder** (`CNN_fuse.py`):  
   - Parallel CNN branches for sound/vibration feature extraction
   - CAME module for cross-modal attention calculation
2. **Dynamic Fusion Controller** (`Transformer_fuse.py`):  
   - Adaptive selection between weight fusion and hybrid input
   - Threshold update via gradient-based regularization
3. **Transformer Decoder** (`visual_relation_all.py`):  
   - Temporal dependency modeling with compressed attention
   - Fault classification with interpretable feature maps

---

## 🛠 Installation
### Prerequisites
- Python ≥ 3.8
- NVIDIA GPU with CUDA ≥ 11.3
- PyTorch ≥ 2.0

### Step-by-Step Setup
```bash
# Clone repository
git clone https://github.com/Yks151/CAME-TD.git
cd CAME-TD

# Create conda environment
conda create -n cametd python=3.8
conda activate cametd

# Install core dependencies
pip install -r requirements.txt

# Install signal processing libraries
pip install librosa>=0.10.0 scipy>=1.11.0
