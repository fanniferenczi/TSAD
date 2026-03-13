# Efficient Time-Series Anomaly Detection for IoT Systems

> **Status: Ongoing Research** — This is an active master's thesis project. Results, findings, and model evaluations are continuously being updated.

---

## Motivation

Time-series anomaly detection is critical for IoT systems, enabling predictive maintenance, security monitoring, and system optimization. While recent deep learning models achieve impressive accuracy on benchmark datasets, they face significant challenges in real-world IoT deployments. Edge devices operate under strict resource constraints—limited computational power, memory, and energy—and often require low-latency responses.

This creates a fundamental gap: existing research optimizes for detection accuracy, while IoT deployments require balancing accuracy with practical constraints like computational efficiency, latency, and resource consumption.

---

## Objectives

This thesis explores and compares different time-series anomaly detection models with a focus on their suitability for IoT deployment. The goal is to examine diverse model architectures and understand their characteristics beyond accuracy — considering factors relevant to resource-constrained environments. By analyzing how different models perform under IoT constraints, this work aims to develop insights into why certain models are more suitable for constrained environments.

Specifically, this research aims to:

- **Research** deep learning models for anomaly detection under resource-constrained IoT environments.
- **Evaluate** multiple time-series anomaly detection models comparatively, analyzing trade-offs in accuracy, latency, memory footprint, and computational cost.
- **Assess** model suitability for real-world deployment on edge devices.
- **Profile** model benchmarks, computational characteristics, and performance under IoT-relevant constraints.

---

## Model Candidates

The following eight models are evaluated and compared in this study:

| Model            | Architecture Type             | Key Characteristics                                                                                                 |
| ---------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **TimesNet**     | CNN-based (2D temporal)       | Transforms 1D time series into 2D tensors using multi-periodicity; uses inception blocks for 2D variation modeling. |
| **DeepAnT**      | CNN-based                     | Lightweight convolutional model designed for unsupervised anomaly detection in time series.                         |
| **TCN-ms**       | TCN variant                   | Temporal Convolutional Network with multi-scale feature extraction.                                                 |
| **TinyAd**       | Lightweight / Edge-optimized  | Designed for deployment on constrained devices with minimal resource usage.                                         |
| **ModernTCN**    | Modern TCN                    | Updated temporal convolutional architecture with improved representational capacity.                                |
| **DACAD**        | Contrastive / Self-supervised | Domain-adversarial or contrastive learning approach to anomaly detection.                                           |
| **TranAD**       | Transformer-based             | Transformer-based deep autoencoder for anomaly detection with adversarial training.                                 |
| **RANSynCoders** | Synchronization-based         | Uses random sampling and encoder synchronization for multivariate anomaly detection.                                |

---

## Datasets

| Dataset | Domain                         | Anomaly Rate |
| ------- | ------------------------------ | ------------ |
| SMD     | Server machine monitoring      | ~4%          |
| MSL     | Spacecraft telemetry           | ~10%         |
| SMAP    | Spacecraft telemetry           | ~13%         |
| SWaT    | Water treatment infrastructure | ~12%         |
| PSM     | Server machine monitoring      | ~27%         |
| GECCO   | IoT water quality (custom)     | ~0.45%       |
