# Transformer-IDS: Cross-Dataset Intrusion Detection with AUC-Oriented Optimization

This repository provides the official implementation of the paper:

**"An Improved Transformer-Based Intrusion Detection Model with Cross-Dataset Generalization and AUC-Oriented Optimization for Network Security"**

The code supports reproducible experiments for evaluating Transformer-based intrusion detection models under **within-dataset**, **cross-dataset**, and **leave-one-dataset-out (LODO)** settings.

---

# Overview

Intrusion Detection Systems (IDS) based on deep learning have achieved strong results on benchmark datasets. However, models often suffer from significant performance degradation when deployed in unseen network environments due to **domain shift**.

This repository implements a **Transformer-based IDS framework** designed to investigate:

- Cross-dataset generalization
- AUC-oriented optimization
- Threshold calibration
- Computational efficiency
- Class imbalance robustness

The framework is evaluated on three widely used IDS benchmark datasets:

- NSL-KDD
- UNSW-NB15
- CIC-IDS2017

---

# Repository Structure
