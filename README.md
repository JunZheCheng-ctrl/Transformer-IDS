# Transformer-IDS: Cross-Dataset Intrusion Detection with AUC-Oriented Optimization

Official implementation of the **Transformer-IDS** framework for cross-dataset intrusion detection.

This repository accompanies the research paper:

**"An Improved Transformer-Based Intrusion Detection Model with Cross-Dataset Generalization and AUC-Oriented Optimization for Network Security"**

The project provides reproducible experimental scripts and models for evaluating intrusion detection systems under **cross-dataset generalization settings**.

---

# Overview

Deep learning–based intrusion detection systems (IDS) have achieved strong performance on benchmark datasets. However, many models experience significant performance degradation when deployed in unseen network environments due to **distribution shift** between datasets.

This repository implements a **Transformer-based IDS framework** designed to analyze:

- cross-dataset generalization ability  
- AUC-oriented optimization strategies  
- threshold calibration behavior  
- computational efficiency of model architectures  
- robustness under class imbalance  

The framework supports systematic evaluation across multiple IDS benchmark datasets.

---

# Key Features

The proposed Transformer-IDS framework includes:

- Transformer-based feature interaction modeling  
- Cross-dataset evaluation pipeline  
- AUC-oriented optimization strategy  
- Threshold calibration analysis  
- Computational complexity evaluation  
- Class imbalance robustness experiments  

The model contains approximately **74k parameters**, enabling efficient inference suitable for practical IDS deployment scenarios.

---
Transformer-IDS
│
├── models
│ improved_transformer.py
│ cnn_model.py
│ lstm_model.py
│
├── experiments
│ cross_dataset_experiment.py
│ lodo_experiment.py
│ model_comparison.py
│
├── analysis
│ feature_importance.py
│ plot_ablation.py
│
├── utils
│ coral_utils.py
│ loss_balanced.py
│ threshold_utils.py
│ preprocess_harmonize.py
│
├── figures
│ figure1_architecture.png
│ figure2_roc.png
│ figure3_pr.png
│ figure4_ablation.png
│
├── training
│ train_nn.py
│ train_ift.py
│
├── requirements.txt
└── README.md

---

# Environment Requirements

Recommended environment:

Python ≥ 3.9

Main dependencies:

- pytorch
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

Install dependencies:
pip install -r requirements.txt

---

# Datasets

The experiments use three widely adopted intrusion detection datasets.

### NSL-KDD

Download:

https://www.unb.ca/cic/datasets/nsl.html

---

### UNSW-NB15

Download:

https://research.unsw.edu.au/projects/unsw-nb15-dataset

---

### CIC-IDS2017

Download:

https://www.unb.ca/cic/datasets/ids-2017.html

---

# Data Preprocessing

Before training, datasets must be harmonized to align feature spaces.

Run:
python preprocess_harmonize.py

This script performs:

- feature normalization  
- feature alignment across datasets  
- label transformation  

---

# Training

Train the baseline neural IDS model:
python train_nn.py

Train the model using the **AUC-oriented optimization objective**:
python train_ift.py

---

# Experiments

### Cross-Dataset Evaluation

Train on one dataset and evaluate on another.

Example:

Train: NSL-KDD  
Test: UNSW-NB15

Run:
python cross_dataset_experiment.py


---

### Leave-One-Dataset-Out (LODO)

Train on two datasets and evaluate on the third unseen dataset.

Run:


python lodo_experiment.py


---

### Model Comparison

Compare Transformer-IDS with baseline models:

- CNN  
- LSTM  
- Transformer  

Run:


python model_comparison.py


---

### Ablation Study

Evaluate the contribution of model components.

Run:


python plot_ablation.py


---

# Reproducibility

To reproduce the main experiments:


python preprocess_harmonize.py
python train_nn.py
python cross_dataset_experiment.py
python lodo_experiment.py
python plot_ablation.py


---

# Experimental Settings

The framework evaluates IDS models under three scenarios:

### 1. Within-Dataset Evaluation

Training and testing performed on the same dataset.

---

### 2. Cross-Dataset Evaluation

Training on one dataset and testing on a different dataset.

---

### 3. Leave-One-Dataset-Out (LODO)

Training on two datasets and evaluating on an unseen dataset.

---

# Citation

If you use this repository in your research, please cite:


@article{cheng2026transformerids,
title={An Improved Transformer-Based Intrusion Detection Model with Cross-Dataset Generalization and AUC-Oriented Optimization for Network Security},
author={Cheng, Junzhe},
journal={Applied Sciences},
year={2026}
}


---

# Code Availability

The implementation code for the Transformer-IDS model and the experimental pipeline is publicly available at:

https://github.com/JunZheCheng-ctrl/Transformer-IDS

This repository contains the scripts required to reproduce the cross-dataset experiments and ablation studies reported in the paper.

---

# License

This project is released under the MIT License.

---

# Contact

Author: Junzhe Cheng

Email:

18530970921@163.com

---

# Disclaimer

This repository is intended for **academic research and educational purposes only**.  
The authors are not responsible for misuse of the provided code.

---

# Status

Project Status: **Research Prototype**

The repository will be updated with additional improvements and experimental results.


# Repository Structure
