# 🧠 Engineering Intelligence: Reproducible ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange)
![Status](https://img.shields.io/badge/Status-Working-brightgreen)

## 📖 Overview
This project demonstrates a **complete machine learning pipeline** — from dataset generation to training, evaluation, and visualization.  
It is designed to showcase **software engineering discipline** applied to machine learning:
- Modular code structure
- Reproducibility
- Automated evaluation
- Visual reporting

---

## ⚙️ Features
- **Synthetic dataset generator** (`generate_structured_dataset.py`)  
- **Data loader** (`src/data_loader.py`) with train/test split  
- **Model builder** (`src/model.py`) — configurable hidden layers + dropout  
- **Training loop** (`src/train.py`) — tracks loss per epoch  
- **Evaluation** (`src/evaluate.py`) — accuracy, F1, confusion matrix, classification report  
- **Visualization** (`docs/`) — training curve + confusion matrix heatmap  
- **Metrics saving** (`docs/evaluation.json`)  

---

## 🚀 Quickstart

```bash
# 1. Create structured dataset
py generate_structured_dataset.py

# 2. Run pipeline
py main.py --data_path data/structured_data.csv --epochs 50 --hidden_dim 128