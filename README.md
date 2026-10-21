# brain-tumor-mri-cnn

Brain tumor classification from MRI slices (glioma, meningioma, pituitary, healthy) using CNNs in PyTorch. Compares a custom CNN, with an ablation on depth and kernel size, against fine-tuned pretrained networks, evaluated with medical-oriented metrics on 7,023 images.

> Mini-project 1 of the course *Técnicas de Deep Learning* (MAIA-4213), MSc in Artificial Intelligence, Universidad de los Andes.

## Problem

Each MRI volume is a stack of 2D slices. Instead of processing full 3D volumes, this project classifies **individual 2D slices** into one of four classes:

| Class | Description |
|---|---|
| Glioma | Abnormal masses with irregular borders and tissue heterogeneity |
| Meningioma | Well-delimited tumors originating in the meninges |
| Pituitary | Pituitary adenomas in the sellar region |
| Healthy | Normal brain tissue without anomalies |

**Dataset:** [Brain Tumor (MRI Scans) — Kaggle](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans), 7,023 images.

## Approach (planned)

1. **Data pipeline:** loading, stratified train / validation / test split with a fixed seed, preprocessing and data augmentation.
2. **Custom CNN:** convolution + ReLU + max-pooling blocks, with an ablation over the number of blocks and the kernel size.
3. **Transfer learning:** fine-tuning of pretrained networks (e.g. ResNet, DenseNet).
4. **Evaluation:** accuracy, macro F1, per-class precision / recall, confusion matrix, and qualitative analysis of hits and misses.

## Repository structure

```
brain-tumor-mri-cnn/
├── data/
│   ├── raw/          # Original dataset, as downloaded from Kaggle (see data/README.md)
│   └── splits/       # CSV files with the train / val / test split (versioned for reproducibility)
├── notebooks/        # Jupyter notebooks (exploration, training, evaluation)
├── outputs/
│   ├── figures/      # Figures used in the report
│   ├── tables/       # Metrics tables (CSV)
│   └── models/       # Trained weights (.pt, not versioned)
├── report/           # IEEE / CVPR-style report (LaTeX)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

See [`data/README.md`](data/README.md) for how to obtain the data.

## Results

_To be completed._
