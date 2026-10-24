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
├── notebooks/        # 01_solucion_cnn_mri.ipynb (v1) and 01_solucion_cnn_mri_v2.ipynb (v2, report results)
├── outputs/
│   ├── figures/      # Figures used in the report
│   ├── tables/       # Metrics tables (CSV)
│   └── models/       # Trained weights (v1: not versioned; v2: weights and results versioned)
├── report/           # IEEE-style report (LaTeX source, figures in report/figs/, compiled PDF)
├── docs/             # Assignment statement and course report templates (Word / LaTeX)
└── requirements.txt
```

## Dataset findings

The exploratory analysis found that the dataset contains copies of the same slice (exact duplicates, rescaled or contrast-modified versions), concentrated in the *healthy* class: at least 52 % of its images are redundant. A random split would place copies of the same slice in both train and test, inflating the metrics. For this reason the project uses a **group-aware stratified split** (`data/splits/split_grupos.csv`), where all copies of a slice fall in the same partition, and quantifies the effect of leakage against a random split.

## How to run

The main notebook is [`notebooks/01_solucion_cnn_mri.ipynb`](notebooks/01_solucion_cnn_mri.ipynb). It runs end to end and writes every table, figure and model to `outputs/`. **An NVIDIA GPU is required** for the full run.

**Local (NVIDIA GPU):**

```bash
git clone https://github.com/panamiguell/brain-tumor-mri-cnn.git
cd brain-tumor-mri-cnn
# 1. Install PyTorch with CUDA following https://pytorch.org/get-started/locally/
# 2. Install the remaining dependencies
pip install -r requirements.txt jupyter
jupyter notebook notebooks/01_solucion_cnn_mri.ipynb   # or open it in VS Code
```

Then run all cells (*Run All*). If the run is interrupted, running it again reloads the already trained models from `outputs/models/` instead of retraining them.

**Notebook v2 (`notebooks/01_solucion_cnn_mri_v2.ipynb`):** produces the results and figures of the report (`report/`). It runs on CPU, CUDA or Apple MPS and can be opened from `notebooks/` or from the repository root. Its trained weights and experiment results are versioned in `outputs/models/`, so a full run reloads them instead of retraining (set `RETRAIN = True` and delete `outputs/models/exp_*.json` to retrain from scratch, ~4 h on CPU).

**Google Colab:** open the notebook from GitHub, set *Runtime > Change runtime type > GPU* and run all cells. The first cell clones this repository, data included.

**Smoke test without GPU:** set the environment variable `MP1_QUICK=1` to run the whole pipeline on a small subset for one epoch (outputs go to `outputs/_smoke_test/`, not versioned).

## Results

_To be completed._
