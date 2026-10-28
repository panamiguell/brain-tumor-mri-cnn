# brain-tumor-mri-cnn

Brain tumor classification from MRI slices (glioma, meningioma, pituitary, healthy) with a compact CNN trained from scratch in PyTorch, using a duplicate-aware data split, multi-seed evaluation, a data-leakage experiment and Grad-CAM interpretability on 7,023 images.

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

## Approach

1. **Duplicate detection:** exact copies (MD5) and near-duplicates (dHash + Pearson correlation ≥ 0.95) are grouped; 7,023 images form 5,835 groups.
2. **Data pipeline:** group-aware stratified split 70/15/15 (5,016 / 1,004 / 1,003), grayscale, pad-to-square, 128×128, train-only normalization and augmentation.
3. **Custom CNN:** 4 VGG-style blocks ([Conv3×3, BatchNorm, ReLU]×2 + max pooling; 16-32-64-128 filters), global average pooling, dropout and a linear classifier (294,228 parameters). AdamW, cosine annealing, label smoothing, model selection by validation macro F1.
4. **Evaluation:** accuracy with bootstrap CI, balanced accuracy, macro F1, one-vs-rest AUC, per-class recall, tumor-to-healthy false negatives, 3 seeds, a random-vs-group split leakage experiment, and Grad-CAM.
5. **Transfer learning (ResNet-18):** fine-tuning code is included in the notebook (Section 7) as an optional alternative; it is not part of the reported results and is listed as future work in the report.

## Repository structure

```
brain-tumor-mri-cnn/
├── data/
│   ├── raw/          # Original dataset, as downloaded from Kaggle (see data/README.md)
│   └── splits/       # CSV files with the train / val / test split (versioned for reproducibility)
├── notebooks/        # MicroProyecto1_CNN_MRI.ipynb (deliverable, report results); archive/ holds an earlier draft
├── outputs/
│   └── models/       # Trained weights, training history and experiment results (versioned)
├── report/           # IEEE-style report (LaTeX source, figures in report/figs/ exported by the notebook, compiled PDF)
├── docs/             # Assignment statement and course report templates (Word / LaTeX)
└── requirements.txt
```

## Dataset findings

The exploratory analysis found that the dataset contains copies of the same slice (exact duplicates, rescaled or contrast-modified versions), concentrated in the *healthy* class: at least 52 % of its images are redundant. A random split would place copies of the same slice in both train and test, inflating the metrics. For this reason the project uses a **group-aware stratified split** (`data/splits/split_grupos.csv`), where all copies of a slice fall in the same partition, and quantifies the effect of leakage against a random split.

## How to run

The deliverable notebook is [`notebooks/MicroProyecto1_CNN_MRI.ipynb`](notebooks/MicroProyecto1_CNN_MRI.ipynb). It produces every number, table and figure of the report (`report/`), exporting the report figures to `report/figs/` and a summary of the cited figures to `outputs/models/resumen_informe.json`. It runs on CPU, NVIDIA GPU (CUDA) or Apple Silicon (MPS) and can be opened from `notebooks/` or from the repository root.

```bash
git clone https://github.com/panamiguell/brain-tumor-mri-cnn.git
cd brain-tumor-mri-cnn
# 1. Install PyTorch for your platform following https://pytorch.org/get-started/locally/
# 2. Install the remaining dependencies
pip install -r requirements.txt jupyter
jupyter notebook notebooks/MicroProyecto1_CNN_MRI.ipynb   # or open it in VS Code
```

Then run all cells (*Run All*). The trained weights and experiment results are versioned in `outputs/models/`, so a full run reloads them instead of retraining. To retrain from scratch, set `RETRAIN = True` and delete `outputs/models/exp_*.json` (~4 h on a 2-core CPU; much less with a GPU or MPS).

`notebooks/archive/01_solucion_cnn_mri_v1.ipynb` is an earlier, unexecuted draft kept for reference only; it is not part of the submission.

## Results

Test set of 1,003 images, group-aware split (no duplicates across partitions):

| Model | Accuracy | Balanced acc. | Macro F1 | AUC (OvR) |
|---|---|---|---|---|
| Custom CNN, seed 42 | 0.9661 (95 % CI 0.954-0.976) | 0.9663 | 0.9657 | 0.9945 |
| Custom CNN, 3 seeds (mean ± std) | **0.9711 ± 0.0062** | 0.9710 ± 0.0061 | 0.9707 ± 0.0063 | 0.9946 ± 0.0008 |

- Per-class recall (seed 42): glioma 0.987, meningioma 0.949, pituitary 0.964, healthy 0.965. Only 5 of 717 tumors (0.7 %) are classified as healthy.
- **Data leakage:** with a random split the same CNN reaches 0.9781; test images with a near-duplicate in train reach 0.9909 vs 0.9745 without one (Fisher exact test p = 0.19).
- **Main limitation:** the healthy class comes from a different source (lower resolution, higher intensity, mostly non-square), so part of the healthy/tumor separation may rely on acquisition shortcuts rather than pathology.

Full report: [`report/Informe_MicroProyecto_CNN_MRI.pdf`](report/Informe_MicroProyecto_CNN_MRI.pdf).
