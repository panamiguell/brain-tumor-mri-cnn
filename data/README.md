# Data

**Source:** [Brain Tumor (MRI Scans) — Kaggle](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans)

## Layout

Place the dataset **unzipped and exactly as it comes from Kaggle** inside `data/raw/`, keeping its original folder structure (one folder per class):

```
data/raw/
└── <dataset folders as downloaded>/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── healthy/
```

The notebooks infer the class from the name of the folder that contains each image, so folder names must not be changed.

## How to upload the data to this repository

- Upload the **unzipped images**, not the `.zip`: GitHub rejects individual files larger than 100 MB.
- The GitHub web interface accepts a limited number of files per upload, so for ~7,000 images use `git` from the command line or GitHub Desktop:

```bash
git add data/raw
git commit -m "Add raw MRI dataset"
git push
```

## Alternative: download inside the notebook

```python
import kagglehub
DATA_DIR = kagglehub.dataset_download("rm1000/brain-tumor-mri-scans")
```

## `splits/`

CSV files with the path and label of each image for the train / validation / test partitions. They are versioned so that every experiment uses exactly the same split.
