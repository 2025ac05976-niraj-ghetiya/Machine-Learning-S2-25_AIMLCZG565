# Dataset: UCI Dry Bean Dataset

**Source (official):** UCI Machine Learning Repository, dataset id 602
https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

**Citation:**
Koklu, M. and Ozkan, I.A. (2020). *Multiclass Classification of Dry Beans Using
Computer Vision and Machine Learning Techniques.* Computers and Electronics in
Agriculture, 174, 105507. https://doi.org/10.1016/j.compag.2020.105507

## Files in this folder

- `dry_bean_dataset.csv` - the full dataset (13,543 rows after removing 68 exact
  duplicate rows found in the original 13,611-row release), committed here so
  the training pipeline is reproducible directly from this repository.
- `raw/Dry_Bean_Dataset.xlsx` - the original file as downloaded from UCI
  (git-ignored; not committed, since `dry_bean_dataset.csv` above is the
  cleaned, reproducible artifact actually used for training). To regenerate it,
  download the dataset zip from the UCI link above, extract it, and place
  `Dry_Bean_Dataset.xlsx` at `data/raw/Dry_Bean_Dataset.xlsx` before running
  `src/train_models.py`.

## Summary

| Property | Value |
|---|---|
| Instances (raw) | 13,611 |
| Instances (after de-duplication) | 13,543 |
| Features | 16 (all numeric, geometric measurements extracted from bean images) |
| Target | `Class` (7 categories) |
| Classes | Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira |
| Missing values | None |

The 16 features are: `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`,
`AspectRation`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`,
`Solidity`, `roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`,
`ShapeFactor3`, `ShapeFactor4` - 12 dimensional measurements and 4 derived
shape factors per grain, as described in the original publication.
