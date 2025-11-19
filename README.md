This repository provides a complete, reproducible pipeline for cleaning,
visualizing, clustering, and classifying Ubuntu dialogue text data. It is
designed for experimentation with multiple unsupervised and supervised
machine-learning techniques, including TF-IDF, embeddings, UMAP, HDBSCAN,
KMeans, XGBoost, and deep MLP models.

## Features

### Data Preprocessing
- Cleans raw Ubuntu dialogue CSVs
- Removes noise, empty rows, malformed texts
- Produces `ubuntu_cleaned.csv`

### Visualization
Generates exploratory plots:
- Word count distributions
- Message length histograms
- Category distributions
- Saved automatically in `/plots`

### Multi-Method Clustering
Implemented in `clusterings.py`:
1. Method 1 – PCA → HDBSCAN
2. Method 2 – UMAP (2D) → HDBSCAN
3. Method 3 – UMAP (10D → 2D) → HDBSCAN
4. Method 4 – KMeans with silhouette-based K selection

Each method saves cluster CSVs, distribution plots, and PCA/UMAP scatter plots.

### Multi-Model Classification
Implemented in `classification.py`:

| Method | Vectorizer | Model |
|--------|------------|--------|
| method1 | TF-IDF | Logistic Regression |
| method2 | TF-IDF | Linear SVM |
| method3 | TF-IDF | XGBoost |
| method4 | Embeddings | MLP neural network |

Targets:
- fault_type
- severity

Outputs include reports, confusion matrices, and prediction CSVs.

## Installation

```
pip install -r requirements.txt
```

## Download the Ubuntu Dialogue Corpus

```
pip install kaggle
kaggle datasets download -d rtatman/ubuntu-dialogue-corpus
unzip ubuntu-dialogue-corpus.zip
```

## Preprocess the Data

```
python3 preprocess.py --input ubuntu_annotated.csv --output ubuntu_cleaned.csv
```

## Generate Visualizations

```
python3 visualization.py --input ubuntu_cleaned.csv
```

## Run All Clustering Methods

```
python3 clusterings.py --input ubuntu_cleaned.csv --method all --outdir results
```

## Run All Classification Methods

```
python3 classification.py --input ubuntu_cleaned.csv --outdir class_results --methods all
```

## Requirements

See `requirements.txt`.

## Dataset

- **Ubuntu Dialogue Corpus**
  - Source: https://www.kaggle.com/datasets/rtatman/ubuntu-dialogue-corpus
