###########################################
# Multi-method classification for:
#  - fault_type
#  - severity
###########################################

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
from sentence_transformers import SentenceTransformer


###########################################################
# Utility: ensure output folder exists
###########################################################
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


###########################################################
# Plot confusion matrix
###########################################################
def save_confusion_matrix(y_true, y_pred, labels, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


###########################################################
# TF-IDF helper
###########################################################
def build_tfidf(texts):
    vect = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vect.fit_transform(texts)
    return vect, X


###########################################################
# Embedding helper
###########################################################
def embed_texts(texts, model_name="all-MiniLM-L6-v2"):
    print(f"[EMBED] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True)
    return np.array(emb)


###########################################################
# Generic classifier runner
###########################################################
def run_classifier(df, target_name, method_name, X, outdir):
    ensure_dir(outdir)

    # Encode labels
    y = df[target_name].astype(str).fillna("unknown")

    # Remove classes with <2 samples (required for stratify)
    class_counts = y.value_counts()

    too_small = class_counts[class_counts < 2].index.tolist()
    if too_small:
        print(f"[WARN] Removing classes with <2 samples for '{target_name}': {too_small}")
        df = df[~df[target_name].isin(too_small)]
        y = df[target_name].astype(str)
        # Also slice X to match df
        X = X[df.index]

    # If target empty after filtering → skip
    if df.empty or y.nunique() < 2:
        print(f"[SKIP] Not enough data to classify '{target_name}'. Skipping.")
        return

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split safely
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Select model
    if method_name == "logreg":
        model = LogisticRegression(max_iter=300)
    elif method_name == "svm":
        model = LinearSVC()
    elif method_name == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            eval_metric="mlogloss"
        )
    elif method_name == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(256, 128),
                             activation="relu", max_iter=200)
    else:
        raise ValueError("Unknown method.")

    print(f"[{method_name.upper()}] Training on {target_name} with {len(y)} samples...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Save classification report
    unique_test_labels = np.unique(y_test)
    test_class_names = le.inverse_transform(unique_test_labels)

    report_path = os.path.join(outdir, f"{target_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(
            classification_report(
                y_test,
                preds,
                labels=unique_test_labels,
                target_names=test_class_names,
                zero_division=0
            )
        )
    print(f"[SAVE] {report_path}")
    # Save confusion matrix
    cm_path = os.path.join(outdir, f"{target_name}_confusion.png")
    save_confusion_matrix(y_test, preds, list(range(len(le.classes_))), cm_path)
    print(f"[SAVE] {cm_path}")

    # Save predictions
    pred_csv = os.path.join(outdir, f"{target_name}_predictions.csv")
    pd.DataFrame({
        "true": le.inverse_transform(y_test),
        "pred": le.inverse_transform(preds)
    }).to_csv(pred_csv, index=False)
    print(f"[SAVE] {pred_csv}")



###########################################################
# Pipeline runner for all methods
###########################################################
def run_all_methods(df, outdir):

    # Prepare TF-IDF
    vect, X_tfidf = build_tfidf(df["text"].tolist())

    # Prepare embeddings for method 4
    X_emb = embed_texts(df["text"].tolist(), model_name="all-MiniLM-L6-v2")

    # METHODS
    methods = {
        "method1": ("logreg", X_tfidf),
        "method2": ("svm", X_tfidf),
        "method3": ("xgboost", X_tfidf),
        "method4": ("mlp", X_emb),   # Embedding + MLP
    }

    targets = ["fault_type", "severity"]

    for method_id, (method_name, X) in methods.items():
        print(f"\n===== RUNNING {method_id.upper()} =====")
        method_dir = os.path.join(outdir, method_id)
        ensure_dir(method_dir)

        for target in targets:
            print(f"--- Classifying {target} ---")
            run_classifier(df, target, method_name, X, method_dir)

    print("\n==== ALL METHODS COMPLETED ====")


###########################################################
# MAIN
###########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str,
                        help="Path to cleaned CSV")
    parser.add_argument("--outdir", default="results_classification",
                        help="Folder to save output")
    parser.add_argument("--methods", default="all",
                        choices=["all", "tfidf", "svm", "xgb", "embed"],
                        help="Choose classification methods")

    args = parser.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.input)

    print("Starting classification on:", args.input)

    run_all_methods(df, args.outdir)
