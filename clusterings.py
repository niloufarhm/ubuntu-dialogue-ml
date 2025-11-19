import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import hdbscan
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#############################################
# UTILITIES
#############################################

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


#############################################
# LOADING & CLEANING
#############################################

def load_and_clean(path):
    print(f"[LOAD] {path}")
    df = pd.read_csv(path)

    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() > 3]

    print(f"[CLEAN] Rows kept: {len(df)}")
    return df


#############################################
# EMBEDDING
#############################################

def embed(texts, model_name):
    print(f"[EMBED] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=32, show_progress_bar=True)


#############################################
# METHODS
#############################################

### METHOD 1 — PCA + HDBSCAN
def method1_cluster(emb):
    print("[METHOD1] PCA 50D")
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(emb)

    print("[METHOD1] HDBSCAN")
    labels = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(reduced)
    return reduced, labels


### METHOD 2 — UMAP 2D + HDBSCAN
def method2_cluster(emb):
    print("[METHOD2] UMAP 2D")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=25,
        metric="cosine",
        random_state=42
    )
    reduced = reducer.fit_transform(emb)

    print("[METHOD2] HDBSCAN")
    labels = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(reduced)
    return reduced, labels


### METHOD 3 — UMAP(10D) → UMAP(2D) + HDBSCAN
def method3_cluster(emb):
    print("[METHOD3] UMAP 10D")
    reducer10 = umap.UMAP(
        n_components=10,
        n_neighbors=20,
        metric="cosine",
        random_state=42
    )
    reduced10 = reducer10.fit_transform(emb)

    print("[METHOD3] UMAP 2D on 10D")
    reducer2 = umap.UMAP(
        n_components=2,
        n_neighbors=25,
        metric="cosine",
        random_state=42
    )
    reduced2 = reducer2.fit_transform(reduced10)

    print("[METHOD3] HDBSCAN")
    labels = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(reduced2)

    return reduced2, labels


#############################################
# PLOTTING
#############################################

def plot_distribution(labels, path):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=labels, palette="viridis")
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_2d(emb2d, labels, path):
    plt.figure(figsize=(10, 7))
    unique = sorted(set(labels))

    cmap = plt.get_cmap("tab10")
    for i, cluster_id in enumerate(unique):
        mask = labels == cluster_id
        plt.scatter(
            emb2d[mask, 0], emb2d[mask, 1], 
            s=60, alpha=0.8, color=cmap(i % 10),
            label=f"C{cluster_id} (n={mask.sum()})"
        )

    plt.legend()
    plt.title("Cluster Visualization (2D)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def method4_kmeans(df, outdir):
    print("\n===== METHOD 4: KMEANS CLUSTERING =====")

    os.makedirs(outdir, exist_ok=True)

    texts = df["text"].tolist()

    # ---------------------------------------------------------
    # TF–IDF Vectorization
    # ---------------------------------------------------------
    print("[METHOD4] Building TF-IDF vectors...")
    vect = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    X = vect.fit_transform(texts)

    # ---------------------------------------------------------
    # Best K search with silhouette score
    # ---------------------------------------------------------
    print("[METHOD4] Searching best K (3–10)...")
    best_k = None
    best_score = -1
    best_model = None

    for k in range(3, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"   K={k}: silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    print(f"[METHOD4] Best K = {best_k} (silhouette={best_score:.4f})")

    labels = best_model.labels_
    df_out = df.copy()
    df_out["cluster"] = labels

    csv_path = os.path.join(outdir, "clusters_method4.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"[SAVE] {csv_path}")

    # ---------------------------------------------------------
    # Plot cluster histogram
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))
    df_out["cluster"].value_counts().sort_index().plot(kind="bar", color="steelblue")
    plt.title(f"KMeans Cluster Counts (K={best_k})")
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.tight_layout()

    hist_path = os.path.join(outdir, "clusters_plot.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"[SAVE] {hist_path}")

    # ---------------------------------------------------------
    # PCA 2D visualization
    # ---------------------------------------------------------
    print("[METHOD4] PCA 2D plotting...")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X.toarray())

    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap("tab10")

    for cid in sorted(set(labels)):
        mask = labels == cid
        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=60,
            alpha=0.75,
            color=cmap(cid % 10),
            label=f"Cluster {cid}"
        )

    plt.title("KMeans PCA Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    pca_path = os.path.join(outdir, "pca_kmeans.png")
    plt.savefig(pca_path)
    plt.close()
    print(f"[SAVE] {pca_path}")

    print("[DONE] Method 4")
    return labels
#############################################
# RUN METHOD
#############################################

def run_method(method_id, emb, df, outdir):
    print(f"\n===== RUNNING METHOD {method_id} =====")

    # FIX: ensure the directory exists
    ensure_dir(outdir)
    emb2d = None
    if method_id == 1:
        emb2d, labels = method1_cluster(emb)
    elif method_id == 2:
        emb2d, labels = method2_cluster(emb)
    elif method_id == 3:
        emb2d, labels = method3_cluster(emb)
    elif method_id == 4:
        labels = method4_kmeans(df, outdir)
    else:
        raise ValueError("Invalid method")

    # Save CSV
    df_out = df.copy()
    #df_out["cluster"] = labels
    
    df_out["cluster"] = labels
    df_out_path = os.path.join(outdir, f"clusters_method{method_id}.csv")
    df_out.to_csv(df_out_path, index=False)
    print(f"[SAVE] {df_out_path}")

    # Save plots
    if emb2d is not None:
        plot_distribution(labels, os.path.join(outdir, f"dist_method{method_id}.png"))
        plot_2d(emb2d, labels, os.path.join(outdir, f"plot_method{method_id}.png"))

    print(f"[DONE] Method {method_id}")


#############################################
# MAIN
#############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="all-mpnet-base-v2")
    parser.add_argument("--method", default="all", help="1,2,3,4 or all")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    df = load_and_clean(args.input)
    texts = df["text"].tolist()

    emb = embed(texts, args.model)

    if args.method == "all":
        for m in [1, 2, 3, 4]:
            run_method(m, emb, df, os.path.join(args.outdir, f"method{m}"))
    else:
        m = int(args.method)
        run_method(m, emb, df, os.path.join(args.outdir, f"method{m}"))
