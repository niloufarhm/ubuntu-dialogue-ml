import polars as pl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import argparse

plt.style.use("ggplot")

def save_plot(name):
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_bar(series, title, xlabel, filename, top_n=20):
    counts = series.drop_nulls().value_counts().to_pandas()
    counts = counts.sort_values("count", ascending=False).head(top_n)

    plt.figure(figsize=(10,6))
    plt.barh(counts.index, counts["count"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.gca().invert_yaxis()
    save_plot(filename)


def plot_wordcloud(series, title, filename):
    text = " ".join(series.drop_nulls().to_list())

    if len(text.strip()) < 5:
        print(f"[WARN] Not enough text for wordcloud: {filename}")
        return

    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        max_words=200
    ).generate(text)

    plt.figure(figsize=(12, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(title)
    plt.axis("off")
    save_plot(filename)


def plot_message_length(df):
    lengths = df["text"].str.len_bytes().to_numpy()

    plt.figure(figsize=(10,6))
    plt.hist(lengths, bins=40)
    plt.title("Message Length Distribution")
    plt.xlabel("Length (bytes)")
    plt.ylabel("Frequency")
    save_plot("message_length_distribution")


def plot_heatmap(df):
    import pandas as pd
    import seaborn as sns

    # encode categories to numbers
    numeric_df = df.select([
        pl.col("severity").cast(pl.Categorical),
        pl.col("fault_type").cast(pl.Categorical),
        pl.col("component").cast(pl.Categorical),
    ]).to_pandas()

    for col in numeric_df.columns:
        numeric_df[col] = numeric_df[col].cat.codes

    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save_plot("correlation_heatmap")


def visualize(input_path):
    print("Loading:", input_path)
    df = pl.read_csv(input_path)
    print("Rows:", df.shape)

    # ----------------------------------------------------------
    # 1. Severity distribution
    # ----------------------------------------------------------
    plot_bar(
        df["severity"],
        "Severity Distribution",
        "Count",
        "severity_distribution"
    )

    # ----------------------------------------------------------
    # 2. Most common fault types
    # ----------------------------------------------------------
    plot_bar(
        df["fault_type"],
        "Top Fault Types",
        "Count",
        "fault_type_frequency"
    )

    # ----------------------------------------------------------
    # 3. Most common components
    # ----------------------------------------------------------
    plot_bar(
        df["component"],
        "Top Components",
        "Count",
        "component_frequency"
    )

    # ----------------------------------------------------------
    # 4. Most common symptoms
    # ----------------------------------------------------------
    plot_bar(
        df["symptom"],
        "Top Symptoms",
        "Count",
        "symptom_frequency"
    )

    # ----------------------------------------------------------
    # 5. Resolution frequency
    # ----------------------------------------------------------
    plot_bar(
        df["resolution"],
        "Top Resolutions",
        "Count",
        "resolution_frequency"
    )

    # ----------------------------------------------------------
    # 6. Word cloud of symptoms
    # ----------------------------------------------------------
    plot_wordcloud(
        df["symptom"],
        "Symptom Word Cloud",
        "wordcloud_symptom"
    )

    # ----------------------------------------------------------
    # 7. Word cloud of fault types
    # ----------------------------------------------------------
    plot_wordcloud(
        df["fault_type"],
        "Fault Type Word Cloud",
        "wordcloud_fault_type"
    )

    # ----------------------------------------------------------
    # 8. Message length distribution
    # ----------------------------------------------------------
    plot_message_length(df)

    # ----------------------------------------------------------
    # 9. Correlation heatmap
    # ----------------------------------------------------------
    plot_heatmap(df)

    print("All plots saved to /plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize annotated dataset")

    parser.add_argument(
        "--input", "-i",
        type=str,
        default="ubuntu_clean.csv",
        help="Input cleaned dataset"
    )

    args = parser.parse_args()
    visualize(args.input)
