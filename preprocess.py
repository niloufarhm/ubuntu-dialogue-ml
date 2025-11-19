import polars as pl
import argparse


def clean(input_path, output_path):
    print(f"Loading: {input_path}")
    df = pl.read_csv(input_path)

    print("Initial size:", df.shape)

    # Drop empty rows
    df = df.drop_nulls()
    df = df.filter(pl.col("text").str.len_bytes() > 3)

    # Normalize labels (Polars >= 1.0 API)
    df = df.with_columns([
        pl.col("fault_type").str.to_lowercase().str.strip_chars(),
        pl.col("component").str.to_lowercase().str.strip_chars(),
        pl.col("symptom").str.to_lowercase().str.strip_chars(),
        pl.col("resolution").str.to_lowercase().str.strip_chars(),
        pl.col("severity").str.to_lowercase().str.strip_chars(),
    ])

    # Replace weird values
    CLEAN_VALUES = ["", "none", "n/a", "na", "null", "unknown"]
    df = df.with_columns([
        pl.when(pl.col(c).is_in(CLEAN_VALUES))
          .then(None)
          .otherwise(pl.col(c))
          .alias(c)
        for c in ["fault_type", "component", "symptom", "resolution", "severity"]
    ])

    print("Final size:", df.shape)
    df.write_csv(output_path)
    print("Saved cleaned data to:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean annotated Ubuntu dataset")

    parser.add_argument(
        "--input", "-i", 
        type=str, 
        default="ubuntu_annotated_qwen.csv",
        help="Path to the annotated CSV file"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="ubuntu_clean.csv",
        help="Path to save cleaned CSV output"
    )

    args = parser.parse_args()
    clean(args.input, args.output)
