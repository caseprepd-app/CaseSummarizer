"""
ML Model Diagnostics Script

Analyzes feedback data and model feature importances to identify
issues with the vocabulary preference learner.
"""

import pickle
from pathlib import Path

import pandas as pd

# Paths
DEFAULT_FEEDBACK = Path("config/default_feedback.csv")
USER_FEEDBACK = Path.home() / "AppData/Roaming/CasePrepd/data/feedback/user_feedback.csv"
MODEL_PATH = Path.home() / "AppData/Roaming/CasePrepd/data/models/vocab_meta_learner.pkl"


def get_bin(count):
    """Get count bin name for a given occurrence count."""
    if count == 1:
        return "bin_1"
    if count == 2:
        return "bin_2"
    if count == 3:
        return "bin_3"
    if 4 <= count <= 6:
        return "bin_4_6"
    if 7 <= count <= 20:
        return "bin_7_20"
    if 21 <= count <= 50:
        return "bin_21_50"
    return "bin_51_plus"


def analyze_feedback():
    """Analyze feedback data distribution."""
    print("=" * 60)
    print("FEEDBACK DATA ANALYSIS")
    print("=" * 60)

    # Load default feedback
    dfs = []
    if DEFAULT_FEEDBACK.exists():
        df_default = pd.read_csv(DEFAULT_FEEDBACK)
        df_default["source"] = "default"
        dfs.append(df_default)
        print(f"\nDefault feedback: {len(df_default)} samples")

    if USER_FEEDBACK.exists():
        df_user = pd.read_csv(USER_FEEDBACK)
        df_user["source"] = "user"
        dfs.append(df_user)
        print(f"User feedback: {len(df_user)} samples")
    else:
        print("User feedback: Not found")

    if not dfs:
        print("No feedback data found!")
        return None

    df = pd.concat(dfs, ignore_index=True)

    print(f"\nTotal samples: {len(df)}")
    pos_count = (df["feedback"] == 1).sum()
    neg_count = (df["feedback"] == -1).sum()
    print(f"Positive (+1): {pos_count} ({100 * pos_count / len(df):.1f}%)")
    print(f"Negative (-1): {neg_count} ({100 * neg_count / len(df):.1f}%)")

    # Frequency distribution by feedback
    print("\n" + "-" * 60)
    print("FREQUENCY DISTRIBUTION BY FEEDBACK")
    print("-" * 60)
    pos = df[df["feedback"] == 1]["in_case_freq"]
    neg = df[df["feedback"] == -1]["in_case_freq"]
    print("\nPositive samples (+1):")
    print(f"  Mean:   {pos.mean():.1f}")
    print(f"  Median: {pos.median():.1f}")
    print(f"  Min:    {pos.min()}")
    print(f"  Max:    {pos.max()}")
    print("\nNegative samples (-1):")
    print(f"  Mean:   {neg.mean():.1f}")
    print(f"  Median: {neg.median():.1f}")
    print(f"  Min:    {neg.min()}")
    print(f"  Max:    {neg.max()}")

    # High-frequency examples
    print("\n" + "-" * 60)
    print("HIGH FREQUENCY EXAMPLES (count > 50)")
    print("-" * 60)
    high_freq = df[df["in_case_freq"] > 50].sort_values("in_case_freq", ascending=False)
    if len(high_freq) == 0:
        print("No high-frequency examples found.")
    else:
        for _, row in high_freq.iterrows():
            label = "+1" if row["feedback"] == 1 else "-1"
            term = str(row["term"])[:25]
            print(f"  {label} | {term:25s} | count={int(row['in_case_freq']):3d}")

    # Count bin distribution
    print("\n" + "-" * 60)
    print("COUNT BIN DISTRIBUTION (positive vs negative)")
    print("-" * 60)
    df["count_bin"] = df["in_case_freq"].apply(get_bin)
    crosstab = pd.crosstab(df["count_bin"], df["feedback"])
    # Reorder bins
    bin_order = ["bin_1", "bin_2", "bin_3", "bin_4_6", "bin_7_20", "bin_21_50", "bin_51_plus"]
    crosstab = crosstab.reindex(bin_order)
    crosstab.columns = ["-1 (negative)", "+1 (positive)"]
    print(crosstab)

    # Calculate positive rate by bin
    print("\n  Positive rate by bin:")
    for bin_name in bin_order:
        bin_data = df[df["count_bin"] == bin_name]
        if len(bin_data) > 0:
            pos_rate = (bin_data["feedback"] == 1).mean() * 100
            print(f"    {bin_name:12s}: {pos_rate:5.1f}% positive (n={len(bin_data)})")

    return df


def analyze_feature_patterns(df):
    """Analyze patterns in features that have unexpected coefficients."""
    print("\n" + "=" * 60)
    print("FEATURE PATTERN ANALYSIS")
    print("=" * 60)

    # BM25 Analysis
    print("\n" + "-" * 60)
    print("BM25 DETECTION ANALYSIS")
    print("-" * 60)
    print("(Model gives has_bm25 a NEGATIVE coefficient - investigating why)")

    # Check if BM25_detection column exists, otherwise check algorithms
    if "BM25_detection" in df.columns:
        df["has_bm25"] = df["BM25_detection"].astype(bool)
    else:
        df["has_bm25"] = df["algorithms"].str.contains("BM25", case=False, na=False)

    bm25_pos = df[(df["has_bm25"]) & (df["feedback"] == 1)]
    bm25_neg = df[(df["has_bm25"]) & (df["feedback"] == -1)]
    no_bm25_pos = df[(~df["has_bm25"]) & (df["feedback"] == 1)]
    no_bm25_neg = df[(~df["has_bm25"]) & (df["feedback"] == -1)]

    total_bm25 = len(bm25_pos) + len(bm25_neg)
    total_no_bm25 = len(no_bm25_pos) + len(no_bm25_neg)

    print(f"\n  With BM25: {total_bm25} samples")
    if total_bm25 > 0:
        print(f"    Positive: {len(bm25_pos)} ({100 * len(bm25_pos) / total_bm25:.1f}%)")
        print(f"    Negative: {len(bm25_neg)} ({100 * len(bm25_neg) / total_bm25:.1f}%)")

    print(f"\n  Without BM25: {total_no_bm25} samples")
    if total_no_bm25 > 0:
        print(f"    Positive: {len(no_bm25_pos)} ({100 * len(no_bm25_pos) / total_no_bm25:.1f}%)")
        print(f"    Negative: {len(no_bm25_neg)} ({100 * len(no_bm25_neg) / total_no_bm25:.1f}%)")

    # Show BM25 negative examples
    if len(bm25_neg) > 0:
        print("\n  BM25-detected NEGATIVE examples:")
        for _, row in bm25_neg.iterrows():
            term = str(row["term"])[:30]
            count = int(row["in_case_freq"])
            print(f"    -1 | {term:30s} | count={count:3d}")

    # Show BM25 positive examples (for comparison)
    if len(bm25_pos) > 0:
        print("\n  BM25-detected POSITIVE examples (sample):")
        for _, row in bm25_pos.head(10).iterrows():
            term = str(row["term"])[:30]
            count = int(row["in_case_freq"])
            print(f"    +1 | {term:30s} | count={count:3d}")

    # Hyphen Analysis
    print("\n" + "-" * 60)
    print("HYPHEN ANALYSIS")
    print("-" * 60)
    print("(Model gives contains_hyphen a NEGATIVE coefficient - investigating why)")

    df["has_hyphen"] = df["term"].str.contains("-", na=False)

    hyph_pos = df[(df["has_hyphen"]) & (df["feedback"] == 1)]
    hyph_neg = df[(df["has_hyphen"]) & (df["feedback"] == -1)]
    no_hyph_pos = df[(~df["has_hyphen"]) & (df["feedback"] == 1)]
    no_hyph_neg = df[(~df["has_hyphen"]) & (df["feedback"] == -1)]

    total_hyph = len(hyph_pos) + len(hyph_neg)
    total_no_hyph = len(no_hyph_pos) + len(no_hyph_neg)

    print(f"\n  With hyphen: {total_hyph} samples")
    if total_hyph > 0:
        print(f"    Positive: {len(hyph_pos)} ({100 * len(hyph_pos) / total_hyph:.1f}%)")
        print(f"    Negative: {len(hyph_neg)} ({100 * len(hyph_neg) / total_hyph:.1f}%)")

    print(f"\n  Without hyphen: {total_no_hyph} samples")
    if total_no_hyph > 0:
        print(f"    Positive: {len(no_hyph_pos)} ({100 * len(no_hyph_pos) / total_no_hyph:.1f}%)")
        print(f"    Negative: {len(no_hyph_neg)} ({100 * len(no_hyph_neg) / total_no_hyph:.1f}%)")

    # Show hyphenated examples
    if len(hyph_neg) > 0:
        print("\n  Hyphenated NEGATIVE examples:")
        for _, row in hyph_neg.iterrows():
            term = str(row["term"])[:40]
            print(f"    -1 | {term}")

    if len(hyph_pos) > 0:
        print("\n  Hyphenated POSITIVE examples:")
        for _, row in hyph_pos.iterrows():
            term = str(row["term"])[:40]
            print(f"    +1 | {term}")

    # Trailing digit analysis
    print("\n" + "-" * 60)
    print("TRAILING DIGIT ANALYSIS")
    print("-" * 60)
    print("(Model gives has_trailing_digit a NEGATIVE coefficient)")

    df["has_trailing_digit"] = df["term"].str.match(r".*\d$", na=False)

    td_samples = df[df["has_trailing_digit"]]
    if len(td_samples) > 0:
        pos_rate = (td_samples["feedback"] == 1).mean() * 100
        print(f"\n  Samples with trailing digit: {len(td_samples)}")
        print(f"  Positive rate: {pos_rate:.1f}%")
        print("\n  Examples:")
        for _, row in td_samples.head(15).iterrows():
            label = "+1" if row["feedback"] == 1 else "-1"
            term = str(row["term"])[:40]
            print(f"    {label} | {term}")

    # Internal digits analysis
    print("\n" + "-" * 60)
    print("INTERNAL DIGITS ANALYSIS")
    print("-" * 60)

    def has_internal_digit(term):
        if pd.isna(term) or len(str(term)) < 3:
            return False
        middle = str(term)[1:-1]
        return any(c.isdigit() for c in middle)

    df["has_internal_digit"] = df["term"].apply(has_internal_digit)

    id_samples = df[df["has_internal_digit"]]
    if len(id_samples) > 0:
        pos_rate = (id_samples["feedback"] == 1).mean() * 100
        print(f"\n  Samples with internal digit: {len(id_samples)}")
        print(f"  Positive rate: {pos_rate:.1f}%")
        print("\n  Examples:")
        for _, row in id_samples.head(15).iterrows():
            label = "+1" if row["feedback"] == 1 else "-1"
            term = str(row["term"])[:40]
            print(f"    {label} | {term}")

    # All caps analysis
    print("\n" + "-" * 60)
    print("ALL CAPS ANALYSIS")
    print("-" * 60)

    def is_all_caps(term):
        if pd.isna(term):
            return False
        alpha = [c for c in str(term) if c.isalpha()]
        return len(alpha) > 0 and all(c.isupper() for c in alpha)

    df["is_all_caps"] = df["term"].apply(is_all_caps)

    caps_samples = df[df["is_all_caps"]]
    if len(caps_samples) > 0:
        pos_rate = (caps_samples["feedback"] == 1).mean() * 100
        print(f"\n  All-caps samples: {len(caps_samples)}")
        print(f"  Positive rate: {pos_rate:.1f}%")
        print("\n  Examples:")
        for _, row in caps_samples.iterrows():
            label = "+1" if row["feedback"] == 1 else "-1"
            term = str(row["term"])[:40]
            print(f"    {label} | {term}")
    else:
        print("\n  No all-caps samples found.")

    # Person analysis by frequency
    print("\n" + "-" * 60)
    print("PERSON STATUS BY FREQUENCY BIN")
    print("-" * 60)

    df["count_bin"] = df["in_case_freq"].apply(get_bin)
    bin_order = ["bin_1", "bin_2", "bin_3", "bin_4_6", "bin_7_20", "bin_21_50", "bin_51_plus"]

    print("\n  Positive rate for PERSONS by frequency bin:")
    for bin_name in bin_order:
        persons_in_bin = df[(df["count_bin"] == bin_name) & (df["is_person"] == 1)]
        if len(persons_in_bin) > 0:
            pos_rate = (persons_in_bin["feedback"] == 1).mean() * 100
            print(f"    {bin_name:12s}: {pos_rate:5.1f}% positive (n={len(persons_in_bin)})")

    print("\n  Positive rate for NON-PERSONS by frequency bin:")
    for bin_name in bin_order:
        non_persons_in_bin = df[(df["count_bin"] == bin_name) & (df["is_person"] == 0)]
        if len(non_persons_in_bin) > 0:
            pos_rate = (non_persons_in_bin["feedback"] == 1).mean() * 100
            print(f"    {bin_name:12s}: {pos_rate:5.1f}% positive (n={len(non_persons_in_bin)})")


def analyze_model():
    """Analyze trained model feature importances."""
    print("\n" + "=" * 60)
    print("MODEL ANALYSIS")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"\nModel not found at: {MODEL_PATH}")
        return

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    print(f"\nModel path: {MODEL_PATH}")
    print(f"Total samples trained on: {model_data.get('total_sample_count', 'unknown')}")
    print(f"User samples: {model_data.get('user_sample_count', 'unknown')}")
    print(f"Ensemble enabled: {model_data.get('ensemble_enabled', False)}")

    feature_names = model_data.get("feature_names", [])
    lr_model = model_data.get("lr_model")
    rf_model = model_data.get("rf_model")

    if lr_model is not None and hasattr(lr_model, "coef_"):
        print("\n" + "-" * 60)
        print("LOGISTIC REGRESSION COEFFICIENTS")
        print("-" * 60)
        coefs = lr_model.coef_[0]
        importance = list(zip(feature_names, coefs))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)

        print("\n  Top 15 features by absolute coefficient:")
        for i, (name, coef) in enumerate(importance[:15], 1):
            direction = "+" if coef > 0 else "-"
            print(f"    {i:2d}. {name:30s} {direction}{abs(coef):6.3f}")

        # Specifically look at count-related features
        print("\n  Count-related features:")
        count_features = [
            (name, coef)
            for name, coef in zip(feature_names, coefs)
            if "count" in name.lower() or "bin" in name.lower() or "occurrence" in name.lower()
        ]
        for name, coef in count_features:
            direction = "+" if coef > 0 else "-"
            print(f"    {name:30s} {direction}{abs(coef):6.3f}")

    if rf_model is not None:
        print("\n" + "-" * 60)
        print("RANDOM FOREST FEATURE IMPORTANCES")
        print("-" * 60)
        rf_importance = list(zip(feature_names, rf_model.feature_importances_))
        rf_importance.sort(key=lambda x: x[1], reverse=True)

        print("\n  Top 15 features:")
        for i, (name, imp) in enumerate(rf_importance[:15], 1):
            print(f"    {i:2d}. {name:30s} {imp:6.3f}")

        # Count-related features
        print("\n  Count-related features:")
        count_features = [
            (name, imp)
            for name, imp in zip(feature_names, rf_model.feature_importances_)
            if "count" in name.lower() or "bin" in name.lower() or "occurrence" in name.lower()
        ]
        for name, imp in count_features:
            print(f"    {name:30s} {imp:6.3f}")


def main():
    df = analyze_feedback()
    if df is not None:
        analyze_feature_patterns(df)
    analyze_model()


if __name__ == "__main__":
    main()
