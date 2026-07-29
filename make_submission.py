"""
Optimized pipeline - fast version
Uses sparse matrix operations throughout for speed
"""

import re, ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"
SAMPLE_PATH = "sample_submission.csv"
OUTPUT_PATH = "submission.csv"
K = 10


def strip(review, course):
    return re.sub(re.escape(course), " ", review, flags=re.IGNORECASE)


def clean(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def main():
    print("Loading...")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    train["Reviews"] = train["Reviews"].apply(clean)
    test["Reviews"]  = test["Reviews"].apply(clean)
    train["stripped"] = train.apply(lambda r: strip(r["Reviews"], r["Course"]), axis=1)

    # ── 1. Classify course ───────────────────────────────────────────────────
    print("Classifier...")
    cv = TfidfVectorizer(ngram_range=(1, 2), max_features=80000, sublinear_tf=True)
    Xc = cv.fit_transform(train["stripped"])
    clf = LogisticRegression(max_iter=1000, C=10.0, solver="lbfgs")
    clf.fit(Xc, train["Course"])

    Xt_clf = cv.transform(test["Reviews"])
    test["pred_course"] = clf.predict(Xt_clf)
    print("  courses predicted")

    # ── 2. Retrieval vectorizer – fit on ALL stripped train ──────────────────
    print("TF-IDF vectorizer for retrieval...")
    rv = TfidfVectorizer(ngram_range=(1, 2), max_features=80000, sublinear_tf=True)
    Xr_train = rv.fit_transform(train["stripped"])   # (109776, V) sparse
    Xr_test  = rv.transform(test["Reviews"])          # (10977, V) sparse
    print("  done, shape:", Xr_train.shape)

    # ── 3. Per-course index ──────────────────────────────────────────────────
    course_rows = train.groupby("Course").indices   # course -> array of row positions
    train_ids   = train["Index"].values

    # ── 4. Batch retrieval ───────────────────────────────────────────────────
    print("Retrieving top-10 per test row...")
    # Group test rows by predicted course and process course by course
    test["row_i"] = np.arange(len(test))
    predictions = [None] * len(test)

    for course, row_positions in course_rows.items():
        test_subset = test[test["pred_course"] == course]
        if len(test_subset) == 0:
            continue

        test_row_indices = test_subset["row_i"].values
        Qc = Xr_test[test_row_indices]          # (n_test_in_course, V)
        Dc = Xr_train[row_positions]            # (n_train_in_course, V)

        # Cosine similarity in one batch
        sims = cosine_similarity(Qc, Dc)        # (n_test, n_train_course)

        for j, (ti, row) in enumerate(zip(test_row_indices, test_subset.itertuples())):
            sim_row = sims[j]
            top_local  = np.argsort(sim_row)[::-1][:K]
            top_global = row_positions[top_local]
            predictions[ti] = train_ids[top_global].tolist()

        print(f"  course '{course}': {len(test_subset)} test rows done", flush=True)

    # ── 5. Write submission ──────────────────────────────────────────────────
    sub = pd.DataFrame({
        "Index": test["Index"].tolist(),
        "Index_list": [str(p) for p in predictions],
    })[sample.columns]
    sub.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}  shape={sub.shape}")

    # validation
    check = pd.read_csv(OUTPUT_PATH)
    for v in check["Index_list"].head(20):
        assert len(ast.literal_eval(v)) == K
    print("✓ all rows have 10 items")
    print("\nFirst 3 predictions:")
    for i in range(3):
        print(f"  {check['Index'].iloc[i]}: {check['Index_list'].iloc[i]}")


if __name__ == "__main__":
    main()
