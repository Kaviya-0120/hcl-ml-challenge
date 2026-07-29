"""solution.py
Stronger ensemble for retrieval by combining:
- Top-3 course prediction (TF-IDF + LogisticRegression)
- Sparse retrieval: TF-IDF + BM25
- Dense retrieval: SentenceTransformers bi-encoder + ANN (faiss if available)
- Cross-Encoder reranking (pretrained) to score (query, candidate) pairs

This script:
1. Loads train.csv and test.csv
2. Trains course classifier (predicts top-3 courses per test row)
3. Builds retrieval indices per course (to restrict search) for TF-IDF, BM25, and dense
4. For each test row, produce candidate pool = union(top-K from TF-IDF, BM25, dense) across top-3 predicted courses
5. Rerank candidate pool with CrossEncoder and output top-10 final predictions
6. Writes submission.csv matching sample_submission.csv format

Notes:
- Designed to run locally / Colab with optional GPU acceleration for CrossEncoder
- Save intermediate artifacts under ./artifacts
- Adjust model names and batch sizes for your environment

Usage:
    pip install -r requirements.txt
    python solution.py --train train.csv --test test.csv --sample sample_submission.csv --out submission.csv

"""

import os
import argparse
import gc
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

# rank_bm25 and sentence-transformers are optional heavy deps
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:
    SentenceTransformer = None
    CrossEncoder = None


def simple_preprocess(text: str) -> str:
    if pd.isna(text):
        return ""
    # keep it simple; TF-IDF and transformers are robust
    return text.replace('\n', ' ').strip()


def train_course_classifier(train_texts: List[str], train_courses: List[str]):
    tfidf_clf = TfidfVectorizer(max_features=80000, ngram_range=(1,2), stop_words='english')
    X = tfidf_clf.fit_transform(train_texts)
    le = LabelEncoder()
    y = le.fit_transform(train_courses)
    clf = LogisticRegression(C=10, max_iter=1000, multi_class='ovr', n_jobs=-1)
    clf.fit(X, y)
    return tfidf_clf, clf, le


def build_tfidf_index(texts: List[str], max_features=100000) -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), stop_words='english')
    X = vec.fit_transform(texts)
    return vec, X


def build_bm25_index(tokenized_texts: List[List[str]]):
    if BM25Okapi is None:
        raise RuntimeError('rank_bm25 not installed. Install via pip install rank_bm25')
    bm25 = BM25Okapi(tokenized_texts)
    return bm25


def embed_texts(model, texts: List[str], batch_size=64) -> np.ndarray:
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc='embedding'):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_emb.append(emb)
    return np.vstack(all_emb)


def get_topk_sparse(tfidf_vec: TfidfVectorizer, tfidf_matrix, query_texts: List[str], topk=50) -> List[List[int]]:
    Q = tfidf_vec.transform(query_texts)
    # compute cosine similarity via dot product (tfidf vectors are L2 normed by default? not necessarily)
    # We'll use (Q * tfidf_matrix.T) dense computation in batches
    results = []
    batch = 256
    for i in range(0, Q.shape[0], batch):
        qbatch = Q[i:i+batch]
        sims = qbatch.dot(tfidf_matrix.T)
        for row in sims:
            # row is sparse matrix 1xN
            if hasattr(row, 'toarray'):
                arr = np.asarray(row.toarray()).ravel()
            else:
                arr = row.ravel()
            idx = np.argpartition(-arr, range(min(topk, len(arr))))[:topk]
            # sort these
            idx = idx[np.argsort(-arr[idx])]
            results.append(idx.tolist())
    return results


def get_topk_bm25(bm25, tokenized_queries: List[List[str]], topk=50) -> List[List[int]]:
    results = []
    for q in tokenized_queries:
        scores = bm25.get_scores(q)
        idx = np.argpartition(-scores, range(min(topk, len(scores))))[:topk]
        idx = idx[np.argsort(-scores[idx])]
        results.append(idx.tolist())
    return results


def get_topk_dense_ann(corpus_emb: np.ndarray, query_emb: np.ndarray, topk=50):
    # Use sklearn's NearestNeighbors for portability; user can swap to Faiss
    nbr = NearestNeighbors(n_neighbors=min(topk, corpus_emb.shape[0]), metric='cosine', algorithm='auto', n_jobs=-1)
    nbr.fit(corpus_emb)
    dists, idxs = nbr.kneighbors(query_emb)
    # convert to index lists
    return idxs.tolist(), dists.tolist()


def batch_crossencoder_score(cross_encoder, queries: List[str], candidates: List[str], batch_size=512) -> List[float]:
    # cross-encoder expects pairs; we will score pairwise by batching pairs
    pairs = [[q, c] for q, c in zip(queries, candidates)]
    scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    return scores


def main(args):
    # load
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)

    # basic preprocess
    train['text'] = train['Reviews'].fillna('').apply(simple_preprocess)
    test['text'] = test['Reviews'].fillna('').apply(simple_preprocess)
    # strip course names from train text to reduce leakage (copying baseline behaviour)
    train['text_stripped'] = train.apply(lambda r: r['text'].replace(str(r['Course']), '').strip(), axis=1)

    # train course classifier
    print('Training course classifier...')
    tfidf_clf, course_clf, le = train_course_classifier(train['text_stripped'].tolist(), train['Course'].tolist())

    # predict top-3 courses for test
    X_test_tfidf = tfidf_clf.transform(test['text'].tolist())
    proba = course_clf.predict_proba(X_test_tfidf)
    top3 = np.argsort(-proba, axis=1)[:, :3]
    top3_courses = [[le.inverse_transform([i])[0] for i in row] for row in top3]

    # build per-course indices (we'll build TF-IDF and BM25 and dense per course)
    print('Building per-course indices...')
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)

    # map course -> train indices
    course_to_indices = {}
    for idx, course in enumerate(train['Course']):
        course_to_indices.setdefault(course, []).append(idx)

    # TF-IDF global (will subset using indices)
    tfidf_global = TfidfVectorizer(max_features=100000, ngram_range=(1,2), stop_words='english')
    X_global = tfidf_global.fit_transform(train['text_stripped'].tolist())

    # prepare tokenized texts for BM25 per course
    tokenized_texts = [t.split() for t in train['text_stripped'].tolist()]
    # we will build BM25 per course lazily when needed

    # dense model
    if SentenceTransformer is None:
        print('sentence-transformers not installed; dense retrieval disabled. Install sentence-transformers for best results.')
        dense_model = None
        train_embeddings = None
    else:
        dense_name = args.dense_model
        print('Loading dense model:', dense_name)
        dense_model = SentenceTransformer(dense_name)
        train_embeddings = embed_texts(dense_model, train['text_stripped'].tolist(), batch_size=128)
        np.save(artifacts_dir / 'train_embeddings.npy', train_embeddings)

    # CrossEncoder
    if CrossEncoder is None:
        print('sentence-transformers CrossEncoder not available; please install sentence-transformers.')
        cross_encoder = None
    else:
        cross_name = args.cross_model
        print('Loading CrossEncoder:', cross_name)
        cross_encoder = CrossEncoder(cross_name, device=args.device)

    # For each test row, gather candidates from top-3 courses
    final_predictions = []
    K_sparse = args.k_sparse  # per-method candidates
    K_dense = args.k_dense
    K_union = args.k_union  # pool size before rerank

    for i, row in tqdm(test.iterrows(), total=len(test), desc='Processing queries'):
        q_text = row['text']
        courses = top3_courses[i]
        candidate_set = set()
        candidate_scores = {}

        for course in courses:
            indices = course_to_indices.get(course, [])
            if not indices:
                continue
            # subset matrices
            # TF-IDF
            sub_X = X_global[indices]
            q_vec = tfidf_global.transform([q_text])
            sims = q_vec.dot(sub_X.T)
            arr = np.asarray(sims.toarray()).ravel()
            topk = min(K_sparse, len(indices))
            idx_local = np.argpartition(-arr, range(topk))[:topk]
            idx_sorted = idx_local[np.argsort(-arr[idx_local])]
            for pos, j in enumerate(idx_sorted):
                doc_idx = indices[j]
                candidate_set.add(doc_idx)
                candidate_scores.setdefault(doc_idx, 0)
                candidate_scores[doc_idx] = max(candidate_scores[doc_idx], float(arr[j]))

            # BM25
            if BM25Okapi is not None:
                # build BM25 for this course lazily
                course_texts = [train['text_stripped'].iloc[j] for j in indices]
                tokenized = [t.split() for t in course_texts]
                bm25 = BM25Okapi(tokenized)
                q_tokens = q_text.split()
                scores = bm25.get_scores(q_tokens)
                topk = min(K_sparse, len(scores))
                idx_local = np.argpartition(-scores, range(topk))[:topk]
                idx_sorted = idx_local[np.argsort(-scores[idx_local])]
                for j in idx_sorted:
                    doc_idx = indices[j]
                    candidate_set.add(doc_idx)
                    candidate_scores.setdefault(doc_idx, 0)
                    candidate_scores[doc_idx] = max(candidate_scores[doc_idx], float(scores[j]))

            # Dense
            if dense_model is not None and train_embeddings is not None:
                q_emb = dense_model.encode([q_text], convert_to_numpy=True)
                # compute cosine similarity to embeddings for this course
                sub_emb = train_embeddings[indices]
                # cosine similarity
                norm_sub = np.linalg.norm(sub_emb, axis=1)
                norm_q = np.linalg.norm(q_emb)
                sims = (sub_emb @ q_emb.T).ravel() / (norm_sub * norm_q + 1e-9)
                topk = min(K_dense, len(indices))
                idx_local = np.argpartition(-sims, range(topk))[:topk]
                idx_sorted = idx_local[np.argsort(-sims[idx_local])]
                for j in idx_sorted:
                    doc_idx = indices[j]
                    candidate_set.add(doc_idx)
                    candidate_scores.setdefault(doc_idx, 0)
                    candidate_scores[doc_idx] = max(candidate_scores[doc_idx], float(sims[j]))

        # Now we have candidate_set; limit to K_union highest by candidate_scores
        if not candidate_set:
            final_predictions.append([])
            continue
        cand_list = list(candidate_set)
        scores_arr = np.array([candidate_scores.get(c, 0.0) for c in cand_list])
        topu = min(K_union, len(cand_list))
        top_local = np.argpartition(-scores_arr, range(topu))[:topu]
        top_cands = [cand_list[j] for j in top_local[np.argsort(-scores_arr[top_local])]]

        # Rerank with CrossEncoder if available
        if cross_encoder is not None and len(top_cands) > 0:
            pairs = [[q_text, train['text_stripped'].iloc[c]] for c in top_cands]
            scores = cross_encoder.predict(pairs, batch_size=args.cross_batch)
            # sort by score descending
            order = np.argsort(-np.array(scores))
            ranked = [top_cands[o] for o in order]
        else:
            # fallback: use candidate_scores
            ranked = sorted(top_cands, key=lambda x: -candidate_scores.get(x, 0.0))

        final_predictions.append(ranked[:10])

    # Prepare submission.csv
    out_df = pd.DataFrame({'Index': test['Index'], 'Index_list': [str([int(x) for x in lst]) for lst in final_predictions]})
    # ensure same ordering as sample
    if 'Index' in sample.columns:
        out_df = out_df.set_index('Index').reindex(sample['Index']).reset_index()
    out_df.to_csv(args.out, index=False)
    print('Wrote', args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train.csv')
    parser.add_argument('--test', default='test.csv')
    parser.add_argument('--sample', default='sample_submission.csv')
    parser.add_argument('--out', default='submission.csv')
    parser.add_argument('--dense_model', default='sentence-transformers/paraphrase-MiniLM-L6-v2')
    parser.add_argument('--cross_model', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--k_sparse', type=int, default=50)
    parser.add_argument('--k_dense', type=int, default=50)
    parser.add_argument('--k_union', type=int, default=200)
    parser.add_argument('--cross_batch', type=int, default=128)

    args = parser.parse_args()
    main(args)
