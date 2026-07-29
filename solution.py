"""solution.py
Aggressive recall-optimized retrieval pipeline to push Recall@10 towards >=95%.
Changes over previous version:
- Expand course candidates to top-N (default top_courses=10)
- Add global retrieval signals (global TF-IDF and BM25) in addition to per-course
- Add fuzzy matching (rapidfuzz) to find near-duplicate reviews
- Add n-gram Jaccard overlap signal (character 5-grams) as additional feature
- Increase candidate pool sizes (k_sparse/k_dense/k_union defaults much higher)
- Combine signals via score normalization and weighted sum when CrossEncoder unavailable
- Deterministic finalization and robust popularity fallback

Notes:
- This script is heavy; run on GPU/large-memory machine for speed. Use FAISS if available for dense ANN (optional integration point commented).
- Install rapidfuzz for fuzzy matching: pip install rapidfuzz

Usage example:
  python solution.py --train train.csv --test test.csv --sample sample_submission.csv --out submission.csv --device cuda --top_courses 10 --k_sparse 300 --k_dense 300 --k_union 1500

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

# optional heavy deps
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:
    SentenceTransformer = None
    CrossEncoder = None

try:
    from rapidfuzz import process as rf_process
except Exception:
    rf_process = None


def simple_preprocess(text: str) -> str:
    if pd.isna(text):
        return ""
    return text.replace('\n', ' ').strip()


def train_course_classifier(train_texts: List[str], train_courses: List[str]):
    vec = TfidfVectorizer(max_features=80000, ngram_range=(1,2), stop_words='english')
    X = vec.fit_transform(train_texts)
    le = LabelEncoder()
    y = le.fit_transform(train_courses)
    clf = LogisticRegression(C=10, max_iter=1000, multi_class='ovr', n_jobs=-1)
    clf.fit(X, y)
    return vec, clf, le


def embed_texts(model, texts: List[str], batch_size=64) -> np.ndarray:
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc='embedding'):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_emb.append(emb)
    return np.vstack(all_emb)


def get_topk_tfidf_batch(tfidf_vec: TfidfVectorizer, corpus_X, q_texts: List[str], topk=100) -> List[List[int]]:
    Q = tfidf_vec.transform(q_texts)
    results = []
    batch = 128
    for i in range(0, Q.shape[0], batch):
        qbatch = Q[i:i+batch]
        sims = qbatch.dot(corpus_X.T)
        for row in sims:
            arr = np.asarray(row.toarray()).ravel() if hasattr(row, 'toarray') else row.ravel()
            k = min(topk, len(arr))
            idx = np.argpartition(-arr, range(k))[:k]
            idx = idx[np.argsort(-arr[idx])]
            results.append(idx.tolist())
    return results


def build_bm25(texts: List[str]):
    if BM25Okapi is None:
        return None, None
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized


def get_topk_bm25_single(bm25: BM25Okapi, tokenized_corpus: List[List[str]], q_tokens: List[str], topk=100):
    scores = bm25.get_scores(q_tokens)
    k = min(topk, len(scores))
    idx = np.argpartition(-scores, range(k))[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist(), scores


def ngram_set(text: str, n=5):
    s = text
    return set([s[i:i+n] for i in range(max(0, len(s) - n + 1))])


def normalize_scores(arr: np.ndarray):
    if len(arr) == 0:
        return arr
    mn = arr.min()
    mx = arr.max()
    if mx <= mn:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn + 1e-12)


def main(args):
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)

    train['text'] = train['Reviews'].fillna('').apply(simple_preprocess)
    test['text'] = test['Reviews'].fillna('').apply(simple_preprocess)
    train['text_stripped'] = train.apply(lambda r: r['text'].replace(str(r['Course']), '').strip(), axis=1)

    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)

    # exact map
    exact_map = {}
    for idx, txt in zip(train['Index'], train['text']):
        exact_map.setdefault(txt, []).append(int(idx))

    # train course classifier
    print('Training course classifier...')
    tfidf_clf, course_clf, le = train_course_classifier(train['text_stripped'].tolist(), train['Course'].tolist())
    X_test_tfidf = tfidf_clf.transform(test['text'].tolist())
    proba = course_clf.predict_proba(X_test_tfidf)
    topN = args.top_courses
    topN_idx = np.argsort(-proba, axis=1)[:, :topN]
    topN_courses = [[le.inverse_transform([i])[0] for i in row] for row in topN_idx]

    # global TF-IDF
    print('Building global TF-IDF...')
    tfidf_global = TfidfVectorizer(max_features=200000, ngram_range=(1,2), stop_words='english')
    X_global = tfidf_global.fit_transform(train['text_stripped'].tolist())

    # global BM25
    print('Building BM25...')
    bm25_global, tokenized_global = build_bm25(train['text_stripped'].tolist())

    # per-course indices map for quick subset
    course_to_indices = {}
    for idx, course in enumerate(train['Course']):
        course_to_indices.setdefault(course, []).append(idx)

    # dense embeddings
    dense_model = None
    train_embeddings = None
    if SentenceTransformer is not None:
        print('Loading dense model:', args.dense_model)
        dense_model = SentenceTransformer(args.dense_model)
        emb_path = artifacts_dir / 'train_embeddings.npy'
        if emb_path.exists():
            train_embeddings = np.load(emb_path)
        else:
            train_embeddings = embed_texts(dense_model, train['text_stripped'].tolist(), batch_size=256)
            np.save(emb_path, train_embeddings)
    else:
        print('Dense disabled (sentence-transformers not found)')

    # cross-encoder
    cross_encoder = None
    if CrossEncoder is not None:
        print('Loading cross-encoder:', args.cross_model)
        cross_encoder = CrossEncoder(args.cross_model, device=args.device)
    else:
        print('CrossEncoder not found; will use weighted heuristic scores')

    # prepare global ngram sets for a lightweight subset (optional caching)
    if args.use_ngrams:
        print('Precomputing ngram sets...')
        n = args.ngram_n
        ngram_corpus = [ngram_set(t, n=n) for t in tqdm(train['text_stripped'].tolist())]
    else:
        ngram_corpus = None

    # prepare global popular fallback
    global_popular = train['Index'].tolist()[:500]
    course_popular = {c: group['Index'].tolist()[:500] for c, group in train.groupby('Course')}

    final_predictions = []

    for qi, qrow in tqdm(test.iterrows(), total=len(test), desc='queries'):
        q_text = qrow['text']
        pool_scores = {}  # key: train index, value: dict of scores

        # 1) exact matches (highest priority)
        if q_text in exact_map:
            for tid in exact_map[q_text]:
                pool_scores[int(tid)] = pool_scores.get(int(tid), {})
                pool_scores[int(tid)]['exact'] = 1.0

        # 2) per-topN-course retrieval
        courses = topN_courses[qi]
        for course in courses:
            indices = course_to_indices.get(course, [])
            if not indices:
                continue
            # tf-idf subset
            sub_X = X_global[indices]
            qv = tfidf_global.transform([q_text])
            sims = qv.dot(sub_X.T)
            arr = np.asarray(sims.toarray()).ravel()
            k = min(args.k_sparse, len(indices))
            if k > 0:
                idxs = np.argpartition(-arr, range(k))[:k]
                for j in idxs:
                    tidx = int(train['Index'].iloc[indices[j]])
                    pool_scores.setdefault(tidx, {})
                    pool_scores[tidx]['tfidf'] = max(pool_scores[tidx].get('tfidf', 0.0), float(arr[j]))
            # BM25 subset
            if bm25_global is not None:
                # get BM25 scores for indices subset: use full bm25 scores then pick subset
                qtokens = q_text.split()
                scores = bm25_global.get_scores(qtokens)
                # scores align with train order
                # filter by indices
                sub_scores = [(idx, scores[idx]) for idx in indices]
                sub_scores.sort(key=lambda x: -x[1])
                for idx_j, sc in sub_scores[:args.k_sparse]:
                    tidx = int(train['Index'].iloc[idx_j])
                    pool_scores.setdefault(tidx, {})
                    pool_scores[tidx]['bm25'] = max(pool_scores[tidx].get('bm25', 0.0), float(sc))
            # dense subset
            if dense_model is not None and train_embeddings is not None:
                q_emb = dense_model.encode([q_text], convert_to_numpy=True)
                sub_emb = train_embeddings[indices]
                norm_sub = np.linalg.norm(sub_emb, axis=1)
                norm_q = np.linalg.norm(q_emb)
                sims = (sub_emb @ q_emb.T).ravel() / (norm_sub * norm_q + 1e-12)
                k = min(args.k_dense, len(indices))
                idxs = np.argpartition(-sims, range(k))[:k]
                for j in idxs:
                    tidx = int(train['Index'].iloc[indices[j]])
                    pool_scores.setdefault(tidx, {})
                    pool_scores[tidx]['dense'] = max(pool_scores[tidx].get('dense', 0.0), float(sims[j]))

        # 3) global retrieval (to catch missed courses)
        # TF-IDF global
        tfidf_global_hits = get_topk_tfidf_batch(tfidf_global, X_global, [q_text], topk=args.k_sparse)[0]
        for j in tfidf_global_hits:
            tidx = int(train['Index'].iloc[j])
            pool_scores.setdefault(tidx, {})
            # compute similarity quickly
            qv = tfidf_global.transform([q_text])
            sim = (qv.dot(X_global[j].T)).toarray().ravel()[0]
            pool_scores[tidx]['tfidf'] = max(pool_scores[tidx].get('tfidf', 0.0), float(sim))

        # BM25 global
        if bm25_global is not None:
            qtokens = q_text.split()
            scores = bm25_global.get_scores(qtokens)
            top_idx = np.argpartition(-scores, range(args.k_sparse))[:args.k_sparse]
            for j in top_idx:
                tidx = int(train['Index'].iloc[j])
                pool_scores.setdefault(tidx, {})
                pool_scores[tidx]['bm25'] = max(pool_scores[tidx].get('bm25', 0.0), float(scores[j]))

        # Dense global
        if dense_model is not None and train_embeddings is not None:
            q_emb = dense_model.encode([q_text], convert_to_numpy=True)
            # compute similarities in batches to avoid memory blow
            batch = 8192
            sims_all = []
            for s in range(0, train_embeddings.shape[0], batch):
                block = train_embeddings[s:s+batch]
                norm_block = np.linalg.norm(block, axis=1)
                simb = (block @ q_emb.T).ravel() / (norm_block * (np.linalg.norm(q_emb)) + 1e-12)
                sims_all.append(simb)
            sims_all = np.concatenate(sims_all)
            top_idx = np.argpartition(-sims_all, range(args.k_dense))[:args.k_dense]
            for j in top_idx:
                tidx = int(train['Index'].iloc[j])
                pool_scores.setdefault(tidx, {})
                pool_scores[tidx]['dense'] = max(pool_scores[tidx].get('dense', 0.0), float(sims_all[j]))

        # 4) fuzzy matching (rapidfuzz) to detect near duplicates
        if rf_process is not None:
            try:
                # use train texts list
                choices = train['text'].tolist()
                topf = rf_process.extract(q_text, choices, limit=args.k_fuzzy)
                for match_txt, score, j in topf:
                    tidx = int(train['Index'].iloc[j])
                    pool_scores.setdefault(tidx, {})
                    pool_scores[tidx]['fuzzy'] = max(pool_scores[tidx].get('fuzzy', 0.0), float(score)/100.0)
            except Exception:
                pass

        # 5) ngram overlap
        if args.use_ngrams:
            q_ng = ngram_set(q_text, n=args.ngram_n)
            for tidx, ng in enumerate(ngram_corpus):
                # skip heavy compare; only check if tidx in pool candidates to limit cost
                train_index = int(train['Index'].iloc[tidx])
                if train_index not in pool_scores:
                    continue
                inter = len(q_ng & ng)
                union = len(q_ng | ng) + 1e-12
                jacc = inter / union
                pool_scores[train_index]['ngram'] = jacc

        # assemble pool list and compute aggregate scores
        if not pool_scores:
            # fallback: course popular then global
            pool = []
            for c in topN_courses[qi]:
                pool.extend(course_popular.get(c, [])[:10])
            pool.extend(global_popular[:10])
            pool = list(dict.fromkeys(pool))[:10]
            final_predictions.append([int(x) for x in pool])
            continue

        cand_ids = np.array(list(pool_scores.keys()))
        # build feature arrays
        tfidf_arr = np.array([pool_scores[c].get('tfidf', 0.0) for c in cand_ids])
        bm25_arr = np.array([pool_scores[c].get('bm25', 0.0) for c in cand_ids])
        dense_arr = np.array([pool_scores[c].get('dense', 0.0) for c in cand_ids])
        fuzzy_arr = np.array([pool_scores[c].get('fuzzy', 0.0) for c in cand_ids])
        exact_arr = np.array([pool_scores[c].get('exact', 0.0) for c in cand_ids])
        ngram_arr = np.array([pool_scores[c].get('ngram', 0.0) for c in cand_ids])

        # normalize each
        tfidf_n = normalize_scores(tfidf_arr)
        bm25_n = normalize_scores(bm25_arr)
        dense_n = normalize_scores(dense_arr)
        fuzzy_n = normalize_scores(fuzzy_arr)
        exact_n = exact_arr  # binary
        ngram_n = normalize_scores(ngram_arr)

        # weighted sum ensemble (weights tuned for recall)
        w_tfidf = args.w_tfidf
        w_bm25 = args.w_bm25
        w_dense = args.w_dense
        w_fuzzy = args.w_fuzzy
        w_exact = args.w_exact
        w_ngram = args.w_ngram

        combined = (w_tfidf * tfidf_n + w_bm25 * bm25_n + w_dense * dense_n + w_fuzzy * fuzzy_n + w_exact * exact_n + w_ngram * ngram_n)

        # if cross-encoder is available, rerank top M by cross-encoder
        order = np.argsort(-combined)
        topM = min(args.cross_pool, len(order))
        top_candidates = cand_ids[order][:topM]

        if cross_encoder is not None and len(top_candidates) > 0:
            pairs = [[q_text, train.loc[train['Index'] == int(c),'text_stripped'].iloc[0]] for c in top_candidates]
            ce_scores = cross_encoder.predict(pairs, batch_size=args.cross_batch)
            ce_order = np.argsort(-np.array(ce_scores))
            final_ranked = [int(top_candidates[i]) for i in ce_order]
        else:
            final_ranked = [int(x) for x in cand_ids[np.argsort(-combined)]]

        # ensure unique and fill to 10 with course/global popular
        seen = []
        for x in final_ranked:
            if x not in seen:
                seen.append(x)
            if len(seen) >= 10:
                break
        # fill
        for c in topN_courses[qi]:
            for p in course_popular.get(c, [])[:50]:
                if int(p) not in seen:
                    seen.append(int(p))
                if len(seen) >= 10:
                    break
            if len(seen) >= 10:
                break
        j = 0
        while len(seen) < 10 and j < len(global_popular):
            gp = int(global_popular[j])
            if gp not in seen:
                seen.append(gp)
            j += 1
        # pad if still <10
        if len(seen) == 0:
            seen = [int(train['Index'].iloc[0])] * 10
        while len(seen) < 10:
            seen.append(seen[-1])

        final_predictions.append(seen[:10])

    out_df = pd.DataFrame({'Index': test['Index'], 'Index_list': [str([int(x) for x in lst]) for lst in final_predictions]})
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
    parser.add_argument('--dense_model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--cross_model', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--top_courses', type=int, default=10)
    parser.add_argument('--k_sparse', type=int, default=300)
    parser.add_argument('--k_dense', type=int, default=300)
    parser.add_argument('--k_fuzzy', type=int, default=50)
    parser.add_argument('--k_union', type=int, default=1500)
    parser.add_argument('--cross_pool', type=int, default=500)
    parser.add_argument('--cross_batch', type=int, default=128)
    parser.add_argument('--use_ngrams', action='store_true')
    parser.add_argument('--ngram_n', type=int, default=5)

    # ensemble weights (tune these to favor recall)
    parser.add_argument('--w_tfidf', type=float, default=1.0)
    parser.add_argument('--w_bm25', type=float, default=1.0)
    parser.add_argument('--w_dense', type=float, default=1.0)
    parser.add_argument('--w_fuzzy', type=float, default=1.0)
    parser.add_argument('--w_exact', type=float, default=3.0)
    parser.add_argument('--w_ngram', type=float, default=0.5)

    args = parser.parse_args()
    main(args)
