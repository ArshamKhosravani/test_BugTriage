"""
utils.py – universal helpers for DL-based bug-assignment
───────────────────────────────────────────────────────────────────────────────
This file is a **superset** of the original: every pre-existing helper
(batch_iter, pad_sentences, build_vocab, embedding loaders, …) is still here,
and a brand-new data-loader now supports:
  1. Classic single-CSV corpora  (gcc_data.csv, jdt_data.csv, …)
  2. “*_my” corpora              (eclipse_my.csv, office_my.csv, mozilla_my.csv)
  3. Mozilla multi-part bundle   (mozilla_my1.csv  +  mozilla_my_doc_l.csv
                                                  +  mozilla_my_cc_l.csv)
Any future dataset that follows the same idea—one main CSV plus optional
companions sharing the same prefix and bug-ID column—will load automatically.
"""
from __future__ import annotations

import os, re, glob, io, json, pickle, itertools, random, math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Iterable, Dict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1 ─ Text cleaning (unchanged)
# --------------------------------------------------------------------------- #
import nltk
from nltk.corpus import stopwords
# Make sure stop-word list is downloaded once
try:
    _STOP = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    _STOP = set(stopwords.words("english"))

_re_multi_space = re.compile(r"\s{2,}")

def clean_str(text: str) -> str:
    """Basic token-level cleaning → lower-case, strip stop-words, squash spaces."""
    text = re.sub(r"[^A-Za-z(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " had", text)
    text = re.sub(r"\'ll", " will", text)
    text = _re_multi_space.sub(" ", text).strip().lower()
    return " ".join(tok for tok in text.split() if tok not in _STOP and len(tok) > 1)

# --------------------------------------------------------------------------- #
# 2 ─ NEW: universal dataset loader
# --------------------------------------------------------------------------- #
_TEXT_PAT  = re.compile(r"(summary|desc|text|comment|doc|extra|context)", re.I)
_LABEL_PAT = re.compile(r"(assignee|owner|developer|assigned_to|cc|login)", re.I)
_ID_PAT    = re.compile(r"(bug[_ ]?id$|^id$)", re.I)

def _pick_id(df: pd.DataFrame) -> str | None:
    """Return the first column that looks like a Bug-ID."""
    for col in df.columns:
        if _ID_PAT.search(col):
            return col
    return None

def _read_csv(path: str, *, id_name: str | None = None) -> pd.DataFrame:
    """Read CSV as strings (avoids pandas’ mixed-dtype slowing things down)."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if id_name and id_name not in df.columns:
        alt = [c for c in df.columns if _ID_PAT.search(c)]
        if alt:
            df = df.rename(columns={alt[0]: id_name})
    return df

def load_data_and_labels(main_csv: str) -> Tuple[List[str],
                                                 np.ndarray,
                                                 List[str],
                                                 List[str]]:
    """
    Parameters
    ----------
    main_csv : str
        Path to *any* CSV in the corpus (e.g. mozilla_my1.csv OR eclipse_my.csv).

    Returns
    -------
    texts_clean : List[str]
    y_onehot    : np.ndarray   shape = (n_samples, n_labels)
    labels_raw  : List[str]
    label_set   : List[str]
    """
    base_dir, fname = os.path.split(main_csv)
    base_dir = base_dir or "."
    # common root, e.g.  "mozilla_my"
    root = re.sub(r"(_(?:doc|cc).*?)?\.csv$", "", fname)

    # ── gather all sibling CSVs sharing the prefix ───────────────────────────
    csv_files = sorted(glob.glob(os.path.join(base_dir, f"{root}*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for prefix “{root}”")

    primary = Path(main_csv).resolve() if Path(main_csv).exists() else Path(csv_files[0])
    df_main = _read_csv(str(primary))
    bug_id  = _pick_id(df_main) or "row_id"
    if bug_id not in df_main.columns:
        df_main[bug_id] = np.arange(len(df_main))

    # ── left-join companion files on bug-ID, concatenate relevant columns ───
    for csv_path in csv_files:
        if Path(csv_path).resolve() == primary:
            continue
        df_c = _read_csv(csv_path, id_name=bug_id)
        tcols = [c for c in df_c.columns if _TEXT_PAT.search(c)]
        lcols = [c for c in df_c.columns if _LABEL_PAT.search(c)]
        if not tcols and not lcols:
            continue
        agg: Dict[str, dict] = defaultdict(lambda: {"_comp_text": "", "_comp_label": ""})
        for _, row in df_c.iterrows():
            bid = row[bug_id]
            if tcols:
                agg[bid]["_comp_text"]  += " ".join(str(row[c]) for c in tcols if pd.notna(row[c])) + " "
            if lcols:
                agg[bid]["_comp_label"] += " ".join(str(row[c]) for c in lcols if pd.notna(row[c])) + " "
        df_join = (
            pd.DataFrame.from_dict(agg, orient="index")
            .reset_index()
            .rename(columns={"index": bug_id})
        )
        df_main = df_main.merge(df_join, on=bug_id, how="left")

    # ── if we still have no text, fall back to *.clean.txt ──────────────────
    text_cols = [c for c in df_main.columns if _TEXT_PAT.search(c)]
    if "_comp_text" in df_main.columns:
        text_cols.append("_comp_text")

    if text_cols:
        texts = (
            df_main[text_cols]
            .fillna("")
            .agg(" ".join, axis=1)
            .map(clean_str)
            .tolist()
        )
    else:
        txt_fallback = os.path.join(base_dir, f"{root}.clean.txt")
        if not os.path.exists(txt_fallback):
            raise FileNotFoundError(f"No text columns and no fallback {txt_fallback}")
        with io.open(txt_fallback, encoding="utf-8") as fh:
            texts = [clean_str(line) for line in fh if line.strip()]

    # ── label extraction ────────────────────────────────────────────────────
    label_col = next((c for c in df_main.columns if _LABEL_PAT.search(c)), None)
    if label_col is None and "_comp_label" in df_main.columns:
        label_col = "_comp_label"

    labels = (
        df_main[label_col].fillna("unknown").astype(str).tolist()
        if label_col else ["unknown"] * len(texts)
    )

    # ── one-hot encode ──────────────────────────────────────────────────────
    label_set = sorted(set(labels))
    lab_to_idx = {lab: i for i, lab in enumerate(label_set)}
    y = np.zeros((len(labels), len(label_set)), dtype=np.int8)
    for i, lab in enumerate(labels):
        y[i, lab_to_idx[lab]] = 1

    return texts, y, labels, label_set

# --------------------------------------------------------------------------- #
# 3 ─ Original helpers (unchanged)
# --------------------------------------------------------------------------- #
def pad_sentences(sentences: List[List[int]],
                  max_length: int | None = None,
                  padding_idx: int = 0) -> np.ndarray:
    """Pad / truncate to a uniform length so we can stack into an array."""
    if max_length is None:
        max_length = max(len(s) for s in sentences)
    padded = np.full((len(sentences), max_length), padding_idx, dtype=np.int64)
    for i, s in enumerate(sentences):
        padded[i, : min(len(s), max_length)] = s[:max_length]
    return padded

def batch_iter(data: List | np.ndarray,
               batch_size: int,
               num_epochs: int,
               shuffle: bool = True):
    """Generator that yields batches for each epoch."""
    data = np.array(data)
    data_size = len(data)
    num_batches = math.ceil(data_size / batch_size)
    for epoch in range(num_epochs):
        idx = np.random.permutation(np.arange(data_size)) if shuffle else np.arange(data_size)
        shuffled = data[idx]
        for b in range(num_batches):
            start = b * batch_size
            end   = min((b + 1) * batch_size, data_size)
            yield shuffled[start:end]

def build_vocab(sentences: Iterable[str],
                max_words: int | None = None) -> Tuple[dict, list]:
    """Tokenise and build word→index map."""
    counter = Counter(itertools.chain.from_iterable(s.split() for s in sentences))
    vocab = {w: i + 2 for i, (w, _) in enumerate(counter.most_common(max_words))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    indices = [[vocab.get(tok, 1) for tok in s.split()] for s in sentences]
    return vocab, indices

# ---- Embedding loaders (Word2Vec, GloVe, fastText, …) --------------------- #
# The original project already had these; they are reproduced verbatim below.
# If you added new embeddings to `config.yml`, just extend the dict paths.

def _load_word2vec_bin(path: str, vocab: dict, dim: int) -> np.ndarray:
    kv = KeyedVectors.load_word2vec_format(path, binary=True)
    embedding = np.random.normal(0, 0.05, (len(vocab), dim)).astype(np.float32)
    for word, idx in vocab.items():
        if word in kv.key_to_index:
            embedding[idx] = kv[word]
    return embedding

def _load_glove_txt(path: str, vocab: dict, dim: int) -> np.ndarray:
    embedding = np.random.normal(0, 0.05, (len(vocab), dim)).astype(np.float32)
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            if word in vocab:
                embedding[vocab[word]] = np.asarray(parts[1:], dtype=np.float32)
    return embedding

def load_embedding_matrix(embedding_name: str,
                          vocab: dict,
                          config: dict,
                          dim: int = 300) -> np.ndarray:
    """
    Load (or create) an embedding matrix *aligned* with the vocabulary indices.
    Currently supports: word2vec-bin, glove-txt, random.
    """
    if embedding_name == "random":
        return np.random.normal(0, 0.05, (len(vocab), dim)).astype(np.float32)

    info = config["embeddings"].get(embedding_name)
    if info is None:
        raise ValueError(f"Unknown embedding “{embedding_name}”")

    path = info["path"]
    if embedding_name.startswith("word2vec"):
        return _load_word2vec_bin(path, vocab, info["dim"])
    elif embedding_name.startswith("glove"):
        return _load_glove_txt(path, vocab, info["dim"])
    else:
        raise NotImplementedError(f"Loader for {embedding_name} not implemented")

# --------------------------------------------------------------------------- #
# 4 ─ Metrics (top-k accuracy) – unchanged
# --------------------------------------------------------------------------- #
def top_k_accuracy(preds: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Compute top-k accuracy given probability or logit matrix."""
    topk = np.argsort(preds, axis=1)[:, -k:]
    match = sum(1 for i, labs in enumerate(labels) if any(labs[topk[i]]))
    return match / len(labels)
