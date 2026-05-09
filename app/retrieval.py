from dataclasses import dataclass
from typing import Dict, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .catalog import CatalogItem
from .utils import tokenize


@dataclass
class ScoredItem:
    item: CatalogItem
    bm25_score: float
    vector_score: float
    score: float


class HybridRetriever:
    def __init__(self, items: List[CatalogItem], model_name: str):
        self.items = items
        self._tokenized_corpus = [tokenize(item.doc_text) for item in items]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._embedder = SentenceTransformer(model_name)
        self._vectors = self._embed_texts([item.doc_text for item in items])
        self._index = self._build_index(self._vectors)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = self._embedder.encode(texts, normalize_embeddings=True)
        return vectors.astype("float32")

    def _build_index(self, vectors: np.ndarray) -> faiss.Index:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def search(self, query: str, top_k: int, bm25_weight: float, vector_weight: float) -> List[ScoredItem]:
        bm25_scores = self._bm25.get_scores(tokenize(query))
        bm25_norm = _min_max_normalize(bm25_scores)

        query_vec = self._embed_texts([query])
        search_k = min(top_k, len(self.items))
        scores, indices = self._index.search(query_vec, search_k)
        vector_scores = np.zeros(len(self.items), dtype="float32")
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            vector_scores[idx] = score
        vector_norm = _min_max_normalize(vector_scores)

        merged: List[ScoredItem] = []
        for idx, item in enumerate(self.items):
            combined = bm25_weight * bm25_norm[idx] + vector_weight * vector_norm[idx]
            merged.append(
                ScoredItem(
                    item=item,
                    bm25_score=float(bm25_norm[idx]),
                    vector_score=float(vector_norm[idx]),
                    score=float(combined),
                )
            )

        merged = _simple_rerank(merged, query)
        return merged[:top_k]


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val - min_val <= 1e-6:
        return np.zeros_like(values, dtype="float32")
    return (values - min_val) / (max_val - min_val)


def _simple_rerank(items: List[ScoredItem], query: str) -> List[ScoredItem]:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        items.sort(key=lambda entry: entry.score, reverse=True)
        return items

    for entry in items:
        overlap = len(query_tokens.intersection(entry.item.normalized_tokens))
        if overlap:
            entry.score += min(0.15, 0.02 * overlap)

    items.sort(key=lambda entry: entry.score, reverse=True)
    return items
