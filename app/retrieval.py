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

        query_lower = (query or "").lower()
        merged: List[ScoredItem] = []
        for idx, item in enumerate(self.items):
            combined = bm25_weight * bm25_norm[idx] + vector_weight * vector_norm[idx]
            combined += _keyword_boost(query_lower, item)
            merged.append(
                ScoredItem(
                    item=item,
                    bm25_score=float(bm25_norm[idx]),
                    vector_score=float(vector_norm[idx]),
                    score=float(combined),
                )
            )

        merged.sort(key=lambda entry: entry.score, reverse=True)
        return merged[:top_k]


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val - min_val <= 1e-6:
        return np.zeros_like(values, dtype="float32")
    return (values - min_val) / (max_val - min_val)


def _keyword_boost(query_lower: str, item: CatalogItem) -> float:
    if not query_lower:
        return 0.0
    item_name = item.name.lower()
    item_keys = " ".join(item.keys or []).lower()
    item_text = f"{item_name} {item_keys}"

    boost = 0.0
    if item_name and item_name in query_lower:
        boost += 0.3

    phrase_map = {
        "opq": "opq",
        "sjt": "situational",
        "situational judgement": "situational",
        "situational judgment": "situational",
        "numerical reasoning": "numerical",
        "verbal reasoning": "verbal",
        "inductive reasoning": "inductive",
        "deductive reasoning": "deductive",
        "personality": "personality",
        "cognitive": "cognitive",
        "g+": "g+",
    }

    for phrase, token in phrase_map.items():
        if phrase in query_lower and token in item_text:
            boost += 0.08

    return boost
