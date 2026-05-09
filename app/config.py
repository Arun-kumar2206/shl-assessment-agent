import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    embedding_model: str
    bm25_weight: float
    vector_weight: float
    top_k: int
    max_recs: int


def get_settings() -> Settings:
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        bm25_weight=float(os.getenv("BM25_WEIGHT", "0.45")),
        vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.55")),
        top_k=int(os.getenv("TOP_K", "5")),
        max_recs=int(os.getenv("MAX_RECS", "10")),
    )
