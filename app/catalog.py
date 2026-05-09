import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .utils import normalize_text


@dataclass(frozen=True)
class CatalogItem:
    entity_id: str
    name: str
    url: str
    description: str
    job_levels: List[str]
    languages: List[str]
    duration: str
    keys: List[str]
    test_type: str
    raw: Dict[str, Any]

    @property
    def doc_text(self) -> str:
        parts = [self.name, self.description]
        if self.job_levels:
            parts.append("Job levels: " + ", ".join(self.job_levels))
        if self.keys:
            parts.append("Keys: " + ", ".join(self.keys))
        if self.duration:
            parts.append("Duration: " + self.duration)
        if self.languages:
            parts.append("Languages: " + ", ".join(self.languages))
        if self.test_type:
            parts.append("Test type: " + self.test_type)
        return normalize_text(". ".join(parts))


def infer_test_type(item: Dict[str, Any]) -> str:
    text_parts = [item.get("name", ""), item.get("description", "")]
    keys = item.get("keys", []) or []
    text_parts.extend(keys)
    text = normalize_text(" ".join(text_parts)).lower()

    personality_terms = [
        "personality",
        "behavior",
        "behaviour",
        "opq",
        "values",
        "motivation",
        "temperament",
        "traits",
    ]
    situational_terms = [
        "situational judgment",
        "situational judgement",
        "sjt",
        "scenario",
        "workplace situations",
    ]
    knowledge_terms = [
        "knowledge",
        "technical",
        "cognitive",
        "ability",
        "reasoning",
        "numerical",
        "verbal",
        "logical",
        "programming",
        "software",
        "skills",
        "aptitude",
    ]

    if any(term in text for term in personality_terms):
        return "P"
    if any(term in text for term in situational_terms):
        return "S"
    if any(term in text for term in knowledge_terms):
        return "K"
    return ""


def load_catalog(path: str) -> List[CatalogItem]:
    with open(path, "r", encoding="utf-8") as handle:
        raw_items = json.load(handle)

    items: List[CatalogItem] = []
    for raw in raw_items:
        url = raw.get("link", "")
        if not url:
            continue
        test_type = infer_test_type(raw)
        items.append(
            CatalogItem(
                entity_id=str(raw.get("entity_id", "")),
                name=raw.get("name", "").strip(),
                url=url,
                description=raw.get("description", "").strip(),
                job_levels=raw.get("job_levels", []) or [],
                languages=raw.get("languages", []) or [],
                duration=raw.get("duration", "").strip(),
                keys=raw.get("keys", []) or [],
                test_type=test_type,
                raw=raw,
            )
        )
    return items
