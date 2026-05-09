import glob
import os
import re
import sys
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.agent import AssessmentAgent


def parse_user_turns(text: str) -> List[str]:
    turns: List[str] = []
    parts = re.split(r"^### Turn\s+\d+\s*$", text, flags=re.MULTILINE)
    for part in parts:
        if "**User**" not in part:
            continue
        user_block = part.split("**User**", 1)[1]
        if "**Agent**" in user_block:
            user_block = user_block.split("**Agent**", 1)[0]
        lines = []
        for line in user_block.splitlines():
            line = line.strip()
            if line.startswith(">"):
                lines.append(line.lstrip(">").strip())
        if lines:
            turns.append(" ".join(lines))
    return turns


def parse_last_table_urls(text: str) -> List[str]:
    lines = text.splitlines()
    table_start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("| #") and "URL" in line:
            table_start = idx
    if table_start is None:
        return []

    urls: List[str] = []
    for line in lines[table_start + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        url_match = re.search(r"<(https?://[^>]+)>", line)
        if url_match:
            urls.append(url_match.group(1))
    return urls


def evaluate_conversation(agent: AssessmentAgent, path: str) -> Tuple[float, Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    user_turns = parse_user_turns(text)
    expected_urls = parse_last_table_urls(text)

    messages: List[Dict[str, str]] = []
    result = None
    for user in user_turns:
        messages.append({"role": "user", "content": user})
        result = agent.respond(messages)
        messages.append({"role": "assistant", "content": result.get("reply", "")})
        if result.get("end_of_conversation"):
            break

    got_urls = []
    if result:
        for rec in result.get("recommendations", []):
            url = getattr(rec, "url", None)
            if not url and isinstance(rec, dict):
                url = rec.get("url")
            if url:
                got_urls.append(url)

    expected_set = set(expected_urls)
    got_set = set(got_urls)
    hits = len(expected_set & got_set)
    total = len(expected_set)
    recall = hits / total if total else 0.0
    return recall, {"hits": hits, "total": total, "returned": len(got_urls)}


def main() -> None:
    catalog_path = os.path.join(BASE_DIR, "data", "raw", "shl_product_catalog.json")
    agent = AssessmentAgent(catalog_path)

    paths = sorted(
        glob.glob(
            os.path.join(BASE_DIR, "data", "GenAI_SampleConversations", "C*.md")
        )
    )
    if not paths:
        print("No sample conversation files found.")
        return

    recalls: List[float] = []
    for path in paths:
        recall, stats = evaluate_conversation(agent, path)
        recalls.append(recall)
        name = os.path.basename(path)
        print(f"{name}: Recall@10={recall:.2f} ({stats['hits']}/{stats['total']}), returned={stats['returned']}")

    mean_recall = sum(recalls) / len(recalls)
    print(f"Mean Recall@10: {mean_recall:.2f}")


if __name__ == "__main__":
    main()
