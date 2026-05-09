from __future__ import annotations

import re
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from .catalog import CatalogItem, load_catalog
from .config import Settings, get_settings
from .llm.base import LLMClient, SimpleLLM
from .llm.groq_client import GroqClient
from .prompts import (
    COMPARISON_SYSTEM,
    COMPARISON_USER,
    REQUIREMENT_EXTRACTION_SYSTEM,
    REQUIREMENT_EXTRACTION_USER,
    RECOMMENDATION_SYSTEM,
    RECOMMENDATION_USER,
)
from .retrieval import HybridRetriever
from .schema import Recommendation
from .utils import normalize_text


class AgentState(TypedDict, total=False):
    messages: List[Dict[str, str]]
    last_user: str
    off_topic: bool
    comparison: bool
    comparison_items: List[CatalogItem]
    requirements: Dict[str, Any]
    enough_context: bool
    clarification_question: str
    retrieved: List[CatalogItem]
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


class AssessmentAgent:
    def __init__(self, catalog_path: str):
        self._settings = get_settings()
        self._catalog = load_catalog(catalog_path)
        self._catalog_by_name = {item.name.lower(): item for item in self._catalog}
        self._retriever = HybridRetriever(
            self._catalog, self._settings.embedding_model
        )
        self._llm = self._build_llm(self._settings)
        self._graph = self._build_graph()

    def _build_llm(self, settings: Settings) -> LLMClient:
        if settings.groq_api_key:
            return GroqClient(settings.groq_api_key, settings.groq_model)
        return SimpleLLM()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("analyze", self._analyze)
        graph.add_node("refuse", self._refuse)
        graph.add_node("compare", self._compare)
        graph.add_node("clarify", self._clarify)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("respond", self._respond)

        graph.set_entry_point("analyze")
        graph.add_conditional_edges(
            "analyze",
            self._route,
            {
                "refuse": "refuse",
                "compare": "compare",
                "clarify": "clarify",
                "retrieve": "retrieve",
            },
        )
        graph.add_edge("refuse", END)
        graph.add_edge("compare", END)
        graph.add_edge("clarify", END)
        graph.add_edge("retrieve", "respond")
        graph.add_edge("respond", END)
        return graph.compile()

    def respond(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        last_user = self._get_last_user_message(messages)
        state: AgentState = {
            "messages": messages,
            "last_user": last_user,
        }
        result = self._graph.invoke(state)
        return {
            "reply": result.get("reply", ""),
            "recommendations": result.get("recommendations", []),
            "end_of_conversation": result.get("end_of_conversation", False),
        }

    def _route(self, state: AgentState) -> str:
        if state.get("off_topic"):
            return "refuse"
        if state.get("comparison"):
            return "compare"
        if not state.get("enough_context"):
            return "clarify"
        return "retrieve"

    def _analyze(self, state: AgentState) -> AgentState:
        last_user = state.get("last_user", "")
        off_topic = self._is_off_topic(last_user)
        comparison_items = self._find_comparison_items(last_user)
        comparison = len(comparison_items) >= 2 and self._is_comparison_request(last_user)
        requirements = self._extract_requirements(state.get("messages", []))
        enough_context = self._has_enough_context(requirements)

        return {
            **state,
            "off_topic": off_topic,
            "comparison": comparison,
            "comparison_items": comparison_items,
            "requirements": requirements,
            "enough_context": enough_context,
        }

    def _refuse(self, state: AgentState) -> AgentState:
        return {
            **state,
            "reply": "I can only help with SHL assessment recommendations and comparisons.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    def _compare(self, state: AgentState) -> AgentState:
        items = state.get("comparison_items", [])[:2]
        if len(items) < 2:
            return {
                **state,
                "reply": "Which two SHL assessments would you like to compare?",
                "recommendations": [],
                "end_of_conversation": False,
            }
        doc_a = _format_item(items[0])
        doc_b = _format_item(items[1])
        prompt = COMPARISON_USER.format(doc_a=doc_a, doc_b=doc_b)
        reply = self._llm.generate_text(COMPARISON_SYSTEM, prompt)
        return {
            **state,
            "reply": reply,
            "recommendations": [],
            "end_of_conversation": False,
        }

    def _clarify(self, state: AgentState) -> AgentState:
        requirements = state.get("requirements", {})
        missing = self._missing_fields(requirements)
        question = self._build_clarification(missing)
        return {
            **state,
            "reply": question,
            "recommendations": [],
            "end_of_conversation": False,
        }

    def _retrieve(self, state: AgentState) -> AgentState:
        requirements = state.get("requirements", {})
        query = self._build_query(state.get("messages", []), requirements)
        scored = self._retriever.search(
            query,
            top_k=self._settings.max_recs,
            bm25_weight=self._settings.bm25_weight,
            vector_weight=self._settings.vector_weight,
        )
        items = [entry.item for entry in scored]
        return {
            **state,
            "retrieved": items,
        }

    def _respond(self, state: AgentState) -> AgentState:
        retrieved = state.get("retrieved", [])
        if not retrieved:
            return {
                **state,
                "reply": "I could not find a close match in the catalog. Can you clarify the role or key skills?",
                "recommendations": [],
                "end_of_conversation": False,
            }
        shortlist = retrieved[: self._settings.top_k]
        recommendations = [
            Recommendation(name=item.name, url=item.url, test_type=item.test_type)
            for item in shortlist
        ]
        summary = self._build_summary(state.get("requirements", {}))
        assessments_text = "\n".join(_format_item(item) for item in shortlist)
        prompt = RECOMMENDATION_USER.format(
            summary=summary, assessments=assessments_text
        )
        reply = self._llm.generate_text(RECOMMENDATION_SYSTEM, prompt)
        return {
            **state,
            "reply": reply,
            "recommendations": recommendations,
            "end_of_conversation": True,
        }

    def _build_query(self, messages: List[Dict[str, str]], requirements: Dict[str, Any]) -> str:
        user_turns = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
        recent_turns = user_turns[-6:] if user_turns else []
        recent_text = " ".join(recent_turns)
        parts = [recent_text]
        expanded = self._expand_query_terms(recent_text)
        if expanded:
            parts.append(expanded)
        for key in ["role", "seniority", "assessment_type", "constraints"]:
            value = requirements.get(key, "")
            if value:
                parts.append(str(value))
        skills = requirements.get("skills", [])
        if skills:
            parts.append("skills: " + ", ".join(skills))
        return normalize_text(" ".join(parts))

    def _expand_query_terms(self, text: str) -> str:
        lowered = (text or "").lower()
        expansions: List[str] = []
        if "contact center" in lowered or "call center" in lowered:
            expansions.extend(["customer service", "call simulation", "svar spoken english"])
        if "customer service" in lowered:
            expansions.extend(["contact center", "call simulation", "svar spoken english"])
        if "sales" in lowered:
            expansions.extend(["opq mq sales report", "sales transformation"])
        if "report" in lowered:
            expansions.extend(["report", "report 2.0", "leadership report"])
        if "safety" in lowered or "dependability" in lowered:
            expansions.extend(["dependability and safety instrument", "safety and dependability"])
        if "hipaa" in lowered or "medical" in lowered:
            expansions.extend(["medical terminology", "hipaa security"])
        if "admin assistant" in lowered or "admin" in lowered:
            expansions.extend(["ms excel", "ms word", "microsoft word"])
        if "java" in lowered:
            expansions.extend(["core java", "spring", "sql", "restful web services"])
        if "rust" in lowered:
            expansions.extend(["live coding", "linux programming", "networking and implementation"])
        if "graduate" in lowered:
            expansions.extend(["graduate scenarios", "verify interactive"])
        if "numerical reasoning" in lowered:
            expansions.extend(["verify interactive numerical reasoning"])
        if "leadership" in lowered or "executive" in lowered:
            expansions.extend(["opq leadership report", "opq32r"])
        return " ".join(dict.fromkeys(expansions))


    def _build_summary(self, requirements: Dict[str, Any]) -> str:
        role = requirements.get("role", "")
        seniority = requirements.get("seniority", "")
        skills = ", ".join(requirements.get("skills", []) or [])
        assessment_type = requirements.get("assessment_type", "")
        constraints = requirements.get("constraints", "")
        parts = [
            f"Role: {role}" if role else "",
            f"Seniority: {seniority}" if seniority else "",
            f"Skills: {skills}" if skills else "",
            f"Assessment focus: {assessment_type}" if assessment_type else "",
            f"Constraints: {constraints}" if constraints else "",
        ]
        return normalize_text(" | ".join(part for part in parts if part))

    def _extract_requirements(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        convo = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in messages if msg.get("content")
        )
        prompt = REQUIREMENT_EXTRACTION_USER.format(conversation=convo)
        data = self._llm.generate_json(REQUIREMENT_EXTRACTION_SYSTEM, prompt)
        role = data.get("role", "") or self._infer_role(convo)
        seniority = data.get("seniority", "") or self._infer_seniority(convo)
        skills = data.get("skills", []) or self._infer_skills(convo)
        if isinstance(skills, str):
            skills = [skill.strip() for skill in skills.split(",") if skill.strip()]
        return {
            "role": role,
            "seniority": seniority,
            "skills": skills,
            "assessment_type": data.get("assessment_type", ""),
            "constraints": data.get("constraints", ""),
        }

    def _has_enough_context(self, requirements: Dict[str, Any]) -> bool:
        role = requirements.get("role", "")
        has_role = bool(role)
        has_signal = bool(
            requirements.get("seniority")
            or requirements.get("skills")
            or requirements.get("assessment_type")
        )
        return has_role and has_signal

    def _missing_fields(self, requirements: Dict[str, Any]) -> List[str]:
        missing = []
        if not requirements.get("role"):
            missing.append("role")
        if not requirements.get("seniority"):
            missing.append("seniority")
        if not requirements.get("skills"):
            missing.append("skills")
        return missing

    def _build_clarification(self, missing: List[str]) -> str:
        if not missing:
            return "Could you share more about the role and key skills?"
        if "role" in missing:
            return "What role are you hiring for?"
        if "seniority" in missing:
            return "What seniority level should I target?"
        if "skills" in missing:
            return "What key skills or technologies are most important?"
        return "Could you share more about the role and key skills?"

    def _is_off_topic(self, text: str) -> bool:
        lowered = (text or "").lower()
        blocked_terms = [
            "legal",
            "salary",
            "compensation",
            "visa",
            "immigration",
            "contract",
            "tax",
            "ignore previous",
            "system prompt",
            "developer instructions",
            "jailbreak",
        ]
        return any(term in lowered for term in blocked_terms)

    def _is_comparison_request(self, text: str) -> bool:
        lowered = (text or "").lower()
        cues = ["compare", "difference", "versus", "vs"]
        return any(cue in lowered for cue in cues)

    def _find_comparison_items(self, text: str) -> List[CatalogItem]:
        lowered = (text or "").lower()
        matches = []
        for name, item in self._catalog_by_name.items():
            if name in lowered:
                matches.append(item)
        return matches

    def _get_last_user_message(self, messages: List[Dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _infer_role(self, text: str) -> str:
        lowered = text.lower()
        role_terms = [
            "developer",
            "engineer",
            "analyst",
            "manager",
            "designer",
            "accountant",
            "sales",
            "support",
        ]
        for term in role_terms:
            if term in lowered:
                return term
        return ""

    def _infer_seniority(self, text: str) -> str:
        lowered = text.lower()
        if any(term in lowered for term in ["entry", "junior", "graduate", "intern"]):
            return "entry"
        if any(term in lowered for term in ["mid", "mid-level", "mid level"]):
            return "mid"
        if any(term in lowered for term in ["senior", "lead", "principal"]):
            return "senior"
        if "manager" in lowered:
            return "manager"
        return ""

    def _infer_skills(self, text: str) -> List[str]:
        lowered = text.lower()
        skill_terms = [
            "java",
            "python",
            "c#",
            ".net",
            "javascript",
            "react",
            "sql",
            "aws",
            "azure",
            "communication",
            "stakeholder",
            "leadership",
        ]
        return [term for term in skill_terms if term in lowered]


def _format_item(item: CatalogItem) -> str:
    return (
        f"Name: {item.name}\n"
        f"URL: {item.url}\n"
        f"Description: {item.description}\n"
        f"Job levels: {', '.join(item.job_levels)}\n"
        f"Duration: {item.duration}\n"
        f"Test type: {item.test_type}"
    )
