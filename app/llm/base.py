from typing import Any, Dict


class LLMClient:
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class SimpleLLM(LLMClient):
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return "I can help with SHL assessment recommendations once I have the key role details."

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return {}
