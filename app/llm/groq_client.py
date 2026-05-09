from typing import Any, Dict

from groq import Groq

from .base import LLMClient
from ..utils import safe_json_loads


class GroqClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        self._client = Groq(api_key=api_key)
        self._model = model

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return safe_json_loads(content)
