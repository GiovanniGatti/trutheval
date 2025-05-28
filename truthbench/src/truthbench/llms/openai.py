from typing import Dict, List

from openai import OpenAI

from truthbench.pipeline import LLM


class GPT(LLM):

    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self._client = client
        self._model = model

    def query(self, messages: List[Dict[str, str]]) -> str:
        completion = self._client.chat.completions.create(model=self._model, messages=messages)
        return completion.choices[0].message.content.strip()
