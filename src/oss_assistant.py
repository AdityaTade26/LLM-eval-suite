"""
OSS Assistant — Qwen2.5 via HuggingFace Inference API
"""

import time
import requests
from typing import List, Dict


SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer questions accurately and concisely. "
    "If you don't know something, say so rather than guessing."
)


class OSSAssistant:
    """
    Wraps the HuggingFace Inference API for open-source models.
    Supports multi-turn conversation with short-term memory.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        hf_token: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.model_id = model_id
        self.hf_token = hf_token
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions"

    def _build_messages(
        self, history: List[Dict], user_input: str
    ) -> List[Dict]:
        """Build messages list with system prompt + history + new user turn."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Include prior conversation (exclude latency metadata)
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        messages.append({"role": "user", "content": user_input})
        return messages

    def chat(self, history: List[Dict], user_input: str) -> Dict:
        """
        Send a message and return response with latency.
        Returns: {"text": str, "latency": float, "error": str|None}
        """
        messages = self._build_messages(history, user_input)

        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        start = time.time()
        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            latency = time.time() - start

            if resp.status_code != 200:
                return {
                    "text": f"[API Error {resp.status_code}]: {resp.text[:200]}",
                    "latency": latency,
                    "error": resp.text,
                }

            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return {"text": text, "latency": latency, "error": None}

        except requests.exceptions.Timeout:
            latency = time.time() - start
            return {
                "text": "[Timeout] The OSS model took too long to respond.",
                "latency": latency,
                "error": "timeout",
            }
        except Exception as e:
            latency = time.time() - start
            return {
                "text": f"[Error]: {str(e)}",
                "latency": latency,
                "error": str(e),
            }

    def single_turn(self, prompt: str) -> Dict:
        """Convenience method for single-turn inference (used in evaluation)."""
        return self.chat([], prompt)
