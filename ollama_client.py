"""
Ollama API client for LLM agent communication.
Based on the reference implementation from 2d-multi-places-simulation-on-fire-public.
"""
import json
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REPEAT_PENALTY = 1.1
DEFAULT_REPEAT_LAST_N = 128
DEFAULT_MIN_P = 0.05
API_TIMEOUT = 120
CONNECTION_CHECK_TIMEOUT = 5


class OllamaClient:
    """Client for the Ollama local LLM API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        repeat_last_n: int = DEFAULT_REPEAT_LAST_N,
        min_p: float = DEFAULT_MIN_P,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        self.min_p = min_p
        self.api_url = f"{self.base_url}/api/generate"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        force_json: bool = True,
    ) -> str:
        """
        Generate a response from Ollama.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system-level instruction.
            temperature: Override instance default.
            max_tokens: Override instance default.
            force_json: Request JSON-formatted output (format: "json").

        Returns:
            Raw response string (expected to be JSON when force_json=True).
        """
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "repeat_penalty": self.repeat_penalty,
                "repeat_last_n": self.repeat_last_n,
                "min_p": self.min_p,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if force_json:
            payload["format"] = "json"

        try:
            response = requests.post(self.api_url, json=payload, timeout=API_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            logger.error("Ollama に接続できません。Ollama が起動しているか確認してください。")
            return ""
        except requests.exceptions.Timeout:
            logger.error(f"Ollama の応答がタイムアウトしました（{API_TIMEOUT}秒）。")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API エラー: {e}")
            return ""
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            return ""

    def check_connection(self) -> bool:
        """Ollama サーバーへの接続を確認する。"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=CONNECTION_CHECK_TIMEOUT,
            )
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Ollama で利用可能なモデルの一覧を返す。"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=CONNECTION_CHECK_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"モデル一覧の取得に失敗しました: {e}")
            return []

    def check_model_exists(self) -> bool:
        """指定されたモデルが Ollama に存在するか確認する。"""
        available = self.list_models()
        # "qwen3:8b" は "qwen3:8b" または "qwen3:8b-..." で登録されている場合がある
        return any(m == self.model or m.startswith(self.model.split(":")[0] + ":") and self.model in m
                   for m in available) or self.model in available
