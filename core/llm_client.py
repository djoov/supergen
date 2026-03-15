import requests
import logging
from core.config import config
from openai import OpenAI

logger = logging.getLogger(__name__)


class LocalLLM:
    """Wrapper for Ollama that provides multiple interfaces:
       1. Raw /api/generate (simple text completion)
       2. OpenAI-compatible /v1 chat (for direct chat_completion calls)
       3. AutoGen-compatible model client (for multi-agent teams)
    """

    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.generate_url = config.OLLAMA_GENERATE_URL

        # OpenAI-compatible client (used by HR agent & simple travel queries)
        self.openai_client = OpenAI(
            base_url=config.OLLAMA_CHAT_URL,
            api_key="ollama"
        )

    def generate(self, prompt, system_prompt=None):
        """Standard text completion using Ollama /generate API"""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        try:
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False
            }
            response = requests.post(self.generate_url, json=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                logger.error(f"Ollama error: HTTP {response.status_code} - {response.text}")
                return f"Error: HTTP {response.status_code}"
        except Exception as e:
            logger.error(f"Failed to generate text from Ollama: {e}")
            return f"Error connecting to local LLM: {str(e)}"

    def chat_completion(self, messages, temperature=0.7, max_tokens=2000):
        """OpenAI-compatible chat completion using Ollama /v1 API"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to chat complete via Ollama: {e}")
            return f"Error: {e}"

    def get_autogen_client(self):
        """Create an AutoGen-compatible model client pointing to local Ollama.
        This wraps Ollama's OpenAI-compatible /v1 endpoint for use with
        autogen_agentchat agents and teams.
        """
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        return OpenAIChatCompletionClient(
            model=self.model,
            api_key="ollama",
            base_url=config.OLLAMA_CHAT_URL,
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": "unknown",
            },
        )


# Singleton Instance
llm = LocalLLM()
