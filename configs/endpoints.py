"""Endpoint registry for vf-eval.

This file contains ONLY model names, base URLs and the names of environment
variables that should hold API keys. Do NOT put secrets here.

How it’s used:
- vf-eval reads this file (default path: ./configs/endpoints.py)
- When you pass -m <name>, it looks up ENDPOINTS[<name>] to get:
  - provider base URL ("url")
  - env var name that holds the API key ("key")
  - the model identifier to send to the provider ("model")
"""

ENDPOINTS = {
    # OpenAI presets (require OPENAI_API_KEY in your environment)
    # Replace models below with IDs available to your account if needed.
    "gpt-5": {
        "model": "gpt-5",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-4.1-mini": {
        "model": "gpt-4.1-mini",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-4o-mini": {
        "model": "gpt-4o-mini",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },

    # Local OpenAI-compatible server (LM Studio default)
    # Start LM Studio’s server and download a small instruct model first.
    # Set LMSTUDIO_API_KEY to any non-empty string.
    "local-qwen": {
        "model": "lmstudio-community/Qwen2.5-1.5B-Instruct",
        "url": "http://localhost:1234/v1",
        "key": "LMSTUDIO_API_KEY",
    },
    "local-llama": {
        "model": "lmstudio-community/Meta-Llama-3.1-8B-Instruct",
        "url": "http://localhost:1234/v1",
        "key": "LMSTUDIO_API_KEY",
    },
}

