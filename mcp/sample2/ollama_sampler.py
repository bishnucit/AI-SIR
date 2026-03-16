import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def sample_llama(prompt: str):

    payload = {
        "model": "mistral:7b", #your model that you have locally
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["response"]
