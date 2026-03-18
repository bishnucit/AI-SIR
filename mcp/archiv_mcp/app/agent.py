import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "anymodel" # you can use ur any model u like

def chunk_text(text, max_length=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunks.append(" ".join(words[i:i+max_length]))
    return chunks

def summarize_text(text):
    try:
        chunks = chunk_text(text, max_length=500)  # smaller chunks
        summaries = []

        for chunk in chunks:
            prompt = f"Summarize this text in 5 bullet points:\n\n{chunk}"

            response = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            })
            response.raise_for_status()
            data = response.json()

            if "response" in data:
                summaries.append(data["response"])
            elif "content" in data:
                summaries.append(data["content"])
            else:
                summaries.append(str(data))

        return "\n".join(summaries)

    except Exception as e:
        print("❌ Ollama API error:", e)
        return "Error: Could not summarize"
