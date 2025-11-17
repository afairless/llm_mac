
import requests

url = "http://127.0.0.1:8000/v1/chat/completions/stream"
data = {
    "model": "Qwen2.5-Coder-3B-Instruct",
    "messages": [{"role": "user", "content": "Compare Rust and Go"}],
    "max_tokens": 500
}


with requests.post(url, json=data, stream=True) as r:
    for chunk in r.iter_content(chunk_size=None):
        print(chunk.decode(), end="")


