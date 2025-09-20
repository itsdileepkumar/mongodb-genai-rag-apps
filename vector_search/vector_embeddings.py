import requests

def get_embeddings(text, model, api_key):
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {"input": text, "model": model, "options": {"wait_for_model": True}}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]
