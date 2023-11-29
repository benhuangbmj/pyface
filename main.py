import requests
import os
API_TOKEN = os.environ['Hugging_Face_Token']
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
data = query({"inputs": "javascript is"})
print(data)