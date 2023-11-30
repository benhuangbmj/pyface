import requests
import os
API_TOKEN = os.environ['Hugging_Face_Token']
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/gpt2-large"
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    print(response)
    return response.json()
data = query({
  "inputs": "dilemma means", 
  "parameters": {
    "max_new_tokens": 250
  }
})
for i in range(len(data)):
  print(data[i]['generated_text'])
    