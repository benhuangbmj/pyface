
#import requests
import os
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']

hub_llm = HuggingFaceHub(repo_id='gpt2')
prompt = PromptTemplate(
  input_variables=["question"],
  template = "Translate English to SQL: {question}"
)
#hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
#print(hub_chain.run("What is the average age of the respondents using a mobile device?"))

text='What would be a good company name for a company that makes colorful socks?'
messages=[HumanMessage(content=text)]
print(hub_llm.invoke(messages))
'''
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
'''