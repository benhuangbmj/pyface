
#import requests
import os
import bs4
from langchain import hub
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings import HuggingFaceHubEmbeddings

API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=API_TOKEN,
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

llm = HuggingFaceHub(repo_id='gpt2')

loader = WebBaseLoader(
  web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
  bs_kwargs=dict(
      parse_only=bs4.SoupStrainer(
          class_=("post-content", "post-title", "post-header")
      )
  ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=hf_embeddings.embed_documents)
retriever = vectorstore.as_retriever()
print('vectorstore', vectorstore)
print('retriever', retriever)
'''


prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#res = rag_chain.invoke("What is Task Decomposition?")
#print(res)
'''

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