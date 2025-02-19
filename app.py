import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


from src.logger import logging
from src.exception import CustomException

from src.chains import create_rag_chain,ask_question,create_retriever
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()
index_name = "medibot"

retriver=create_retriever(index_name,embeddings)

rag_chain,memory=create_rag_chain(retriver)

query="Blood Types?"
response=ask_question(query,rag_chain,memory)
print(response)
