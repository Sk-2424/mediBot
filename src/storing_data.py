import os

from data_ingestion import data_loader,text_spliter
from embeddings import create_vector_db,add_embeddings_to_db

from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

pdf_path = os.path.join(os.getcwd(), "Data", "Medical_book.pdf") 

# Data Ingestion
documents = data_loader(pdf_path)
docs = text_spliter(documents,500,50)

#Vector DB and Embeddings Creations
index_name = "medibot"
index=create_vector_db(PINECONE_API_KEY,index_name)
add_embeddings_to_db(docs,index_name)


