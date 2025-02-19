import os
import sys

from logger import logging
from exception import CustomException

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


pdf_path = os.path.join(os.getcwd(), "Data", "Medical_book.pdf")  # Medical_data_path

# Loading of data
def data_loader(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logging.info("PDF document loaded sucessfully")
        return documents
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)
    
def text_spliter(documents,chunk_size,chunk_overlap):
    try:
        text_spliter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap)
        docs = text_spliter.split_documents(documents)
        logging.info(f"Documents are splitted into chunks with chunk size of {chunk_size} and chunk overlap of {chunk_overlap}")
        total_chunks = len(docs)
        logging.info(f"Total no. of chunks are {total_chunks}")
        return docs
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)
    


    



