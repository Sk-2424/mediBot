import sys

from logger import logging
from exception import CustomException

from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore


def create_retriever(index_name,embeddings):
    docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
    logging.info("Retriever is created")
    return retriever


def create_rag_chain(retriever):
    try:
        llm = ChatOpenAI(model='gpt-4o-mini',temperature=0.2, max_tokens=500)

        ### Contextualize Question ###
        contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt)

        #General Question Answer prompt
        system_prompt = (
        "You are an Assistant for question-answering task related to health or medicines"
        " Always use the retrieved documents to answer"
        " If you don't know the answer. Say that I don't know  "
        "You can do normal conversation with your user. But never answer any questions which you don't find in retrieved documents"
        "Answer maximum in 2-3 lines in concise way."
        "\n\n"
        "Context: {context}")

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        #Use ConversationBufferWindowMemory instead of manual dictionary storage
        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5,return_messages=True)
        
        #creating Final chain
        bot_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
        
        logging.info("Final Rag Chain is created")
        return bot_chain,memory
    except Exception as e:
        # print(f"Error in create_rag_chain: {e}")
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)


def ask_question(query,bot_chain,memory):
    try:   
        input_data = {"chat_history": memory.load_memory_variables({})["chat_history"], "input": query}
        response = bot_chain.invoke(input_data)
        logging.info("Got the response of the query")

        memory.save_context({"input": query}, {"answer": response["answer"]})
        logging.info("Latest conversation is saved in memory ")
        return response['answer']
    except Exception as e:
        # pass
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)
    
def clear_memory(memory):
    memory.clear()
    logging.info("Memory is cleared")