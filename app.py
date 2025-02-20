import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory

st.set_page_config(page_title="MediBot - AI Health Assistant", page_icon="💬", layout="centered")
# Add the 'src' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Import custom modules
from src.logger import logging
from src.chains import create_rag_chain, ask_question, create_retriever, clear_memory

# Load environment variables
load_dotenv()

# Get API keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize embeddings and retriever
embeddings = OpenAIEmbeddings()
index_name = "medibot"

# Cache RAG Chain
@st.cache_resource
def get_rag_chain(index_name, _embeddings):
    retriever = create_retriever(index_name, _embeddings)
    rag_chain = create_rag_chain(retriever)
    return rag_chain

rag_chain = get_rag_chain(index_name, embeddings)

# Initialize memory in session state
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)

# ---- UI Styling ---- #
st.markdown(
    """
    <style>
        .main {background-color: #f5f7fa;}
        .stTextInput>div>div>input {
            font-size: 18px; 
            padding: 10px;
            border-radius: 10px;
        }
        .stButton>button {
            font-size: 16px;
            padding: 10px 15px;
            border-radius: 8px;
            background-color: #007BFF;
            color: white;
            border: none;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .chat-bubble {
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        font-size: 16px;
        max-width: 75%;
        word-wrap: break-word;
        display: inline-block;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1)
        }
        .bot-response {
            background-color: #f8d7da;
            text-align: left;
            border-left: 5px solid #52a645; /* Green border for ai messages */
            float: right;
        }
        .user-query {
            background-color: #d1e7dd;
            text-align: left;
            color: #155724;
            border-left: 5px solid #28a745; /* Green border for user messages */
            float: left;
            clear: both;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- UI Layout ---- #
st.title("💬 MediBot - AI Health Assistant")
st.write("🤖 **Ask me anything about health or medicines!**")

query = st.text_input("Enter your question:", "")

# ---- Buttons: Submit & Clear Conversation ---- #
col1, col2 = st.columns([0.7, 0.3])
with col1:
    submit = st.button("💬 Submit", use_container_width=True)
with col2:
    clear_chat = st.button("🗑️ Clear Conversation", use_container_width=True)

# ---- Processing Query ---- #
if submit:
    if query.strip():
        with st.spinner("Thinking... 💭"):
            try:
                input_data = {"chat_history": st.session_state["memory"].load_memory_variables({})["chat_history"], "input": query}
                response = ask_question(input_data, rag_chain)
                st.session_state["memory"].save_context({"input": query}, {"answer": response})                
                logging.info("Latest conversation saved in memory.")
            except Exception as e:
                st.error("⚠️ An error occurred. Please try again.")
                st.text(f"Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter a valid question.")

# ---- Clearing Conversation ---- #
if clear_chat:
    clear_memory(st.session_state["memory"])
    st.session_state["memory"] = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)
    st.success("✅ Conversation cleared!")

# ---- Display Chat History ---- #
if st.session_state["memory"].buffer:
    st.write("📝 **Conversation History:**")
    for msg in st.session_state["memory"].buffer:
        if msg.type == "human":
            st.markdown(f'<div class="chat-bubble user-query">👤 <strong>You:</strong> <br> {msg.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble bot-response">🤖 <strong>Bot:</strong> <br> {msg.content}</div>', unsafe_allow_html=True)
