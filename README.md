# Medibot 

## 🤖 About Medibot  
Medibot is an AI-powered medical advisory bot designed to assist users with health-related queries. It leverages advanced AI models to provide insightful and accurate medical advice. Additionally, it maintains the context of previous conversations, ensuring a seamless and personalized user experience.

## 🏗 Architecture
Medibot is built using:
- **OpenAI API** for AI-driven medical insights
- **Pinecone Vector Database** for efficient retrieval and context management
- **Streamlit** for a user-friendly interface

![image](https://github.com/user-attachments/assets/49533bc4-a6dc-425e-9236-b62a183faf7e)


## 🚀 How to Run
Follow these steps to set up and run Medibot on your local machine:

### 1️⃣ Prerequisites
Ensure you have the following installed:
- Python 3.10
- OpenAI API Key
- Pinecone API Key
- Required dependencies from `requirements.txt`

### 2️⃣ Installation
Clone this repository and install the required dependencies:
```sh
$ git clone https://github.com/Sk-2424/medibot.git
$ cd medibot
$ pip install -r requirements.txt
```

### 3️⃣ Setup API Keys
Add your API keys to the environment variables:
```sh
$ export OPENAI_API_KEY="your_openai_api_key"
$ export PINECONE_API_KEY="your_pinecone_api_key"
```
Or create a `.env` file and add:
```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 4️⃣ Store Data in Pinecone
Run the following command to store embeddings in Pinecone:
```sh
$ python storing_data.py
```

### 5️⃣ Run the Application
Start the Medibot app using Streamlit:
```sh
$ streamlit run app.py
```

## 🎥 Demo
![image](https://github.com/user-attachments/assets/8fec5efe-646e-49c3-89a8-28436983fd90)


## 📌 Features
- AI-driven medical advisory system
- Context-aware conversation handling
- Fast and efficient retrieval using Pinecone
- User-friendly interface with Streamlit

## 🔧 Tech Stack
- **Programming Language:** Python
- **AI Model:** OpenAI GPT
- **Vector Database:** Pinecone
- **Frontend:** Streamlit
- **Architecture:** RAG-based (Retrieval-Augmented Generation)

