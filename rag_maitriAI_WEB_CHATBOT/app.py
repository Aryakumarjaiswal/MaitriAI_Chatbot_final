import os
import re
import uuid
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check for the Google API Key
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")


# Define session store
store = {}

# Load PDF and create vectorstore
pdf_loader = PyPDFLoader("rag_maitriAI_WEB_CHATBOT\data_v2.pdf")
docs = pdf_loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
try:
    vectorstore = Chroma.from_documents(documents=splits, embedding=gemini_embeddings)
    print("Vectorstore successfully created.")
except ValueError as e:
    raise ValueError(f"Error embedding content: {e}")

# Define the LLM and Chat Chain components
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]

# Create conversational chain
def create_conversational_chain():
    retriever = vectorstore.as_retriever()
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_system_prompt = """
You are Alex, a professional assistant for MaitriAI.
Start the conversation by politely asking for the user's name, email address, and mobile number. Inform the user that if they wish to update any of these three details later, they must update all three for consistency.

If the user inquires about connecting, consulting, or engaging in business with MaitriAI, assume their interest is specifically regarding MaitriAI’s offerings, and respond accordingly.

MaitriAI is a company specializing in AI-driven applications that enhance business efficiency and innovation through artificial intelligence. Our AI services include products like Credisence, AI Avatar, Customer Assistant Chatbot, LMS, AI Interviewer, OCR, Object Detection-based products, and more.

Using machine learning, natural language processing, computer vision, and data analytics, MaitriAI develops applications that can adapt to complex business environments. Provide concise answers with 2-3 sentences unless the user requests further detail.

**Guidelines for Responses:**
- **Clarifying Intent:** If a user’s question seems unclear or lacks detail, kindly ask them for more specifics. For instance: "Could you provide a bit more detail so I can assist you accurately?"
- **Professional Empathy:** If a user appears frustrated or expresses dissatisfaction, respond professionally with empathy, e.g., "I understand your concern. I'm here to help and ensure you get the assistance you need. Could you clarify how I can assist further?"
- **Ambiguity and Consistency:** If a question is too broad or general, provide a brief overview and offer to elaborate on specific areas if the user is interested.
- **Non-MaitriAI Queries:** If the question is unrelated to MaitriAI, respond with: "I'm here to assist with queries related to MaitriAI. Could I help with something specific about our services?"
-**Font-Style:** Since you're representing an organisation thus when needed you can change font to impress user.
Lastly, if the user uses inappropriate language or becomes aggressive, politely suggest they visit MaitriAI’s official website or mail at contact@maitriai.com. for further assistance.

If a query is outside your capabilities, suggest they visit the "Contact" section on MaitriAI’s website.{context}
"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
   
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
   
    return conversational_rag_chain

conversational_rag_chain = create_conversational_chain()





            ##################  APP
def main():
    st.set_page_config(page_title="MaitriAI Chatbot", page_icon="image.png", layout="centered")

    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
        text-align: center;
        margin: 0 auto;
        display: block;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .voice-input-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main content
  

    st.image("rag_maitriAI_WEB_CHATBOT\Logo.png", width=200)
    st.title("Chat with Us 🤖")
    st.write("Ask me anything about MaitriAI's services!")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("What would you like to know?")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            
            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "MaitriAI_Test-II"}}
            )["answer"]
            
            message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        

    


main()


