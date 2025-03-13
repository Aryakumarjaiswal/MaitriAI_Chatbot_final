from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma   
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check for the Google API Key
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Instantiate FastAPI app
app = FastAPI()

# Define session store
store = {}

# Load PDF and create vectorstore
pdf_loader = PyPDFLoader("MaitriDoc.pdf")
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
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

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
    You are Alex a MaitriAI assistant,
    representing MaitriAI is a leading software company specializing in web & AI application development, website design, logo design, and cutting-edge AI applications.
    You are Alex, a MaitriAI assistant. MaitriAI specializes in web & AI development, design, and AI applications.

    Provide brief, concise answers using the following context. Limit responses to 2-3 sentences unless more detail is explicitly requested.
    DONOT print any emoji.If your are  asked any question that is not in document  then simple answer "Sorry for inconvenience but I'm designed to answer your Queries that related to MaitriAI.What else I can assist you with?\nWe would be Happy to solve your query with respect to MaitriAI.".
    .
    {context}"""
    
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

# Helper functions
def clean_text(text):
    return re.sub(r"[*\t]+", " ", text)

def text_2_speech_converter(text):
    engine = pyttsx3.init()
    statement2print = text.split(".")

    for statement in statement2print:
        cleaned_statement = clean_text(statement)
        engine.say(cleaned_statement)
        engine.runAndWait()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Define request and response models
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

class ChatResponse(BaseModel):
    response: str

class VoiceInputRequest(BaseModel):
    session_id: str

# FastAPI Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    session_id = chat_request.session_id
    user_input = chat_request.user_input

    # Invoke the conversational chain
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    return ChatResponse(response=response)

@app.post("/voice-input", response_model=ChatResponse)
async def voice_input_endpoint(voice_input_request: VoiceInputRequest):
    session_id = voice_input_request.session_id

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            # Listen to audio input
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)

            # Process the recognized text
            response_after_speech_to_text = conversational_rag_chain.invoke(
                {"input": text},
                config={"configurable": {"session_id": session_id}},
            )["answer"]

            # Convert response to speech
            text_2_speech_converter(response_after_speech_to_text)

            return ChatResponse(response=response_after_speech_to_text)

        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Speech not understood")
        except sr.RequestError:
            raise HTTPException(status_code=500, detail="Failed to process speech request")

# Main function
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
