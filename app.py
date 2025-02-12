import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq 

load_dotenv()

# Load Groq API Key
groq_api_key = os.environ['GROQ_API_KEY']

st.title("Chat With Simon")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-70b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to download embeddings
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Function to load book and create FAISS embeddings
def vector_embeddings():
    with st.spinner("Waking up Simon... Loading and embedding the book. Please wait."):
        if "vectors" not in st.session_state:
            st.session_state.embeddings = download_hugging_face_embeddings()
            st.session_state.loader = PyPDFDirectoryLoader("book")       
            st.session_state.docs = st.session_state.loader.load()

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            
            st.session_state.db_ready = True  # Mark FAISS as ready

# Ensure session state variable exists
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

# Button to initialize FAISS DB
if st.button("Wake up Simon, he might be sleeping!"):
    vector_embeddings()
    st.success("Simon is UP and Ready!")

# Display input only when FAISS DB is ready
if st.session_state.db_ready:
    st.write("Simon is UP and Ready")
    prompt1 = st.text_input("Ask your question ??")

    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time:", time.process_time() - start)

        st.write(response['answer'])

        # With a Streamlit expander
        with st.expander("References from the Simon's book"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

