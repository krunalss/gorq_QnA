import streamlit as st
from langchain_groq import ChatGroq 
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
load_dotenv()

#load groq env varibale
groq_api_key=os.environ['GROQ_API_KEY']

st.title("Chat With Simon")

llm=ChatGroq(groq_api_key=groq_api_key,
             model="Llama3-70b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
        
    """)

#Download the Embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = download_hugging_face_embeddings()
        st.session_state.loader=PyPDFDirectoryLoader("/book")       
        st.session_state.docs=st.session_state.loader.load()
        print("docs count")
        print(len(st.session_state.docs))

        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        print("final doc count")
        print(len(st.session_state.final_documents))
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


button=True
if st.button("Wake up Simon, he might be sleeping! !!!"):
    vector_embeddings()
    button=True
    st.write("Simon is UP and Ready")

if button:
    st.write("Simon is UP and Ready")
    prompt1=st.text_input("Ask your question ??")   

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("References from the Simon's book"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")



