import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()

## load the GROQ API KEY from the .env file
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="gemma2-9b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
    )

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        st.session_state.loader = PyPDFDirectoryLoader(path="./example_pdf") ## Data Ingestion

        st.session_state.docs = st.session_state.loader.load() ## Document Loading

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                                        chunk_overlap=200) ## Chunk Creation
        
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents=st.session_state.docs) ## Splitting

        st.session_state.vectors = FAISS.from_documents(documents=st.session_state.final_documents, 
                                                        embedding=st.session_state.embeddings) ## Google Embeddings vector


if st.button("Create Vector Store"):
    vector_embeddings()
    st.success("Vector Store DB Is Ready")

prompt1 = st.text_input("What are you looking for in the documents?")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm=llm, 
                                                  prompt=prompt)
    
    retriever = st.session_state.vectors.as_retriever()

    retriever_chain = create_retrieval_chain(retriever=retriever, 
                                             combine_docs_chain=document_chain)

    start_time = time.process_time()
    response = retriever_chain.invoke({"input": prompt1})
    print("Response time :", time.process_time() - start_time)
    st.write(response["answer"])

    ## With a streamlit expander
    with st.expander("Document Similarity Search"):
        ## Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")