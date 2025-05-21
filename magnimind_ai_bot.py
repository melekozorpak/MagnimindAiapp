import json
import os
import sys
import boto3
import streamlit as st
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_aws.embeddings import BedrockEmbeddings
import time
import re

# Bedrock Clients - connection to bedrock runtime
bedrock = boto3.client(service_name="bedrock-runtime",
                       region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
                        )
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Function Definitions
def extract_tables(documents):
    tables_as_text = []
    for doc in documents:
        doc_path = doc.metadata["source"]
        try:
            with pdfplumber.open(doc_path) as pdf:
                for page in pdf.pages:
                    extracted_tables = page.extract_tables()
                    if extracted_tables:
                        for table in extracted_tables:
                            table_text = '\n'.join(['\t'.join([str(cell) if cell is not None else '' for cell in row]) for row in table])
                            tables_as_text.append(table_text)
        except Exception as pdfplumber_exception:
            print(f"Error extracting tables with PDFPlumber: {pdfplumber_exception}")
    return tables_as_text

def data_ingestion():
    loader = PyPDFDirectoryLoader("Pdf_meeting_notes")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    tables = extract_tables(documents)
    concatenated_docs = []
    for doc, table_text in zip(docs, tables):
        combined_text = doc.page_content + "\n\n" + table_text
        concatenated_docs.append(Document(page_content=combined_text, metadata=doc.metadata))
    return concatenated_docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# LLM Definitions

def get_llama3_7B_chat_llm():
    return Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})

def get_llama3_70B_chat_llm():
    return Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})

def get_Mistral_7B_llm():
    return Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock, model_kwargs={'max_tokens': 512})

def get_Mistral_Large_llm():
    return Bedrock(model_id="mistral.mistral-large-2402-v1:0", client=bedrock, model_kwargs={'max_tokens': 512})

# Prompt Template
prompt_template = """ 

Human: You are an AI assistant helping users answer questions based only on the provided PDF document context. 

<context>
{context}
</context>

User Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Main Streamlit App
st.set_page_config(page_title="Magnimind AI Bot", page_icon="emoji.png")  # Use your uploaded image

# st.image("emoji.png", width=80)  # Display image in the header
# st.header("Ask Magni-Bot")  # No need for a Unicode emoji now

col1, col2 = st.columns([1,1])  # Adjust column widths for spacing

with col1:
    st.header("Ask Magni-Bot")  # Title on the left

with col2:
    st.image("emoji.png", width=60)  # Image on the right




# Sidebar
with st.sidebar:
    st.title("LLM Settings")

    # Vector Update Button

    show_update_vectors = True  # Change to True if you want to show it

    if show_update_vectors:
        if st.button("Update Vectors"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated!")

    # LLM Selection
    selected_model = st.selectbox(
        "Select an LLM",
        options=["Llama3 7B", "Llama3 70B", "Mistral 7B", "Mistral Large"]
    )

    # Clear Chat History Button
    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.success("Chat history cleared!")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_question = st.text_input("Ask a question from the PDF files:", key="user_question")

if user_question:
    with st.spinner("Generating response..."):
        faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
        
        # Load selected LLM
        
        if selected_model == "Llama3 7B":
            llm = get_llama3_7B_chat_llm()
        elif selected_model == "Llama3 70B":
            llm = get_llama3_70B_chat_llm()
        elif selected_model == "Mistral 7B":
            llm = get_Mistral_7B_llm()
        elif selected_model == "Mistral Large":
            llm = get_Mistral_Large_llm()

        # Generate response with typing effect
        response = get_response_llm(llm, faiss_index, user_question)
        st.session_state["chat_history"].insert(0, {"question": user_question, "answer": response, "model": selected_model})

        # Display typing effect for the generated response
        placeholder = st.empty()
        displayed_text = ""
        for char in response:
            displayed_text += char
            placeholder.text(displayed_text)
            time.sleep(0.01)  # Faster typing effect (adjust speed as needed)

# Display Chat History
if st.session_state["chat_history"]:
    st.subheader("Chat History:")
    for chat in st.session_state["chat_history"]:
        st.write(f"**Question:** {chat['question']}")
        st.write(f"**Answer (Model - {chat['model']}):** {chat['answer']}")
        st.write("---")
