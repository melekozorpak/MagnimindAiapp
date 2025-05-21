import os
import json
import time
import re
import boto3
import pdfplumber
from pdf2image import convert_from_path
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_aws.embeddings import BedrockEmbeddings
from flask import Flask, request, jsonifypython 

app = Flask(__name__)

# ------------------------------
# AWS Bedrock Setup and Clients
# ------------------------------
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# ------------------------------
# Function Definitions
# ------------------------------
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
                            table_text = '\n'.join([
                                '\t'.join([str(cell) if cell is not None else '' for cell in row])
                                for row in table
                            ])
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
    # Combine the split document text with the extracted tables
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

# ------------------------------
# Flask API Endpoints
# ------------------------------

@app.route('/update_vectors', methods=['POST'])
def update_vectors():
    """
    Endpoint to re-ingest data from PDFs and update the vector store.
    """
    try:
        docs = data_ingestion()
        get_vector_store(docs)
        return jsonify({"message": "Vector store updated successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/qa', methods=['POST'])
def qa_endpoint():
    """
    Endpoint that receives a JSON request with a question and an optional LLM model choice,
    then returns the generated answer.
    
    Expected JSON format:
    {
      "query": "Your question here",
      "model": "Llama3 7B"  // optional; defaults to Llama3 7B if missing
    }
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in the request payload."}), 400

    query = data["query"]
    selected_model = data.get("model", "Llama3 7B")

    try:
        # Load the locally stored FAISS index.
        faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

        # Select LLM based on the provided model choice.
        if selected_model == "Llama3 7B":
            llm = get_llama3_7B_chat_llm()
        elif selected_model == "Llama3 70B":
            llm = get_llama3_70B_chat_llm()
        elif selected_model == "Mistral 7B":
            llm = get_Mistral_7B_llm()
        elif selected_model == "Mistral Large":
            llm = get_Mistral_Large_llm()
        else:
            return jsonify({"error": f"Unsupported model: {selected_model}."}), 400

        # Generate the response using the selected LLM.
        answer = get_response_llm(llm, faiss_index, query)
        return jsonify({
            "question": query,
            "answer": answer,
            "model": selected_model
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------
# Run the Flask App
# ------------------------------
if __name__ == '__main__':
    # When deploying to EC2, ensure the security group allows inbound traffic on the specified port (e.g., 5000).
    app.run(host="0.0.0.0", port=5000)
