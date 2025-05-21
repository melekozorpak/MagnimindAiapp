# MagnimindAiapp  
## Internal AI Assistant for Meeting Notes

This AI-powered assistant helps your team retrieve insights and summaries from company meeting notes stored as PDFs. It uses AWS Bedrock, LangChain, and FAISS for natural language querying over structured and unstructured content.

---

## Features

- Extracts text and tables from PDFs with high accuracy  
- Embeds documents using Amazon Titan via Bedrock  
- Supports multiple LLMs (LLaMA 3, Mistral)  
- Stores vectors in FAISS for fast similarity search  
- Exposes two endpoints via a Flask API:
  - `/update_vectors` to re-index documents  
  - `/qa` to ask questions and get AI-generated answers  

---

## Project Structure

meeting-notes-ai/
├── app.py # Main Flask backend
├── requirements.txt # Python package list
├── Dockerfile # Container build instructions
├── .env.example # Example env vars for AWS access
├── Pdf_meeting_notes/ # Folder with meeting PDF files
├── faiss_index/ # Generated vector index
└── README.md # Project documentation

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/meeting-notes-ai.git
cd meeting-notes-ai

2. Set Up Environment Variables
Rename .env.example to .env and fill in your AWS credentials:
AWS_ACCESS_KEY_ID=your_aws_access_key  
AWS_SECRET_ACCESS_KEY=your_aws_secret_key  
AWS_DEFAULT_REGION=us-east-1

3. Install Python Dependencies
Make sure you're using Python 3.9+:
pip install -r requirements.txt

Docker Setup
Build the Docker Image
docker build -t ai-meeting-assistant .

Run the Docker Container
docker run -p 5000:5000 --env-file .env \
  -v $(pwd)/Pdf_meeting_notes:/app/Pdf_meeting_notes \
  ai-meeting-assistant
Note: poppler-utils and tesseract-ocr are already included in the Dockerfile.

Supported LLM Models
| Model Name    | Parameter Size | ID Used in Code                    |
| ------------- | -------------- | ---------------------------------- |
| LLaMA 3 7B    | 7 Billion      | `meta.llama3-8b-instruct-v1:0`     |
| LLaMA 3 70B   | 70 Billion     | `meta.llama3-70b-instruct-v1:0`    |
| Mistral 7B    | 7 Billion      | `mistral.mistral-7b-instruct-v0:2` |
| Mistral Large | Large-scale    | `mistral.mistral-large-2402-v1:0`  |

API Endpoints
POST /update_vectors
Re-ingests all PDFs in Pdf_meeting_notes/ and regenerates the FAISS vector store.
curl -X POST http://localhost:5000/update_vectors

POST /qa
Asks a natural language question and receives an AI-generated answer.

Request Example:
{
  "query": "Summarize the Q1 marketing meeting.",
  "model": "Llama3 7B"
}

Response Example:
{
  "question": "...",
  "answer": "...",
  "model": "..."
}

Curl Example:
curl -X POST http://localhost:5000/qa \
  -H "Content-Type: application/json" \
  -d '{"query": "What decisions were made in the last leadership meeting?", "model": "Mistral Large"}'

PDF Processing Workflow
1- Load PDFs from the Pdf_meeting_notes/ folder
2- Extract tables with pdfplumber
3- Split text using RecursiveCharacterTextSplitter
4- Embed content using Amazon Titan (Bedrock)
5- Store vectors in FAISS for semantic search

System Requirements (Handled in Docker)
. poppler-utils – for PDF to image conversion
. tesseract-ocr – for OCR tasks
. libgl1-mesa-glx – required by OpenCV

License:
This is an internal tool developed for private use within the organization. Do not distribute without approval.


