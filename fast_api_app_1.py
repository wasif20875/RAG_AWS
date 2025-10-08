# app.py
from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
import os

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama  # lightweigh local model
import os

# ----------------------------
# 3. Function to query vector database
# ----------------------------
import os
import boto3
from botocore.exceptions import ClientError
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ----------------------------
# 1. Load and chunk PDF recursively
# ----------------------------
def load_and_chunk_pdf(pdf_path, chunk_size=10000, chunk_overlap=500):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(docs)
    return chunks

# ----------------------------
# 2. Vectorize and create FAISS index
# ----------------------------


def get_vectorstore(docs=None, 
                    doc_name=None, 
                    embedding_model="all-MiniLM-L6-v2", 
                    base_dir="faiss_indexes", 
                    bucket_name="vectorstores3"):
    """
    Load FAISS vectorstore from local/S3 if available, otherwise create and save.

    Args:
        docs: Documents to index (only needed if creating new index).
        doc_name: Name of the uploaded document (used as folder name).
        embedding_model: Embedding model to use.
        base_dir: Local storage folder.
        bucket_name: S3 bucket name.

    Returns:
        FAISS vectorstore instance.
    """

    safe_name = os.path.splitext(os.path.basename(doc_name))[0]
    persist_dir = os.path.join(base_dir, safe_name)

    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    s3 = boto3.client("s3")

    # 1. Check if already exists locally
    if os.path.exists(persist_dir):
        print(f"Loading vectorstore locally from {persist_dir}")
        return FAISS.load_local(persist_dir, embeddings,allow_dangerous_deserialization=True)

    # 2. Check if exists in S3
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{safe_name}/")
        if "Contents" in response:
            print(f"Downloading vectorstore from s3://{bucket_name}/{safe_name}/")
            os.makedirs(persist_dir, exist_ok=True)

            # Download all files
            for obj in response["Contents"]:
                key = obj["Key"]
                local_path = os.path.join(persist_dir, os.path.basename(key))
                s3.download_file(bucket_name, key, local_path)

            return FAISS.load_local(persist_dir, embeddings,allow_dangerous_deserialization=True)
    except ClientError as e:
        print(f"Error checking S3: {e}")

    # 3. If not found anywhere, create a new one
    if docs is None:
        raise ValueError("Docs must be provided if index does not already exist.")

    print("Creating new vectorstore...")
    vector_store = FAISS.from_documents(docs, embeddings)

    os.makedirs(persist_dir, exist_ok=True)
    vector_store.save_local(persist_dir)
    print(f"Saved new vectorstore locally at {persist_dir}")

    # Upload to S3
    try:
        for root, _, files in os.walk(persist_dir):
            for file in files:
                local_path = os.path.join(root, file)
                s3_key = f"{safe_name}/{file}"
                s3.upload_file(local_path, bucket_name, s3_key)
        print(f"Uploaded new vectorstore to s3://{bucket_name}/{safe_name}/")
    except ClientError as e:
        print(f"Error uploading to S3: {e}")

    return vector_store

# ----------------------------
# 3. Function to query vector database
# ----------------------------

def query_vectorstore(vector_store, query, top_k=3):
    """
    Returns top_k most similar documents for the query
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    relevant_docs = retriever.get_relevant_documents(query)
    return relevant_docs

# ----------------------------
# 4. Function to pass docs to LLM and get answer
# ----------------------------
def generate_answer(llm, query, vector_store,top_k=3):
    """
    Takes the retrieved documents and the question, returns the answer
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    # You can also use RetrievalQA chain directly
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,  # docs already retrieved
        chain_type="stuff"
    )
    # Join the text from documents for the LLM
    # context = " ".join([doc.page_content for doc in docs])
    # prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    # answer = llm(prompt)
    return qa_chain.run(query)

# ----------------------------
# Example usage
# ----------------------------
# if __name__ == "__main__":
    # pdf_path = "example.pdf"
    # chunks = load_and_chunk_pdf("E:\Wasif\AWS_proj\Kotak_TULIP_Brochure.pdf")
    
    # # Create FAISS vectorstore
    # vector_store = create_vectorstore(chunks)
    
    # # Initialize LLM (lightweight local model)
    # llm = Ollama(model="llama3")
    
    # # Query example
    # user_query = "Explain the main topic of this PDF."
    # # relevant_docs = query_vectorstore(vector_store, user_query, top_k=3)
    # answer = generate_answer(llm, user_query, vector_store=vector_store,top_k=3)
    # print("\nAnswer:\n", answer)

    # ----------------------------
    # FastAPI Setup
    # ----------------------------
app = FastAPI(title="PDF QA API", version="1.0")

vector_store = None
llm = Ollama(model="llama3")
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int]= 3

@app.get("/")
def root():
    return {"message": "Welcome to the PDF QA API. Use /upload_pdf to upload a PDF and /ask to ask questions."}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    """
    Upload and index a PDF for QA.
    """
    global vector_store
    pdf_path = f"temp_{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    chunks = load_and_chunk_pdf(pdf_path)
    vector_store = get_vectorstore(chunks,file.filename)
    return {"status": "PDF processed successfully", "filename": file.filename}

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    """
    Ask a question to the indexed PDF.
    """
    global vector_store
    if vector_store is None:
        return {"error": "No PDF has been uploaded yet."}

    answer = generate_answer(llm, request.query, vector_store, request.top_k)
    return {"query": request.query, "answer": answer}

