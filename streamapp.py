import streamlit as st
import requests
import os

# FastAPI base URL
# FASTAPI_URL = "http://localhost:8000"  # replace with your EC2 or ngrok URL if needed
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000") 

st.title("PDF Question-Answering App")

# --- Upload PDF ---
st.header("Step 1: Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    if st.button("Upload PDF"):
        with st.spinner("Uploading PDF to FastAPI..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{FASTAPI_URL}/upload_pdf/", files=files)
                if response.status_code == 200:
                    st.success(f"PDF '{uploaded_file.name}' uploaded successfully!")
                    st.write(response.json())
                else:
                    st.error(f"Error uploading PDF: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

# --- Ask Question ---
st.header("Step 2: Ask a Question")
query = st.text_input("Enter your question")

top_k = st.number_input("Number of top answers to retrieve", min_value=1, max_value=10, value=3)

if st.button("Ask Question") and query:
    with st.spinner("Querying FastAPI..."):
        try:
            payload = {"query": query, "top_k": top_k}
            response = requests.post(f"{FASTAPI_URL}/ask/", json=payload)
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"Answer: {result.get('answer', 'No answer returned')}")
            else:
                st.error(f"Error from API: {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")
