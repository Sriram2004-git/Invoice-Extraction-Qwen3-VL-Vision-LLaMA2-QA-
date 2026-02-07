import streamlit as st
import base64
import os
import io
from PIL import Image
from ollama import Client
import fitz  

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# OLLAMA CLIENT

ollama_client = Client()

DB_DIR = "./invoice_db"


# UTILITIES

def image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")



# QWEN3-VL EXTRACTION

def extract_with_qwen_vision(img):
    image_base64 = image_to_base64(img)

    messages = [
        {
            "role": "user",
            "content": """
Extract the following invoice information clearly:

project:
Client:
Engineer:
contractor:
Location:
RFI NO:
Structure ID:
Date of installation:
Bearing manufacturer:
Dimension of Bearing :

Span ID: 
Bearing IDs
Mumbai end:
B01:
B02:
B03:
B04:
Ahmedabad end:
B05:
B06:
B07:
B08:
""",
            "images": [image_base64]
        }
    ]

    response = ollama_client.chat(
        model="qwen3-vl:235b-cloud",
        messages=messages
    )

    return response["message"]["content"]



# VECTOR STORE CREATION

def create_vectorstore(extracted_text):
    if os.path.exists(DB_DIR):
        vectordb = Chroma(
            persist_directory=DB_DIR,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
        )
        vectordb.delete_collection()

    docs = [Document(page_content=extracted_text)]

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    return vectordb



# LLAMA2 QA

def qa_with_llama(vectordb, query):
    llm = Ollama(model="llama2")

    retriever = vectordb.as_retriever()

    template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer: """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)



# STREAMLIT UI

st.set_page_config(page_title="Invoice Extraction + QA", layout="wide")

st.title("Invoice Extraction & QA (Qwen3-VL + LLaMA2)")

uploaded_file = st.file_uploader(
    "Upload Invoice Image or PDF",
    type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    
    if file_path.lower().endswith('.pdf'):
        # Extract first page as image
        pdf_doc = fitz.open(file_path)
        page = pdf_doc.load_page(0)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        pdf_doc.close()
    else:
        img = Image.open(file_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Uploaded Invoice")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("🔍 Extracted Text")
        with st.spinner("Extracting using qwen3-vl:235b-cloud"):
            extracted_text = extract_with_qwen_vision(img)

        st.text_area(
            "Invoice Content",
            extracted_text,
            height=350
        )

        vectordb = create_vectorstore(extracted_text)
        st.success("Extraction & indexing completed")

    st.divider()

    st.subheader(" Ask Questions from Invoice")

    user_query = st.text_input("Enter your question")

    if user_query:
        with st.spinner("Thinking with LLaMA2..."):
            answer = qa_with_llama(vectordb, user_query)

        st.markdown("###  Answer")
        st.write(answer)

