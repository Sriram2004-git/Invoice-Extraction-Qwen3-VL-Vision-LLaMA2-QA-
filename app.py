import os
import io
import base64
import uuid
import numpy as np
from PIL import Image
from ollama import Client
import fitz

from flask import Flask, request, session, jsonify, render_template
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

OLLAMA_API_KEY = os.getenv("ollama_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
OLLAMA_BASE_URL = "https://api.ollama.com"
MODEL_NAME = "qwen3-vl:235b-cloud"

ollama_client = Client(
    host=OLLAMA_BASE_URL,
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

vectorstore_cache = {}


# EMBEDDINGS

class NomicEmbeddings(Embeddings):
    def __init__(self):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=HF_TOKEN
        )

    def _embed(self, text):
        result = self.client.feature_extraction(
            text, model="flax-sentence-embeddings/all_datasets_v3_mpnet-base"
        )
        arr = np.array(result)
        vec = arr.mean(axis=0) if arr.ndim == 2 else arr
        return vec.tolist()

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def get_embeddings():
    return NomicEmbeddings()


# UTILITIES

def image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def file_to_pil(file_bytes, filename):
    if filename.lower().endswith(".pdf"):
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = pdf_doc.load_page(0)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        pdf_doc.close()
    else:
        img = Image.open(io.BytesIO(file_bytes))
    return img


# EXTRACTION

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

    response = ollama_client.chat(model=MODEL_NAME, messages=messages)
    return response["message"]["content"]


# VECTOR STORE

def create_vectorstore(extracted_text):
    embeddings = get_embeddings()
    docs = [Document(page_content=extracted_text)]
    return Chroma.from_documents(documents=docs, embedding=embeddings)


# QA

def qa_with_qwen(vectordb, query):
    llm = Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    )

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


# ROUTES

@app.route("/")
def index():
    if "id" not in session:
        session["id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "id" not in session:
        session["id"] = str(uuid.uuid4())

    file = request.files.get("file")
    if not file:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file_bytes = file.read()
    filename = file.filename

    if filename == session.get("last_file") and session.get("extracted_text"):
        return jsonify({
            "success": True,
            "extracted_text": session["extracted_text"],
            "cached": True
        })

    try:
        img = file_to_pil(file_bytes, filename)
        img_b64 = image_to_base64(img)
        extracted_text = extract_with_qwen_vision(img)
        vectordb = create_vectorstore(extracted_text)

        session["extracted_text"] = extracted_text
        session["last_file"] = filename
        vectorstore_cache[session["id"]] = vectordb

        return jsonify({
            "success": True,
            "extracted_text": extracted_text,
            "image_b64": img_b64
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    sid = session.get("id")
    vectordb = vectorstore_cache.get(sid)

    if not vectordb:
        return jsonify({"success": False, "error": "No invoice loaded. Please upload a file first."}), 400

    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"success": False, "error": "Empty query"}), 400

    try:
        answer = qa_with_qwen(vectordb, query)
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
