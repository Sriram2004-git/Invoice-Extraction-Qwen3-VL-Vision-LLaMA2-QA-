Handwritten and scanned INVOICE EXTRACTION using Qwen3-vl:235b-cloud (Vision) + LLaMA2 (QA) LOCALLY(Ollama):

The application allows users to upload invoice images, handwritten or scanned PDFs, extract structured information using a vision-language modelBand then ask natural language questions about the extracted content.

KEY FEATURES:

-Invoice Image & PDF Upload
-Vision-based Extraction using Qwen3-vl:235b-cloud
-Structured Field Extraction
-Retrieval-Based Question Answering using LLaMA2
-Vector Storage with ChromaDB
-Interactive Streamlit UI
-Fully Local Execution (Ollama-based)

ARCHITECTURE OVERVIEW:

Invoice Image / PDF
        ↓
Qwen3-vl:235b-cloud (Vision Language Model)
        ↓
Structured Text Extraction
        ↓
Vector Embeddings (nomic-embed-text)
        ↓
Chroma Vector Store
        ↓
LLaMA2 Question Answering


| Component      | Technology        |
| -------------- | ----------------- |
| UI             | Streamlit         |
| Vision Model   | Qwen3-VL (Ollama) |
| Language Model | LLaMA2 (Ollama)   |
| Embeddings     | nomic-embed-text  |
| Vector DB      | ChromaDB          |
| PDF Processing | PyMuPDF (fitz)    |
| Image Handling | Pillow            |


INSTALLATION:
1️.Install Dependencies

pip install streamlit ollama langchain langchain-community chromadb pillow pymupdf

2️.Pull Required Ollama Models

ollama pull qwen3-vl:235b-cloud
ollama pull llama2
ollama pull nomic-embed-text

3.RUN APPLICATION:
Start Ollama:
ollama serve

4.RUN STREAMLIT:
streamlit run app.py

WHY THIS APPROACH IS RELIABLE:

-Vision-based extraction avoids OCR-only limitations
-Retrieval-based QA prevents hallucinations
-No external APIs or cloud dependency
-Suitable for sensitive or confidential documents

