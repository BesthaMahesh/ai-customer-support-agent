# Advanced AI RAG Assistant (SaaS)

A professional, enterprise-grade AI Document Assistant featuring **Retrieval-Augmented Generation (RAG)** built on Streamlit, FastAPI, FAISS, and OpenRouter LLMs. It supports dynamic in-memory file parsing, incremental database indexing, Maximal Marginal Relevance (MMR) search, and a modern ChatGPT-style chat workspace.

---

## Technical Features

1. **Dynamic In-Memory Ingestion**: PDF, DOCX, TXT, CSV, XLSX, PPTX, and MD files are read directly from memory bytes without writing files to local disk folders.
2. **Incremental Index Updates**: Automatically removes stale chunks of a file and appends new ones on re-upload, rebuilding the FAISS index on-the-fly.
3. **Maximal Marginal Relevance (MMR)**: Custom-implemented search algorithm balancing query semantic similarity and diversity of retrieved context chunks.
4. **Persistant Chat Memory**: ChatGPT-like conversational UI storing user inputs and streaming LLM tokens in real-time.
5. **Citations & Confidence Scores**: Displays exact citation paths (Source file, Page/Slide/Sheet index, Chunk number, and exact Cosine Confidence percentage).
6. **Exports**: Instant PDF and plain text downloads of active chat histories.
7. **Resource Metrics Monitoring**: Logs live system memory RSS metrics directly in the sidebar.

---

## File Sitemap

```
AI-Customer-Support-Agent/
│
├── app.py                 # Streamlit UI Dashboard (streaming, chat, cards, settings)
├── backend.py             # FastAPI REST Server
├── config.py              # Configuration & constant defaults loader
├── utils.py               # Resource tracking, L2 confidence scores, logger
├── document_loader.py     # In-memory parsing (PDF, DOCX, TXT, CSV, XLSX, PPTX, MD)
├── embeddings.py          # SentenceTransformers model cache
├── vector_store.py        # FAISS search operations (Rebuild, MMR, Standard L2)
├── chatbot.py             # OpenRouter Client (Blocking & Streaming calls)
├── rag.py                 # RAG orchestrator, Prompt builder, RecursiveTextSplitter
│
├── requirements.txt       # Unified Python dependencies
└── .env                   # Configuration parameters (API Key, Model, Backend URL)
```

---

## Setup & Installation

1. **Clone/Open workspace**: Ensure you are in the project folder.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install tf-keras
   ```
3. **Configure Settings**:
   Add your OpenRouter API Key to `.env` in the root:
   - `OPENROUTER_API_KEY`: Your key starting with `sk-or-v1-...`
   - `OPENROUTER_MODEL`: `tencent/hy3:free` (Recommended model for free-tier users)

---

## Running the Application

### 1. Launch FastAPI Backend
Start the server in port 8000:
```bash
python -m uvicorn backend:app --reload --port 8000
```

### 2. Launch Streamlit UI
Start the web dashboard in a separate terminal:
```bash
streamlit run app.py
```
This launches the application on `http://localhost:8501`.
