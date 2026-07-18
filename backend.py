import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

import config
from vector_store import FAISSVectorStore
from document_loader import parse_in_memory_file
from rag import chunk_document, query_rag
from embeddings import get_embeddings
from utils import get_logger

logger = get_logger("BackendAPI")

app = FastAPI(title="Advanced AI RAG Assistant Backend")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize and load index from disk
vector_store = FAISSVectorStore()
vector_store.load()

class QueryRequest(BaseModel):
    query: str
    k: int = config.DEFAULT_K

@app.post("/upload")
async def upload_document_endpoint(file: UploadFile = File(...)):
    """Receives an uploaded file, extracts text in-memory, chunks, embeds, and saves incrementally."""
    logger.info(f"Received API upload request for file: '{file.filename}'")
    
    try:
        # Read file bytes directly from stream
        file_bytes = await file.read()
        
        # 1. Parse document from memory bytes
        pages = parse_in_memory_file(file_bytes, file.filename)
        if not pages:
            raise HTTPException(
                status_code=400, 
                detail=f"Parsed content from '{file.filename}' was empty or format is unsupported."
            )
            
        # 2. Chunk using RecursiveCharacterTextSplitter
        chunks = chunk_document(pages)
        if not chunks:
            return {
                "message": f"Document '{file.filename}' yielded no text chunks.",
                "chunks_count": 0
            }
            
        # 3. Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = get_embeddings(texts)
        
        # 4. Add to FAISS index (handling duplicate file overwrite) and save
        # We reload index from disk first to ensure we don't overwrite concurrent changes
        vector_store.load()
        vector_store.add_documents(embeddings, chunks, file.filename)
        vector_store.save()
        
        return {
            "message": f"Successfully parsed, chunked, and indexed '{file.filename}'",
            "chunks_count": len(chunks)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error indexing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """Retrieves relevant chunks via MMR search and returns LLM response."""
    try:
        # Reload index from disk to get latest files
        vector_store.load()
        
        if vector_store.index.ntotal == 0:
            return {
                "answer": "No documents have been indexed yet. Please upload files (PDF, DOCX, TXT, CSV, XLSX, PPTX, MD) first.",
                "sources": []
            }
            
        response = query_rag(req.query, vector_store, k=req.k)
        return response
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    """Lists unique names of all loaded documents in the index."""
    # Reload index to get latest files
    vector_store.load()
    unique_sources = sorted(list(set(meta["source"] for meta in vector_store.metadata)))
    return {
        "documents": unique_sources,
        "chunks_total": len(vector_store.metadata)
    }

@app.post("/clear")
async def clear_index():
    """Clears all indexed documents from the database."""
    try:
        vector_store.clear()
        return {"message": "Vector index and metadata successfully wiped."}
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {str(e)}")
