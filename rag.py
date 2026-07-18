import os
from typing import List, Dict, Any, Tuple, Generator
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from embeddings import get_embedding
from chatbot import get_chat_response, get_chat_response_stream
from utils import get_logger

logger = get_logger("RAGPipeline")

SYSTEM_PROMPT = """You are an expert AI Document Assistant.
Your goal is to answer the user's question accurately, politely, and clearly, using ONLY the retrieved context provided below.

Rules:
1. Base your answer strictly on the retrieved context. Do not make up facts or use external knowledge.
2. If the context does not contain the answer, politely state: "I'm sorry, but I couldn't find that information in the uploaded documents. How else can I assist you today?"
3. If the answer partially exists in the context, summarize only the parts that exist. Do not fill in missing parts.
4. When relevant, reference the source document names and page numbers (e.g. "According to [filename] (Page X)...").
5. Keep your answers structured, using Markdown lists, bullet points, or tables if appropriate.

Retrieved Context:
-----------------
{context}
-----------------
"""

def chunk_document(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chunks loaded pages using LangChain's RecursiveCharacterTextSplitter."""
    if not pages:
        return []
        
    logger.info(f"Chunking document containing {len(pages)} pages using RecursiveCharacterTextSplitter")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    chunks = []
    for page in pages:
        split_texts = splitter.split_text(page["text"])
        for idx, text in enumerate(split_texts):
            chunks.append({
                "text": text,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": idx + 1
            })
            
    logger.info(f"Generated {len(chunks)} chunks from document.")
    return chunks

def query_rag(query_text: str, vector_store, k: int = 5) -> Dict[str, Any]:
    """Retrieves relevant chunks via MMR search and returns a blocking chat response along with sources."""
    # 1. Embed query
    query_emb = get_embedding(query_text)
    
    # 2. Search FAISS index using MMR
    results = vector_store.search_mmr(query_emb, k=k, lambda_val=config.MMR_LAMBDA)
    
    # 3. Format retrieved chunks for the prompt context
    context_chunks = []
    sources = []
    
    for i, (meta, dist) in enumerate(results):
        source_info = f"Source: {meta['source']} (Page: {meta['page']}) (Chunk: {meta['chunk_id']})"
        context_chunks.append(f"[{i+1}] {source_info}\nContent: {meta['text']}")
        sources.append({
            "source": meta["source"],
            "page": meta["page"],
            "chunk_id": meta["chunk_id"],
            "text": meta["text"],
            "score": float(dist)
        })
        
    context = "\n\n".join(context_chunks)
    
    # 4. Inject context and construct user prompt
    user_prompt = f"User Question: {query_text}\nAnswer:"
    formatted_system = SYSTEM_PROMPT.format(context=context if context else "No context available.")
    
    # 5. Call LLM (blocking)
    answer = get_chat_response(formatted_system, user_prompt)
    
    return {
        "answer": answer,
        "sources": sources
    }

def query_rag_stream(query_text: str, vector_store, k: int = 5) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
    """Retrieves relevant chunks via MMR and returns a token stream generator along with sources list."""
    # 1. Embed query
    query_emb = get_embedding(query_text)
    
    # 2. Search FAISS index using MMR
    results = vector_store.search_mmr(query_emb, k=k, lambda_val=config.MMR_LAMBDA)
    
    # 3. Format retrieved chunks for the prompt context
    context_chunks = []
    sources = []
    
    for i, (meta, dist) in enumerate(results):
        source_info = f"Source: {meta['source']} (Page: {meta['page']}) (Chunk: {meta['chunk_id']})"
        context_chunks.append(f"[{i+1}] {source_info}\nContent: {meta['text']}")
        sources.append({
            "source": meta["source"],
            "page": meta["page"],
            "chunk_id": meta["chunk_id"],
            "text": meta["text"],
            "score": float(dist)
        })
        
    context = "\n\n".join(context_chunks)
    
    # 4. Inject context and construct user prompt
    user_prompt = f"User Question: {query_text}\nAnswer:"
    formatted_system = SYSTEM_PROMPT.format(context=context if context else "No context available.")
    
    # 5. Get LLM token stream generator
    stream_generator = get_chat_response_stream(formatted_system, user_prompt)
    
    return stream_generator, sources
