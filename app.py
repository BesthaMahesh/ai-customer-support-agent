import os
import streamlit as st
import requests
import io
import time
from datetime import datetime
from fpdf import FPDF
from typing import List, Dict, Any

# Set Page Config
st.set_page_config(
    page_title="RAG.SaaS Enterprise Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configurations and modules
import config
from utils import get_system_metrics, l2_distance_to_confidence, get_logger
import json

logger = get_logger("SaaSFrontendUI")

def backend_query_stream(prompt: str):
    """Queries the backend streaming endpoint and yields text tokens while saving sources."""
    try:
        res = requests.post(
            f"{config.BACKEND_URL}/query/stream",
            json={"query": prompt, "k": config.DEFAULT_K},
            stream=True
        )
        if res.status_code == 200:
            for line in res.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "sources" in data:
                        st.session_state.last_sources = data["sources"]
                    elif "token" in data:
                        yield data["token"]
        else:
            yield f"Error from backend API: {res.text}"
    except Exception as e:
        yield f"Backend connection failed: {str(e)}"

# Inject Billion-Dollar SaaS Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* 1. Global Reset & Body styling */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0B1020 !important;
        color: #FFFFFF !important;
    }
    
    /* Hide Streamlit Default Brandings */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    button[title="View fullscreen"] {display: none !important;}
    
    /* 2. Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0B1020;
    }
    ::-webkit-scrollbar-thumb {
        background: #273449;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #7C3AED;
    }
    
    /* 3. Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #131A2A !important;
        border-right: 1px solid #273449 !important;
        padding-top: 1.5rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1rem;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* 4. Top Sticky Navbar */
    .top-navbar {
        position: sticky;
        top: 0;
        z-index: 99;
        background: rgba(11, 16, 32, 0.85);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid #273449;
        padding: 12px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
    }
    .profile-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #7C3AED 0%, #22D3EE 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.85rem;
        border: 2px solid #273449;
    }
    
    /* 5. Custom Analytics Dashboard Cards */
    .analytics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    .analytics-card {
        background: #131A2A;
        border: 1px solid #273449;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .analytics-card:hover {
        transform: translateY(-4px);
        border-color: #7C3AED;
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.15);
    }
    .card-meta {
        font-size: 0.75rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 0.5rem;
        background: linear-gradient(90deg, #FFFFFF, #94A3B8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 6. Document Inventory List Cards */
    .doc-saas-card {
        background: #1B2335;
        border: 1px solid #273449;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: border-color 0.2s ease;
    }
    .doc-saas-card:hover {
        border-color: #8B5CF6;
    }
    .doc-meta {
        font-size: 0.75rem;
        color: #94A3B8;
    }
    
    /* 7. Chat Windows Styling (ChatGPT-like) */
    .message-container {
        display: flex;
        gap: 16px;
        padding: 1.5rem;
        border-bottom: 1px solid #1B2335;
        align-items: flex-start;
    }
    .message-container.user {
        background-color: #131A2A;
    }
    .message-container.assistant {
        background-color: #0B1020;
        border-left: 3px solid #7C3AED;
    }
    .avatar-wrapper {
        width: 38px;
        height: 38px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        flex-shrink: 0;
        font-size: 1rem;
    }
    .avatar-wrapper.user {
        background-color: #1B2335;
        color: #22D3EE;
        border: 1px solid #273449;
    }
    .avatar-wrapper.assistant {
        background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%);
        color: white;
    }
    .msg-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .msg-author {
        font-weight: 600;
        font-size: 0.95rem;
        color: #FFFFFF;
    }
    .msg-time {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-left: 10px;
    }
    .msg-body {
        font-size: 0.95rem;
        color: #E2E8F0;
        line-height: 1.6;
    }
    
    /* 8. Citation cards & Progress bars (Right Column) */
    .citation-card {
        background: #131A2A;
        border: 1px solid #273449;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    .confidence-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.8rem;
    }
    .confidence-bar-outer {
        width: 100%;
        height: 5px;
        background: #1B2335;
        border-radius: 10px;
        overflow: hidden;
    }
    .confidence-bar-inner {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Custom input bar drop shadow styling */
    .stChatInputContainer {
        border-radius: 24px !important;
        border: 1px solid #273449 !important;
        background-color: #131A2A !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.3s ease !important;
        padding: 4px 10px !important;
    }
    .stChatInputContainer:focus-within {
        border-color: #7C3AED !important;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.4) !important;
    }
    
    /* Styling Buttons */
    .stButton>button {
        background: #131A2A;
        color: #FFFFFF;
        border: 1px solid #273449;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #1B2335;
        border-color: #7C3AED;
        color: #7C3AED;
    }
    
    /* Primary Gradient Button Action */
    .gradient-btn-wrapper .stButton>button {
        background: linear-gradient(135deg, #7C3AED 0%, #8B5CF6 100%) !important;
        color: white !important;
        border: none !important;
    }
    .gradient-btn-wrapper .stButton>button:hover {
        box-shadow: 0 0 12px rgba(124, 58, 237, 0.6) !important;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Helper: Export chat history to PDF
def export_chat_pdf(messages: List[Dict[str, str]]) -> bytes:
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, text="RAG.SaaS Workspace - Exported Chat", ln=True, align='C')
        pdf.ln(10)
        
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.cell(200, 8, text=f"{role}:", ln=True)
            pdf.set_font("Helvetica", size=10)
            
            content_cleaned = msg["content"].encode("latin1", errors="ignore").decode("latin1")
            pdf.multi_cell(0, 6, text=content_cleaned)
            pdf.ln(4)
            
        return bytes(pdf.output())
    except Exception as e:
        logger.error(f"Error compiling export PDF: {e}")
        return b""

# Helper: Export chat to TXT
def export_chat_txt(messages: List[Dict[str, str]]) -> bytes:
    lines = []
    lines.append(f"RAG.SaaS Enterprise Conversation Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("="*50 + "\n")
    for msg in messages:
        role = "USER" if msg["role"] == "user" else "ASSISTANT"
        lines.append(f"[{role}]:\n{msg['content']}\n")
        lines.append("-"*50 + "\n")
    return "\n".join(lines).encode("utf-8")

# Initialize Session variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# Fetch stats from backend API
try:
    res = requests.get(f"{config.BACKEND_URL}/documents")
    if res.status_code == 200:
        backend_data = res.json()
        documents_dict = backend_data.get("documents", {})
        unique_docs = sorted(list(documents_dict.keys()))
        total_chunks_count = backend_data.get("chunks_total", 0)
    else:
        unique_docs = []
        documents_dict = {}
        total_chunks_count = 0
except Exception as e:
    unique_docs = []
    documents_dict = {}
    total_chunks_count = 0
    logger.error(f"Failed to fetch stats from backend: {e}")

unique_docs_count = len(unique_docs)
sys_metrics = get_system_metrics()

# ----------------- TOP NAVBAR HEADER -----------------
st.markdown(f"""
<div class="top-navbar">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.3rem;">⚡</span>
        <span style="font-weight: 700; font-size: 1.2rem; letter-spacing:-0.03rem; background: linear-gradient(90deg, #FFFFFF, #94A3B8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">RAG.SaaS Workspace</span>
    </div>
    <div style="display: flex; align-items: center; gap: 16px;">
        <span class="badge" style="background: rgba(124, 58, 237, 0.15); color: #8B5CF6; border: 1px solid rgba(124, 58, 237, 0.3);">🤖 Model: {config.OPENROUTER_MODEL.split("/")[-1]}</span>
        <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3);">🟢 status: online</span>
        <div class="profile-avatar">S</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- ANALYTICS DASHBOARD ROW -----------------
st.markdown(f"""
<div class="analytics-grid">
    <div class="analytics-card">
        <div class="card-meta">Total Documents</div>
        <div class="card-value">{unique_docs_count}</div>
    </div>
    <div class="analytics-card">
        <div class="card-meta">Total Chunks</div>
        <div class="card-value">{total_chunks_count}</div>
    </div>
    <div class="analytics-card">
        <div class="card-meta">Memory Usage</div>
        <div class="card-value">{sys_metrics['process_memory_mb']} MB</div>
    </div>
    <div class="analytics-card">
        <div class="card-meta">Vector DB Size</div>
        <div class="card-value">{total_chunks_count} vectors</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    # SaaS Branding Logo
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 2rem;">
            <img src="https://img.icons8.com/nolan/64/artificial-intelligence.png" width="38"/>
            <h3 style="margin: 0; font-weight: 700; color: #FFFFFF; font-size: 1.15rem; letter-spacing: -0.02rem;">RAG.SaaS Enterprise</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Ingest Knowledge Base")
    
    # Drag-and-drop File Ingester
    uploaded_files = st.file_uploader(
        "Ingest support files (PDF, DOCX, TXT, CSV, XLSX, PPTX, MD)",
        type=["pdf", "docx", "txt", "csv", "xlsx", "pptx", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        new_upload = False
        for uploaded_file in uploaded_files:
            file_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
            if file_key not in st.session_state:
                new_upload = True
                st.session_state[file_key] = True
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    file_bytes = uploaded_file.getvalue()
                    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                    try:
                        res = requests.post(f"{config.BACKEND_URL}/upload", files=files)
                        if res.status_code == 200:
                            st.toast(f"Indexed: {uploaded_file.name}", icon="✅")
                        else:
                            st.error(f"Error '{uploaded_file.name}': {res.json().get('detail', 'Error')}")
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
        
        if new_upload:
            st.toast("Vector Database updated!", icon="🗂️")
            time.sleep(0.5)
            st.rerun()

    # Document card list with inline delete triggers!
    st.markdown("### Document Index Manager")
    if unique_docs:
        for doc in unique_docs:
            # Render a custom file display card row
            col_doc_name, col_doc_del = st.columns([8, 2])
            with col_doc_name:
                doc_chunks = documents_dict.get(doc, 0)
                st.markdown(f"""
                    <div style="font-size: 0.85rem; font-weight: 500; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">
                        📄 {doc}<br/>
                        <span style="color:#94A3B8; font-size:0.7rem;">Chunks: {doc_chunks}</span>
                    </div>
                """, unsafe_allow_html=True)
            with col_doc_del:
                if st.button("🗑️", key=f"del_{doc}", help=f"Wipe {doc} from vector index"):
                    try:
                        res = requests.post(f"{config.BACKEND_URL}/delete", params={"filename": doc})
                        if res.status_code == 200:
                            st.toast(f"Removed {doc}!", icon="🗑️")
                            # Reset dynamic file uploaded session states
                            for key in list(st.session_state.keys()):
                                if doc in key:
                                    del st.session_state[key]
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {doc}: {res.json().get('detail', 'Error')}")
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")
    else:
        st.info("No files indexed. Drag and drop documents to start.")
        
    st.markdown("---")
    st.markdown("### Workspace Configs")
    
    col_clear_chat, col_clear_db = st.columns(2)
    with col_clear_chat:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()
            
    with col_clear_db:
        if st.button("Purge DB", use_container_width=True):
            try:
                res = requests.post(f"{config.BACKEND_URL}/clear")
                if res.status_code == 200:
                    for key in list(st.session_state.keys()):
                        if key.startswith("processed_"):
                            del st.session_state[key]
                    st.session_state.messages = []
                    st.session_state.last_sources = []
                    st.toast("Database purged!", icon="🧹")
                    time.sleep(0.5)
                    st.rerun()
            except Exception as e:
                st.error(f"Purge failed: {e}")
                
    # Download Chat History
    if st.session_state.messages:
        st.markdown("#### Conversation Export")
        col_pdf, col_txt = st.columns(2)
        with col_pdf:
            pdf_bytes = export_chat_pdf(st.session_state.messages)
            if pdf_bytes:
                st.download_button(
                    label="PDF Export",
                    data=pdf_bytes,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        with col_txt:
            txt_bytes = export_chat_txt(st.session_state.messages)
            st.download_button(
                label="TXT Export",
                data=txt_bytes,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
    # Settings drawer
    st.session_state.debug_mode = st.toggle("Retrieval Debug Console", value=st.session_state.debug_mode)
    
    st.markdown("""
        <div style="text-align: center; font-size: 0.7rem; color: #94A3B8; margin-top: 1.5rem;">
            RAG.SaaS Enterprise Workspace v2.0
        </div>
    """, unsafe_allow_html=True)

# ----------------- MAIN AREA WORKSPACE -----------------
col_workspace, col_citations = st.columns([65, 35], gap="large")

with col_workspace:
    # 1. RAG Conversation Bubbles (ChatGPT theme)
    for idx, msg in enumerate(st.session_state.messages):
        container_class = "message-container user" if msg["role"] == "user" else "message-container assistant"
        avatar_class = "avatar-wrapper user" if msg["role"] == "user" else "avatar-wrapper assistant"
        avatar_label = "U" if msg["role"] == "user" else "AI"
        author_label = "You" if msg["role"] == "user" else "RAG Assistant"
        timestamp = datetime.now().strftime("%I:%M %p") # Mock timestamp representation
        
        st.markdown(f"""
            <div class="{container_class}">
                <div class="{avatar_class}">{avatar_label}</div>
                <div style="flex-grow: 1;">
                    <div class="msg-header">
                        <span class="msg-author">{author_label}</span>
                        <span class="msg-time">{timestamp}</span>
                    </div>
                    <div class="msg-body">{msg['content']}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 2. Main Chat Input Prompt
    if prompt := st.chat_input("Ask about company policies, resumes, or uploaded research papers..."):
        # Render user prompt immediately
        timestamp = datetime.now().strftime("%I:%M %p")
        st.markdown(f"""
            <div class="message-container user">
                <div class="avatar-wrapper user">U</div>
                <div style="flex-grow: 1;">
                    <div class="msg-header">
                        <span class="msg-author">You</span>
                        <span class="msg-time">{timestamp}</span>
                    </div>
                    <div class="msg-body">{prompt}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Handle assistant stream completion
        with st.chat_message("assistant", avatar="🤖"):
            if total_chunks_count == 0:
                answer = "No document index is active yet. Please ingest files in the sidebar."
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.last_sources = []
            else:
                try:
                    # Stream tokens word-by-word from backend
                    stream_gen = backend_query_stream(prompt)
                    full_response = st.write_stream(stream_gen)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    logger.error(f"RAG query execution failed: {e}")
                    error_msg = f"API Service failure: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.last_sources = []
                    
        st.rerun()

# ----------------- CITATIONS DRAWER (RIGHT COLUMN) -----------------
with col_citations:
    st.markdown("### 🔍 Retrieved Context Citations")
    
    if st.session_state.last_sources:
        st.markdown("Context snippets fetched using MMR (Maximal Marginal Relevance) Diversity ranking:")
        
        for idx, src in enumerate(st.session_state.last_sources):
            # L2 flat distance converted to confidence %
            confidence = l2_distance_to_confidence(src["score"])
            
            # Map color of the progress bar based on confidence score
            if confidence >= 70:
                progress_color = "#10B981"  # Success (Green)
            elif confidence >= 40:
                progress_color = "#F59E0B"  # Warning (Amber)
            else:
                progress_color = "#EF4444"  # Danger (Red)
                
            st.markdown(f"""
                <div class="citation-card">
                    <div style="font-weight: 600; font-size: 0.85rem; color:#FFFFFF; margin-bottom: 6px;">
                        📄 {src['source']} (Page/Slide: {src['page']})
                    </div>
                    <div class="confidence-container">
                        <span style="color:#94A3B8;">Similarity Confidence</span>
                        <span style="color:{progress_color}; font-weight:700;">{confidence}%</span>
                    </div>
                    <div class="confidence-bar-outer">
                        <div class="confidence-bar-inner" style="width: {confidence}%; background-color: {progress_color};"></div>
                    </div>
                    <div style="font-size:0.75rem; color:#94A3B8; margin-top:8px;">
                        Chunk Index: <code>{src['chunk_id']}</code> | L2 Dist: <code>{src['score']:.4f}</code>
                    </div>
                    <div style="font-size: 0.85rem; color:#E2E8F0; background: #1B2335; border: 1px solid #273449; border-radius: 8px; padding: 8px; margin-top:10px; font-style:italic;">
                        "{src['text'][:250]}..."
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background:#131A2A; border:1px dashed #273449; border-radius:12px; padding: 1.5rem; text-align:center; color:#94A3B8; font-size:0.85rem; margin-top:1rem;">
                No query has been sent yet.<br/>Citations, relevance statistics, and confidence bars will render here.
            </div>
        """, unsafe_allow_html=True)
        
    # Suggested Questions (General queries)
    st.markdown("### 💡 Quick Suggestions")
    suggestions = [
        "What is the main aim of this paper?",
        "which projects are there in resume",
        "What certifications are listed?",
        "What is the refund window policy?"
    ]
    for sug in suggestions:
        if st.button(sug, key=f"sug_{sug}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": sug})
            with st.chat_message("user"):
                st.write(sug)
                
            with st.chat_message("assistant"):
                if total_chunks_count == 0:
                    answer = "Please upload documents first in the sidebar before asking suggestions."
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_sources = []
                else:
                    stream_gen = backend_query_stream(sug)
                    full_response = st.write_stream(stream_gen)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

# ----------------- RETRIEVAL DEBUG PANEL -----------------
if st.session_state.debug_mode:
    st.markdown("---")
    st.subheader("🛠️ RAG Retrieval Debug Console")
    
    col_settings, col_dist = st.columns(2)
    with col_settings:
        st.markdown("**Workspace RAG Configurations:**")
        st.write(f"- Chunk size: `{config.CHUNK_SIZE}` characters")
        st.write(f"- Chunk overlap: `{config.CHUNK_OVERLAP}` characters")
        st.write(f"- Active context limit (K): `{config.DEFAULT_K}`")
        st.write(f"- MMR diversity parameter lambda: `{config.MMR_LAMBDA}`")
    with col_dist:
        if st.session_state.last_sources:
            st.markdown("**MMR Vector Distance Allocation:**")
            for idx, src in enumerate(st.session_state.last_sources):
                confidence = l2_distance_to_confidence(src["score"])
                st.write(f"Chunk {idx+1} [{src['source']}]: {confidence}% confidence")
                st.progress(int(confidence))
