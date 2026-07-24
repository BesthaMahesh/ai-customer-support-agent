import os
import streamlit as st
import requests
import io
import time
from datetime import datetime
from typing import List, Dict, Any
import config
from utils import get_system_metrics, l2_distance_to_confidence, get_logger
import json

logger = get_logger("SaaSFrontendUI")

st.set_page_config(
    page_title="RAG Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global helper for streaming queries
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

# Helper to process uploaded files
def process_uploaded_files(files_list):
    if not files_list:
        return False
    new_upload = False
    for uploaded_file in files_list:
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
    return new_upload

# Initialize Session variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

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

# CSS Injection for Premium Polished SaaS Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* 1. Global Reset & Typography Standards */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC !important;
        color: #111827 !important;
        font-size: 15px;
    }
    
    .stApp {
        background-color: #F8FAFC !important;
    }
    
    /* Hide Streamlit default headers & branding */
    [data-testid="stHeader"] {display: none !important;}
    footer {visibility: hidden;}
    
    /* 2. Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #F1F5F9;
    }
    ::-webkit-scrollbar-thumb {
        background: #E2E8F0;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #4F46E5;
    }
    
    /* 3. Left Sidebar Customization (Width restricted to 270px) */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
        padding-top: 16px;
        width: 270px !important;
        min-width: 270px !important;
        max-width: 270px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 270px !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8 !important;
        margin-top: 24px;
        margin-bottom: 8px;
    }
    
    /* Search Bar with Vector Icon styling */
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        border-radius: 12px !important;
        border: 1px solid #1E293B !important;
        padding-left: 35px !important;
        background: #1E293B url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2394A3B8'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z'/%3E%3C/svg%3E") no-repeat 10px center !important;
        background-size: 16px 16px !important;
        color: #F8FAFC !important;
        height: 38px !important;
    }
    
    /* Sidebar List Document styling */
    .sidebar-doc-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 8px;
        border-radius: 8px;
        margin-bottom: 4px;
    }
    .sidebar-doc-row:hover {
        background-color: #1E293B;
    }
    .sidebar-doc-name {
        font-size: 13px;
        color: #E2E8F0;
        text-overflow: ellipsis;
        overflow: hidden;
        white-space: nowrap;
        max-width: 140px;
        display: inline-block;
    }
    
    /* Navigation menu links styling */
    .sidebar-menu {
        margin-top: 16px;
    }
    .menu-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 12px;
        color: #94A3B8 !important;
        font-size: 14px;
        font-weight: 500;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 4px;
    }
    .menu-item:hover {
        background-color: #1E293B;
        color: #FFFFFF !important;
    }
    
    /* 4. Main Header */
    .main-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
        border-radius: 12px;
    }
    .workspace-title {
        font-size: 24px;
        font-weight: 600;
        color: #111827;
    }
    .header-right {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .settings-icon {
        font-size: 1.25rem;
        color: #6B7280;
        cursor: pointer;
    }
    .profile-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #4F46E5;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    /* 5. Metrics Cards (100px Height, consistent spacing) */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 16px;
        min-height: 100px;
        height: 100px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #111827;
    }
    
    /* 6. Chat Bubbles */
    .chat-row {
        display: flex;
        width: 100%;
        margin-bottom: 16px;
        align-items: flex-start;
    }
    .user-row {
        justify-content: flex-end;
    }
    .assistant-row {
        justify-content: flex-start;
        gap: 12px;
    }
    .chat-bubble {
        max-width: 80%;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 15px;
        line-height: 1.6;
    }
    .user-bubble {
        background-color: #4F46E5;
        color: #FFFFFF;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .assistant-bubble {
        background-color: #FFFFFF;
        color: #111827;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .assistant-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: #E5E7EB;
        color: #4F46E5;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.75rem;
        flex-shrink: 0;
    }
    
    /* Action buttons under Assistant response */
    .stButton>button {
        background: transparent !important;
        color: #6B7280 !important;
        border: none !important;
        font-size: 13px !important;
        padding: 2px 4px !important;
        margin-top: -8px !important;
        border-radius: 8px !important;
    }
    .stButton>button:hover {
        color: #4F46E5 !important;
        background: #EFF6FF !important;
    }
    
    /* Custom button in sidebar upload zone to prevent overflow */
    [data-testid="stSidebar"] .stButton>button {
        background: #4F46E5 !important;
        color: white !important;
        border-radius: 12px !important;
        width: 100% !important;
        height: 40px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        margin-top: 0px !important;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background: #4338CA !important;
    }
    
    /* 7. Source Cards (Right Column) */
    .source-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .source-title {
        font-size: 14px;
        font-weight: 500;
        color: #111827;
        margin-bottom: 4px;
    }
    .source-page {
        font-size: 12px;
        color: #6B7280;
        margin-bottom: 8px;
    }
    .source-preview {
        font-size: 13px;
        color: #374151;
        line-height: 1.5;
        margin-bottom: 8px;
    }
    
    /* 8. Empty State (Centering and reduced height) */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 32px 24px;
        text-align: center;
        background: #FFFFFF;
        border: 1px dashed #E5E7EB;
        border-radius: 12px;
        min-height: 220px;
        height: 220px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .empty-state-illustration {
        font-size: 2.5rem;
        margin-bottom: 12px;
        color: #4F46E5;
    }
    .empty-state-title {
        font-size: 18px;
        font-weight: 500;
        color: #111827;
        margin-bottom: 8px;
    }
    .empty-state-text {
        font-size: 15px;
        color: #6B7280;
        margin-bottom: 0px;
    }
    
    /* Custom input bar overrides */
    .stChatInputContainer {
        border-radius: 12px !important;
        border: 1px solid #E5E7EB !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03) !important;
        padding: 4px 10px !important;
    }
    .stChatInputContainer:focus-within {
        border-color: #4F46E5 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    # SaaS Branding Logo
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 2rem;">
            <span style="font-size: 1.5rem; color: #4F46E5;">⚡</span>
            <h3 style="margin: 0; font-weight: 600; color: #FFFFFF; font-size: 1.2rem;">RAG Workspace</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Hide sidebar upload area if DB is empty to prevent duplicates.
    # Uploader is conditionally rendered.
    sidebar_upload = False
    if unique_docs_count > 0:
        st.markdown("### Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload Document",
            type=["pdf", "docx", "txt", "csv", "xlsx", "pptx", "md"],
            accept_multiple_files=True,
            key="sidebar_uploader",
            label_visibility="collapsed"
        )
        sidebar_upload = process_uploaded_files(uploaded_files)

    # Search within documents
    doc_search = st.text_input("Search documents...", placeholder="Search documents...", label_visibility="collapsed")
    
    # Filter documents based on search
    if unique_docs:
        filtered_docs = [d for d in unique_docs if doc_search.lower() in d.lower()]
    else:
        filtered_docs = []

    st.markdown("### Uploaded Documents")
    if unique_docs_count > 0:
        if filtered_docs:
            for doc in filtered_docs:
                col_doc_name, col_doc_del = st.columns([8, 2])
                with col_doc_name:
                    st.markdown(f"""
                        <div class="sidebar-doc-row">
                            <span class="sidebar-doc-name" title="{doc}">📄 {doc}</span>
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
            st.caption("No matching documents.")
    else:
        st.caption("No uploaded documents yet.")

    st.markdown("---")
    
    # Professional Menu Links replacement for technical K/MMR configs
    st.markdown("### Navigation")
    st.markdown("""
        <div class="sidebar-menu">
            <div class="menu-item">⚙️ Workspace Settings</div>
            <div class="menu-item">🎨 Theme</div>
            <div class="menu-item">👤 Profile</div>
            <div class="menu-item">🚪 Logout</div>
        </div>
    """, unsafe_allow_html=True)

# ----------------- MAIN AREA WORKSPACE -----------------
# Main Header
st.markdown("""
<div class="main-header">
    <div class="workspace-title">RAG Workspace</div>
    <div class="header-right">
        <div class="settings-icon">⚙️</div>
        <div class="profile-avatar">S</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Statistics Overview (only 2 cards, 100px height, consistent layout)
st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card">
        <div class="metric-label">Documents</div>
        <div class="metric-value">{unique_docs_count}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Indexed Files</div>
        <div class="metric-value">{unique_docs_count}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Workspace layout: Left (75%) Chat, Right (25%) Sources (Reduced right column width)
col_workspace, col_citations = st.columns([75, 25], gap="large")

# Render conversation or empty state
main_upload = False
with col_workspace:
    if unique_docs_count == 0:
        # Empty State card with reduced height, centered text and uploader
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-illustration">📂</div>
            <div class="empty-state-title">No Documents Indexed</div>
            <div class="empty-state-text">Upload your first document to start chatting with your knowledge base.</div>
        </div>
        """, unsafe_allow_html=True)
        
        main_files = st.file_uploader(
            "Upload first document to start",
            type=["pdf", "docx", "txt", "csv", "xlsx", "pptx", "md"],
            accept_multiple_files=True,
            key="main_uploader",
            label_visibility="collapsed"
        )
        main_upload = process_uploaded_files(main_files)
    else:
        # Render Chat Bubbles
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f"""
                    <div class="chat-row user-row">
                        <div class="chat-bubble user-bubble">{msg['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-row assistant-row">
                        <div class="assistant-avatar">AI</div>
                        <div class="chat-bubble assistant-bubble">{msg['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Copy & Regenerate Actions row
                col_space, col_copy, col_regen = st.columns([8, 1, 1])
                with col_copy:
                    if st.button("📋 Copy", key=f"copy_{idx}", use_container_width=True):
                        st.toast("Response copied to clipboard!")
                with col_regen:
                    if st.button("🔄 Regen", key=f"regen_{idx}", use_container_width=True):
                        # Rollback chat state to right before this assistant message
                        st.session_state.messages = st.session_state.messages[:idx]
                        st.rerun()

# Rerun triggers if uploads occurred
if sidebar_upload or main_upload:
    time.sleep(0.5)
    st.rerun()

# Stream Generation Block (automatically triggered if last message is a user message)
with col_workspace:
    if unique_docs_count > 0 and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_prompt = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant", avatar="🤖"):
            try:
                stream_gen = backend_query_stream(user_prompt)
                full_response = st.write_stream(stream_gen)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"RAG Query failure: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.last_sources = []
        st.rerun()

# Input panel (Only visible when not waiting for an assistant response)
with col_workspace:
    if unique_docs_count > 0:
        if (st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant") or not st.session_state.messages:
            if prompt := st.chat_input("Ask about company policies, resumes, or uploaded research papers..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

# Retrieved Sources Panel (Right Column 25%, display friendly empty state)
with col_citations:
    st.markdown("### Retrieved Sources")
    if st.session_state.last_sources:
        for idx, src in enumerate(st.session_state.last_sources):
            # Clean preview
            preview = src['text'][:250].replace('\n', ' ') + "..."
            st.markdown(f"""
                <div class="source-card">
                    <div class="source-title">📄 {src['source']}</div>
                    <div class="source-page">Page/Slide {src['page']}</div>
                    <div class="source-preview">"{preview}"</div>
                    <details style="margin-top: 10px; cursor: pointer;">
                        <summary style="font-size: 13px; color: #4F46E5; font-weight: 500;">View Source</summary>
                        <div style="font-size: 13px; color: #374151; margin-top: 8px; line-height: 1.5; background: #F8FAFC; border: 1px solid #E5E7EB; padding: 10px; border-radius: 6px; font-style: italic;">
                            "{src['text']}"
                        </div>
                    </details>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background:#FFFFFF; border:1px dashed #E5E7EB; border-radius:12px; padding: 1.5rem; text-align:center; color:#6B7280; font-size:0.85rem;">
                No sources fetched yet. Send a query to see citations.
            </div>
        """, unsafe_allow_html=True)
