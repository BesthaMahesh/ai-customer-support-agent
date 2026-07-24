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
if "questions_count" not in st.session_state:
    st.session_state.questions_count = 12

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

# CSS Injection for World-Class $100M SaaS UI Look
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
        background: #CBD5E1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #4F46E5;
    }
    
    /* 3. Left Sidebar Customization (280px Wide, Dark #111827) */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1F2937 !important;
        padding-top: 24px;
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 280px !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #F9FAFB !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9CA3AF !important;
        margin-top: 24px;
        margin-bottom: 8px;
    }
    
    /* Search Bar inside Sidebar */
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        border-radius: 12px !important;
        border: 1px solid #374151 !important;
        padding-left: 35px !important;
        background: #1F2937 url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%239CA3AF'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z'/%3E%3C/svg%3E") no-repeat 10px center !important;
        background-size: 16px 16px !important;
        color: #F9FAFB !important;
        height: 48px !important;
    }
    
    /* Sidebar List Document styling */
    .sidebar-doc-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 12px;
        border-radius: 12px;
        margin-bottom: 4px;
    }
    .sidebar-doc-row:hover {
        background-color: #1F2937;
    }
    .sidebar-doc-name {
        font-size: 13px;
        color: #F3F4F6;
        text-overflow: ellipsis;
        overflow: hidden;
        white-space: nowrap;
        max-width: 150px;
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
        padding: 10px 12px;
        color: #9CA3AF !important;
        font-size: 14px;
        font-weight: 500;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 4px;
    }
    .menu-item:hover {
        background-color: #1F2937;
        color: #FFFFFF !important;
    }
    
    /* Recent conversation mock item */
    .recent-chat-item {
        font-size: 14px;
        color: #D1D5DB;
        padding: 8px 12px;
        border-radius: 12px;
        cursor: pointer;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .recent-chat-item:hover {
        background: #1F2937;
        color: #FFFFFF;
    }
    
    /* Sidebar Profile Card */
    .sidebar-profile {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px;
        border-radius: 16px;
        background: #1F2937;
        margin-top: 24px;
    }
    .sidebar-profile-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #4F46E5;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 14px;
    }
    .sidebar-profile-info {
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    .sidebar-profile-name {
        font-size: 14px;
        font-weight: 500;
        color: #FFFFFF;
    }
    .sidebar-profile-email {
        font-size: 12px;
        color: #9CA3AF;
        text-overflow: ellipsis;
        overflow: hidden;
    }
    
    /* New Chat and Sidebar Buttons (48px height, 12px radius) */
    [data-testid="stSidebar"] .stButton>button {
        background: #4F46E5 !important;
        color: white !important;
        border-radius: 12px !important;
        width: 100% !important;
        height: 48px !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        margin-top: 0px !important;
        border: none !important;
        transition: background 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background: #4338CA !important;
    }
    
    /* Secondary sidebar delete button styling */
    [data-testid="stSidebar"] .doc-delete-btn button {
        background: transparent !important;
        color: #9CA3AF !important;
        width: auto !important;
        height: 32px !important;
        padding: 4px !important;
        font-size: 14px !important;
        margin-top: 0px !important;
    }
    [data-testid="stSidebar"] .doc-delete-btn button:hover {
        color: #EF4444 !important;
        background: transparent !important;
    }

    /* 4. Top Sticky Navbar (72px height, White Glassmorphism, Soft Shadow) */
    .top-navbar {
        height: 72px;
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(229, 231, 235, 0.5);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
    }
    .navbar-title {
        font-size: 22px;
        font-weight: 600;
        color: #111827;
    }
    .navbar-right {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .navbar-icon {
        font-size: 1.25rem;
        color: #6B7280;
        cursor: pointer;
        padding: 8px;
        border-radius: 12px;
        transition: background 0.2s;
    }
    .navbar-icon:hover {
        background: #F1F5F9;
        color: #111827;
    }
    .navbar-avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: #E0E7FF;
        color: #4F46E5;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 14px;
        border: 2px solid #FFFFFF;
        box-shadow: 0 0 0 2px #E0E7FF;
    }
    
    /* navbar search box styling */
    .navbar-search {
        display: flex;
        align-items: center;
        background: #F1F5F9;
        border-radius: 12px;
        padding: 8px 12px;
        width: 240px;
    }
    .navbar-search input {
        border: none;
        background: transparent;
        font-size: 14px;
        outline: none;
        width: 100%;
        color: #111827;
    }
    .navbar-search-icon {
        margin-right: 8px;
        color: #9CA3AF;
        font-size: 14px;
    }

    /* 5. Metrics Cards (Analytics grid, 16px radius, hover animation, trendlines) */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #4F46E5;
    }
    .metric-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        color: #6B7280;
    }
    .metric-icon {
        font-size: 18px;
        color: #4F46E5;
        background: #EEF2F6;
        padding: 6px;
        border-radius: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 4px;
    }
    .metric-trend {
        font-size: 12px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .trend-up {
        color: #22C55E;
    }
    .trend-neutral {
        color: #4F46E5;
    }
    
    /* 6. Chat Bubbles & Actions */
    .chat-row {
        display: flex;
        width: 100%;
        margin-bottom: 24px;
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
        border-radius: 16px;
        padding: 16px 20px;
        font-size: 15px;
        line-height: 1.6;
    }
    .user-bubble {
        background-color: #4F46E5;
        color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.1);
    }
    .assistant-bubble {
        background-color: #FFFFFF;
        color: #111827;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .assistant-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background-color: #EEF2F6;
        color: #4F46E5;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.85rem;
        flex-shrink: 0;
        border: 1px solid #E5E7EB;
    }
    
    /* Action items underneath assistant bubble */
    .assistant-actions {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 8px;
        margin-left: 48px;
    }
    .assistant-action-btn {
        background: transparent !important;
        border: none !important;
        color: #9CA3AF !important;
        font-size: 13px !important;
        cursor: pointer;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        transition: all 0.2s;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .assistant-action-btn:hover {
        color: #4F46E5 !important;
        background: #F1F5F9 !important;
    }
    .assistant-action-btn.dislike:hover {
        color: #EF4444 !important;
    }

    /* 7. Source Cards (Right Column) */
    .source-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .source-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-color: #4F46E5;
    }
    .source-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }
    .source-icon {
        color: #4F46E5;
        font-size: 14px;
    }
    .source-title {
        font-size: 14px;
        font-weight: 600;
        color: #111827;
        text-overflow: ellipsis;
        overflow: hidden;
        white-space: nowrap;
    }
    .source-page {
        font-size: 12px;
        color: #6B7280;
        margin-bottom: 12px;
    }
    .source-preview {
        font-size: 13px;
        color: #374151;
        line-height: 1.5;
        background: #F8FAFC;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid #F1F5F9;
        font-style: italic;
    }
    
    /* 8. Empty State (Centered premium documentation look) */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 48px 32px;
        text-align: center;
        background: #FFFFFF;
        border: 1px dashed #E5E7EB;
        border-radius: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        max-width: 600px;
        margin: 40px auto;
    }
    .empty-state-illustration {
        font-size: 3.5rem;
        margin-bottom: 16px;
        color: #4F46E5;
    }
    .empty-state-title {
        font-size: 22px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 8px;
    }
    .empty-state-text {
        font-size: 15px;
        color: #6B7280;
        margin-bottom: 24px;
        line-height: 1.5;
    }
    .empty-state-buttons {
        display: flex;
        gap: 16px;
        justify-content: center;
    }
    
    /* Custom button in empty state */
    .primary-saas-btn button {
        background: #4F46E5 !important;
        color: white !important;
        height: 48px !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        padding: 0 24px !important;
        border: none !important;
        transition: background 0.2s ease !important;
    }
    .primary-saas-btn button:hover {
        background: #4338CA !important;
    }
    .secondary-saas-btn button {
        background: #FFFFFF !important;
        color: #4F46E5 !important;
        height: 48px !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        padding: 0 24px !important;
        border: 1px solid #4F46E5 !important;
        transition: all 0.2s ease !important;
    }
    .secondary-saas-btn button:hover {
        background: #F5F7FF !important;
        color: #4338CA !important;
    }

    /* Custom input bar overrides */
    .stChatInputContainer {
        border-radius: 16px !important;
        border: 1px solid #E5E7EB !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03) !important;
        padding: 8px 16px !important;
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
    
    # 1. New Chat Action Button (48px height, 12px radius, Indigo)
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.toast("Chat reset successfully!", icon="🧹")
        time.sleep(0.5)
        st.rerun()
        
    st.markdown("### Knowledge Base")
    
    # Hide sidebar upload area if DB is empty to prevent duplicates.
    sidebar_upload = False
    if unique_docs_count > 0:
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
                    # Renders inline delete button inside container
                    st.markdown('<div class="doc-delete-btn">', unsafe_allow_html=True)
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
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("No matching documents.")
    else:
        st.caption("No uploaded documents yet.")

    # Recent Conversations list
    st.markdown("### Recent Conversations")
    st.markdown("""
        <div class="recent-chat-item">💬 Q3 Strategy Planning</div>
        <div class="recent-chat-item">💬 Resume Screening</div>
        <div class="recent-chat-item">💬 Technical Documentation</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Professional Menu Links replacement for technical configs
    st.markdown("### Navigation")
    st.markdown("""
        <div class="sidebar-menu">
            <div class="menu-item">⚙️ Workspace Settings</div>
            <div class="menu-item">🎨 Theme</div>
            <div class="menu-item">👤 Profile</div>
            <div class="menu-item">🚪 Logout</div>
        </div>
    """, unsafe_allow_html=True)
    
    # User Profile avatar card at bottom
    st.markdown("""
        <div class="sidebar-profile">
            <div class="sidebar-profile-avatar">S</div>
            <div class="sidebar-profile-info">
                <div class="sidebar-profile-name">Support User</div>
                <div class="sidebar-profile-email">support@saas.com</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ----------------- MAIN AREA WORKSPACE -----------------
# 72px Glassmorphism Top Navbar
st.markdown("""
<div class="top-navbar">
    <div class="navbar-title">RAG Workspace</div>
    <div class="navbar-right">
        <div class="navbar-search">
            <span class="navbar-search-icon">🔍</span>
            <input type="text" placeholder="Search files & queries..."/>
        </div>
        <div class="navbar-icon" title="Notifications">🔔</div>
        <div class="navbar-icon" title="Workspace Settings">⚙️</div>
        <div class="navbar-avatar">SU</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Process uploads
sidebar_upload = process_uploaded_files(uploaded_files) if unique_docs_count > 0 else False

# 4 Analytics Cards Grid (16px radius, soft shadow, hover lift, trendlines)
st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card">
        <div class="metric-top">
            <div class="metric-label">Documents</div>
            <div class="metric-icon">📂</div>
        </div>
        <div class="metric-value">{unique_docs_count}</div>
        <div class="metric-trend"><span class="trend-up">▲ 12%</span> <span style="color:#6B7280;">this week</span></div>
    </div>
    <div class="metric-card">
        <div class="metric-top">
            <div class="metric-label">Indexed Files</div>
            <div class="metric-icon">📑</div>
        </div>
        <div class="metric-value">{unique_docs_count}</div>
        <div class="metric-trend"><span class="trend-up">▲ 15%</span> <span style="color:#6B7280;">index growth</span></div>
    </div>
    <div class="metric-card">
        <div class="metric-top">
            <div class="metric-label">Questions Asked</div>
            <div class="metric-icon">❓</div>
        </div>
        <div class="metric-value">{st.session_state.questions_count}</div>
        <div class="metric-trend"><span class="trend-up">▲ 8%</span> <span style="color:#6B7280;">activity rise</span></div>
    </div>
    <div class="metric-card">
        <div class="metric-top">
            <div class="metric-label">Recent Activity</div>
            <div class="metric-icon">🔄</div>
        </div>
        <div class="metric-value">Active</div>
        <div class="metric-trend"><span class="trend-neutral">● Synced</span> <span style="color:#6B7280;">just now</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Workspace layout: Left (70%) Chat, Right (30%) Sources
col_workspace, col_citations = st.columns([70, 30], gap="large")

# Render conversation or empty state
main_upload = False
with col_workspace:
    if unique_docs_count == 0:
        # Centered premium Empty State block
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-illustration">📂</div>
            <div class="empty-state-title">Knowledge Base</div>
            <div class="empty-state-text">Upload documents to build your AI knowledge base.<br/>Supported formats: PDF, DOCX, TXT, CSV, XLSX, PPTX, Markdown</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Centered layout buttons
        col_btn_upload, col_btn_doc = st.columns([1, 1])
        with col_btn_upload:
            st.markdown('<div class="primary-saas-btn">', unsafe_allow_html=True)
            main_files = st.file_uploader(
                "Upload Documents",
                type=["pdf", "docx", "txt", "csv", "xlsx", "pptx", "md"],
                accept_multiple_files=True,
                key="main_uploader",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            main_upload = process_uploaded_files(main_files)
        with col_btn_doc:
            st.markdown('<div class="secondary-saas-btn">', unsafe_allow_html=True)
            if st.button("View Documentation", use_container_width=True):
                st.toast("Documentation is available at https://render.com/docs")
            st.markdown('</div>', unsafe_allow_html=True)
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
                
                # Copy & Regenerate Actions row styled cleanly
                st.markdown('<div class="assistant-actions">', unsafe_allow_html=True)
                col_btn_1, col_btn_2, col_btn_3, col_btn_4 = st.columns([1, 1, 1, 7])
                with col_btn_1:
                    if st.button("📋 Copy", key=f"copy_{idx}", use_container_width=True):
                        st.toast("Response copied to clipboard!")
                with col_btn_2:
                    if st.button("🔄 Regen", key=f"regen_{idx}", use_container_width=True):
                        st.session_state.messages = st.session_state.messages[:idx]
                        st.rerun()
                with col_btn_3:
                    if st.button("👍", key=f"like_{idx}", use_container_width=True):
                        st.toast("Feedback registered, thank you!")
                with col_btn_4:
                    if st.button("👎", key=f"dislike_{idx}", use_container_width=True):
                        st.toast("Feedback registered, thank you!")
                st.markdown('</div>', unsafe_allow_html=True)

# Rerun triggers if uploads occurred
if sidebar_upload or main_upload:
    time.sleep(0.5)
    st.rerun()

# Stream Generation Block (automatically triggered if last message is a user message)
with col_workspace:
    if unique_docs_count > 0 and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_prompt = st.session_state.messages[-1]["content"]
        # Increment questions asked
        st.session_state.questions_count += 1
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
            if prompt := st.chat_input("Ask anything about your uploaded documents..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

# Retrieved Sources Panel (Right Column 30%, premium source layout, display friendly empty state)
with col_citations:
    st.markdown("### Retrieved Sources")
    if st.session_state.last_sources:
        for idx, src in enumerate(st.session_state.last_sources):
            # Clean preview
            preview = src['text'][:250].replace('\n', ' ') + "..."
            st.markdown(f"""
                <div class="source-card">
                    <div class="source-header">
                        <span class="source-icon">📄</span>
                        <div class="source-title" title="{src['source']}">{src['source']}</div>
                    </div>
                    <div class="source-page">Page {src['page']}</div>
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
            <div style="background:#FFFFFF; border:1px dashed #E5E7EB; border-radius:16px; padding: 2rem; text-align:center; color:#6B7280; font-size:0.85rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                No sources fetched yet. Send a query to see citations.
            </div>
        """, unsafe_allow_html=True)
