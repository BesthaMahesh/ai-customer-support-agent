import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# API Parameters
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "tencent/hy3:free")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Database Directories
DB_DIR = "vector_db"
INDEX_PATH = os.path.join(DB_DIR, "faiss_index.bin")
META_PATH = os.path.join(DB_DIR, "metadata.pkl")
TEMP_DIR = "temp_uploads"

# RAG Splitter & Search configs
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DEFAULT_K = 5
MMR_LAMBDA = 0.5  # Parameter for MMR search (0 = max diversity, 1 = max similarity)
