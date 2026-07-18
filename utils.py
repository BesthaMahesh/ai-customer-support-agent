import os
import sys
import psutil
import logging
from typing import Dict, Any

# Setup Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger with the specified name."""
    return logging.getLogger(name)

logger = get_logger("SystemUtils")

def get_system_metrics() -> Dict[str, Any]:
    """Retrieves system resource metrics (memory usage of current process, CPU percent, total physical memory)."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
        cpu_usage = process.cpu_percent(interval=None)
        
        sys_mem = psutil.virtual_memory()
        sys_mem_percent = sys_mem.percent
        
        return {
            "process_memory_mb": round(rss_mb, 2),
            "process_cpu_percent": round(cpu_usage, 2),
            "system_memory_percent": sys_mem_percent
        }
    except Exception as e:
        logger.error(f"Error reading system metrics: {e}")
        return {
            "process_memory_mb": 0.0,
            "process_cpu_percent": 0.0,
            "system_memory_percent": 0.0
        }

def l2_distance_to_confidence(l2_dist: float) -> float:
    """Converts a FAISS L2 flat distance to a confidence percentage.
    
    Since SentenceTransformer embeddings are unit-normalized:
      L2 = 2 - 2 * CosineSimilarity
      => CosineSimilarity = 1 - (L2 / 2)
    This yields a similarity score from -1.0 to 1.0, which we scale to 0-100%.
    """
    try:
        cosine_sim = 1.0 - (l2_dist / 2.0)
        # Clip to [0, 1] for confidence percentage
        confidence = max(0.0, min(1.0, cosine_sim)) * 100.0
        return round(confidence, 2)
    except Exception as e:
        logger.error(f"Error converting distance to confidence: {e}")
        return 0.0
