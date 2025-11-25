import os

# --- FILE PATHS ---
CLEAN_TEXT_DIR = "./clean_prose_chapters"
INDEX_DIR = "./rag_indices"

# --- API AND MODEL CONFIGURATION ---
GEMINI_API_KEY = "my_secret_key" 

# Model for summarization and final generation (Now a Chat Model)
GENERATION_MODEL = "gemini-2.5-flash-preview-09-2025" 

# Model for embedding (no change)
EMBEDDING_MODEL = "text-embedding-004"

# --- RAG PARAMETERS ---
K1_ROUTER_CHAPTERS = 3
K2_CHUNKS_PER_CHAPTER = 2 
CHUNK_SIZE = 1500 
CHUNK_OVERLAP = 100 

if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":

    print("🚨 WARNING: Please set your GEMINI_API_KEY in config.py")
