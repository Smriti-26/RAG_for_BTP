import os
import json
import time
from typing import List
from config import (
    GEMINI_API_KEY, 
    GENERATION_MODEL, 
    EMBEDDING_MODEL, 
    CLEAN_TEXT_DIR, 
    INDEX_DIR, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP
)

# --- CORRECTED IMPORTS ---
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate 
from langchain_core.documents import Document 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# -------------------------

# Define file paths for the vector stores
ROUTER_INDEX_PATH = os.path.join(INDEX_DIR, "router_index")
SUBRAG_INDEX_PATH = os.path.join(INDEX_DIR, "subrag_index")

# --- LLM UTILITY ---

def call_gemini_api(prompt: str, model_name: str = GENERATION_MODEL, max_retries: int = 3) -> str:
    """Handles API call with exponential backoff using the Chat Model."""
    if not GEMINI_API_KEY:
        return "Gemini API key is missing."
    
    # Use the Chat Model (ChatGoogleGenerativeAI) instead of the base LLM (GoogleGenAI)
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GEMINI_API_KEY)
    
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt).content # .content is used for Chat Models
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  [API Error] Attempt {attempt+1} failed ({e}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  [API Error] Final attempt failed. Could not summarize. Error: {e}")
                return "Error: Could not generate summary."
    return "Error: API call failed after multiple retries."

# --- STAGE 1 & 2 (summarization and chunking functions) ---
def create_chapter_summaries() -> List[Document]:
    chapter_summary_docs = []
    
    SUMMARY_PROMPT = PromptTemplate.from_template(
        "You are an expert academic research assistant. Summarize the following chapter content "
        "in one concise paragraph (max 300 words). The summary MUST focus on the key "
        "topics, concepts, and technical terms so a semantic search engine can accurately "
        "determine the chapter's content for a relevant user query.\n\n"
        "CHAPTER CONTENT:\n---\n{chapter_content}\n---"
    )

    text_files = [f for f in os.listdir(CLEAN_TEXT_DIR) if f.endswith('.txt')]
    
    print(f"\n--- STAGE 1: Creating {len(text_files)} Chapter Summaries ---")
    
    for filename in sorted(text_files):
        filepath = os.path.join(CLEAN_TEXT_DIR, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            chapter_content = f.read()
            
        print(f"  -> Summarizing: {filename}...")
        
        prompt = SUMMARY_PROMPT.format(chapter_content=chapter_content[:30000])
        summary_text = call_gemini_api(prompt)
        
        if summary_text.startswith("Error"):
            print(f"  -> Skipping {filename} due to API error.")
            continue
            
        summary_doc = Document(
            page_content=summary_text,
            metadata={"source": filename, "type": "router_summary"}
        )
        chapter_summary_docs.append(summary_doc)
        
        if not os.path.exists(INDEX_DIR):
             os.makedirs(INDEX_DIR)
        with open(os.path.join(INDEX_DIR, f"{filename}_summary.txt"), 'w', encoding='utf-8') as sf:
            sf.write(summary_text)

    return chapter_summary_docs

def create_detailed_chunks() -> List[Document]:
    text_files = [os.path.join(CLEAN_TEXT_DIR, f) for f in os.listdir(CLEAN_TEXT_DIR) if f.endswith('.txt')]
    
    print(f"\n--- STAGE 2: Loading and Chunking Documents ---")
    raw_docs = []
    for filepath in text_files:
        loader = TextLoader(filepath, encoding='utf-8')
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(filepath)
        raw_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    detailed_chunks = text_splitter.split_documents(raw_docs)
    
    print(f"  -> Total chunks created for Sub-RAG: {len(detailed_chunks)}")
    
    return detailed_chunks

# --- STAGE 3: BUILD VECTOR STORES ---

def build_indices(router_docs: List[Document], subrag_chunks: List[Document]):
    if not router_docs or not subrag_chunks:
        print("Index build failed: No documents found.")
        return

    # Use the correct embedding class
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
    
    print("\n--- STAGE 3: Building Router Index (Chapter Summaries) ---")
    router_vectorstore = FAISS.from_documents(router_docs, embeddings)
    router_vectorstore.save_local(ROUTER_INDEX_PATH)
    print(f"  ✅ Router Index saved to: {ROUTER_INDEX_PATH}")
    
    print("\n--- STAGE 4: Building Sub-RAG Index (Detailed Chunks) ---")
    subrag_vectorstore = FAISS.from_documents(subrag_chunks, embeddings)
    subrag_vectorstore.save_local(SUBRAG_INDEX_PATH)
    print(f"  ✅ Sub-RAG Index saved to: {SUBRAG_INDEX_PATH}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("🔴 ERROR: Please set your GEMINI_API_KEY in config.py before running.")
    else:
        router_documents = create_chapter_summaries()
        subrag_documents = create_detailed_chunks()
        build_indices(router_documents, subrag_documents)
        print("\n🎉 Indexing Process Complete. Proceed to retrieval_and_generation.py.")