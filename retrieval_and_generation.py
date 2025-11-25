import os
import json
from typing import List, Dict
from config import (
    GEMINI_API_KEY, 
    GENERATION_MODEL, 
    EMBEDDING_MODEL, 
    INDEX_DIR, 
    K1_ROUTER_CHAPTERS, 
    K2_CHUNKS_PER_CHAPTER
)

# --- CORRECTED IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI # <-- Using the correct Chat Model
# -------------------------

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document 
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.llm import LLMChain

# Define file paths for the vector stores
ROUTER_INDEX_PATH = os.path.join(INDEX_DIR, "router_index")
SUBRAG_INDEX_PATH = os.path.join(INDEX_DIR, "subrag_index")

# --- INITIALIZATION ---

def initialize_rag_components():
    """Load the necessary indices and components."""
    if not GEMINI_API_KEY or not os.path.exists(ROUTER_INDEX_PATH) or not os.path.exists(SUBRAG_INDEX_PATH):
        raise FileNotFoundError("RAG setup is incomplete. Run indexing.py first and check your API key.")

    # Use the correct embedding class
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
    
    # Load Vector Stores
    router_vectorstore = FAISS.load_local(ROUTER_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    subrag_vectorstore = FAISS.load_local(SUBRAG_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Initialize the LLM for final generation
    llm = ChatGoogleGenerativeAI(model=GENERATION_MODEL, google_api_key=GEMINI_API_KEY)
    
    return router_vectorstore, subrag_vectorstore, llm

# --- TWO-STAGE RETRIEVAL (remains the same) ---

def hierarchical_retrieve(query: str, router_vs: FAISS, subrag_vs: FAISS) -> List[Document]:
    """Executes the two-stage retrieval process."""
    
    print("\n[Stage 1] Executing Router Search...")
    
    router_results = router_vs.similarity_search(query, k=K1_ROUTER_CHAPTERS)
    selected_chapters = set(doc.metadata["source"] for doc in router_results)
    
    if not selected_chapters:
        print("  -> No relevant chapters found.")
        return []
        
    print(f"  -> Router selected {len(selected_chapters)} chapters (k={K1_ROUTER_CHAPTERS}): {selected_chapters}")
    
    print("\n[Stage 2] Executing Detailed Sub-RAG Search...")
    
    all_final_chunks = []
    
    for chapter_file in selected_chapters:
        
        chapter_specific_chunks = subrag_vs.similarity_search_with_score(
            query, 
            k=K2_CHUNKS_PER_CHAPTER * len(selected_chapters) * 2
        )

        filtered_chunks = [
            doc for doc, score in chapter_specific_chunks if doc.metadata.get("source") == chapter_file
        ][:K2_CHUNKS_PER_CHAPTER]

        all_final_chunks.extend(filtered_chunks)
        print(f"  -> Retrieved {len(filtered_chunks)} chunks from {chapter_file}")

    print(f"\n[Stage 2] Total final chunks retrieved: {len(all_final_chunks)}")
    return all_final_chunks

# --- GENERATION (uses ChatGoogleGenerativeAI via LLMChain) ---

def generate_answer(query: str, chunks: List[Document], llm: ChatGoogleGenerativeAI) -> str:
    """Generates the final answer using the retrieved context."""
    
    if not chunks:
        return "I could not find any relevant information in the provided handbook chapters to answer your question."

    context_text = "\n---\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" for d in chunks])

    GENERATION_PROMPT = PromptTemplate.from_template(
        "You are an expert Q&A assistant specializing in the ICAR Handbook of Agriculture. "
        "Use ONLY the following retrieved context to answer the user's question. "
        "Synthesize a concise, comprehensive answer. If the context does not contain the answer, "
        "state explicitly that the information is not available in the provided sources. "
        "Cite the specific source files (e.g., Chapter 3.txt) you used for your answer.\n\n"
        "RETRIEVED CONTEXT:\n{context}\n\n"
        "USER QUESTION: {query}"
    )

    rag_chain = LLMChain(llm=llm, prompt=GENERATION_PROMPT)
    
    print("\n[Stage 3] Generating final answer...")
    try:
        # LLMChain invoke returns a dictionary; extract the output key (usually 'text')
        response = rag_chain.invoke({"context": context_text, "query": query})
        return response['text'].strip() 
    except Exception as e:
        return f"Error during generation: {e}"


# --- MAIN EXECUTION (remains the same) ---
if __name__ == "__main__":
    try:
        router_vs, subrag_vs, llm = initialize_rag_components()
    except FileNotFoundError as e:
        print(f"🔴 ERROR: {e}")
        exit()

    print("--- Hierarchical RAG System Initialized ---")
    
    test_query = "What were the major changes in Indian food production between 1965 and 2008, and what were the associated environmental impacts like water table decline?"
    print(f"\n❓ QUERY: {test_query}")
    
    retrieved_chunks = hierarchical_retrieve(test_query, router_vs, subrag_vs)
    final_answer = generate_answer(test_query, retrieved_chunks, llm)
    
    print("\n==========================================")
    print("✨ FINAL ANSWER:")
    print(final_answer)
    print("==========================================")