import os
from collections import defaultdict
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# Load environment variables (API keys)
load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and LLM
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
model = ChatOpenAI(model="gpt-4o")

# Connect to the Chroma Vector Store
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# ==========================================
# HYBRID SEARCH SETUP (BM25 + DENSE VECTOR)
# ==========================================
# 1. Create the Dense Vector Retriever
vector_retriever = db.as_retriever(search_kwargs={"k": 5})

# 2. Create the BM25 Keyword Retriever
# We extract the existing documents from Chroma to build the local BM25 keyword index
chroma_data = db.get()
bm25_docs = [
    Document(page_content=content, metadata=meta or {}) 
    for content, meta in zip(chroma_data['documents'], chroma_data['metadatas'])
]

if bm25_docs:
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 5
    
    # 3. Combine them into an Ensemble Retriever
    # This weights the keyword search and vector search equally (50/50)
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5] 
    )
    print(f"✅ Hybrid Search (BM25 + Vector) Initialized with {len(bm25_docs)} documents!")
else:
    print("⚠️ No documents found in database. Falling back to Vector Search only.")
    retriever = vector_retriever

class QueryVariations(BaseModel):
    queries: List[str]

# Bind the structured output to our model for query generation
model_with_tools = model.with_structured_output(QueryVariations)

# Global chat history state
chat_history = []

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):
    """
    Takes multiple lists of retrieved documents, scores them based on their rank 
    in each list, and fuses them into a single sorted list.
    """
    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
    
    rrf_scores = defaultdict(float)  # Stores: {chunk_content: rrf_score}
    all_unique_chunks = {}           # Stores: {chunk_content: actual_chunk_object}
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result list (from each query variation)
    for query_idx, chunks in enumerate(chunk_lists, 1):
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1): 
            chunk_content = chunk.page_content
            
            # Assign a simple ID for verbose logging
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
                all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1 / (k + position)
            position_score = 1 / (k + position)
            rrf_scores[chunk_content] += position_score
    
    # Sort chunks by their total accumulated RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[content], score) for content, score in rrf_scores.items()],
        key=lambda x: x[1], 
        reverse=True 
    )
    
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
        
    return sorted_chunks

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # ==========================================
    # STEP 1: History-Aware Query Reformulation
    # ==========================================
    if chat_history:
        print("\n[Step 1] Reformulating question based on chat history...")
        reformulation_messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question. Do not answer it.")
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(reformulation_messages)
        search_question = result.content.strip()
        print(f"Standalone Query: {search_question}")
    else:
        search_question = user_question

    # ==========================================
    # STEP 2: Multi-Query Generation
    # ==========================================
    print("\n[Step 2] Generating query variations...")
    prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents from a vector database:
    Original query: {search_question}
    Return 3 alternative queries that rephrase or approach the same question from different angles."""
    
    response = model_with_tools.invoke(prompt)
    
    # Combine the standalone question with the 3 variations
    all_queries = [search_question] + response.queries
    for i, q in enumerate(all_queries, 1):
        print(f"  {i}. {q}")

    # ==========================================
    # STEP 3: Multi-Query Retrieval
    # ==========================================
    print("\n[Step 3] Retrieving documents for all queries...")
    all_retrieval_results = [] 
    
    for query in all_queries:
        docs = retriever.invoke(query)
        all_retrieval_results.append(docs)

    # ==========================================
    # STEP 4: Reciprocal Rank Fusion (RRF)
    # ==========================================
    # Pass the lists of documents to our RRF function
    fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60, verbose=False)
    
    # Extract just the top 5 documents from the fused, re-ranked list
    top_fused_docs = [doc for doc, score in fused_results[:5]]
    
    print(f"\n[Step 4] Top 5 documents selected via RRF.")
    for i, doc in enumerate(top_fused_docs, 1):
        preview = doc.page_content.replace('\n', ' ')[:80]
        print(f"  Doc {i}: {preview}...")

    # ==========================================
    # STEP 5: Final Answer Generation
    # ==========================================
    print("\n[Step 5] Generating final answer...")
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in top_fused_docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    final_messages = [
        SystemMessage(content="You are a helpful assistant that answers questions strictly based on provided documents and conversation history.")
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(final_messages)
    answer = result.content

    # Update global chat history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
        
    print(f"\n🤖 Answer: {answer}")
    return answer
    
# ==========================================
# Application Entry Point
# ==========================================
def start_chat():
    print("="*60)
    print("Advanced RAG Chat Started (Multi-Query + RRF + Memory)")
    print("="*60)
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()