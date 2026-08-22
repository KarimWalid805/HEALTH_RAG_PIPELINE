import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import warnings
# since langchain_community is deprecated, we will ignore deprecation warnings for now, might update this in the future with langchain_unstructured
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

from langchain_community.document_loaders import TextLoader, DirectoryLoader
load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt", # can be changed to "*.pdf" or other formats if needed
        loader_cls=TextLoader, # specify the loader class for text files
        loader_kwargs={"encoding": "utf-8"}
         
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=512, chunk_overlap=51):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=70
)
    recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],  # Multiple separators
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
    
    #chunks = semantic_splitter.split_documents(documents) #used for semantic splitting, but can be changed to recursive_splitter for character-based splitting. (MORE EXPENSIVE, uses embeddings)
    chunks = recursive_splitter.split_documents(documents) #used for character-based splitting, but can be changed to semantic_splitter for semantic splitting. (LESS EXPENSIVE, does not use embeddings)
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
      # Step 1: Load documents
    documents = load_documents(docs_path="docs")

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Create vector store
    vectorstore = create_vector_store(chunks, persist_directory="db/chroma_db")

if __name__ == "__main__":
    main()
   