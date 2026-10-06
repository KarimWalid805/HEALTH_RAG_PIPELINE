from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from time import perf_counter

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    NEON_WRITE_BATCH_SIZE,
    get_neon_connection_url,
    get_embeddings,
)


def load_documents(docs_path=DOCS_DIR):
    """Load all PDF documents from the project docs directory."""
    docs_path = Path(docs_path)
    if not docs_path.exists():
        raise FileNotFoundError(f"The documents directory does not exist: {docs_path}")

    loader = DirectoryLoader(
        path=str(docs_path),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    if not documents:
        raise FileNotFoundError(f"No .pdf files found in {docs_path}.")

    print(f"Loaded {len(documents)} source documents from {docs_path}.")
    return documents


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split source documents into stable, retrieval-sized chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def document_id(document, index):
    """Build a deterministic ID so rerunning ingestion updates, not duplicates, chunks."""
    source = Path(document.metadata.get("source", "unknown"))
    try:
        source_key = source.resolve().relative_to(DOCS_DIR.resolve()).as_posix()
    except ValueError:
        source_key = source.name
    identity = f"{source_key}:{index}:{document.page_content}"
    return str(uuid5(NAMESPACE_URL, identity))


def create_vector_store(chunks):
    """Embed in API-sized batches and upsert bounded batches into Neon."""
    store_start = perf_counter()
    embeddings = get_embeddings()
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=get_neon_connection_url(),
        use_jsonb=True,
    )
    store_init_seconds = perf_counter() - store_start

    ids = [document_id(chunk, index) for index, chunk in enumerate(chunks)]
    lookup_start = perf_counter()
    existing_ids = {
        document.id for document in vector_store.get_by_ids(ids) if document.id
    }
    lookup_seconds = perf_counter() - lookup_start
    pending = [
        (chunk, chunk_id)
        for chunk, chunk_id in zip(chunks, ids)
        if chunk_id not in existing_ids
    ]

    if not pending:
        print(f"Neon collection '{COLLECTION_NAME}' is already up to date.")
        print(
            f"Timing: store setup={store_init_seconds:.2f}s; "
            f"existing-ID lookup={lookup_seconds:.2f}s; new chunks=0."
        )
        return vector_store

    embedding_seconds = 0.0
    write_seconds = 0.0
    for embed_start in range(0, len(pending), EMBEDDING_BATCH_SIZE):
        embed_batch = pending[embed_start : embed_start + EMBEDDING_BATCH_SIZE]
        batch_docs = [document for document, _ in embed_batch]
        batch_ids = [document_id for _, document_id in embed_batch]
        batch_texts = [document.page_content for document in batch_docs]
        embed_started = perf_counter()
        batch_embeddings = embeddings.embed_documents(batch_texts)
        embedding_seconds += perf_counter() - embed_started

        for write_start in range(0, len(embed_batch), NEON_WRITE_BATCH_SIZE):
            write_end = write_start + NEON_WRITE_BATCH_SIZE
            write_started = perf_counter()
            vector_store.add_embeddings(
                texts=batch_texts[write_start:write_end],
                embeddings=batch_embeddings[write_start:write_end],
                metadatas=[document.metadata for document in batch_docs[write_start:write_end]],
                ids=batch_ids[write_start:write_end],
            )
            write_seconds += perf_counter() - write_started

        saved = min(embed_start + len(embed_batch), len(pending))
        print(f"Embedded and saved {saved}/{len(pending)} new chunks.")

    print(
        f"Upserted {len(pending)} new chunks into Neon collection '{COLLECTION_NAME}' "
        f"({len(chunks) - len(pending)} unchanged)."
    )
    print(
        f"Timing: store setup={store_init_seconds:.2f}s; "
        f"existing-ID lookup={lookup_seconds:.2f}s; embeddings={embedding_seconds:.2f}s; "
        f"Neon writes={write_seconds:.2f}s."
    )
    return vector_store


def main():
    start_time = perf_counter()
    load_start = perf_counter()
    documents = load_documents()
    load_seconds = perf_counter() - load_start
    split_start = perf_counter()
    chunks = split_documents(documents)
    split_seconds = perf_counter() - split_start
    create_vector_store(chunks)
    end_time = perf_counter()
    print(
        f"Timing: document loading={load_seconds:.2f}s; "
        f"chunking={split_seconds:.2f}s; "
        f"total={end_time - start_time:.2f}s."
    )

if __name__ == "__main__":
    main()

