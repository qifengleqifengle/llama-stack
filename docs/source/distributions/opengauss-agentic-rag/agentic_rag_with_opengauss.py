"""
Agentic RAG demo using Llama Stack with OpenGauss as the Vector IO backend.

This script demonstrates how to:
1. Connect to a running Llama Stack server
2. Register a vector database backed by the remote::opengauss provider
3. Ingest sample documents into the vector store
4. Create an agent with the builtin::rag toolgroup configured for hybrid search
5. Run a multi-turn agentic session and print streamed responses

Prerequisites:
  - A running OpenGauss instance with the vector extension enabled
  - A running Llama Stack server configured with the remote::opengauss vector IO provider

How to run:
  export LLAMA_STACK_HOST=localhost
  export LLAMA_STACK_PORT=8321
  export MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
  export OPENGAUSS_HOST=localhost
  export OPENGAUSS_PORT=5432
  export OPENGAUSS_DB=postgres
  export OPENGAUSS_USER=gaussdb
  export OPENGAUSS_PASSWORD=your_password

  python agentic_rag_with_opengauss.py
"""

import os
import uuid

from llama_stack_client import LlamaStackClient
from llama_stack_client.types.agent_create_params import AgentConfig

# ---------------------------------------------------------------------------
# Configuration – override via environment variables
# ---------------------------------------------------------------------------

LLAMA_STACK_HOST = os.environ.get("LLAMA_STACK_HOST", "localhost")
LLAMA_STACK_PORT = int(os.environ.get("LLAMA_STACK_PORT", "8321"))
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")

OPENGAUSS_HOST = os.environ.get("OPENGAUSS_HOST", "localhost")
OPENGAUSS_PORT = os.environ.get("OPENGAUSS_PORT", "5432")
OPENGAUSS_DB = os.environ.get("OPENGAUSS_DB", "postgres")
OPENGAUSS_USER = os.environ.get("OPENGAUSS_USER", "gaussdb")
OPENGAUSS_PASSWORD = os.environ.get("OPENGAUSS_PASSWORD", "")

VECTOR_DB_ID = f"opengauss_demo_{uuid.uuid4().hex[:8]}"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ---------------------------------------------------------------------------
# Sample documents to ingest
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    {
        "document_id": "doc-1",
        "content": (
            "Python is a high-level, general-purpose programming language. "
            "Its design philosophy emphasizes code readability with the use of significant indentation. "
            "Python is dynamically typed and garbage-collected."
        ),
        "mime_type": "text/plain",
    },
    {
        "document_id": "doc-2",
        "content": (
            "OpenGauss is an open-source relational database management system. "
            "It is PostgreSQL-compatible and supports advanced features such as native vector storage "
            "via the pgvector extension, full-text search with tsvector, and JSONB."
        ),
        "mime_type": "text/plain",
    },
    {
        "document_id": "doc-3",
        "content": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval "
            "with large language models. A retriever fetches relevant documents from a knowledge base, "
            "and the LLM uses them as context to generate a grounded response."
        ),
        "mime_type": "text/plain",
    },
    {
        "document_id": "doc-4",
        "content": (
            "Hybrid search combines dense vector similarity search with sparse keyword-based search "
            "such as BM25 or PostgreSQL full-text search. Reranking strategies like Reciprocal Rank "
            "Fusion (RRF) or weighted scoring are used to merge results from both retrievers."
        ),
        "mime_type": "text/plain",
    },
]

# ---------------------------------------------------------------------------
# Questions to ask the agent
# ---------------------------------------------------------------------------

QUESTIONS = [
    "What is OpenGauss and what features does it support?",
    "How does hybrid search work and what reranking strategies are available?",
    "Explain Retrieval-Augmented Generation in simple terms.",
]


def main():
    # 1. Connect to the Llama Stack server
    base_url = f"http://{LLAMA_STACK_HOST}:{LLAMA_STACK_PORT}"
    print(f"Connecting to Llama Stack server at {base_url} ...")
    client = LlamaStackClient(base_url=base_url)

    # 2. Register a vector database backed by remote::opengauss
    print(f"\nRegistering vector database '{VECTOR_DB_ID}' with remote::opengauss ...")
    client.vector_dbs.register(
        vector_db_id=VECTOR_DB_ID,
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIMENSION,
        provider_id="remote::opengauss",
        provider_vector_db_id=VECTOR_DB_ID,
    )
    print(f"Vector database '{VECTOR_DB_ID}' registered.")

    # 3. Ingest sample documents
    print("\nIngesting sample documents ...")
    client.tool_runtime.rag_tool.insert(
        documents=SAMPLE_DOCUMENTS,
        vector_db_id=VECTOR_DB_ID,
        chunk_size_in_tokens=256,
    )
    print(f"Inserted {len(SAMPLE_DOCUMENTS)} documents into '{VECTOR_DB_ID}'.")

    # 4. Create an agent configured for hybrid RAG with RRF reranking
    print("\nCreating RAG agent ...")
    agent_config = AgentConfig(
        model=MODEL_ID,
        instructions="You are a helpful assistant. Use the knowledge base to answer questions accurately.",
        toolgroups=[
            {
                "name": "builtin::rag",
                "args": {
                    "vector_db_ids": [VECTOR_DB_ID],
                    "query_config": {
                        "mode": "hybrid",
                        "ranker": {"strategy": "rrf", "params": {"k": 60}},
                        "max_chunks": 5,
                    },
                },
            }
        ],
        enable_session_persistence=False,
    )
    agent = client.agents.create(agent_config=agent_config)
    session = client.agents.sessions.create(
        agent_id=agent.agent_id,
        session_name="opengauss_demo_session",
    )
    print(f"Agent '{agent.agent_id}' created, session '{session.session_id}' started.")

    # 5. Run a multi-turn agentic session
    print("\n" + "=" * 60)
    print("Starting multi-turn agentic RAG session")
    print("=" * 60)

    for question in QUESTIONS:
        print(f"\nUser: {question}")
        print("Assistant: ", end="", flush=True)

        response = client.agents.turns.create(
            agent_id=agent.agent_id,
            session_id=session.session_id,
            messages=[{"role": "user", "content": question}],
            stream=True,
        )

        for chunk in response:
            if hasattr(chunk, "event") and chunk.event:
                event = chunk.event
                if hasattr(event, "payload") and event.payload:
                    payload = event.payload
                    if hasattr(payload, "delta") and payload.delta:
                        delta = payload.delta
                        if hasattr(delta, "text") and delta.text:
                            print(delta.text, end="", flush=True)
        print()  # newline after streamed response

    # 6. Clean up
    print("\nCleaning up: unregistering vector database ...")
    client.vector_dbs.unregister(vector_db_id=VECTOR_DB_ID)
    print("Done.")


if __name__ == "__main__":
    main()
