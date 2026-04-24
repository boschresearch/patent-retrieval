# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


CUSTOM_API_URL = "http://localhost:59749/v1"
# Initialize OpenAI embeddings pointing to custom URL
encoder = OpenAIEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  # your deployed model
    openai_api_base=CUSTOM_API_URL,
    openai_api_key="EMPTY"  # vLLM usually doesn't require a real key
)

#encoder = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B",show_progress=True)
# Embed multiple documents
texts = ["Hi", "bye"]
#doc_embeddings = encoder.embed_documents(docs)
#print("Document embeddings:", doc_embeddings)

# Embed a single query
query = "Hey"
query_embedding = encoder.embed_query(query)
print("Query embedding:", query_embedding)
