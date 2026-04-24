# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from pathlib import Path
import os
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from transformers import AutoTokenizer

from langchain_core.documents import Document
import torch
from typing import List, Optional, Any
from langchain_qdrant import  QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import faiss
import txtai.scoring
from pydantic import Field, ConfigDict
from patent_retrieval import utils as utils, encoder as encoder
from openai import OpenAI

OPENAI_API_URL = os.getenv("OPENAI_API_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logger = utils.get_logger(__name__)

class MyHuggingFaceEmbeddings(HuggingFaceEmbeddings):

    tokenizer: Any = Field(default=None, exclude=True)
    max_length: int = Field(default=512)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self,max_length: int = 512, **kwargs):
        super().__init__( **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_length = max_length
    
    def truncate_text(self, text: str) -> str:
        tokens = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
      #  print(texts[0])
        truncated = [self.truncate_text(text) for text in texts]
      #  print(truncated[0])
      #  print([len(txt.split(" ")) for txt in truncated])
        return super().embed_documents(truncated)
    
    def embed_query(self, text: str) -> List[float]:
        truncated = self.truncate_text(text)
        return super().embed_query(truncated)




class DenseEncoder(encoder.Encoder):

    def __init__(self, backend: str = "openai",store_type: str = "faiss", model_name: str = None, index_dir: str = "index",
                 tokenizer: str = "Qwen/Qwen3-Embedding-4B", base_url: str = None, api_key: str = None):

        logger.info(f"Encoder backend: {backend}")
        self.max_tokens = 32760 if "llama" in model_name.lower() else 40960 #40960 -10 #32768-10  #8192 - 10  # leave some margin
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True,max_length=self.max_tokens )
       # self.max_tokens = self.tokenizer.model_max_length
        print(f"max tokens: {self.max_tokens}")
        self.chat_client = OpenAI(base_url="http://localhost:43572/v1", api_key=OPENAI_API_KEY)
        self.backend = backend.lower()
        if self.backend == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=model_name or "text-embedding-3-small",
                openai_api_base=base_url or OPENAI_API_URL,
                openai_api_key=api_key or OPENAI_API_KEY,
                #embedding_ctx_length=32000,
                tiktoken_enabled=False,
                check_embedding_ctx_length=False
            )

        elif backend == "huggingface":
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name or "Qwen/Qwen3-Embedding-0.6B",
               # max_length=12000,
                model_kwargs={
                    "device": "cuda",
                },
                encode_kwargs={
                    "normalize_embeddings":True,
                    "batch_size": 256                }
            )

        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        self.store_type = store_type
        self.index_dir = index_dir
        self.vector_store = None
        # Qdrant client (local storage using default file persistence)

        #self.vector_store: Optional[Qdrant] = None
        # Lazy initialized vector store
        #self.vector_store = None




    def _create_vectorstore(self):

        dim = len(self.embeddings.embed_query("test"))
        logger.info(f"Embedding dimension: {dim}")

        if self.store_type == "faiss":

            logger.info("Creating EMPTY FAISS index (cosine similarity)...")
            index = faiss.IndexFlatIP(dim)  
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore= InMemoryDocstore(),
                index_to_docstore_id={},
                normalize_L2=True,  
            )
            return

        if self.store_type == "qdrant":
            self.client = QdrantClient(path=str(self.index_dir))
            self.collection_name = "index"
            try:
            
                self.client.get_collection(self.collection_name)
                logger.info(f"Qdrant collection '{self.collection_name}' already exists. Using it.")

            except Exception:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' (cosine distance)...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )

            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )
            return
        
        raise ValueError("store_type must be 'faiss' or 'qdrant'")
    
    def _truncate(self, text: str) -> str:
        """
        Internal helper: Tokenize -> Truncate -> Decode
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_tokens:

            #logger.warning(f"Truncating document from {len(tokens)} to {self.max_tokens} tokens.")
            tokens = tokens[:self.max_tokens-10]
            # Decode back to string for the API
            text = self.tokenizer.decode(tokens)
            print(f"Truncated text to {len(tokens)} tokens.")
        return text
        
    def encode_docs(self, docs: List[Document], duplicates=False) -> List[List[float]]:

        if duplicates:
            docs = [d for d in docs if d.metadata["number"] not in self.indexed_ids]

        if self.vector_store is None:
            self._create_vectorstore()
        if self.backend == "openai":
            docs = [Document(page_content=self._truncate(d.page_content), metadata=d.metadata) for d in docs]
        else: 
            docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in docs]
        self.vector_store.add_documents(docs,ids=[d.metadata["number"] for d in docs])

        if self.store_type == "faiss":
            logger.info(f"Added {len(docs)} docs to FAISS. Total stored: {self.vector_store.index.ntotal}")
        else:
            count = self.client.count(self.collection_name, exact=True).count
            logger.info(
                f"Added {len(docs)} docs to Qdrant. Total stored: {count}"
            )

        
    def encode_query(self, text: str) -> List[float]:

        return self.embeddings.embed_query(text)
        
    def save_index(self, path: str = "index"):

        if self.vector_store is None:
            raise ValueError("No FAISS index to save.")
        if isinstance(self.vector_store, FAISS):
            self.vector_store.save_local(path)
            logger.info(f"Saved FAISS index to: {path}")

        elif isinstance(self.vector_store, QdrantVectorStore):
            logger.info(f"Qdrant index is already persisted")
    
    def get_indices(self) -> List[str]:
        return list(self.vector_store.index_to_docstore_id.values())

    def load_index(self, path: str = "faiss_idx",store_type: str="faiss" ):
        if store_type == "faiss":
            self.vector_store = FAISS.load_local(path, embeddings=self.embeddings,allow_dangerous_deserialization=True)
            logger.info(f"Loaded FAISS index from: {path}")
        elif store_type == "qdrant":
            client = QdrantClient(path=str(path))
            self.client.get_collection(self.collection_name)
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name="index",
                embedding=self.embeddings
            )
            logger.info(f"Loaded Qdrant index from: {path}")

        else:
            raise ValueError(f"Unsupported vestor store: {store_type}")
    def rewrite_query(self,query: str,language: str="ENGLISH") -> str:
        
        system_prompt = f"""
### ROLE
You are a Patent Intelligence Engine. Your task is to synthesize raw patent application into a "Dense Technical Query" optimized for vector embedding models (specifically Qwen-based architectures).

### OBJECTIVE
Create a single, high-entropy paragraph that captures the technical essence of the invention. Additionaly, provide a list of keywords in English, German and French that are relevant to the invention. 

### CONSTRAINTS
1. ELIMINATE ALL LEGALESE: Remove filing words such as "comprising," "according," and "arranged such that."
2. TECHNICAL FOCUS: Replace vague terms with specific engineering terminology.
3. NOVELTY WEIGHTING: Explicitly emphasize the "Point of Novelty" (the delta).
4. OUTPUT FORMAT: A single, continuous paragraph of 150-250 words. No bullet points. No labels.

### INSTRUCTIONS
Synthesize the input into a single Search Query Paragraph using this 4-part structure:
1. TECHNICAL FIELD: Define the specific domain (e.g., "Edge-computing low-latency data routing").
2. STRUCTURAL SKELETON: List the essential components and their physical/logical arrangement.
3. FUNCTIONAL MECHANISM: Explain the physics, logic, or algorithm that drives the invention.
4. CROSS-LINGUAL ANCHORING: Integrate keywords in English, German and French that are central to the invention's novelty and utility. 

[QUERY]
<Final Search Query Paragraph goes here, following the structure and constraints outlined above.>
Keywords: [List of keywords in English, German and French]

DO NOT output introductory text ("Here is the summary..."). Output ONLY the format above.

    """
        #4.  **Language**: The output must be in the **EXACT SAME LANGUAGE** as the input patent. Do not translate.
        user_prompt = f"<query>:\n{query}\n</query>\n\n"
        response = self.chat_client.chat.completions.create(
        model="Qwen/Qwen3.5-27B-FP8", 
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, 
            max_tokens=40000 # Increased to allow for reasoning text
        )
        
        full_output = response.choices[0].message.content
        
        # Simple parser to separate the thought process from the search query
        if "[QUERY]" in full_output:
            logger.info("Rewriting query for dense retrieval...")
            parts = full_output.split("[QUERY]")
            reasoning = parts[0].replace("[REASONING]", "").strip()
            final_query = parts[1].strip()
            
            # Optional: Print reasoning to console for debugging/education
            # print(f"DEBUG REASONING:\n{reasoning}\n")
            logger.info(f"Rewritten Query: {final_query}")
            
            return final_query
        else:
            #logger.info("Could not find [QUERY] section in LLM output.")
            # Fallback if format is missed
            return query
    
    def prepend_instruct(self, task_description: str="", query: str="") -> str:
        if not task_description:
                task_description = ""
        return f'<Instruct>: Given a patent, perform a prior art search and identify relevant existing patents regardless of the language. \n<Query>:{query}'

    def search(self,query: str,k:int= 10,fetch_k:int=200,filter_dict:dict=dict(),text=False,rewrite=False,language="EN") -> List[tuple]:
        #query = self._truncate(query)
        if rewrite:
            query = self.rewrite_query(query,language=language)
            query = self.prepend_instruct(query=query)
        else:
            query = self.prepend_instruct(query=query)
            query = self._truncate(query)
            
            
        if text:
            return self.vector_store.similarity_search_with_score(query,k=k,fetch_k=fetch_k,filter=filter_dict)
        else:
            match_docs = self.vector_store.similarity_search_with_score(query,k=k,fetch_k=fetch_k,filter=filter_dict)
            return [(doc.metadata["number"], score) for doc,score in match_docs]
    


if __name__ == "__main__":

    encoder = DenseEncoder(backend="openai",model_name="Qwen/Qwen3-Embedding-0.6B")
    texts = ["Hi", "bye"]

    try:
        embeddings = encoder.encode(texts)
        logger.info(f"texts were encoded successfully")
        logger.info(f"embeddings :{embeddings[0]}")
    
    except Exception as e:
        logger.error(f"Error: {e}")

    