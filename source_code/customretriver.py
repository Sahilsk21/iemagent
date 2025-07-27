
from langchain.schema import BaseRetriever, Document
from typing import List
from pydantic import BaseModel, Extra
from rank_bm25 import BM25Okapi

from langchain_community.vectorstores import Chroma

import numpy as np

import json

from typing import List, Dict, Any

from rank_bm25 import BM25Okapi


import re
with open("./static/bm25_corpus2.json", "r", encoding="utf-8") as f:
        bm25_corpus =json.load(f)

class HybridRetriever(BaseRetriever, BaseModel):
    k: int = 5

    def __init__(self, db, embeddings,  k=5, **kwargs):
        """
        Initialize with ChromaDB instance, embeddings, and BM25 corpus.
        :param db: Chroma collection (vector store)
        :param embeddings: LangChain-compatible embedding model
        :param bm25_corpus: List of text chunks used for BM25 search
        """
        super().__init__(**kwargs)
        self.db = db  # This is your Chroma vector DB instance
        self.embeddings = embeddings
        self.k = k

        # Initialize BM25
        self.bm25 = BM25Okapi([doc.split() for doc in bm25_corpus])
        self.bm25_docs = bm25_corpus

    def get_relevant_documents(self, query: str) -> List[Document]:
        # === Vector Search with Chroma ===
        vector_results = self.db.similarity_search(query, k=self.k)
        vector_docs = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vector_results
        ]

        # === BM25 Search ===
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:self.k]
        bm25_docs = [
            Document(page_content=self.bm25_docs[i], metadata={"source": f"BM25-{i}"})
            for i in bm25_top_indices
        ]

        # === Merge and return ===
        all_docs = vector_docs + bm25_docs
        return all_docs[:self.k]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented.")

    class Config:
        extra = Extra.allow

