"""Hybrid BM25 + embedding search over a corpus's papers."""

import re

import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from papers_mcp.corpus import Paper

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens for BM25."""
    return TOKEN_RE.findall(text.lower())


class SearchIndex:
    """Ranks papers by reciprocal-rank fusion of BM25 and embedding-cosine ranks."""

    def __init__(self, papers: list[Paper], model: SentenceTransformer) -> None:
        self.papers = papers
        self.model = model
        texts = [f"{p.title} {p.abstract}" for p in papers]
        self.bm25 = BM25Okapi([tokenize(f"{text} {p.authors}") for text, p in zip(texts, papers)])
        self.embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    def search(self, query: str, limit: int) -> list[Paper]:
        """Return the top *limit* papers for *query* by fused BM25 + cosine rank."""
        bm25_scores = torch.tensor(self.bm25.get_scores(tokenize(query)))
        query_embedding = self.model.encode(
            [query], convert_to_tensor=True, normalize_embeddings=True
        )
        cosine_scores = (self.embeddings @ query_embedding.T).flatten()

        fused = torch.zeros(len(self.papers))
        for scores in (bm25_scores, cosine_scores):
            ranks = scores.argsort(descending=True).argsort()
            fused += 1.0 / (RRF_K + 1 + ranks)

        top = fused.argsort(descending=True)[:limit]
        return [self.papers[i] for i in top.tolist()]
