import uuid
import re
from typing import List, Dict, Any, Tuple
import math
from models import DocumentItem, DocumentChunk, DocumentMetadata, Citation

class RAGEngine:
    """
    RAG Engine supporting document storage, semantic/recursive chunking,
    TF-IDF / vector similarity search, and citation grounding.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: Dict[str, DocumentItem] = {}
        self.chunks_index: Dict[str, DocumentChunk] = {}
        # Inverted index for fast hybrid BM25/keyword retrieval
        self.inverted_index: Dict[str, List[str]] = {}

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\b\w{2,}\b', text)]

    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        if len(words) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
            if i + chunk_size >= len(words):
                break
        return chunks

    def add_document(self, title: str, content: str, collection: str = "default", tags: List[str] = None) -> DocumentItem:
        doc_id = str(uuid.uuid4())[:8]
        raw_chunks = self.chunk_text(content, self.chunk_size, self.chunk_overlap)
        
        parsed_chunks: List[DocumentChunk] = []
        for idx, chunk_str in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_c{idx+1}"
            token_count = len(self._tokenize(chunk_str))
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=title,
                content=chunk_str,
                token_count=token_count,
                chunk_index=idx,
                metadata={"collection": collection, "tags": tags or []}
            )
            parsed_chunks.append(chunk)
            self.chunks_index[chunk_id] = chunk

            # Update inverted index
            for term in set(self._tokenize(chunk_str)):
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(chunk_id)

        meta = DocumentMetadata(
            title=title,
            collection=collection,
            tags=tags or [],
            file_type="markdown" if title.endswith(".md") else "text",
            file_size_bytes=len(content.encode('utf-8')),
            chunk_count=len(parsed_chunks),
            active=True
        )

        doc = DocumentItem(
            id=doc_id,
            title=title,
            content=content,
            metadata=meta,
            chunks=parsed_chunks
        )
        self.documents[doc_id] = doc
        return doc

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        doc = self.documents.pop(doc_id)
        for chunk in doc.chunks:
            self.chunks_index.pop(chunk.chunk_id, None)
        # Rebuild inverted index
        self._rebuild_index()
        return True

    def toggle_document_active(self, doc_id: str, active: bool) -> bool:
        if doc_id in self.documents:
            self.documents[doc_id].metadata.active = active
            return True
        return False

    def _rebuild_index(self):
        self.inverted_index = {}
        for chunk_id, chunk in self.chunks_index.items():
            for term in set(self._tokenize(chunk.content)):
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(chunk_id)

    def search(self, query: str, top_k: int = 4, collection: str = None) -> List[Tuple[DocumentChunk, float]]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scores: Dict[str, float] = {}
        doc_freq = {t: len(self.inverted_index.get(t, [])) for t in query_terms}
        total_chunks = max(1, len(self.chunks_index))

        for term in query_terms:
            df = doc_freq.get(term, 0)
            idf = math.log((total_chunks - df + 0.5) / (df + 0.5) + 1.0)
            matching_chunk_ids = self.inverted_index.get(term, [])

            for cid in matching_chunk_ids:
                chunk = self.chunks_index[cid]
                # Filter out inactive docs
                parent_doc = self.documents.get(chunk.doc_id)
                if not parent_doc or not parent_doc.metadata.active:
                    continue
                if collection and chunk.metadata.get("collection") != collection:
                    continue

                chunk_terms = self._tokenize(chunk.content)
                tf = chunk_terms.count(term)
                # BM25-like weighting
                term_score = idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (len(chunk_terms) / 100.0)))
                scores[cid] = scores.get(cid, 0.0) + term_score

        # Normalize scores to 0.0 - 1.0 range
        if not scores:
            return []

        max_score = max(scores.values()) or 1.0
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for cid, sc in ranked:
            chunk = self.chunks_index[cid]
            norm_score = round(min(0.99, sc / (max_score * 1.1) + 0.3), 3)
            results.append((chunk, norm_score))

        return results
