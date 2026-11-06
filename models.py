from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentMetadata(BaseModel):
    title: str
    collection: str = "default"
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    file_type: str = "text"
    file_size_bytes: int = 0
    chunk_count: int = 0
    active: bool = True

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    content: str
    token_count: int
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity_score: Optional[float] = None

class DocumentItem(BaseModel):
    id: str
    title: str
    content: str
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = Field(default_factory=list)

class QueryRequest(BaseModel):
    query: str
    collection: Optional[str] = None
    agent_mode: Literal["auto", "rag_only", "task_extractor", "synthesizer", "audit"] = "auto"
    top_k: int = 4
    include_tasks: bool = True
    temperature: float = 0.2

class AgentStepTrace(BaseModel):
    agent_name: str
    role: str
    action_type: str  # "planning", "tool_call", "synthesis", "verification"
    thought: str
    tool_inputs: Optional[Dict[str, Any]] = None
    tool_outputs: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: int = 0

class ExtractedTask(BaseModel):
    task_id: str
    title: str
    description: str
    assignee: Optional[str] = None
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    status: Literal["todo", "in_progress", "done"] = "todo"
    due_date: Optional[str] = None
    source_doc: Optional[str] = None
    source_chunk_id: Optional[str] = None

class Citation(BaseModel):
    doc_id: str
    doc_title: str
    chunk_id: str
    snippet: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    agent_traces: List[AgentStepTrace]
    citations: List[Citation]
    extracted_tasks: List[ExtractedTask]
    referenced_chunks: List[DocumentChunk]
    latency_seconds: float
