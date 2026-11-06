import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from models import DocumentItem, QueryRequest, QueryResponse, ExtractedTask
from rag_engine import RAGEngine
from agents import MultiAgentOrchestrator

app = FastAPI(
    title="RAG Multi-Agent Chatbot & Document Manager API",
    description="Enterprise API for RAG document ingestion, hybrid semantic search, and autonomous multi-agent task execution.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine(chunk_size=400, chunk_overlap=40)
orchestrator = MultiAgentOrchestrator(rag_engine)

# In-memory global task store for tracking agent-created tasks
task_store: List[ExtractedTask] = []

# Preload starter enterprise documents
rag_engine.add_document(
    title="2026_Enterprise_AI_Strategy.md",
    content="""# Enterprise AI Strategy 2026
This document outlines our corporate AI roadmap, RAG architecture migration, and autonomous agent orchestration rollout.
Primary Objectives:
1. Transition all departmental knowledge silos into unified vector stores by Q1 2026.
2. Deploy multi-task agent supervisors to automate internal document workflows, incident summaries, and compliance filings.
3. Target 40% reduction in knowledge lookup latency across engineering and customer support teams.
Budget & Milestones:
- Total allocated CAPEX: $4.2M across compute infrastructure and embedding pipeline licensing.
- Phase 1 Pilot: HR & SecOps agent deployment by end of Q1.
- Phase 2 Enterprise-Wide: Cross-functional deployment with strict RBAC and fine-grained citation auditing by Q3.""",
    collection="Strategy",
    tags=["strategy", "executive", "roadmap", "2026"]
)

rag_engine.add_document(
    title="Q3_Financial_Review.pdf",
    content="""# Q3 Financial & Operational Performance
Revenue reached $18.6M, representing a 24% year-over-year growth led by Cloud Solutions and Enterprise Software subscriptions.
Operating Expenses:
- R&D investment totaled $5.1M, reflecting expanded hiring in AI engineering and LLM infrastructure.
- Infrastructure and cloud hosting expenses decreased by 8% due to hybrid caching and optimized model quantization.
Key Actions for Q4:
- Finance team must consolidate cloud vendor contracts before November 30 to secure 15% committed-use discounts.
- Implement automated billing reconciliation agent to audit vendor invoices against usage logs.""",
    collection="Finance",
    tags=["finance", "q3", "revenue", "budget"]
)

rag_engine.add_document(
    title="Cybersecurity_Policy_v4.2.txt",
    content="""# Enterprise Cybersecurity Policy v4.2
Data Classification and Vector Storage Standards:
All documents ingested into LLM or RAG systems must undergo strict PII scrubbing and data classification checks prior to embedding generation.
Access Controls:
1. Role-Based Access Control (RBAC) must be enforced at the chunk level. Confidential executive documents cannot be retrieved by unauthorized worker agents.
2. All agent actions that modify document state or export external records require cryptographic audit logging.
Incident Response Protocol:
In case of unauthorized context leakage or prompt injection attempt, the supervisory router agent must immediately isolate the affected session and dispatch an alert to SecOps.""",
    collection="Security",
    tags=["security", "compliance", "policy", "secops"]
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "RAG Multi-Agent Document Chatbot API",
        "indexed_documents": len(rag_engine.documents),
        "total_chunks": len(rag_engine.chunks_index)
    }

@app.get("/api/documents", response_model=List[DocumentItem])
def get_documents():
    return list(rag_engine.documents.values())

@app.post("/api/documents/upload", response_model=DocumentItem)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form("default"),
    tags: str = Form("")
):
    try:
        content_bytes = await file.read()
        text_content = content_bytes.decode("utf-8", errors="ignore")
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        doc = rag_engine.add_document(
            title=file.filename,
            content=text_content,
            collection=collection,
            tags=tag_list
        )
        return doc
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    success = rag_engine.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"}

@app.post("/api/chat", response_model=QueryResponse)
def chat_query(request: QueryRequest):
    response = orchestrator.execute_workflow(
        query=request.query,
        agent_mode=request.agent_mode,
        top_k=request.top_k,
        collection=request.collection
    )
    # Save any extracted tasks to task store
    if response.extracted_tasks:
        task_store.extend(response.extracted_tasks)
    return response

@app.get("/api/tasks", response_model=List[ExtractedTask])
def get_tasks():
    return task_store

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
