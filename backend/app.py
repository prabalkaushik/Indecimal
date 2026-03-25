import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import RAGPipeline

rag = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.initialize()
    yield


app = FastAPI(title="Construction RAG Assistant", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    api_key: str = ""


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "documents": len(rag.doc_names),
        "chunks": len(rag.chunks),
        "document_names": rag.doc_names,
    }


@app.post("/api/query")
async def query(req: QueryRequest):
    return await rag.query(req.question, top_k=req.top_k, api_key=req.api_key)


FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
