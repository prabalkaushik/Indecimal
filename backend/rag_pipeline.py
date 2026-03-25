import os, re, glob, numpy as np, httpx
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pypdf import PdfReader

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

load_dotenv()

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 5
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are a helpful AI assistant for Indecimal, a home construction company.
Answer ONLY using the provided context.
If the answer cannot be found in the provided context, reply exactly:
"I don't have enough information in the provided documents to answer this question."
Do NOT hallucinate. Do NOT make assumptions.
Prices are indicative and may vary.

CONTEXT:
{context}"""


def load_documents():
    docs = []
    for ext in ("*.txt", "*.md", "*.pdf"):
        for fp in sorted(glob.glob(os.path.join(DOCUMENTS_DIR, ext))):
            name = os.path.basename(fp)
            if fp.lower().endswith(".pdf"):
                docs.append({"name": name, "content": extract_pdf_text(fp)})
            else:
                with open(fp, "r", encoding="utf-8") as f:
                    docs.append({"name": name, "content": f.read()})
    return [d for d in docs if d["content"].strip()]


def extract_pdf_text(fp: str) -> str:
    reader = PdfReader(fp)
    pages = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return "\n".join([p for p in pages if p])


def chunk_document(doc):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc["content"]) if s.strip()]
    chunks, current, word_count = [], [], 0

    for sentence in sentences:
        words = len(sentence.split())
        if word_count + words > CHUNK_SIZE and current:
            chunks.append({"text": " ".join(current), "source": doc["name"], "chunk_id": len(chunks)})
            overlap, ow = [], 0
            for s in reversed(current):
                sw = len(s.split())
                if ow + sw > CHUNK_OVERLAP:
                    break
                overlap.insert(0, s)
                ow += sw
            current, word_count = overlap, ow
        current.append(sentence)
        word_count += words

    if current:
        chunks.append({"text": " ".join(current), "source": doc["name"], "chunk_id": len(chunks)})
    return chunks


async def generate_answer(question, chunks, api_key=""):
    # Priority: 1. Passed api_key (from frontend), 2. Environment variable
    effective_api_key = api_key or os.getenv("GROQ_API_KEY")
    
    if not effective_api_key:
        return "⚠️ Please set your Groq API key in the sidebar or backend/.env to enable answers."

    context = "\n\n---\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(GROQ_API_URL, json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.replace("{context}", context)},
                    {"role": "user", "content": question},
                ],
                "temperature": 0.2, "max_tokens": 1024,
            }, headers={
                "Authorization": f"Bearer {effective_api_key}",
                "Content-Type": "application/json",
            })
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return "⚠️ Invalid Groq API key. Please check your key in the sidebar."
        return f"⚠️ Groq API error (HTTP {e.response.status_code}): {e.response.text}"
    except Exception as e:
        return f"⚠️ Error: {e}"


class RAGPipeline:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.embeddings = None
        self.chunks = []
        self.doc_names = []
        self._ready = False

    def initialize(self):
        if self._ready:
            return
        docs = load_documents()
        self.doc_names = [d["name"] for d in docs]
        for doc in docs:
            self.chunks.extend(chunk_document(doc))

        texts = [c["text"] for c in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")

        self.embeddings = embeddings
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
        else:
            self.index = None
        self._ready = True

    async def query(self, question, top_k=TOP_K, api_key=""):
        if not self._ready:
            self.initialize()

        q_emb = self.model.encode([question], convert_to_numpy=True).astype("float32")
        k = min(top_k, len(self.chunks))
        retrieved = []

        if self.index is not None:
            distances, indices = self.index.search(q_emb, k)
            for i, idx in enumerate(indices[0]):
                if idx == -1:
                    continue
                c = self.chunks[idx].copy()
                c["score"] = float(distances[0][i])
                c["rank"] = i + 1
                retrieved.append(c)
        else:
            emb = self.embeddings
            emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            q = q_emb[0]
            q_norm = q / (np.linalg.norm(q) + 1e-12)
            scores = emb_norm @ q_norm
            top_idx = np.argsort(scores)[::-1][:k]
            for i, idx in enumerate(top_idx):
                c = self.chunks[int(idx)].copy()
                c["score"] = float(scores[idx])
                c["rank"] = i + 1
                retrieved.append(c)

        answer = await generate_answer(question, retrieved, api_key=api_key)

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": [
                {"rank": c["rank"], "source": c["source"], "text": c["text"], "score": round(c["score"], 4)}
                for c in retrieved
            ],
            "model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": top_k,
        }
