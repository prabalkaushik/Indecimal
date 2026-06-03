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
# Relevance filter for real estate queries
RELEVANT_KEYWORDS = {
    "property",
    "properties",
    "real estate",
    "realestate",
    "house",
    "homes",
    "home",
    "apartment",
    "apartments",
    "condo",
    "condos",
    "construction",
    "materials",
    "material",
    "price",
    "prices",
    "mortgage",
    "lease",
    "sale",
    "sales",
    "marketplace",
    "building",
    "buildings",
    "architect",
    "architects",
    "project",
    "projects",
    "rent",
    "rental",
    "housing",
    "developer",
    "development",
    "Hi",
    "hello",
    "hey",
    "how are you",
    "who are you",
    "what can you do",
    "what is indecimal",
    "what is your name"
}

def is_relevant(question: str) -> bool:
    """Simple keyword‑based check to determine if the question is about real estate or related marketplace topics.
    It performs a case‑insensitive substring match against a broadened set of keywords.
    """
    q_lower = question.lower()
    return any(word in q_lower for word in RELEVANT_KEYWORDS)
SYSTEM_PROMPT = """
You are an AI assistant for Indecimal, a home construction and real estate services company.

Your primary task is to answer questions using the provided context.

Guidelines:

1. If the retrieved context contains information relevant to the user's question, provide a clear and concise answer based only on that information.

2. Minor spelling mistakes, abbreviations, wording variations, and informal language should be interpreted generously.

3. If the context partially answers the question, provide the available information and clearly indicate any missing details.

4. For simple greetings, introductions, or general conversational queries (e.g., "hi", "hello", "who are you", "what can you do"), respond naturally and explain that you can assist with home construction, pricing, materials, specifications, project planning, and related topics.

5. For basic calculations related to construction or property planning, perform the calculation when sufficient information is provided.

6. Do not invent facts, prices, specifications, policies, or project details that are not present in the context.

7. Only respond with the exact sentence below when the retrieved context does not contain information relevant to the user's request:

"I don't have enough information in the provided documents to answer this question."

CONTEXT:
{context}
"""

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


import math

class BM25SparseRetriever:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.N = len(chunks)
        self.stop_words = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
            "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
            "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
            "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
            "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
            "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
            "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
            "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
            "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
            "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
            "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
            "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
            "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
            "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
            "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
            "yourselves"
        }
        self.tokenize_regex = re.compile(r'\w+')
        
        # Preprocess documents
        self.doc_lens = []
        self.doc_tfs = []  # list of dicts: term -> count
        self.df = {}  # term -> document count
        
        for chunk in chunks:
            tokens = self._tokenize(chunk["text"])
            self.doc_lens.append(len(tokens))
            
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_tfs.append(tf)
            
            for token in tf.keys():
                self.df[token] = self.df.get(token, 0) + 1
                
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 1.0
        
        # Precompute IDF values
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            if self.idf[term] < 0:
                self.idf[term] = 0.0001
                
    def _tokenize(self, text):
        tokens = self.tokenize_regex.findall(text.lower())
        return [t for t in tokens if t not in self.stop_words]
        
    def get_scores(self, query):
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.N
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            
            for idx in range(self.N):
                tf_dict = self.doc_tfs[idx]
                tf_val = tf_dict.get(token, 0)
                doc_len = self.doc_lens[idx]
                
                denom = tf_val + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                if denom > 0:
                    scores[idx] += idf_val * (tf_val * (self.k1 + 1.0)) / denom
                    
        return scores


class RAGPipeline:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.embeddings = None
        self.chunks = []
        self.doc_names = []
        self.sparse_retriever = None
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
            
        self.sparse_retriever = BM25SparseRetriever(self.chunks)
        self._ready = True

    async def query(self, question, top_k=TOP_K, api_key=""):
        if not self._ready:
            self.initialize()

        num_chunks = len(self.chunks)
        if num_chunks == 0:
            return {
                "question": question,
                "answer": "⚠️ No documents found. Please add document files in the backend/documents folder.",
                "retrieved_chunks": [],
                "model": LLM_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "top_k": top_k,
            }

        # 1. Dense retrieval scores & ranking
        q_emb = self.model.encode([question], convert_to_numpy=True).astype("float32")
        dense_results = [] # list of (chunk_idx, score)
        
        if self.index is not None:
            # Search all chunks to rank them
            distances, indices = self.index.search(q_emb, num_chunks)
            for i, idx in enumerate(indices[0]):
                if idx == -1:
                    continue
                dense_results.append((int(idx), float(distances[0][i])))
            # Sort dense results: distance ascending (smaller is better)
            dense_results.sort(key=lambda x: x[1])
        else:
            emb = self.embeddings
            emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            q = q_emb[0]
            q_norm = q / (np.linalg.norm(q) + 1e-12)
            scores = emb_norm @ q_norm
            for idx, score in enumerate(scores):
                dense_results.append((idx, float(score)))
            # Sort dense results: similarity descending (larger is better)
            dense_results.sort(key=lambda x: x[1], reverse=True)

        dense_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_results)}
        dense_scores_dict = {idx: score for idx, score in dense_results}

        # 2. Sparse retrieval scores & ranking
        sparse_scores = self.sparse_retriever.get_scores(question)
        sparse_results = [(idx, score) for idx, score in enumerate(sparse_scores)]
        # Sort sparse results: BM25 score descending (larger is better)
        sparse_results.sort(key=lambda x: x[1], reverse=True)
        
        sparse_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(sparse_results)}
        sparse_scores_dict = {idx: score for idx, score in sparse_results}

        # 3. Reciprocal Rank Fusion (RRF) Re-ranking
        # RRF formula: Score(d) = sum_{m in models} 1 / (60 + rank_m(d))
        rrf_scores = {}
        for idx in range(num_chunks):
            # Dense rank term
            r_dense = dense_ranks.get(idx, num_chunks)
            dense_rrf = 1.0 / (60.0 + r_dense)
            
            # Sparse rank term (only count if lexical score is > 0 to filter irrelevant keywords)
            s_score = sparse_scores_dict.get(idx, 0.0)
            if s_score > 0.0:
                r_sparse = sparse_ranks.get(idx, num_chunks)
                sparse_rrf = 1.0 / (60.0 + r_sparse)
            else:
                sparse_rrf = 0.0
                
            rrf_scores[idx] = dense_rrf + sparse_rrf

        # Sort by RRF score descending
        re_ranked_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Take the top_k
        k = min(top_k, num_chunks)
        top_indices = re_ranked_indices[:k]
        
        retrieved = []
        for i, idx in enumerate(top_indices):
            c = self.chunks[idx].copy()
            c["rank"] = i + 1
            c["dense_rank"] = dense_ranks.get(idx)
            c["dense_score"] = dense_scores_dict.get(idx)
            c["sparse_rank"] = sparse_ranks.get(idx) if sparse_scores_dict.get(idx, 0.0) > 0.0 else None
            c["sparse_score"] = sparse_scores_dict.get(idx)
            c["rrf_score"] = rrf_scores[idx]
            retrieved.append(c)

        # If no chunks were retrieved, respond with fallback
        if not retrieved:
            return {
                "question": question,
                "answer": "I don't have enough information in the provided documents to answer this question.",
                "retrieved_chunks": [],
                "model": LLM_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "top_k": top_k,
            }
        answer = await generate_answer(question, retrieved, api_key=api_key)

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": [
                {
                    "rank": c["rank"],
                    "source": c["source"],
                    "text": c["text"],
                    "dense_rank": c["dense_rank"],
                    "dense_score": round(c["dense_score"], 4),
                    "sparse_rank": c["sparse_rank"],
                    "sparse_score": round(c["sparse_score"], 4),
                    "rrf_score": round(c["rrf_score"], 4)
                }
                for c in retrieved
            ],
            "model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": top_k,
        }

