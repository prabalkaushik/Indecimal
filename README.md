# Indecimal AI — Home Construction RAG Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot for Indecimal that answers user questions by retrieving relevant information from internal company documents and generating grounded responses using an LLM.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)
]![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🏗️ Architecture

```
User Question
    │
    ▼
┌──────────────────────┐
│   FastAPI Backend     │
│                       │
│  1. Embed query       │   ← sentence-transformers (all-MiniLM-L6-v2)
│  2. Vector search     │   ← FAISS (Flat L2 index)
│  3. Retrieve top-k    │
│  4. Generate answer   │   ← OpenAI LLM (gpt-4o-mini)
└──────────┬───────────┘
           │
           ▼
   Answer + Retrieved Context
    │
    ▼
┌──────────────────────┐
│  Custom Frontend UI   │   ← HTML/CSS/JS chatbot interface
│  (Glassmorphism Dark) │
└──────────────────────┘
```

---

## 📦 Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **Embedding Model** | `all-MiniLM-L6-v2` (sentence-transformers) | Lightweight (~80MB), runs locally with no API key, produces high-quality 384-dim embeddings optimized for semantic similarity tasks |
| **Vector Search** | FAISS (`IndexFlatL2`) | Industry-standard library by Meta, fast exact nearest-neighbor search, easy to set up with no external services |
| **LLM** | `gpt-4o-mini` via OpenAI API | Lightweight chat model with strong instruction-following |
| **Backend** | FastAPI + Uvicorn | Async support, auto-generated docs, fast Python web framework |
| **Frontend** | Vanilla HTML/CSS/JS | No build step needed, premium glassmorphism dark UI |

---

## 📄 Internal Documents

The RAG pipeline is powered by 3 Indecimal internal documents:

| Document | Contents |
|----------|----------|
| `company_overview_customer_journey.md` | Company overview, operating principles, customer journey (10 stages), and FAQs |
| `package_comparison_specifications.md` | 4 packages (Essential/Premier/Infinia/Pinnacle) with pricing, material specs, kitchen/bathroom wallets, doors, windows, painting, flooring |
| `customer_protection_quality_guarantees.md` | Escrow payments, delay management, 445+ quality checkpoints, zero-cost maintenance, financing, partner onboarding |

---

## 📄 Document Chunking & Retrieval

### Chunking Strategy
- **Sentence-boundary-aware splitting**: Text is split into sentences using regex, then sentences are grouped into chunks of ~300 words with 50-word overlap
- **Why overlapping chunks?** Ensures that information spanning two chunks isn't lost at chunk boundaries
- **Metadata preserved**: Each chunk retains its source document name and chunk index

### Retrieval
- All chunks are embedded into 384-dimensional vectors using `all-MiniLM-L6-v2`
- Vectors are indexed using FAISS `IndexFlatL2` (exact L2 distance search)
- For each query, the user's question is embedded and the **top-5 nearest chunks** are retrieved
- Retrieved chunks are passed as context to the LLM

---

## 🛡️ Grounding & Hallucination Prevention

The LLM is explicitly instructed via a strict system prompt to:
1. **Answer ONLY from the provided context** — no external knowledge
2. **Explicitly state** when the answer cannot be found in the documents
3. **Cite the source document** when possible
4. **Note that prices are indicative** when quoting wallet amounts or package pricing

System prompt excerpt:
```
You are a helpful AI assistant for Indecimal, a home construction company.
You MUST answer questions ONLY using the provided context below.
If the answer cannot be found in the provided context, say:
"I don't have enough information in the provided documents to answer this question."
Do NOT use any external knowledge. Do NOT make assumptions or hallucinate information.
When quoting prices or wallet amounts, always mention that they are indicative and may vary.
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.10+
- An OpenAI API key (create one at https://platform.openai.com/)

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd indecimal

# 2. Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
cd backend
pip install -r requirements.txt

# 4. Configure API key
copy .env.example .env
# Edit .env and set your OPENAI_API_KEY

# 5. Run the server
python app.py
```

### Access
- **Chatbot UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

---

## 📁 Project Structure

```
indecimal/
├── backend/
│   ├── app.py                  # FastAPI server
│   ├── rag_pipeline.py         # Core RAG logic
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # API key template
│   └── documents/              # Indecimal internal docs
│       ├── company_overview_customer_journey.md
│       ├── package_comparison_specifications.md
│       └── customer_protection_quality_guarantees.md
├── frontend/
│   ├── index.html              # Chatbot UI
│   ├── style.css               # Premium dark theme
│   └── script.js               # Frontend logic
└── README.md
```

---

## 🔬 Quality Analysis

### Test Questions & Observations

| # | Question | Relevant Chunks Retrieved? | Grounded Answer? | Notes |
|---|----------|---------------------------|-------------------|-------|
| 1 | What are Indecimal's construction packages and pricing? | ✅ Yes | ✅ Yes | Lists all 4 packages: Essential (₹1,851), Premier (₹1,995), Infinia (₹2,250), Pinnacle (₹2,450) |
| 2 | How does Indecimal ensure quality assurance? | ✅ Yes | ✅ Yes | 445+ checkpoints, structural integrity, safety compliance, execution accuracy |
| 3 | What is the customer journey at Indecimal? | ✅ Yes | ✅ Yes | Covers all 10 stages from request to maintenance |
| 4 | What steel brands are used in the Premier package? | ✅ Yes | ✅ Yes | JSW or Jindal Neo up to ₹74,000/MT |
| 5 | How does the escrow payment system work? | ✅ Yes | ✅ Yes | Payments to escrow → PM verification → disbursement to partner |
| 6 | What is the zero cost maintenance program? | ✅ Yes | ✅ Yes | Covers plumbing, electrical, wardrobe, masonry, painting, etc. |
| 7 | What cement is used in the Pinnacle package? | ✅ Yes | ✅ Yes | ACC, Ultratech, Ramco or equivalent up to ₹400/bag |
| 8 | How does Indecimal handle construction delays? | ✅ Yes | ✅ Yes | Integrated PM system, daily tracking, instant flagging, penalisation |
| 9 | What are the flooring options for the Infinia package? | ✅ Yes | ✅ Yes | Living: tiles/granite/marble up to ₹140/sqft |
| 10 | What financing support does Indecimal provide? | ✅ Yes | ✅ Yes | Dedicated RM, minimal docs, ~7 days confirmation, ~30 days disbursal |
| 11 | What are the bathroom specifications for the Essential package? | ✅ Yes | ✅ Yes | Dado ₹40/sqft, sanitary ₹32,000/1000sqft (Cera/Hindware) |
| 12 | How does Indecimal onboard construction partners? | ✅ Yes | ✅ Yes | 4-stage process: verification, background check, agreement, onboarding |
| 13 | What painting brands are used in each package? | ✅ Yes | ✅ Yes | Ranges from Tractor Emulsion (Essential) to Royale Emulsion (Pinnacle) |
| 14 | Tell me about cryptocurrency trading | ✅ N/A | ✅ Refused | Correctly states info not in documents |

### Key Observations
- **Retrieval quality**: FAISS with sentence-transformers produces highly relevant chunk retrieval for Indecimal-specific questions
- **Hallucination resistance**: The strict system prompt effectively prevents the LLM from going beyond the provided context
- **Out-of-scope handling**: Questions unrelated to Indecimal are correctly identified and refused
- **Price transparency**: The system consistently notes that prices are indicative when quoting wallet amounts

---

## 📝 API Reference

### `POST /api/query`
```json
{
    "question": "What are Indecimal's construction packages?",
    "top_k": 5
}
```

Response:
```json
{
    "question": "What are Indecimal's construction packages?",
    "answer": "According to the package comparison document...",
    "retrieved_chunks": [
        {
            "rank": 1,
            "source": "package_comparison_specifications.md",
            "text": "...",
            "score": 0.4231
        }
    ],
    "model": "gpt-4o-mini",
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k": 5
}
```

### `GET /api/health`
Returns system status, document count, and index size.

---

## 📜 License

MIT License
