# NCERT PCMB → Structured Dataset Pipeline

A fully automated, modular AI pipeline that converts any NCERT chapter (Physics, Chemistry, Mathematics, Biology) into a structured knowledge graph, question bank, and LLM fine-tuning dataset — **using only free tools and APIs**.

---

## 🏗️ Architecture

```
NCERT PDF / Web
      │
      ▼
┌─────────────┐
│  loader.py  │  PyPDFLoader + RecursiveCharacterTextSplitter
└──────┬──────┘
       │ chunks (500 chars, 50 overlap)
       ▼
┌─────────────┐
│   rag.py    │  HuggingFace Embeddings + FAISS vector store
└──────┬──────┘
       │ retriever (k=3 similarity search)
       ▼
┌───────────────┐
│ generator.py  │  Groq LLM (Llama 3 70B) — FREE API
│               │  ├─ Concept extraction
│               │  ├─ Theory generation
│               │  ├─ SCQ generation
│               │  └─ Subjective Q generation
└──────┬────────┘
       │
       ▼
┌─────────────┐
│   main.py   │  Orchestrator + output writer
└──────┬──────┘
       │
  ┌────┴─────┐
  │          │
  ▼          ▼
structured_  hierarchy.
output.json  jsonl
```

---

## 🆓 Free Tools Used

| Component | Tool | Cost |
|-----------|------|------|
| LLM | Groq API (Llama 3 70B) | **FREE** (30 req/min) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | **FREE** (local) |
| Vector Store | FAISS (CPU) | **FREE** (local) |
| PDF Loading | LangChain PyPDFLoader | **FREE** |
| Web Loading | LangChain WebBaseLoader | **FREE** |

> **No OpenAI key needed.** No paid API required.

---

## 📦 Installation

```bash
# 1. Clone / download this folder
cd ncert_pipeline

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your free Groq API key
# Sign up at https://console.groq.com (completely free)
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

---

## 🚀 Usage

### Basic run (Physics Chapter 1 — Electric Charges and Fields)
```bash
python main.py \
  --source https://ncert.nic.in/textbook/pdf/leph101.pdf \
  --subject physics \
  --chapter 1 \
  --output-dir output
```

### Chemistry example
```bash
python main.py \
  --source https://ncert.nic.in/textbook/pdf/lech101.pdf \
  --subject chemistry \
  --chapter 1
```

### Biology example
```bash
python main.py \
  --source https://ncert.nic.in/textbook/pdf/lebo101.pdf \
  --subject biology \
  --chapter 1
```

### From local PDF
```bash
python main.py \
  --source /path/to/chapter.pdf \
  --subject mathematics \
  --chapter 3
```

### All CLI options
```
--source        URL or path to NCERT chapter PDF (default: leph101.pdf)
--subject       physics | chemistry | mathematics | biology
--chapter       Chapter number (integer, for labelling)
--output-dir    Output directory (default: output/)
--no-persist-faiss   Skip saving FAISS index to disk
```

---

## 📁 Output Files

### `structured_output.json`
Complete knowledge graph per chapter:
```json
{
  "Concept: Electric Charges and Fields": {
    "Subconcept: Coulomb's Law": {
      "Summary": {
        "Theory": "Coulomb's law states...",
        "VisualContent": {
          "Needed": true,
          "Content": {
            "THEORY_1": "Diagram showing two point charges..."
          }
        }
      },
      "SCQs": {
        "Question 1": {
          "isBackExercise": false,
          "Difficulty": "Easy",
          "Question": "...",
          "Options": {"A": "...", "B": "...", "C": "...", "D": "..."},
          "Correct Option": "A",
          "Solution": "...",
          "VisualContent": {"Needed": false}
        }
      },
      "SubjectiveQuestions": {
        "Question 1": {
          "Difficulty": "Hard",
          "Question": "...",
          "Answer": "..."
        }
      }
    }
  }
}
```

### `hierarchy.jsonl`
One JSON line per chapter — concept-subconcept hierarchy:
```jsonl
{"Chapter 1: Electric Charges and Fields": {"Electric Charges and Fields": ["Electric Charge", "Coulomb's Law", ...], "Electric Field": [...]}}
```

---

## 🧠 Subject-Specific Intelligence

| Subject | Focus |
|---------|-------|
| **Physics** | Derivations, formulas, numericals, graphs, circuits |
| **Chemistry** | Reactions, mechanisms, structural diagrams, equations |
| **Mathematics** | Theorems, proofs, step-by-step problem solving, geometry |
| **Biology** | Diagrams, processes, definitions, life cycles |

The LLM automatically adjusts theory depth, question style, and visual detection based on the subject.

---

## 🔁 Pipeline Flow Detail

```
1. load_and_split()          → chunks the PDF into 500-char pieces
2. build_rag_pipeline()      → HuggingFace embeddings + FAISS index
3. extract_concepts()        → Groq LLM identifies concepts + subconcepts
4. For each subconcept:
   a. retrieve_context()     → k=3 RAG similarity search
   b. generate_theory()      → Groq LLM with subject-aware prompt
   c. generate_scqs()        → 3 MCQs (Easy/Medium/Hard)
   d. generate_subjective()  → 3 descriptive questions
   e. Visual detection       → embedded in theory + SCQ prompts
5. Save structured_output.json
6. Append to hierarchy.jsonl
```

---

## ⚙️ Module Reference

| File | Purpose |
|------|---------|
| `loader.py` | Load PDF/web, split into chunks |
| `rag.py` | Embeddings, FAISS, retriever |
| `generator.py` | LLM prompts, JSON parsing, generation |
| `main.py` | Orchestrator + CLI |

---

## 🐛 Troubleshooting

**`GROQ_API_KEY not set`**
→ Get a free key at https://console.groq.com and add it to `.env`

**PDF download fails**
→ Try downloading the PDF manually and use `--source /path/to/file.pdf`

**JSON parse error from LLM**
→ The pipeline retries 3 times automatically. Groq models occasionally produce malformed JSON under load; the retry logic handles this.

**Rate limit error (429)**
→ Groq free tier: 30 req/min. The pipeline has built-in 1.5s delays. For large chapters, it should stay within limits.

**FAISS index slow on first run**
→ First run downloads the embedding model (~90 MB). Subsequent runs use the cached model.

---

## 📈 Scaling for Production

- **Multiple chapters**: Loop `run_pipeline()` calls in `main.py`
- **Parallel processing**: Use `asyncio` with Groq's async client
- **Persistent FAISS**: Re-use `faiss_index/` across sessions
- **Fine-tuning dataset**: The JSONL file is directly usable for LLM fine-tuning (instruction-following format)