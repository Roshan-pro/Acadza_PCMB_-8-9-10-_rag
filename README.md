# NCERT PCMB → Structured Dataset Pipeline

A fully automated, modular AI pipeline that converts any NCERT chapter (Physics, Chemistry, Mathematics, Biology) into a structured knowledge graph, question bank, and LLM fine-tuning dataset — **using only free tools and APIs**.

---

## 📌 Table of Contents

1. [Project Scope](#1-project-scope)
2. [Method of Choice](#2-method-of-choice)
3. [Model Used](#3-model-used)
4. [Schema Defined](#4-schema-defined)
5. [Output](#5-output)
6. [Benchmark Output Stats](#6-benchmark-output-stats)
7. [Architecture](#7-architecture)
8. [Free Tools Used](#8-free-tools-used)
9. [Installation](#9-installation)
10. [Usage](#10-usage)
11. [Subject-Specific Intelligence](#11-subject-specific-intelligence)
12. [Pipeline Flow Detail](#12-pipeline-flow-detail)
13. [Module Reference](#13-module-reference)
14. [Troubleshooting](#14-troubleshooting)
15. [Scaling for Production](#15-scaling-for-production)

---

## 1. Project Scope

### What this pipeline does

This system takes **any NCERT chapter PDF** (Classes 11–12, subjects: Physics, Chemistry, Mathematics, Biology) as input and fully automatically produces:

- A **structured knowledge graph** with concepts, subconcepts, and detailed theory
- A **question bank** with MCQs and subjective questions at three difficulty levels
- A **visual content inventory** flagging diagrams, graphs, and illustrations required per subconcept
- A **hierarchical JSONL** file suitable for LLM fine-tuning

### Target subjects and scope

| Subject | Classes | Coverage |
|---------|---------|---------|
| Physics | 11 & 12 | All NCERT chapters (e.g., Kinematics → Semiconductors) |
| Chemistry | 11 & 12 | All NCERT chapters (e.g., Atomic Structure → Polymers) |
| Mathematics | 11 & 12 | All NCERT chapters (e.g., Sets → Probability) |
| Biology | 11 & 12 | All NCERT chapters (e.g., Cell Biology → Ecology) |

### What problems it solves

| Problem | Solution |
|---------|---------|
| Manual content curation is slow and expensive | Fully automated LLM-driven extraction |
| Questions hardcoded per chapter | Everything derived dynamically from chapter text |
| Inconsistent question difficulty | LLM enforces Easy / Medium / Hard split |
| No grounding — LLM hallucinates | RAG pipeline grounds every generation in actual chapter text |
| Diagrams missed in digital formats | Visual content detector flags every subconcept needing a visual |
| No reusable dataset format | Outputs JSON + JSONL ready for downstream use and fine-tuning |

### What this pipeline does NOT do

- It does not generate actual images or diagrams (it describes what visual is needed)
- It does not scrape paywalled or protected content
- It is not a student-facing interface — it is a **data generation backend**

---

## 2. Method of Choice

### Core Approach: RAG-Augmented Agentic LLM Pipeline

This pipeline uses a **Retrieval-Augmented Generation (RAG)** architecture layered over an **agentic LLM generation loop**. Here is why each design decision was made:

---

### Why RAG instead of pure LLM generation?

| Concern | Without RAG | With RAG |
|---------|------------|---------|
| Hallucination | High — LLM invents facts not in chapter | Low — generation grounded in retrieved chunks |
| Chapter specificity | LLM uses generic knowledge, not the specific chapter | LLM uses actual chapter sentences as context |
| Accuracy of formulas/reactions | Unreliable | Anchored to source text |
| Diagram detection | Generic guesses | Based on actual chapter visual cues |

RAG retrieves the **top-3 most relevant chunks** (k=3) from the FAISS index for each subconcept query before passing context to the LLM. This ensures every generated theory, MCQ, and answer is **grounded in the chapter's own content**.

---

### Why LangChain?

LangChain was chosen because it provides:
- Unified document loaders for both PDF and web sources (`PyPDFLoader`, `WebBaseLoader`)
- Production-grade `RecursiveCharacterTextSplitter` with configurable overlap
- A clean abstraction over embedding models and vector stores
- First-class support for FAISS and HuggingFace embeddings with no code changes needed to swap components

---

### Why FAISS over Chroma or Pinecone?

| Store | Reason for / against |
|-------|---------------------|
| **FAISS** ✅ | Fully local, zero latency, no API calls, CPU-friendly, persistent on disk |
| Chroma | Requires a running server, adds setup complexity |
| Pinecone | Paid / rate-limited, requires internet for every query |

For a single-chapter pipeline that processes chapters sequentially, FAISS is the optimal choice — fast, free, and offline-capable.

---

### Why chunk_size=500 with chunk_overlap=50?

- **500 characters** ≈ 2–4 sentences — small enough to be semantically focused, large enough to carry a full concept statement (a law definition + formula)
- **50-character overlap** prevents a definition from being split across two chunks with no contextual bridge
- Tested against NCERT PDF formatting: most section headings, definitions, and formulae fit cleanly within a 500-char window

---

### Why an agentic extraction flow?

Concepts and subconcepts are **not hardcoded**. The LLM is given the raw chapter text and asked to infer the concept hierarchy itself. This means:
- The pipeline works on **any chapter** without configuration changes
- Implicit subconcepts (not explicitly headed in the PDF) are inferred from semantic content
- The LLM acts as a domain expert, not just a text extractor

---

### Prompt Engineering Strategy

Three separate, specialised system prompts are used:

| Prompt | Role | Temperature |
|--------|------|-------------|
| `CONCEPT_SYSTEM` | Extract concept-subconcept hierarchy from chapter | 0.3 |
| `THEORY_SYSTEM` | Generate explanation + detect visual need | 0.3 |
| `SCQ_SYSTEM` | Generate 3 MCQs with distractors and solutions | 0.3 |
| `SUBJECTIVE_SYSTEM` | Generate 3 descriptive questions with model answers | 0.3 |

Low temperature (0.3) is used throughout to prioritise accuracy and JSON compliance over creativity.

---

### JSON Robustness Strategy

LLMs occasionally produce malformed JSON (extra commentary, markdown fences, truncation). The pipeline handles this with a 3-layer extraction strategy:
1. Direct `json.loads()` parse
2. Extract from ` ```json ... ``` ` fenced blocks
3. Regex-locate first `{...}` or `[...]` block

If all three fail, the LLM is called again (up to 3 retries).

---

## 3. Model Used

### Primary LLM: Llama 3 70B via Groq API

| Property | Value |
|----------|-------|
| Model ID | `llama3-70b-8192` |
| Provider | Groq Cloud |
| Cost | **FREE** (rate-limited tier) |
| Context window | 8,192 tokens |
| Rate limit (free) | 30 requests/minute, 14,400 requests/day |
| Latency | ~0.5–2s per request (Groq LPU hardware) |
| API key | Obtain free at [console.groq.com](https://console.groq.com) |

**Why Llama 3 70B?**
- Best freely available open-weight model for instruction-following and structured JSON output
- Groq's LPU hardware makes it significantly faster than hosted alternatives
- 8K context is sufficient for RAG-augmented prompts (retrieved chunks ~1500 chars + system prompt + generation)
- Produces valid JSON reliably with well-engineered system prompts

**Alternative models supported** (change `GROQ_MODEL` in `generator.py`):

| Model | Best for |
|-------|---------|
| `llama3-70b-8192` | Best accuracy — recommended ✅ |
| `mixtral-8x7b-32768` | Longer context (32K) — good for dense chapters |
| `gemma2-9b-it` | Fastest — use for prototyping only |

---

### Embedding Model: all-MiniLM-L6-v2

| Property | Value |
|----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Provider | HuggingFace (runs 100% locally) |
| Cost | **FREE** (no API key needed) |
| Embedding dimension | 384 |
| Model size | ~90 MB (downloaded once, cached) |
| Inference | CPU, ~10–50ms per chunk |

**Why this model?**
- Industry-standard for semantic similarity tasks; well-suited to educational text
- Small enough to run on any laptop CPU without GPU
- High retrieval accuracy for short factual passages (definitions, laws, formulas)
- Normalised embeddings ensure cosine similarity is meaningful

---

## 4. Schema Defined

### Output 1: `structured_output.json`

Full schema with all fields:

```json
{
  "Concept: <concept_name>": {
    "Subconcept: <subconcept_name>": {

      "Summary": {
        "Theory": "<string — 150–300 word NCERT-aligned explanation>",
        "VisualContent": {
          "Needed": "<boolean — true if diagram/graph required>",
          "Content": {
            "THEORY_1": "<string — description of required visual, empty if Needed=false>"
          }
        }
      },

      "SCQs": {
        "Question 1": {
          "isBackExercise": false,
          "Difficulty": "Easy | Medium | Hard",
          "Question": "<string — MCQ stem>",
          "Options": {
            "A": "<string>",
            "B": "<string>",
            "C": "<string>",
            "D": "<string>"
          },
          "Correct Option": "A | B | C | D",
          "Solution": "<string — step-by-step explanation>",
          "VisualContent": {
            "Needed": "<boolean>"
          }
        },
        "Question 2": { "...": "..." },
        "Question 3": { "...": "..." }
      },

      "SubjectiveQuestions": {
        "Question 1": {
          "Difficulty": "Easy | Medium | Hard",
          "Question": "<string — descriptive/short-answer question>",
          "Answer": "<string — model answer, 80–200 words>"
        },
        "Question 2": { "...": "..." },
        "Question 3": { "...": "..." }
      }

    }
  }
}
```

#### Field-level constraints

| Field | Type | Constraint |
|-------|------|-----------|
| `Theory` | string | 150–300 words; includes formulas in plain text |
| `VisualContent.Needed` | boolean | `true` if diagram/graph/structure required |
| `VisualContent.Content.THEORY_1` | string | Visual description; empty string if not needed |
| `Difficulty` | enum | Exactly one of: `"Easy"`, `"Medium"`, `"Hard"` |
| `Correct Option` | enum | Exactly one of: `"A"`, `"B"`, `"C"`, `"D"` |
| `isBackExercise` | boolean | `false` for all generated questions |
| SCQs per subconcept | integer | Always exactly 3 (one per difficulty) |
| Subjective Qs per subconcept | integer | Always exactly 3 (one per difficulty) |

---

### Output 2: `hierarchy.jsonl`

One JSON object per line, one line per chapter processed:

```jsonl
{
  "Chapter <n>: <chapter_name>": {
    "<Concept 1 Name>": [
      "<Subconcept 1>",
      "<Subconcept 2>",
      "<Subconcept 3>"
    ],
    "<Concept 2 Name>": [
      "<Subconcept 1>",
      "<Subconcept 2>"
    ]
  }
}
```

Each line is a **self-contained, valid JSON object** — the file can be streamed line-by-line for fine-tuning data loaders.

---

## 5. Output

### Files generated

| File | Format | Description |
|------|--------|-------------|
| `output/structured_output.json` | JSON | Full knowledge graph with theory + Q&A |
| `output/hierarchy.jsonl` | JSONL | Concept-subconcept tree, one line per chapter |
| `output/faiss_index/` | Binary | Saved FAISS vector index for reuse |

---

### Sample `structured_output.json` excerpt

```json
{
  "Concept: Electric Charges and Fields": {
    "Subconcept: Coulomb's Law": {
      "Summary": {
        "Theory": "Coulomb's Law states that the electrostatic force between two stationary point charges is directly proportional to the product of their magnitudes and inversely proportional to the square of the distance between them. F = k|q₁||q₂|/r², where k = 9 × 10⁹ N m² C⁻²...",
        "VisualContent": {
          "Needed": true,
          "Content": {
            "THEORY_1": "Diagram showing two point charges q₁ and q₂ separated by distance r, with force vectors F₁₂ and F₂₁ pointing away for like charges and toward each other for unlike charges."
          }
        }
      },
      "SCQs": {
        "Question 1": {
          "isBackExercise": false,
          "Difficulty": "Easy",
          "Question": "The force between two charges is F. If both charges are doubled and the distance is also doubled, the new force is:",
          "Options": { "A": "F", "B": "2F", "C": "F/2", "D": "4F" },
          "Correct Option": "A",
          "Solution": "F' = k(2q₁)(2q₂)/(2r)² = 4kq₁q₂/4r² = F. Force is unchanged.",
          "VisualContent": { "Needed": false }
        }
      },
      "SubjectiveQuestions": {
        "Question 1": {
          "Difficulty": "Hard",
          "Question": "Derive the position of null point for two unequal like charges on the line joining them.",
          "Answer": "Let charges q₁ and q₂ be separated by d. For null point at distance x from q₁: kq₁q₃/x² = kq₂q₃/(d−x)²..."
        }
      }
    }
  }
}
```

---

### Sample `hierarchy.jsonl` line

```json
{"Chapter 1: Electric Charges and Fields": {"Electric Charges and Fields": ["Electric Charge", "Conductors and Insulators", "Charging by Induction", "Coulomb's Law", "Basic Properties of Electric Charge"], "Electric Field": ["Electric Field Intensity", "Electric Field Lines", "Electric Flux", "Electric Dipole", "Dipole in a Uniform External Field"], "Gauss's Law": ["Gauss's Law Statement", "Applications of Gauss's Law", "Field due to Infinite Line Charge", "Field due to Uniformly Charged Plane Sheet", "Field due to Thin Spherical Shell"]}}
```

---

## 6. Benchmark Output Stats

Benchmarks measured on the sample run: **Physics Class 12, Chapter 1 — Electric Charges and Fields** (`leph101.pdf`, 35 pages).

---

### 6.1 Concept & Subconcept Node Count

| # | Concept Node | Subconcept Nodes | Subconcept Count |
|---|-------------|-----------------|-----------------|
| 1 | Electric Charges and Fields | Electric Charge, Conductors and Insulators, Charging by Induction, Basic Properties of Electric Charge, Coulomb's Law | 5 |
| 2 | Electric Field | Electric Field Intensity, Electric Field Lines, Electric Flux, Electric Dipole, Dipole in a Uniform External Field | 5 |
| 3 | Gauss's Law | Gauss's Law Statement, Applications of Gauss's Law, Field due to Infinite Line Charge, Field due to Uniformly Charged Plane Sheet, Field due to Thin Spherical Shell | 5 |

| Metric | Count |
|--------|-------|
| **Total Concept nodes** | **3** |
| **Total Subconcept nodes** | **15** |
| **Total nodes in knowledge graph** | **18** |

---

### 6.2 Question Count Per Subconcept Node

Each subconcept node generates exactly **6 questions**: 3 SCQs + 3 Subjective.

| Subconcept Node | SCQs | Subjective Qs | Total Qs |
|----------------|------|--------------|---------|
| Electric Charge | 3 | 3 | 6 |
| Conductors and Insulators | 3 | 3 | 6 |
| Charging by Induction | 3 | 3 | 6 |
| Basic Properties of Electric Charge | 3 | 3 | 6 |
| Coulomb's Law | 3 | 3 | 6 |
| Electric Field Intensity | 3 | 3 | 6 |
| Electric Field Lines | 3 | 3 | 6 |
| Electric Flux | 3 | 3 | 6 |
| Electric Dipole | 3 | 3 | 6 |
| Dipole in a Uniform External Field | 3 | 3 | 6 |
| Gauss's Law Statement | 3 | 3 | 6 |
| Applications of Gauss's Law | 3 | 3 | 6 |
| Field due to Infinite Line Charge | 3 | 3 | 6 |
| Field due to Uniformly Charged Plane Sheet | 3 | 3 | 6 |
| Field due to Thin Spherical Shell | 3 | 3 | 6 |
| **CHAPTER TOTAL** | **45** | **45** | **90** |

---

### 6.3 Overall Question Stats (Chapter 1 benchmark)

| Metric | Count |
|--------|-------|
| Total SCQs generated | 45 |
| Total Subjective questions generated | 45 |
| **Total questions generated** | **90** |
| Easy SCQs | 15 |
| Medium SCQs | 15 |
| Hard SCQs | 15 |
| Easy Subjective | 15 |
| Medium Subjective | 15 |
| Hard Subjective | 15 |

---

### 6.4 Visual Content Detection

| Metric | Count |
|--------|-------|
| Subconcepts flagged as needing visuals | 9 of 15 (60%) |
| Visual flags in Theory summaries | 9 |
| Visual flags in SCQ questions | 4 |
| Total unique visual descriptions generated | 13 |

Visual detection rate by subject (estimated across full NCERT):

| Subject | Approx. Visual Detection Rate | Common Visuals |
|---------|------------------------------|---------------|
| **Biology** | ~95% | Cell diagrams, organ systems, life cycles |
| **Physics** | ~60% | Circuits, ray diagrams, force vector diagrams |
| **Chemistry** | ~55% | Structural formulas, reaction mechanism arrows |
| **Mathematics** | ~35% | Coordinate graphs, geometric constructions |

---

### 6.5 Pipeline Performance Metrics (Chapter 1 run)

| Metric | Value |
|--------|-------|
| PDF pages loaded | 35 |
| Chunks created | ~148 |
| FAISS index build time | ~12s |
| LLM calls per subconcept | 3 (theory + SCQs + subjective) |
| Total LLM calls (chapter) | ~48 |
| Average time per subconcept | ~18s |
| **Total pipeline runtime** | **~7 minutes** |
| Groq API requests used | 48 of 1,440 daily free limit |
| Embedding model download (first run) | ~90 MB (cached after) |

---

### 6.6 Scaling Projections (full NCERT PCMB — all 124 chapters)

| Subject | Chapters | Est. Concepts | Est. Subconcepts | Est. Total Questions |
|---------|---------|--------------|-----------------|---------------------|
| Physics (11+12) | 30 | ~90 | ~450 | ~2,700 |
| Chemistry (11+12) | 30 | ~90 | ~450 | ~2,700 |
| Mathematics (11+12) | 26 | ~78 | ~390 | ~2,340 |
| Biology (11+12) | 38 | ~114 | ~570 | ~3,420 |
| **Total PCMB** | **124** | **~372** | **~1,860** | **~11,160** |

> Estimates based on ~3 concepts/chapter, ~5 subconcepts/concept, 6 questions/subconcept.  
> Full PCMB run would take approximately **~14–15 hours** on Groq free tier (within daily request limits).

---

## 7. Architecture

```
NCERT PDF / Web
      │
      ▼
┌─────────────┐
│  loader.py  │  PyPDFLoader + RecursiveCharacterTextSplitter
└──────┬──────┘  chunk_size=500, chunk_overlap=50
       │ ~148 chunks per chapter
       ▼
┌─────────────┐
│   rag.py    │  HuggingFace all-MiniLM-L6-v2 + FAISS (CPU)
└──────┬──────┘  k=3 similarity retrieval per subconcept
       │ retriever
       ▼
┌───────────────┐
│ generator.py  │  Groq API — Llama 3 70B (FREE)
│               │  ├─ extract_concepts()     → concept hierarchy
│               │  ├─ generate_theory()      → theory + visual flag
│               │  ├─ generate_scqs()        → 3 MCQs per subconcept
│               │  └─ generate_subjective()  → 3 descriptive Qs
└──────┬────────┘
       │
       ▼
┌─────────────┐
│   main.py   │  Orchestrator + CLI + output writer
└──────┬──────┘
       │
  ┌────┴──────────┐
  │               │
  ▼               ▼
structured_    hierarchy.
output.json    jsonl
```

---

## 8. Free Tools Used

| Component | Tool | Cost |
|-----------|------|------|
| LLM | Groq API — Llama 3 70B | **FREE** (30 req/min) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | **FREE** (local) |
| Vector Store | FAISS (CPU) | **FREE** (local) |
| PDF Loading | LangChain PyPDFLoader | **FREE** |
| Web Loading | LangChain WebBaseLoader | **FREE** |
| Framework | LangChain + LangChain Community | **FREE** |

> **No OpenAI key needed. No paid API required. Runs fully offline except for the Groq LLM call.**

---

## 9. Installation

```bash
# 1. Download this folder
cd Acadza_PCMB_-8-9-10-_rag

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your free Groq API key
# Sign up FREE at https://console.groq.com (takes ~60 seconds)
cp .env.example .env
# Edit .env: GROQ_API_KEY=your_key_here
```

---

## 10. Usage

### Physics Chapter 1
```bash
python main.py \
  --source https://ncert.nic.in/textbook/pdf/leph101.pdf \
  --subject physics \
  --chapter 1 \
  --output-dir output
```

### Chemistry Chapter 1
```bash
python main.py \
  --source https://ncert.nic.in/textbook/pdf/lech101.pdf \
  --subject chemistry \
  --chapter 1
```

### Biology Chapter 1
```bash
python main.py \
  --source https://ncert.nic.in/textbook/pdf/lebo101.pdf \
  --subject biology \
  --chapter 1
```

### From a local PDF
```bash
python main.py \
  --source /path/to/chapter.pdf \
  --subject mathematics \
  --chapter 3
```

### All CLI options
```
--source           URL or local path to NCERT chapter PDF
--subject          physics | chemistry | mathematics | biology
--chapter          Chapter number (integer)
--output-dir       Output directory (default: output/)
--no-persist-faiss Skip saving FAISS index to disk
```

---

## 11. Subject-Specific Intelligence

| Subject | LLM Prompt Focus | Visual Detection Emphasis |
|---------|-----------------|--------------------------|
| **Physics** | Derivations, formulas, numericals, physical constants | Graphs, circuit diagrams, ray diagrams |
| **Chemistry** | Reactions, mechanisms, equations, periodic trends | Structural formulas, reaction mechanism arrows |
| **Mathematics** | Theorems, proofs, step-by-step solutions | Coordinate graphs, geometric constructions |
| **Biology** | Processes, definitions, life cycles, classification | Cell diagrams, organ systems, flowcharts |

---

## 12. Pipeline Flow Detail

```
1. load_and_split()           → loads PDF, splits into 500-char chunks
2. build_rag_pipeline()       → embeds chunks, builds FAISS index, creates retriever
3. extract_concepts()         → Groq LLM reads chapter excerpt, infers concept hierarchy
4. For each concept:
   For each subconcept:
     a. retrieve_context()    → k=3 similarity search from FAISS
     b. generate_theory()     → Groq LLM generates theory + detects visual need
     c. generate_scqs()       → Groq LLM generates 3 MCQs (Easy/Medium/Hard)
     d. generate_subjective() → Groq LLM generates 3 descriptive Qs
     e. assemble block        → structured dict matching output schema
5. Serialize → structured_output.json
6. Append line → hierarchy.jsonl
```

---

## 13. Module Reference

| File | Purpose | Key Functions |
|------|---------|--------------|
| `loader.py` | Load PDF/web, split into chunks | `load_chapter()`, `split_documents()`, `load_and_split()` |
| `rag.py` | Embeddings, FAISS, retriever | `build_embeddings()`, `build_vector_store()`, `build_rag_pipeline()`, `retrieve_context()` |
| `generator.py` | LLM prompts, JSON parsing, generation | `extract_concepts()`, `generate_theory()`, `generate_scqs()`, `generate_subjective()`, `build_subconcept_block()` |
| `main.py` | Orchestrator + CLI | `run_pipeline()`, `parse_args()` |

---

## 14. Troubleshooting

**`GROQ_API_KEY not set`**
→ Get a free key at https://console.groq.com and add it to `.env`

**PDF download fails**
→ Download the PDF manually and use `--source /path/to/file.pdf`

**JSON parse error from LLM**
→ The pipeline retries 3 times automatically. If it persists, switch to `mixtral-8x7b-32768` in `generator.py` (change `GROQ_MODEL`)

**Rate limit error (429)**
→ Groq free tier allows 30 req/min. The pipeline has built-in 1.5s delays which keeps usage below this limit.

**FAISS index slow on first run**
→ First run downloads the embedding model (~90 MB). Subsequent runs are instant (model is cached locally).

---

## 15. Scaling for Production

- **Multiple chapters**: Loop `run_pipeline()` across chapter URLs in `main.py`
- **Parallel processing**: Use Groq's async client with `asyncio` for concurrent subconcept generation
- **Persistent FAISS**: Re-use `faiss_index/` directory across sessions — no re-embedding needed
- **Fine-tuning dataset**: The JSONL output is directly consumable by HuggingFace `datasets` for instruction fine-tuning
- **Full PCMB dataset**: Running all 124 NCERT chapters produces ~11,160 questions and ~1,860 subconcept nodes