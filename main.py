"""
main.py
-------
Orchestrates the full NCERT chapter → structured dataset pipeline.

Usage:
    python main.py \
        --source https://ncert.nic.in/textbook/pdf/leph101.pdf \
        --subject physics \
        --chapter 1

Outputs:
    structured_output.json   – full knowledge graph with theory + questions
    hierarchy.jsonl          – concept-subconcept hierarchy (one line per chapter)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from loader import load_and_split
from Rag import build_rag_pipeline, retrieve_context
from generator import extract_concepts, build_subconcept_block

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
STRUCTURED_OUTPUT = "structured_output.json"
HIERARCHY_JSONL = "hierarchy.jsonl"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source: str,
    subject: str,
    chapter_number: int,
    output_dir: str = ".",
    persist_faiss: bool = True,
) -> None:
    """
    Full end-to-end pipeline for one NCERT chapter.

    Parameters
    ----------
    source         : URL or local path to the NCERT chapter PDF (or web page)
    subject        : 'physics' | 'chemistry' | 'mathematics' | 'biology'
    chapter_number : Integer chapter number (for labelling)
    output_dir     : Where to save structured_output.json and hierarchy.jsonl
    persist_faiss  : If True, save FAISS index to disk for reuse
    """
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # STEP 1 & 2 – Load and split the chapter
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 1/7 – Loading chapter from: {source}")
    print("=" * 60)
    chunks = load_and_split(source)

    # Concatenate all text for concept extraction (first 6000 chars)
    full_text = "\n".join(c.page_content for c in chunks)
    excerpt = full_text[:6000]

    # -----------------------------------------------------------------------
    # STEP 3 – Build embeddings + FAISS vector store
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2/7 – Building embeddings and FAISS index…")
    print("=" * 60)
    faiss_path = str(output_dir / "faiss_index") if persist_faiss else None
    retriever, _ = build_rag_pipeline(chunks, persist_path=faiss_path)

    # -----------------------------------------------------------------------
    # STEP 4 – Extract concepts and subconcepts (LLM)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3/7 – Extracting concepts and subconcepts…")
    print("=" * 60)
    concept_data = extract_concepts(excerpt, subject=subject)

    chapter_name = concept_data.get("chapter_name", f"Chapter {chapter_number}")
    concepts = concept_data.get("concepts", [])

    if not concepts:
        print("[WARN] No concepts extracted. Check the chapter source.")
        sys.exit(1)

    print(f"\nChapter: {chapter_name}")
    for c in concepts:
        print(f"  Concept: {c['name']}")
        for s in c.get("subconcepts", []):
            print(f"    - {s}")

    # -----------------------------------------------------------------------
    # STEP 5, 6 – For each subconcept: RAG retrieval + generation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4–6/7 – Generating theory, SCQs, and subjective questions…")
    print("=" * 60)

    structured_output: dict = {}
    hierarchy_entry: dict = {f"Chapter {chapter_number}: {chapter_name}": {}}

    for concept in concepts:
        concept_name = concept["name"]
        subconcepts = concept.get("subconcepts", [])
        concept_key = f"Concept: {concept_name}"
        structured_output[concept_key] = {}
        hierarchy_entry[f"Chapter {chapter_number}: {chapter_name}"][concept_name] = subconcepts

        print(f"\n[Pipeline] Concept: '{concept_name}' ({len(subconcepts)} subconcepts)")

        for subconcept in subconcepts:
            # RAG: retrieve relevant context for this subconcept
            query = f"{concept_name} {subconcept}"
            context = retrieve_context(retriever, query)

            # Generate theory + questions
            subconcept_key = f"Subconcept: {subconcept}"
            block = build_subconcept_block(
                subconcept=subconcept,
                concept=concept_name,
                context=context,
                subject=subject,
            )
            structured_output[concept_key][subconcept_key] = block

            # Small rate-limit pause (Groq free tier: 30 req/min)
            time.sleep(1.5)

    # -----------------------------------------------------------------------
    # STEP 7 – Save outputs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7/7 – Saving outputs…")
    print("=" * 60)

    # structured_output.json
    structured_path = output_dir / STRUCTURED_OUTPUT
    with open(structured_path, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=2, ensure_ascii=False)
    print(f"[Output] Structured JSON saved: {structured_path}")

    # hierarchy.jsonl (append one line per chapter)
    hierarchy_path = output_dir / HIERARCHY_JSONL
    with open(hierarchy_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(hierarchy_entry, ensure_ascii=False) + "\n")
    print(f"[Output] Hierarchy JSONL saved/appended: {hierarchy_path}")

    elapsed = time.time() - start
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s")
    print(f"   JSON  → {structured_path}")
    print(f"   JSONL → {hierarchy_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NCERT PCMB Chapter → Structured Dataset Pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("NCERT_PDF_URL", "https://ncert.nic.in/textbook/pdf/leph101.pdf"),
        help="URL or local path to the NCERT chapter PDF",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=os.getenv("SUBJECT", "physics"),
        choices=["physics", "chemistry", "mathematics", "biology"],
        help="Subject type",
    )
    parser.add_argument(
        "--chapter",
        type=int,
        default=int(os.getenv("CHAPTER_NUMBER", "1")),
        help="Chapter number",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files",
    )
    parser.add_argument(
        "--no-persist-faiss",
        action="store_true",
        help="Don't save FAISS index to disk",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate API key early
    if not os.getenv("GROQ_API_KEY"):
        print(
            "\n❌ GROQ_API_KEY not set!\n"
            "  1. Sign up FREE at: https://console.groq.com\n"
            "  2. Copy your API key\n"
            "  3. Either:\n"
            "     a) export GROQ_API_KEY=your_key_here\n"
            "     b) Create a .env file with: GROQ_API_KEY=your_key_here\n"
        )
        sys.exit(1)

    run_pipeline(
        source=args.source,
        subject=args.subject,
        chapter_number=args.chapter,
        output_dir=args.output_dir,
        persist_faiss=not args.no_persist_faiss,
    )
# run : python main.py --source ncert-8.pdf --subject physics --chapter 1