"""
generator.py
------------
LLM-driven generation module using FREE Groq API (Llama 3 70B / Mixtral).
Handles:
  - Concept + Subconcept extraction
  - Theory generation (subject-aware)
  - SCQ (MCQ) generation
  - Subjective question generation
  - Visual content detection
  - JSON output parsing with retry/repair
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq

# ---------------------------------------------------------------------------
# LLM Client – Groq (FREE tier: https://console.groq.com)
# Model options: llama3-70b-8192 | mixtral-8x7b-32768 | gemma2-9b-it
# ---------------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
RETRY_DELAY = 2   # seconds between retries on parse failures


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
        )
    return Groq(api_key=api_key)


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Raw LLM call returning the assistant message string."""
    client = _get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _extract_json(text: str) -> Any:
    """
    Robustly extract a JSON object or array from LLM output.
    Tries:
      1. Direct parse
      2. Extract from ```json ... ``` fences
      3. Find first { ... } or [ ... ] block
    """
    # 1. Direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced block
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Brute-force first {...}
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Brute-force first [...]
    bracket_match = re.search(r"\[[\s\S]*\]", text)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output:\n{text[:500]}")


def _call_with_retry(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Any:
    """Call LLM and parse JSON with retry on failure."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _call_llm(system_prompt, user_prompt, temperature, max_tokens)
            return _extract_json(raw)
        except Exception as e:
            last_error = e
            print(f"  [Generator] Parse attempt {attempt} failed: {e}. Retrying…")
            time.sleep(RETRY_DELAY)
    raise ValueError(f"All {MAX_RETRIES} parse attempts failed. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Subject-specific guidance injected into prompts
# ---------------------------------------------------------------------------

SUBJECT_HINTS = {
    "physics": (
        "Focus on derivations, formulas, numerical problem types, physical constants, "
        "and laws. Mark subconcepts involving graphs, circuits, or ray diagrams as needing visuals."
    ),
    "chemistry": (
        "Emphasise chemical reactions, mechanisms, equations, periodic trends, "
        "and molecular structures. Mark subconcepts with structural formulas or reaction diagrams as needing visuals."
    ),
    "mathematics": (
        "Emphasise theorems, proofs, step-by-step solutions, and problem-solving strategies. "
        "Mark subconcepts involving coordinate geometry, graphs, or geometric constructions as needing visuals."
    ),
    "biology": (
        "Emphasise biological processes, definitions, life cycles, and classification. "
        "Almost all subconcepts in biology require diagrams (cell structures, organ systems, flowcharts)."
    ),
}


def _subject_hint(subject: str) -> str:
    return SUBJECT_HINTS.get(subject.lower(), "")


# ---------------------------------------------------------------------------
# 1. Concept + Subconcept Extraction
# ---------------------------------------------------------------------------

CONCEPT_SYSTEM = """You are an expert NCERT curriculum analyst.
Your job is to read a chapter excerpt and identify its main Concepts and Subconcepts.
Output ONLY valid JSON — no commentary, no markdown fences.

Output format:
{
  "chapter_name": "<inferred chapter name>",
  "concepts": [
    {
      "name": "<Concept Name>",
      "subconcepts": ["<Subconcept 1>", "<Subconcept 2>", ...]
    }
  ]
}

Rules:
- Do NOT hardcode answers; derive everything from the provided text.
- A Concept is a major topic heading.
- Subconcepts are specific ideas, laws, phenomena, or techniques under that concept.
- Infer implicit subconcepts if the text implies them but doesn't name them explicitly.
- Return at least 2 subconcepts per concept.
"""


def extract_concepts(
    chapter_text: str,
    subject: str = "physics",
) -> Dict[str, Any]:
    """
    Extract concepts and subconcepts from raw chapter text.

    Returns
    -------
    dict with keys: chapter_name, concepts (list of {name, subconcepts})
    """
    hint = _subject_hint(subject)
    user_prompt = f"""Subject: {subject.upper()}
Subject guidance: {hint}

Chapter text (first ~3000 chars):
\"\"\"
{chapter_text[:3000]}
\"\"\"

Extract all Concepts and their Subconcepts from this chapter.
"""
    print("[Generator] Extracting concepts and subconcepts…")
    result = _call_with_retry(CONCEPT_SYSTEM, user_prompt)
    print(f"[Generator] Found {len(result.get('concepts', []))} concepts.")
    return result


# ---------------------------------------------------------------------------
# 2. Theory Generation
# ---------------------------------------------------------------------------

THEORY_SYSTEM = """You are an expert NCERT teacher writing educational content.
For a given Subconcept, generate a detailed, accurate theory explanation.
Output ONLY valid JSON — no commentary, no markdown fences.

Output format:
{
  "theory": "<comprehensive explanation, 150-300 words>",
  "visual_needed": true or false,
  "visual_description": "<describe what diagram/visual is required, or empty string if not needed>"
}

Rules:
- Be accurate, clear, and NCERT-aligned.
- Include formulas (use plain text like F = ma) where relevant.
- Set visual_needed=true if the subconcept is best understood with a diagram, graph, or illustration.
- If visual_needed=true, describe the visual in visual_description (e.g. "Circuit diagram showing series connection of resistors").
"""


def generate_theory(
    subconcept: str,
    concept: str,
    context: str,
    subject: str = "physics",
) -> Dict[str, Any]:
    """
    Generate theory for a subconcept using RAG context.

    Returns
    -------
    dict: {theory, visual_needed, visual_description}
    """
    hint = _subject_hint(subject)
    user_prompt = f"""Subject: {subject.upper()}
Subject guidance: {hint}

Concept: {concept}
Subconcept: {subconcept}

Relevant chapter context (RAG-retrieved):
\"\"\"
{context}
\"\"\"

Generate a comprehensive theory explanation for the subconcept "{subconcept}".
"""
    return _call_with_retry(THEORY_SYSTEM, user_prompt)


# ---------------------------------------------------------------------------
# 3. SCQ (MCQ) Generation
# ---------------------------------------------------------------------------

SCQ_SYSTEM = """You are an expert NCERT question paper setter.
Generate  Single Correct Questions (MCQs) for a given subconcept.
Output ONLY valid JSON — no commentary, no markdown fences.

Output format:
{
  "questions": [
    {
      "difficulty": "Easy" | "Medium" | "Hard",
      "question": "<MCQ question text>",
      "options": {
        "A": "<option text>",
        "B": "<option text>",
        "C": "<option text>",
        "D": "<option text>"
      },
      "correct_option": "A" | "B" | "C" | "D",
      "solution": "<step-by-step explanation of why the answer is correct>",
      "visual_needed": true or false
    }
  ]
}

Rules:
- it should include atleast one Easy, one Medium, one Hard question.
- Questions must be original and NCERT exam-relevant.
- Distractors must be plausible.
- Solutions must be clear and educational.
- Set visual_needed=true only if a diagram is genuinely necessary to answer the question.
"""


def generate_scqs(
    subconcept: str,
    concept: str,
    context: str,
    subject: str = "physics",
) -> List[Dict[str, Any]]:
    """
    Generate 3 MCQs for a subconcept.

    Returns
    -------
    List of question dicts
    """
    hint = _subject_hint(subject)
    user_prompt = f"""Subject: {subject.upper()}
Subject guidance: {hint}

Concept: {concept}
Subconcept: {subconcept}

Relevant chapter context (RAG-retrieved):
\"\"\"
{context}
\"\"\"

Generate exactly 3 MCQs (Easy, Medium, Hard) for the subconcept "{subconcept}".
"""
    result = _call_with_retry(SCQ_SYSTEM, user_prompt)
    return result.get("questions", [])


# ---------------------------------------------------------------------------
# 4. Subjective Question Generation
# ---------------------------------------------------------------------------

SUBJECTIVE_SYSTEM = """You are an expert NCERT examiner.
Generate exactly 3 Subjective (descriptive/short-answer) questions for a given subconcept.
Output ONLY valid JSON — no commentary, no markdown fences.

Output format:
{
  "questions": [
    {
      "difficulty": "Easy" | "Medium" | "Hard",
      "question": "<descriptive question>",
      "answer": "<detailed model answer, 80-200 words>"
    }
  ]
}

Rules:
- One Easy, one Medium, one Hard question.
- Answers must be comprehensive and NCERT-aligned.
- Include derivations or step-by-step reasoning where applicable.
"""


def generate_subjective(
    subconcept: str,
    concept: str,
    context: str,
    subject: str = "physics",
) -> List[Dict[str, Any]]:
    """
    Generate 3 subjective questions for a subconcept.

    Returns
    -------
    List of question dicts
    """
    hint = _subject_hint(subject)
    user_prompt = f"""Subject: {subject.upper()}
Subject guidance: {hint}

Concept: {concept}
Subconcept: {subconcept}

Relevant chapter context (RAG-retrieved):
\"\"\"
{context}
\"\"\"

Generate exactly 3 subjective questions (Easy, Medium, Hard) for "{subconcept}".
"""
    result = _call_with_retry(SUBJECTIVE_SYSTEM, user_prompt)
    return result.get("questions", [])


# ---------------------------------------------------------------------------
# 5. Assemble per-subconcept output block
# ---------------------------------------------------------------------------

def build_subconcept_block(
    subconcept: str,
    concept: str,
    context: str,
    subject: str = "physics",
) -> Dict[str, Any]:
    """
    Orchestrate theory + SCQ + subjective generation for one subconcept.
    Returns a fully populated subconcept dict matching the target JSON schema.
    """
    print(f"  [Generator] Processing subconcept: '{subconcept}'")

    # Theory
    theory_data = generate_theory(subconcept, concept, context, subject)

    # Visual content block
    visual_content = {}
    if theory_data.get("visual_needed") and theory_data.get("visual_description"):
        visual_content["THEORY_1"] = theory_data["visual_description"]

    # SCQs
    scq_raw = generate_scqs(subconcept, concept, context, subject)
    scqs = {}
    for i, q in enumerate(scq_raw, start=1):
        scqs[f"Question {i}"] = {
            "isBackExercise": False,
            "Difficulty": q.get("difficulty", "Medium"),
            "Question": q.get("question", ""),
            "Options": q.get("options", {}),
            "Correct Option": q.get("correct_option", "A"),
            "Solution": q.get("solution", ""),
            "VisualContent": {"Needed": q.get("visual_needed", False)},
        }

    # Subjective
    subj_raw = generate_subjective(subconcept, concept, context, subject)
    subjective = {}
    for i, q in enumerate(subj_raw, start=1):
        subjective[f"Question {i}"] = {
            "Difficulty": q.get("difficulty", "Medium"),
            "Question": q.get("question", ""),
            "Answer": q.get("answer", ""),
        }

    return {
        "Summary": {
            "Theory": theory_data.get("theory", ""),
            "VisualContent": {
                "Needed": theory_data.get("visual_needed", False),
                "Content": visual_content,
            },
        },
        "SCQs": scqs,
        "SubjectiveQuestions": subjective,
    }


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    sample_context = (
        "Coulomb's law states that the force between two point charges is "
        "directly proportional to the product of the charges and inversely "
        "proportional to the square of the distance between them. "
        "F = k * q1 * q2 / r^2, where k = 9 × 10^9 N m^2 C^-2."
    )
    block = build_subconcept_block(
        subconcept="Coulomb's Law",
        concept="Electric Charges and Fields",
        context=sample_context,
        subject="physics",
    )
    print(json.dumps(block, indent=2))