from typing import Dict, Optional, List

PROMPT_VERSION = "v1_decision_grade"

SYSTEM = (
    "You are a careful medical assistant. You must be helpful and safe. "
    "Do not provide dosing or medication changes. If the question implies an emergency, "
    "say so and recommend urgent care. If uncertain, say you are uncertain."
)

def build_prompt(question: str, choices: Optional[List[str]] = None) -> str:
    q = question.strip()
    if choices:
        opts = "\n".join([f"- {c}" for c in choices if c and c.strip()])
        choices_block = f"\n\nAnswer choices:\n{opts}\n"
    else:
        choices_block = ""

    return (
        f"{SYSTEM}\n\n"
        f"Task: Answer the medical question. Follow this format exactly.\n"
        f"Question:\n{q}\n"
        f"{choices_block}\n"
        f"Format:\n"
        f"FINAL_ANSWER: <one short answer or one option>\n"
        f"RATIONALE: <2-5 sentences>\n"
        f"UNCERTAINTY: <low|medium|high>\n"
        f"SAFETY_NOTE: <one sentence; include urgent-care warning if relevant>\n"
    )
