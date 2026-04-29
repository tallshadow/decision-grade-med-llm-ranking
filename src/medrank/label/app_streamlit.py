# src/medrank/label/app_streamlit.py

import json
import os
import time
import random
from typing import Dict, Any, List, Optional, Set

import re
import html

import yaml
import streamlit as st
from filelock import FileLock
from pathlib import Path


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def index_by_key(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    return {r[key]: r for r in rows}


def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    """Safe concurrent append for multi-annotator server use."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = path + ".lock"
    with FileLock(lock_path, timeout=10):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_completed_pairs(annotations: List[Dict[str, Any]], annotator_id: str) -> Set[str]:
    """Pairs already touched by this annotator (including skips) to support resume."""
    done: Set[str] = set()
    for a in annotations:
        if a.get("annotator_id") == annotator_id:
            pid = a.get("pair_id")
            if pid:
                done.add(pid)
    return done


def already_labeled_non_skip(annotations: List[Dict[str, Any]], annotator_id: str, pair_id: str) -> bool:
    """Block duplicates for real labels (preference != 'skip')."""
    # iterate reversed to be slightly faster if file is big
    for a in reversed(annotations):
        if a.get("annotator_id") == annotator_id and a.get("pair_id") == pair_id:
            return a.get("preference") != "skip"
    return False


def load_assignment(assign_path: str) -> Optional[Set[str]]:
    assign_path = (assign_path or "").strip()
    if not assign_path:
        return None
    p = Path(assign_path)
    if not p.exists():
        return None
    data = json.load(open(p, "r", encoding="utf-8"))
    if isinstance(data, list):
        return set(str(x) for x in data)
    raise ValueError("Assignment JSON must be a list of pair_id strings.")


def strip_prompt_prefix(text: str) -> str:
    """
    The decoded model output often contains the original prompt + generated answer.
    We keep the last structured answer block starting from FINAL_ANSWER.
    """
    if not text:
        return ""

    idx = text.rfind("FINAL_ANSWER:")
    if idx != -1:
        return text[idx:].strip()

    return text.strip()


def extract_structured_fields(text: str) -> Dict[str, str]:
    """
    Extract FINAL_ANSWER, RATIONALE, UNCERTAINTY, SAFETY_NOTE from the generated answer.
    Uses the last FINAL_ANSWER block to avoid extracting prompt placeholders.
    """
    cleaned = strip_prompt_prefix(text)

    fields = {
        "FINAL_ANSWER": "",
        "RATIONALE": "",
        "UNCERTAINTY": "",
        "SAFETY_NOTE": "",
    }

    pattern = r"(FINAL_ANSWER|RATIONALE|UNCERTAINTY|SAFETY_NOTE)\s*:\s*"
    matches = list(re.finditer(pattern, cleaned))

    if not matches:
        fields["FINAL_ANSWER"] = cleaned[:500]
        return fields

    for i, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        value = cleaned[start:end].strip()

        # remove repeated prompt artifacts if present
        value = value.replace("Format:", "").strip()
        fields[key] = value

    return fields


def render_answer_card(label: str, gen: Dict[str, Any]) -> None:
    """
    Render one answer using native Streamlit components.
    This is much more readable and avoids HTML display bugs.
    """
    raw_text = gen.get("text", "")
    fields = extract_structured_fields(raw_text)

    final_answer = fields.get("FINAL_ANSWER") or "Not clearly provided"
    rationale = fields.get("RATIONALE") or "Not clearly provided"
    uncertainty = fields.get("UNCERTAINTY") or "Not clearly provided"
    safety_note = fields.get("SAFETY_NOTE") or "Not clearly provided"

    st.markdown(f"### Answer {label}")

    with st.container(border=True):
        st.markdown("#### ✅ Final answer")
        st.success(final_answer)

        st.markdown("#### 🧠 Rationale")
        st.write(rationale)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Uncertainty")
            st.info(uncertainty)

        with col2:
            st.markdown("#### 🛡️ Safety note")
            st.warning(safety_note)

    with st.expander(f"Show full raw Answer {label}"):
        st.text(raw_text)


def main() -> None:
    st.set_page_config(page_title="MedRank Labeling", layout="wide")

    with open("configs/labeling.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    # Allow server to override where annotations are stored
    paths["annotations"] = os.getenv("ANNOTATIONS_PATH", paths["annotations"])

    seed = int(cfg["seed"])
    ui_cfg = cfg.get("ui", {})
    labels_cfg = cfg.get("labels", {})
    harm_flag_keys = cfg.get("harm_flags", [])

    questions = load_jsonl(paths["questions"])
    generations = load_jsonl(paths["generations"])
    pairs = load_jsonl(paths["pairs"])
    annotations = load_jsonl(paths["annotations"])

    q_by_id = index_by_key(questions, "qid")
    g_by_id = index_by_key(generations, "gid")

    st.title("Decision-Grade Medical LLM Pairwise Labeling")

    # Sidebar: annotator session info
    st.sidebar.header("Session")
    annotator_id = st.sidebar.text_input("Annotator ID", value="rater_A")
    expertise = st.sidebar.selectbox(
        "Self-rated expertise", ["novice", "intermediate", "advanced"], index=1
    )
    st.sidebar.caption("Use a stable Annotator ID so resume works.")

    # Assignment filtering (optional)
    st.sidebar.subheader("Assignment (optional)")
    assign_path = st.sidebar.text_input("Assignment JSON path", value="")
    assigned: Optional[Set[str]] = None
    if assign_path.strip():
        try:
            assigned = load_assignment(assign_path)
            if assigned is None:
                st.sidebar.error("Assignment file not found.")
            else:
                st.sidebar.success(f"Loaded {len(assigned)} assigned pairs")
        except Exception as e:
            st.sidebar.error(f"Assignment load error: {e}")
            assigned = None

    # Determine which pairs remain for this annotator
    done = get_completed_pairs(annotations, annotator_id)
    remaining = [p for p in pairs if p.get("pair_id") not in done]

    if assigned is not None:
        remaining = [p for p in remaining if p.get("pair_id") in assigned]

    # if debug_pair_id:
    #     remaining = [p for p in pairs if p.get("pair_id") == debug_pair_id]

    # Optional randomize order per annotator (deterministic)
    if ui_cfg.get("randomize_pair_order", True):
        rr = random.Random(seed + (hash(annotator_id) % 10_000))
        rr.shuffle(remaining)

    st.sidebar.write(f"Total pairs: {len(pairs)}")
    st.sidebar.write(f"Labeled by you: {len(done)}")
    st.sidebar.write(f"Remaining: {len(remaining)}")

    if not remaining:
        st.success("All done! No remaining pairs for this annotator.")
        return

    # Pick the next pair
    pair = remaining[0]
    pair_id = pair["pair_id"]
    qid = pair["qid"]

    if qid not in q_by_id:
        st.error(f"Missing question for qid={qid}")
        return
    if pair["gid_a"] not in g_by_id or pair["gid_b"] not in g_by_id:
        st.error("Missing generation record(s) for this pair. Check generations.jsonl.")
        return

    q = q_by_id[qid]
    ga = g_by_id[pair["gid_a"]]
    gb = g_by_id[pair["gid_b"]]

    # Show question
    st.subheader(
        f"Pair: {pair_id}  |  Split: {pair.get('split')}  |  High-risk: {pair.get('is_high_risk')}"
    )
    st.markdown("### Question")
    st.write(q.get("question", ""))

    if q.get("choices"):
        st.markdown("**Choices:**")
        for c in q["choices"]:
            st.write(f"- {c}")

    # Layout answers side-by-side
    colA, colB = st.columns(2)

    with colA:
        render_answer_card("A", ga)

    with colB:
        render_answer_card("B", gb)

    st.divider()
    st.markdown("### Your judgment")

    pref = st.radio(
        "Preference",
        labels_cfg.get("preference_options", ["A", "B", "tie", "both_bad"]),
        horizontal=True,
    )
    conf = st.radio(
        "Confidence",
        labels_cfg.get("confidence_options", ["low", "medium", "high"]),
        horizontal=True,
    )

    # Annotation mode / assistance logging (conference-safe)
    annotation_mode = st.radio("Annotation mode", ["unaided", "assisted"], horizontal=True)
    assistance_used = "none"
    assistance_purpose = ""
    assistance_notes = ""
    if annotation_mode == "assisted":
        assistance_used = st.selectbox("Assistance used", ["web", "llm", "both"])
        assistance_purpose = st.selectbox(
            "Assistance purpose", ["fact_check", "guideline_check", "term_definition", "other"]
        )
        assistance_notes = st.text_input("Assistance notes (optional)", value="")
        st.info(
            "Important: Do NOT ask an LLM to choose between A vs B. "
            "Assistance is for background/fact-check only."
        )

    needs_expert = st.checkbox("Needs expert review (I am not sure about correctness)", value=False)

    st.markdown("#### Scores")
    st.caption("Correctness allows -1 = unknown. Safety/Uncertainty are 0–3.")

    s_cols = st.columns(3)
    with s_cols[0]:
        correctness_scale = [-1, 0, 1, 2, 3]  # -1 = unknown
        correctness_A = st.selectbox("Correctness (A)  (-1 = unknown)", options=correctness_scale, index=3)
        correctness_B = st.selectbox("Correctness (B)  (-1 = unknown)", options=correctness_scale, index=3)

    score_scale = labels_cfg.get("score_scale", [0, 1, 2, 3])
    with s_cols[1]:
        safety_A = st.select_slider("Safety (A)", options=score_scale, value=2)
        safety_B = st.select_slider("Safety (B)", options=score_scale, value=2)
    with s_cols[2]:
        uh_A = st.select_slider("Uncertainty honesty (A)", options=score_scale, value=2)
        uh_B = st.select_slider("Uncertainty honesty (B)", options=score_scale, value=2)

    st.markdown("#### Harm flags (check if present)")
    hf_cols = st.columns(2)

    harmA: Dict[str, bool] = {}
    harmB: Dict[str, bool] = {}
    with hf_cols[0]:
        st.markdown("**Answer A flags**")
        for h in harm_flag_keys:
            harmA[h] = st.checkbox(f"A: {h}", value=False, key=f"A_{h}_{pair_id}")
    with hf_cols[1]:
        st.markdown("**Answer B flags**")
        for h in harm_flag_keys:
            harmB[h] = st.checkbox(f"B: {h}", value=False, key=f"B_{h}_{pair_id}")

    notes = st.text_area("Optional notes (why?)", value="", height=90)

    # Buttons
    btn_cols = st.columns(3)
    with btn_cols[0]:
        save = st.button("Save label ✅", type="primary")
    with btn_cols[1]:
        allow_skip = bool(ui_cfg.get("allow_skip", True))
        skip = st.button("Skip ⏭️") if allow_skip else False
    with btn_cols[2]:
        st.button("Refresh", type="secondary")

    if save:
        # Block duplicates for real labels
        if already_labeled_non_skip(annotations, annotator_id, pair_id):
            st.error("You already labeled this pair. Duplicate labels are blocked.")
            st.stop()

        rec = {
            "ann_id": f"{annotator_id}__{pair_id}__{int(time.time())}",
            "pair_id": pair_id,
            "qid": qid,
            "annotator_id": annotator_id,
            "expertise_self": expertise,
            "annotation_mode": annotation_mode,
            "assistance_used": assistance_used,
            "assistance_purpose": assistance_purpose,
            "assistance_notes": assistance_notes.strip(),
            "needs_expert_review": int(needs_expert),
            "preference": pref,
            "confidence": conf,
            "scores": {
                "correctness_A": int(correctness_A),
                "correctness_B": int(correctness_B),
                "safety_A": int(safety_A),
                "safety_B": int(safety_B),
                "uncertainty_honesty_A": int(uh_A),
                "uncertainty_honesty_B": int(uh_B),
            },
            "harm_flags": {
                "A": {k: int(v) for k, v in harmA.items()},
                "B": {k: int(v) for k, v in harmB.items()},
            },
            "notes": notes.strip(),
            "meta": {
                "split": pair.get("split"),
                "is_high_risk": int(bool(pair.get("is_high_risk", False))),
                "gid_a": pair.get("gid_a"),
                "gid_b": pair.get("gid_b"),
            },
            "ts": time.time(),
            "seed": seed,
        }
        append_jsonl(paths["annotations"], rec)
        st.success("Saved! Reloading next pair...")
        st.rerun()

    if skip:
        # Allow skips but still record them to support resume and analysis
        rec = {
            "ann_id": f"{annotator_id}__{pair_id}__SKIP__{int(time.time())}",
            "pair_id": pair_id,
            "qid": qid,
            "annotator_id": annotator_id,
            "expertise_self": expertise,
            "annotation_mode": annotation_mode,
            "assistance_used": assistance_used if annotation_mode == "assisted" else "none",
            "assistance_purpose": assistance_purpose if annotation_mode == "assisted" else "",
            "assistance_notes": assistance_notes.strip() if annotation_mode == "assisted" else "",
            "needs_expert_review": int(needs_expert),
            "preference": "skip",
            "confidence": conf,
            "scores": {},
            "harm_flags": {},
            "notes": notes.strip(),
            "meta": {
                "split": pair.get("split"),
                "is_high_risk": int(bool(pair.get("is_high_risk", False))),
                "gid_a": pair.get("gid_a"),
                "gid_b": pair.get("gid_b"),
            },
            "ts": time.time(),
            "seed": seed,
        }
        append_jsonl(paths["annotations"], rec)
        st.warning("Skipped. Reloading next pair...")
        st.rerun()


if __name__ == "__main__":
    main()
