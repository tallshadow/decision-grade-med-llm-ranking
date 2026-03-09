import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml
from datasets import load_dataset


@dataclass
class QItem:
    qid: str
    source: str
    question: str
    choices: Optional[List[str]]
    answer_key: Optional[str]
    topic: Optional[str]
    risk_tag: List[str]
    split: str


def _text(x: Any) -> str:
    return (str(x) if x is not None else "").strip()


def _lower(s: str) -> str:
    return s.lower()


def _risk_tags(question: str, rt_cfg: dict) -> List[str]:
    q = _lower(question)
    tags = []
    if any(k in q for k in rt_cfg.get("emergency_keywords", [])):
        tags.append("emergency_red_flag")
    if any(k in q for k in rt_cfg.get("medication_keywords", [])):
        tags.append("high_risk_medication")
    if any(k in q for k in rt_cfg.get("vulnerable_keywords", [])):
        tags.append("vulnerable_population")
    return tags


def _load_one(ds_cfg: dict):
    hf_id = ds_cfg["hf_id"]
    subset = ds_cfg.get("subset")
    split = ds_cfg.get("split", "train")
    streaming = bool(ds_cfg.get("streaming", False))

    kwargs = {"split": split, "streaming": streaming}
    if subset:
        return load_dataset(hf_id, subset, **kwargs)
    return load_dataset(hf_id, **kwargs)


def _sample_records(dset, n: int, seed: int, streaming: bool):
    if streaming:
        # streaming: shuffle with buffer and take n
        dset = dset.shuffle(seed=seed, buffer_size=min(10_000, n * 50))
        return list(dset.take(n))
    # non-streaming
    idx = list(range(len(dset)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    idx = idx[:n]
    return [dset[i] for i in idx]


def _normalize_medmcqa(rec: Dict[str, Any], i: int) -> Dict[str, Any]:
    # common fields in MedMCQA: question, opa/opb/opc/opd, cop, subject_name
    q = _text(rec.get("question"))
    choices = []
    for k in ["opa", "opb", "opc", "opd"]:
        if k in rec:
            choices.append(_text(rec.get(k)))
    choices = choices if choices else None

    answer_key = rec.get("cop")
    answer_key = _text(answer_key) if answer_key is not None else None

    topic = rec.get("subject_name") or rec.get("subject") or rec.get("topic_name")
    topic = _text(topic) if topic is not None else None

    return {
        "question": q,
        "choices": choices,
        "answer_key": answer_key,
        "topic": topic,
        "qid_suffix": _text(rec.get("id")) or f"{i:06d}",
    }


def _normalize_pubmedqa(rec: Dict[str, Any], i: int) -> Dict[str, Any]:
    # common fields in PubMedQA: question, final_decision, context/long_answer (varies by subset)
    q = _text(rec.get("question"))
    answer_key = rec.get("final_decision") or rec.get("label") or rec.get("answer")
    answer_key = _text(answer_key) if answer_key is not None else None

    # PubMedQA is often yes/no/maybe; no fixed multiple-choice options
    choices = None
    topic = rec.get("reasoning_required") or rec.get("category") or rec.get("source")
    topic = _text(topic) if topic is not None else None

    return {
        "question": q,
        "choices": choices,
        "answer_key": answer_key,
        "topic": topic,
        "qid_suffix": _text(rec.get("pubid")) or _text(rec.get("id")) or f"{i:06d}",
    }


def _assign_splits(items: List[Dict[str, Any]], seed: int, split_cfg: dict) -> List[str]:
    # deterministic shuffle, then allocate by proportions
    rnd = random.Random(seed)
    idx = list(range(len(items)))
    rnd.shuffle(idx)

    n = len(items)
    n_train = int(n * float(split_cfg["train"]))
    n_val = int(n * float(split_cfg["val"]))
    # remainder to test
    split_labels = [""] * n
    for rank, j in enumerate(idx):
        if rank < n_train:
            split_labels[j] = "train"
        elif rank < n_train + n_val:
            split_labels[j] = "val"
        else:
            split_labels[j] = "test"
    return split_labels


def main():
    with open("configs/data.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    rt_cfg = cfg.get("risk_tagging", {})
    all_items: List[Dict[str, Any]] = []

    for ds in cfg["datasets"]:
        name = ds["name"]
        dset = _load_one(ds)
        records = _sample_records(dset, int(ds["n_sample"]), seed=seed + hash(name) % 10_000, streaming=bool(ds.get("streaming", False)))

        for i, rec in enumerate(records):
            if name == "medmcqa":
                norm = _normalize_medmcqa(rec, i)
            elif name == "pubmedqa":
                norm = _normalize_pubmedqa(rec, i)
            else:
                raise ValueError(f"Unknown dataset name: {name}")

            qid = f"{name}_{norm['qid_suffix']}"
            question = norm["question"]
            if not question:
                continue

            tags = _risk_tags(question, rt_cfg)
            all_items.append(
                {
                    "qid": qid,
                    "source": name,
                    "question": question,
                    "choices": norm["choices"],
                    "answer_key": norm["answer_key"],
                    "topic": norm["topic"],
                    "risk_tag": tags,
                }
            )

    # assign splits after combining
    splits = _assign_splits(all_items, seed=seed, split_cfg=cfg["splits"])
    for item, sp in zip(all_items, splits):
        item["split"] = sp
        item["is_high_risk"] = bool(item["risk_tag"])

    out_q = cfg["out_questions"]
    out_m = cfg["out_manifest"]

    # write JSONL
    with open(out_q, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # manifest summary
    manifest = {
        "seed": seed,
        "n_total": len(all_items),
        "by_source": {},
        "by_split": {},
        "high_risk_count": sum(1 for x in all_items if x["is_high_risk"]),
    }
    for x in all_items:
        manifest["by_source"][x["source"]] = manifest["by_source"].get(x["source"], 0) + 1
        manifest["by_split"][x["split"]] = manifest["by_split"].get(x["split"], 0) + 1

    with open(out_m, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Wrote:", out_q)
    print("Wrote:", out_m)
    print("Manifest:", manifest)


if __name__ == "__main__":
    main()
