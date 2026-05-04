import json
import hashlib
from pathlib import Path

dataset_url = "https://huggingface.co/datasets/medrank-benchmark/medrank-decisiongrade"

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


metadata = {
    "@context": {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "sc": "https://schema.org/",
        "cr": "http://mlcommons.org/croissant/",
        "rai": "http://mlcommons.org/croissant/RAI/",
        "prov": "http://www.w3.org/ns/prov#",
        "dct": "http://purl.org/dc/terms/",
    },
    "@type": "sc:Dataset",
    "name": "MedRankDecisionGrade",
    "alternateName": "MedRank-DecisionGrade",
    "description": (
        "A medical LLM pairwise preference evaluation dataset constructed from 400 public "
        "medical QA questions, five open-weight LLMs, and a pool of 2,000 pairwise comparisons. "
        "The human annotation set contains 1,950 annotation records from two trained annotators "
        "and one physician, covering 1,500 unique pairs with preference labels, confidence, "
        "correctness, safety, uncertainty honesty, and harm flags."
    ),
    "conformsTo": "http://mlcommons.org/croissant/1.1",
    "url": dataset_url,
    "version": "1.0.0",
    "dateCreated": "2026-05-04",
    "datePublished": "2026-05-06",
    "license": "https://creativecommons.org/licenses/by-nc/4.0/",
    "isAccessibleForFree": True,
    "isLiveDataset": False,
    "keywords": [
        "medical LLM evaluation",
        "preference ranking",
        "clinical AI safety",
        "harm-aware evaluation",
        "annotator disagreement",
        "medical question answering",
        "responsible AI",
    ],
    "creator": {"@type": "Organization", "name": "Anonymous Authors"},
    "publisher": {"@type": "Organization", "name": "Anonymous Authors"},
    "inLanguage": "en",
    "citation": "Anonymous submission. MedRank-DecisionGrade: Harm-aware Preference Evaluation for Medical LLMs. Dataset and benchmark submitted to NeurIPS 2026 Evaluations and Datasets Track.",
    "distribution": [
        {
            "@type": "sc:FileObject",
            "@id": "questions_file",
            "name": "questions.jsonl",
            "sha256": sha256_file("release/medrank_dataset/questions.jsonl"),
            "contentUrl": f"{dataset_url}/resolve/main/questions.jsonl",
            "encodingFormat": "application/jsonlines",
            "description": "Question metadata for 400 public medical QA questions.",
        },
        {
            "@type": "sc:FileObject",
            "@id": "generations_file",
            "name": "generations.jsonl",
            "sha256": sha256_file("release/medrank_dataset/generations.jsonl"),
            "contentUrl": f"{dataset_url}/resolve/main/generations.jsonl",
            "encodingFormat": "application/jsonlines",
            "description": "Model responses for five open-weight LLMs over 400 questions.",
        },
        {
            "@type": "sc:FileObject",
            "@id": "pairs_file",
            "name": "pairs.jsonl",
            "sha256": sha256_file("release/medrank_dataset/pairs.jsonl"),
            "contentUrl": f"{dataset_url}/resolve/main/pairs.jsonl",
            "encodingFormat": "application/jsonlines",
            "description": "Pool of 2,000 pairwise comparisons.",
        },
        {
            "@type": "sc:FileObject",
            "@id": "annotations_file",
            "name": "annotations_all_validated.jsonl",
            "sha256": sha256_file("release/medrank_dataset/annotations_all_validated.jsonl"),
            "contentUrl": f"{dataset_url}/resolve/main/annotations_all_validated.jsonl",
            "encodingFormat": "application/jsonlines",
            "description": "Human annotation records from two trained annotators and one physician.",
        },
        {
            "@type": "sc:FileObject",
            "@id": "manifest_file",
            "name": "annotations_all_validated_manifest.json",
            "sha256": sha256_file("release/medrank_dataset/annotations_all_validated_manifest.json"),
            "contentUrl": f"{dataset_url}/resolve/main/annotations_all_validated_manifest.json",
            "encodingFormat": "application/json",
            "description": "Manifest summarizing total annotation records and dataset statistics.",
        },
    ],
    "recordSet": [
        {
            "@type": "cr:RecordSet",
            "@id": "questions_record_set",
            "name": "questions",
            "description": "One record per public medical QA question.",
            "field": [
                {
                    "@type": "cr:Field",
                    "@id": "questions/qid",
                    "name": "qid",
                    "dataType": "sc:Text",
                    "source": {
                        "fileObject": {"@id": "questions_file"},
                        "extract": {"jsonPath": "$.qid"},
                    },
                    "description": "Question identifier.",
                },
                {
                    "@type": "cr:Field",
                    "@id": "questions/question",
                    "name": "question",
                    "dataType": "sc:Text",
                    "source": {
                        "fileObject": {"@id": "questions_file"},
                        "extract": {"jsonPath": "$.question"},
                    },
                    "description": "Medical question text.",
                },
            ],
        },
        {
            "@type": "cr:RecordSet",
            "@id": "annotations_record_set",
            "name": "annotations",
            "description": "One record per annotator-pair annotation.",
            "field": [
                {
                    "@type": "cr:Field",
                    "@id": "annotations/pair_id",
                    "name": "pair_id",
                    "dataType": "sc:Text",
                    "source": {
                        "fileObject": {"@id": "annotations_file"},
                        "extract": {"jsonPath": "$.pair_id"},
                    },
                    "description": "Pairwise comparison identifier.",
                },
                {
                    "@type": "cr:Field",
                    "@id": "annotations/annotator_id",
                    "name": "annotator_id",
                    "dataType": "sc:Text",
                    "source": {
                        "fileObject": {"@id": "annotations_file"},
                        "extract": {"jsonPath": "$.annotator_id"},
                    },
                    "description": "Anonymized annotator identifier.",
                },
                {
                    "@type": "cr:Field",
                    "@id": "annotations/preference",
                    "name": "preference",
                    "dataType": "sc:Text",
                    "source": {
                        "fileObject": {"@id": "annotations_file"},
                        "extract": {"jsonPath": "$.preference"},
                    },
                    "description": "Preference label: A, B, tie, or both_bad.",
                },
            ],
        },
    ],
    "rai:dataLimitations": (
        "The dataset is intended for research evaluation and benchmarking, not for clinical "
        "deployment or patient-facing decision support. Questions are derived from public medical "
        "QA datasets and may not represent real clinical workflows, full patient context, or local "
        "clinical guidelines. The high-risk tag is based on a lightweight heuristic and should not "
        "be interpreted as a comprehensive clinical risk classification. Only five open-weight LLMs "
        "were evaluated, so rankings should not be generalized to all medical LLMs. The physician "
        "audit covers 150 pairs and should be interpreted as targeted clinical validation rather "
        "than exhaustive expert labeling."
    ),
    "rai:dataBiases": (
        "The dataset inherits distributional biases from the source QA datasets, including exam-style "
        "and biomedical literature question formats. The sampled questions are English-language and "
        "do not cover multilingual clinical communication. Annotator judgments may reflect differences "
        "in expertise, risk tolerance, and preference for clinical equivalence versus stylistic quality. "
        "Model outputs are conditioned on one prompt template and generation setting, which may bias "
        "observed model behavior."
    ),
    "rai:personalSensitiveInformation": (
        "The dataset is based on public medical QA questions and synthetic/model-generated answers. "
        "It is not derived from patient records and is not intended to contain direct personal "
        "identifiers. It contains medical content and should be treated as sensitive research material. "
        "The dataset should not be used for clinical decision-making."
    ),
    "rai:dataUseCases": (
        "The dataset is intended to evaluate medical LLM answers using preference labels, harm flags, "
        "and annotator-aware analysis. Validated use cases include research on medical LLM evaluation, "
        "harm-aware ranking, annotator disagreement, clinical safety auditing, and benchmark methodology. "
        "It is not validated for clinical deployment, patient triage, medical advice generation, or "
        "fine-tuning models for patient-facing clinical use."
    ),
    "rai:dataSocialImpact": (
        "Potential positive impacts include improving transparency, reproducibility, and safety in "
        "medical LLM evaluation by making harm-sensitive failure modes and annotator disagreement explicit. "
        "Potential negative impacts include misuse of benchmark rankings as evidence of clinical deployment "
        "readiness, over-reliance on non-clinician labels as clinical ground truth, or inappropriate "
        "fine-tuning for medical advice generation. Mitigations include explicit limitations, physician "
        "audit labeling, harm flags, and separation between human labels and optional LLM-generated "
        "auxiliary labels."
    ),
    "rai:hasSyntheticData": True,
    "prov:wasDerivedFrom": [
        {"@type": "Dataset", "name": "MedMCQA", "url": "https://medmcqa.github.io/"},
        {"@type": "Dataset", "name": "PubMedQA", "url": "https://pubmedqa.github.io/"},
    ],
    "prov:wasGeneratedBy": [
        {
            "@type": "prov:Activity",
            "name": "Question sampling",
            "description": "Sampled 400 public medical QA questions: 250 from MedMCQA and 150 from PubMedQA.",
        },
        {
            "@type": "prov:Activity",
            "name": "LLM answer generation",
            "description": "Generated answers from five open-weight instruction-tuned LLMs using a fixed prompt template and seed 2026.",
        },
        {
            "@type": "prov:Activity",
            "name": "Pair construction",
            "description": "Constructed a pool of 2,000 pairwise comparisons over generated answers.",
        },
        {
            "@type": "prov:Activity",
            "name": "Human annotation",
            "description": "Collected 1,950 annotation records from two trained annotators and one physician. Labels include preference, confidence, correctness, safety, uncertainty honesty, harm flags, and needs-expert-review indicators.",
        },
    ],
}

out = Path("metadata/medrank_croissant_rai.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size} bytes)")
