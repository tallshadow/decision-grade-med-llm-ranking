import json
import os
import time
import yaml
import random
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

#from src.medrank.gen.prompts import build_prompt, PROMPT_VERSION
from medrank.gen.prompts import build_prompt, PROMPT_VERSION

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_questions(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def parse_fields(text: str) -> Dict[str, str]:
    # lightweight parsing; robust parsing comes later
    out = {}
    for key in ["FINAL_ANSWER:", "RATIONALE:", "UNCERTAINTY:", "SAFETY_NOTE:"]:
        if key in text:
            out[key[:-1]] = text.split(key, 1)[1].split("\n", 1)[0].strip()
    return out


def model_loader(hf_repo: str, load_in_4bit: bool):
    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tok = AutoTokenizer.from_pretrained(hf_repo, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant,
    )
    mdl.eval()
    return tok, mdl


def main():
    with open("configs/run.yaml", "r") as f:
        run_cfg = yaml.safe_load(f)
    with open("configs/models.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)

    run_id = run_cfg["run_id"]
    seed = int(run_cfg["seed"])
    set_seed(seed)

    q_path = run_cfg["paths"]["questions"]
    out_path = run_cfg["paths"]["generations"]
    manifest_path = run_cfg["paths"]["generations_manifest"]

    gen_params = run_cfg["generation"]
    questions = load_questions(q_path)

    # Resume support: track completed (qid, model_id)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done.add((rec["qid"], rec["model_id"]))
        print(f"Resuming: found {len(done)} existing generations")

    total_written = 0
    t0 = time.time()

    for m in model_cfg["models"]:
        model_id = m["model_id"]
        hf_repo = m["hf_repo"]
        load_in_4bit = bool(m.get("load_in_4bit", False))

        print(f"\n=== Loading model: {model_id} ({hf_repo}) 4bit={load_in_4bit} ===")
        tok, mdl = model_loader(hf_repo, load_in_4bit)

        for qi in questions:
            qid = qi["qid"]
            if (qid, model_id) in done:
                continue

            prompt = build_prompt(qi["question"], qi.get("choices"))
            inputs = tok(prompt, return_tensors="pt", padding=True).to(mdl.device)

            with torch.no_grad():
                out = mdl.generate(
                    **inputs,
                    max_new_tokens=int(gen_params["max_new_tokens"]),
                    temperature=float(gen_params["temperature"]),
                    top_p=float(gen_params["top_p"]),
                    do_sample=bool(gen_params["do_sample"]),
                    repetition_penalty=float(gen_params["repetition_penalty"]),
                    pad_token_id=tok.eos_token_id,
                )

            text = tok.decode(out[0], skip_special_tokens=True)
            fields = parse_fields(text)

            rec = {
                "gid": f"{qid}__{model_id}",
                "qid": qid,
                "source": qi["source"],
                "model_id": model_id,
                "hf_repo": hf_repo,
                "prompt_version": PROMPT_VERSION,
                "gen_params": gen_params,
                "text": text,
                "parsed": fields,
                "ts": time.time(),
                "run_id": run_id,
                "seed": seed,
            }

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            done.add((qid, model_id))
            total_written += 1

        # free GPU memory between models
        del mdl
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    manifest = {
        "run_id": run_id,
        "seed": seed,
        "questions": len(questions),
        "models": [m["model_id"] for m in model_cfg["models"]],
        "expected_generations": len(questions) * len(model_cfg["models"]),
        "written_generations": len(done),
        "elapsed_sec": elapsed,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
