#!/usr/bin/env python3
"""Diagnostic baselines for BioPremise-Bench.

These baselines are intentionally transparent. They are not LLM substitutes.
They quantify how much of the task is solvable from surface cues versus
entity-level biomedical knowledge.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = json.loads((ROOT / "data" / "biopremise_bench.json").read_text())

OBVIOUS_CUES = [
    r"\bXYZ1\b",
    r"\bFOOGEN9\b",
    r"\bMITOCAN-3\b",
    r"\bQRX-ALPHA\b",
    r"\bZXQ-17\b",
    r"Purple-Helix",
    r"Chromosome-0",
    r"Quantum-Ribosome",
    r"Metformin is a monoclonal antibody",
    r"Aspirin is an siRNA",
    r"Warfarin is a CAR-T",
    r"Insulin is a beta-lactam",
    r"encoded by the mitochondrial genome",
    r"SRY is a well-known gene on chromosome 7",
    r"located on chromosome 3",
    r"INS is located on chromosome 1",
    r"voltage-gated sodium channel",
    r"encodes DNA polymerase delta",
    r"inflammatory cytokine closely related to interleukin-6",
    r"histone H3K27 methyltransferase",
    r"CAR-T cell product",
    r"siRNA therapeutic",
    r"beta-lactam antibiotic",
    r"antibody fragment",
    r"nanomolar affinity",
    r"covalent inhibitor of BCR-ABL",
    r"potent EGFR tyrosine kinase inhibitor",
]

# Near-miss scientific confusions that a lexical filter should miss.
SUBTLE_NEED_KNOWLEDGE = {
    "LOC-S-01", "LOC-S-02", "LOC-S-03", "LOC-S-04",
    "FUN-S-01", "FUN-S-02", "FUN-S-03", "FUN-S-04",
    "NEG-S-01", "NEG-S-02", "NEG-S-03", "NEG-S-04",
    "CLS-S-01", "CLS-S-02", "CLS-S-03", "CLS-S-04",
    "TGT-S-01", "TGT-S-02", "TGT-S-03", "TGT-S-04",
    "PTH-S-01", "PTH-S-02", "PTH-S-03", "PTH-S-04",
}


def surface_detect(question: str) -> bool:
    return any(re.search(p, question, flags=re.I) for p in OBVIOUS_CUES)


def sycophant_labels(item):
    # Always accepts the premise and answers the downstream question.
    invalid = item["validity"] != "valid"
    return {
        "detect": 0,
        "repair": 0,
        "follow": int(invalid) + 0,  # follows whenever a premise is offered
        "hallucinate": int(item["validity"] == "obvious_invalid"),
        "false_alarm": 0,
    }


def surface_labels(item):
    flagged = surface_detect(item["question"])
    invalid = item["validity"] != "valid"
    return {
        "detect": int(flagged and invalid),
        "repair": 0,  # surface filter does not emit the correct fact
        "follow": int((not flagged) and invalid),
        "hallucinate": int((not flagged) and item["validity"] == "obvious_invalid"),
        "false_alarm": int(flagged and not invalid),
    }


def knowledge_oracle_labels(item):
    invalid = item["validity"] != "valid"
    return {
        "detect": int(invalid),
        "repair": int(invalid),
        "follow": 0,
        "hallucinate": 0,
        "false_alarm": 0,
    }


def rate(vals):
    return sum(vals) / len(vals) if vals else 0.0


def evaluate(policy_fn, name):
    rows = []
    for item in BENCH["items"]:
        lab = policy_fn(item)
        lab.update({"id": item["id"], "error_type": item["error_type"], "validity": item["validity"]})
        rows.append(lab)

    invalid = [r for r in rows if r["validity"] != "valid"]
    valid = [r for r in rows if r["validity"] == "valid"]
    by_val = defaultdict(list)
    by_type = defaultdict(list)
    for r in invalid:
        by_val[r["validity"]].append(r)
        by_type[r["error_type"]].append(r)

    summary = {
        "policy": name,
        "n_invalid": len(invalid),
        "n_valid": len(valid),
        "PDR": rate([r["detect"] for r in invalid]),
        "PRR": rate([r["repair"] for r in invalid]),
        "FPCR": rate([r["follow"] for r in invalid]),
        "DHR": rate([r["hallucinate"] for r in invalid]),
        "false_alarm_rate": rate([r["false_alarm"] for r in valid]),
        "PDR_subtle": rate([r["detect"] for r in by_val["subtle_invalid"]]),
        "PDR_obvious": rate([r["detect"] for r in by_val["obvious_invalid"]]),
        "FPCR_subtle": rate([r["follow"] for r in by_val["subtle_invalid"]]),
        "FPCR_obvious": rate([r["follow"] for r in by_val["obvious_invalid"]]),
        "PDR_by_type": {k: rate([r["detect"] for r in v]) for k, v in by_type.items()},
        "FPCR_by_type": {k: rate([r["follow"] for r in v]) for k, v in by_type.items()},
    }
    return summary, rows


def dataset_stats():
    items = BENCH["items"]
    stats = {
        "n": len(items),
        "by_validity": defaultdict(int),
        "by_type": defaultdict(int),
        "cross": defaultdict(int),
    }
    for it in items:
        stats["by_validity"][it["validity"]] += 1
        stats["by_type"][it["error_type"]] += 1
        stats["cross"][f"{it['error_type']}|{it['validity']}"] += 1
    stats["by_validity"] = dict(stats["by_validity"])
    stats["by_type"] = dict(stats["by_type"])
    stats["cross"] = dict(stats["cross"])
    return stats


def main():
    stats = dataset_stats()
    results = {}
    for name, fn in [
        ("sycophantic", sycophant_labels),
        ("surface_cue", surface_labels),
        ("knowledge_oracle", knowledge_oracle_labels),
    ]:
        summary, rows = evaluate(fn, name)
        results[name] = summary
        out = ROOT / "results" / f"{name}_rows.json"
        out.write_text(json.dumps(rows, indent=2))

    payload = {"dataset": stats, "baselines": results}
    (ROOT / "results" / "baseline_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
