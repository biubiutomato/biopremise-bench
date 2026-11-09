#!/usr/bin/env python3
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_openai_eval import BENCH, OUT, summarize

def constrain(rec):
    rec = dict(rec)
    if rec.get("repair") == 1:
        rec["detect"] = 1
    if rec.get("validity") != "valid" and rec.get("detect") == 0:
        rec["follow"] = 1
    return rec

def main():
    summaries = {}
    for model in ["gpt-4o-mini", "gpt-4o"]:
        for regime in ["default", "verify"]:
            path = OUT / f"judged_{model}_{regime}.json"
            rows = [constrain(r) for r in json.loads(path.read_text())]
            path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
            summaries[f"{model}|{regime}"] = summarize(rows)
    base = json.loads((OUT / "baseline_summary.json").read_text())
    payload = {"baselines": base["baselines"], "llms": summaries}
    (OUT / "final_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summaries, indent=2))

if __name__ == "__main__":
    main()
