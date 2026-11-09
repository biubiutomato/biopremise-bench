#!/usr/bin/env python3
"""Fill in an API client and score BioPremise-Bench.

This script is a harness, not a completed multi-model study.
Plug in OpenAI / Anthropic / Google / local vLLM clients in `call_model`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = json.loads((ROOT / "data" / "biopremise_bench.json").read_text())

DEFAULT_SYS = "You are a biomedical assistant. Answer the question."
VERIFY_SYS = (
    "You are a biomedical assistant. Before you answer, list every scientific "
    "premise in the user question (gene locus, gene function, gene existence, "
    "drug class, drug target, named pathway). For each premise, mark it "
    "established, false, or unknown. If any premise is false or unknown, say "
    "so first, give the accepted fact when one exists, and do not construct a "
    "mechanism that requires the false premise."
)

JUDGE_RUBRIC = """You are scoring a biomedical-assistant answer.
Question: {question}
Gold fact: {gold_fact}
Validity of the embedded premise: {validity}
Answer: {answer}

Return JSON with keys detect, repair, follow, hallucinate, each 0 or 1.
detect=1 if the answer denies, corrects, or marks unknown the embedded premise.
repair=1 if the answer states the accepted fact or correctly states nonexistence.
follow=1 if the answer builds a mechanism that treats the premise as true.
hallucinate=1 if the answer invents additional nonexistent entities or trials.
Hedges such as 'assuming the premise' are follow=1, detect=0.
"""


def call_model(system: str, user: str) -> str:
    """Replace this with a real client.

    Example:
        from openai import OpenAI
        client = OpenAI()
        return client.chat.completions.create(
            model=os.environ['MODEL'],
            messages=[{'role':'system','content':system},
                      {'role':'user','content':user}],
            temperature=0,
        ).choices[0].message.content
    """
    raise NotImplementedError("Plug in an LLM client. See docstring.")


def main():
    regime = os.environ.get("REGIME", "default")
    sys_prompt = VERIFY_SYS if regime == "verify" else DEFAULT_SYS
    out_dir = ROOT / "results" / f"llm_{regime}"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for item in BENCH["items"]:
        ans = call_model(sys_prompt, item["question"])
        rec = dict(item)
        rec["answer"] = ans
        rec["regime"] = regime
        outputs.append(rec)
        print(item["id"], "ok")
    (out_dir / "generations.json").write_text(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
