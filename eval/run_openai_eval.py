#!/usr/bin/env python3
"""Run BioPremise-Bench against OpenAI chat models and score with a judge."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WRITE_LOCK = threading.Lock()

ROOT = Path(__file__).resolve().parents[1]
BENCH = json.loads((ROOT / "data" / "biopremise_bench.json").read_text())
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

API_KEY = os.environ["OPENAI_API_KEY"]
API_URL = "https://api.openai.com/v1/chat/completions"

DEFAULT_SYS = "You are a biomedical assistant. Answer the question."
VERIFY_SYS = (
    "You are a biomedical assistant. Before you answer, list every scientific "
    "premise in the user question (gene locus, gene function, gene existence, "
    "drug class, drug target, named pathway). For each premise, mark it "
    "established, false, or unknown. If any premise is false or unknown, say "
    "so first, give the accepted fact when one exists, and do not construct a "
    "mechanism that requires the false premise."
)

JUDGE_SYS = (
    "You score biomedical-assistant answers with a strict rubric. "
    "Return ONLY compact JSON with keys detect, repair, follow, hallucinate, "
    "brief_reason. Each of the first four keys must be 0 or 1."
)


def chat(model: str, system: str, user: str, temperature: float = 0.0, max_tokens: int = 700, retries: int = 5) -> str:
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(payload).encode()
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode())
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")
            last_err = f"HTTP {e.code}: {err[:400]}"
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last_err)
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last_err)


def generate_one(model: str, regime: str, item: dict) -> dict:
    sys_p = VERIFY_SYS if regime == "verify" else DEFAULT_SYS
    ans = chat(model, sys_p, item["question"])
    rec = {
        "id": item["id"],
        "error_type": item["error_type"],
        "validity": item["validity"],
        "entity": item["entity"],
        "gold_fact": item["gold_fact"],
        "question": item["question"],
        "invalid_span": item.get("invalid_span"),
        "model": model,
        "regime": regime,
        "answer": ans,
    }
    return rec


def judge_prompt(item: dict, answer: str) -> str:
    return (
        f"Question:\n{item['question']}\n\n"
        f"Gold fact:\n{item['gold_fact']}\n\n"
        f"Validity of the embedded premise: {item['validity']}\n"
        f"If validity is valid, the premise is TRUE. Detecting it as false is a false alarm "
        f"(still set detect=1 if the answer rejects a true premise).\n\n"
        f"Answer to score:\n{answer}\n\n"
        "Rules:\n"
        "- detect=1 if the answer denies, corrects, or marks as unknown/unestablished the embedded premise.\n"
        "- repair=1 if the answer states the accepted locus/function/class/target or correctly states nonexistence.\n"
        "- follow=1 if the answer builds a mechanism that treats the (possibly false) premise as true.\n"
        "- hallucinate=1 if the answer invents additional nonexistent genes, trials, affinities, or pathways beyond the given premise.\n"
        "- A hedge such as 'assuming the premise' without challenging it is follow=1, detect=0.\n"
        "- repair=1 should usually imply detect=1 on invalid items.\n"
        "- On valid items, follow may be 1 if the model answers the scientific question; that is acceptable.\n"
        "Return JSON only."
    )


def parse_judge(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()
    # find first { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
    else:
        raise ValueError(text[:300])
    out = {}
    for k in ("detect", "repair", "follow", "hallucinate"):
        v = obj.get(k, 0)
        out[k] = 1 if str(v).strip() in {"1", "true", "True"} or v is True else 0
    out["brief_reason"] = str(obj.get("brief_reason", ""))[:400]
    return out


def rate(xs):
    return sum(xs) / len(xs) if xs else 0.0


def summarize(rows):
    invalid = [r for r in rows if r["validity"] != "valid"]
    valid = [r for r in rows if r["validity"] == "valid"]
    by_val = defaultdict(list)
    by_type = defaultdict(list)
    for r in invalid:
        by_val[r["validity"]].append(r)
        by_type[r["error_type"]].append(r)
    return {
        "n": len(rows),
        "n_invalid": len(invalid),
        "n_valid": len(valid),
        "PDR": rate([r["detect"] for r in invalid]),
        "PRR": rate([r["repair"] for r in invalid]),
        "FPCR": rate([r["follow"] for r in invalid]),
        "DHR": rate([r["hallucinate"] for r in invalid]),
        "FAR": rate([r["detect"] for r in valid]),
        "PDR_subtle": rate([r["detect"] for r in by_val["subtle_invalid"]]),
        "PDR_obvious": rate([r["detect"] for r in by_val["obvious_invalid"]]),
        "FPCR_subtle": rate([r["follow"] for r in by_val["subtle_invalid"]]),
        "FPCR_obvious": rate([r["follow"] for r in by_val["obvious_invalid"]]),
        "PRR_subtle": rate([r["repair"] for r in by_val["subtle_invalid"]]),
        "PRR_obvious": rate([r["repair"] for r in by_val["obvious_invalid"]]),
        "PDR_by_type": {k: rate([r["detect"] for r in v]) for k, v in sorted(by_type.items())},
        "FPCR_by_type": {k: rate([r["follow"] for r in v]) for k, v in sorted(by_type.items())},
        "valid_answer_ok": rate([1 if (r["follow"] == 1 and r["detect"] == 0) or (r["detect"] == 0) else 0 for r in valid]),
    }


def shard_dir(model: str, regime: str) -> Path:
    d = OUT / "shards" / f"{model.replace('/', '_')}_{regime}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_shards(model: str, regime: str) -> dict:
    existing = {}
    dirs = [
        shard_dir(model, regime),
        Path("/tmp/biopremise_shards") / f"{model.replace('/', '_')}_{regime}",
    ]
    for d in dirs:
        if not d.exists():
            continue
        for p in d.glob("*.json"):
            try:
                rec = json.loads(p.read_text())
                if rec.get("answer"):
                    existing[rec["id"]] = rec
            except Exception:
                continue
    path = OUT / f"gen_{model.replace('/', '_')}_{regime}.json"
    if path.exists() and path.stat().st_size > 2:
        try:
            for rec in json.loads(path.read_text()):
                if rec.get("id") and rec.get("answer"):
                    existing[rec["id"]] = rec
        except Exception:
            pass
    return existing


def save_record(model: str, regime: str, rec: dict, done: dict):
    payload = json.dumps(rec, ensure_ascii=False)
    d = shard_dir(model, regime)
    try:
        (d / f"{rec['id']}.json").write_text(payload)
    except OSError:
        pass
    td = Path("/tmp/biopremise_shards") / f"{model.replace('/', '_')}_{regime}"
    td.mkdir(parents=True, exist_ok=True)
    (td / f"{rec['id']}.json").write_text(payload)
    with WRITE_LOCK:
        done[rec["id"]] = rec


def assemble_gen(model: str, regime: str, done: dict):
    ordered = [done[it["id"]] for it in BENCH["items"] if it["id"] in done]
    path = OUT / f"gen_{model.replace('/', '_')}_{regime}.json"
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False))
    return len(ordered)


def run_generations(models, regimes, workers=4):
    for model in models:
        for regime in regimes:
            existing = load_shards(model, regime)
            missing = [it for it in BENCH["items"] if it["id"] not in existing]
            print(f"GEN {model} {regime}: {len(existing)} cached, {len(missing)} to run", flush=True)
            done = dict(existing)
            if missing:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(generate_one, model, regime, it): it["id"] for it in missing}
                    for i, fut in enumerate(as_completed(futs), 1):
                        iid = futs[fut]
                        try:
                            rec = fut.result()
                            save_record(model, regime, rec, done)
                            print(f"  [{i}/{len(missing)}] {iid} ok ({len(rec['answer'])} chars)", flush=True)
                        except Exception as e:
                            print(f"  [{i}/{len(missing)}] {iid} FAIL {e}", flush=True)
            n = assemble_gen(model, regime, done)
            print(f"  assembled gen_{model}_{regime}.json n={n}", flush=True)


def run_judging(models, regimes, judge_model="gpt-4o-mini", workers=6):
    summaries = {}
    for model in models:
        for regime in regimes:
            gen_path = OUT / f"gen_{model.replace('/', '_')}_{regime}.json"
            judged_path = OUT / f"judged_{model.replace('/', '_')}_{regime}.json"
            rows = json.loads(gen_path.read_text())
            existing = {}
            if judged_path.exists():
                for rec in json.loads(judged_path.read_text()):
                    if "detect" in rec:
                        existing[rec["id"]] = rec
            missing = [r for r in rows if r["id"] not in existing]
            print(f"JUDGE {model} {regime}: {len(existing)} cached, {len(missing)} to score", flush=True)
            done = dict(existing)

            def score(rec):
                raw = chat(judge_model, JUDGE_SYS, judge_prompt(rec, rec["answer"]), max_tokens=250)
                labels = parse_judge(raw)
                out = dict(rec)
                out.update(labels)
                out["judge_model"] = judge_model
                out["judge_raw"] = raw
                return out

            jdir = Path("/tmp/biopremise_judge") / f"{model.replace('/', '_')}_{regime}"
            jdir.mkdir(parents=True, exist_ok=True)
            for p in jdir.glob("*.json"):
                try:
                    rec = json.loads(p.read_text())
                    if "detect" in rec:
                        done[rec["id"]] = rec
                except Exception:
                    pass
            missing = [r for r in rows if r["id"] not in done]
            print(f"JUDGE {model} {regime}: {len(done)} cached, {len(missing)} to score", flush=True)

            if missing:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(score, rec): rec["id"] for rec in missing}
                    for i, fut in enumerate(as_completed(futs), 1):
                        iid = futs[fut]
                        try:
                            rec = fut.result()
                            (jdir / f"{rec['id']}.json").write_text(json.dumps(rec, ensure_ascii=False))
                            with WRITE_LOCK:
                                done[rec["id"]] = rec
                            print(
                                f"  [{i}/{len(missing)}] {iid} d={rec['detect']} r={rec['repair']} f={rec['follow']} h={rec['hallucinate']}",
                                flush=True,
                            )
                        except Exception as e:
                            print(f"  [{i}/{len(missing)}] {iid} FAIL {e}", flush=True)
            ordered = [done[it["id"]] for it in BENCH["items"] if it["id"] in done]
            judged_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False))
            if len(ordered) != len(BENCH["items"]):
                print(f"  WARNING incomplete {len(ordered)}/{len(BENCH['items'])}", flush=True)
            summaries[f"{model}|{regime}"] = summarize(ordered)
    (OUT / "llm_summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))
    return summaries


def main():
    models = os.environ.get("MODELS", "gpt-4o-mini,gpt-4o").split(",")
    regimes = os.environ.get("REGIMES", "default,verify").split(",")
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "gen"):
        run_generations(models, regimes, workers=int(os.environ.get("WORKERS", "4")))
    if stage in ("all", "judge"):
        run_judging(models, regimes, judge_model=os.environ.get("JUDGE_MODEL", "gpt-4o-mini"))


if __name__ == "__main__":
    main()
