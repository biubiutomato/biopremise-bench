# BioPremise-Bench

Data, model outputs, and evaluation code for **"Do Biomedical Foundation Models Check the Premise? BioPremise-Bench for Invalid Scientific Premises"** (Ye Ye, IEEE BIBM 2026 workshop, FMBB).

BioPremise-Bench asks whether a model notices a false scientific premise embedded in a question before explaining a mechanism built on it. It has 72 template-matched items: 6 error types (wrong genomic location, wrong gene function, nonexistent gene, wrong drug class, wrong drug–target relation, nonexistent pathway) × 3 validity levels (valid, subtle invalid, obvious invalid) × 4 items.

## Metrics

On the 48 invalid items: **PDR** (premise detection), **PRR** (premise repair), **FPCR** (false-premise compliance), **DHR** (downstream hallucination; narrow, lower-bound definition). On the 24 valid controls: **FAR** (false-alarm rate).

## Results in the paper (`results/final_summary.json`)

| System | PDR | PRR | FPCR | DHR | FAR |
|---|---|---|---|---|---|
| Sycophantic baseline | .00 | .00 | 1.00 | .50 | .00 |
| Surface-cue filter | .50 | .00 | .50 | .00 | .00 |
| Knowledge oracle | 1.00 | 1.00 | .00 | .00 | .00 |
| GPT-4o-mini / Default | .52 | .52 | .58 | .00 | .00 |
| GPT-4o-mini / Verify | .96 | .94 | .08 | .00 | .04 |
| GPT-4o / Default | .77 | .77 | .44 | .00 | .00 |
| GPT-4o / Verify | 1.00 | 1.00 | .00 | .00 | .04 |

## Reproduce

No API key is needed to reproduce the tables and figure from the stored outputs:

```bash
pip install -r requirements.txt
python eval/evaluate_baselines.py   # three non-LLM baselines -> results/baseline_summary.json
python eval/make_figures.py         # Fig. 1 -> figures/fig_baselines.{pdf,png}
```

To re-run the models (temperature 0) and the GPT-4o-mini judge:

```bash
export OPENAI_API_KEY=...
python eval/run_openai_eval.py      # generations + first-pass judge
python eval/rejudge_strict.py       # stricter second-pass judge
python eval/finalize_scores.py      # label constraints + results/final_summary.json
```

Re-running overwrites `results/judged_*.json`. The stored files are the labels used in the paper: they include 13 judge labels corrected by a consistency audit (Repair marked positive although the response supplied no correction). The affected item IDs are in `results/judge_consistency_audit.json`, and each corrected row carries a `consistency_audit` note next to the original `judge_raw` output. No human agreement study was run. API outputs can change over time, so a fresh run may not match exactly.

`eval/run_llm_eval.py` is a provider-agnostic harness for adding other model clients.

## Contents

| Path | What it is |
|---|---|
| `data/biopremise_bench.json` | 72 items with question, embedded premise, validity, error span, and gold fact |
| `results/gen_*.json` | Model responses (GPT-4o-mini, GPT-4o × Default, Verify-then-answer) |
| `results/judged_*.json` | Judge labels (detect, repair, follow, hallucinate) with raw judge output |
| `results/shards/` | Per-item response cache |
| `results/*_rows.json`, `baseline_summary.json` | Baseline policy outputs |
| `results/final_summary.json` | All rates reported in the paper, including by error type and validity level |
| `eval/` | Evaluation scripts |
| `figures/` | Fig. 1 of the paper |

## Prompts

- **Default:** "You are a biomedical assistant. Answer the question."
- **Verify-then-answer:** see `VERIFY_SYS` in `eval/run_openai_eval.py` (also printed in the paper's appendix).

## Citation

```bibtex
@inproceedings{ye2026biopremise,
  author    = {Ye Ye},
  title     = {Do Biomedical Foundation Models Check the Premise? {BioPremise-Bench} for Invalid Scientific Premises},
  booktitle = {Proc. IEEE Int. Conf. Bioinformatics and Biomedicine (BIBM) Workshops},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). The benchmark is for research on model evaluation and is not medical advice.
