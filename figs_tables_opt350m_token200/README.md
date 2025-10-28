# Reproducing Table 2 and Figure 4 (opt-350m, token200)

This directory contains aggregated results (figures + Table 2 CSV/MD) produced from a small-grid experiment on `facebook/opt-350m` using `max_new_tokens=200` ("token200"). The goal is to reproduce the paper-style Table 2 and Figure 4 (ROC overlay, z histograms, boxplots, and TPR/FPR bars) for the multinomial decoding regime.

Files produced here
- `figs/roc_figure4.png` — ROC overlay (approximate) for tested (δ,γ) combos
- `figs/z_histograms.png` — histogram of watermarked vs baseline z-scores
- `figs/z_boxplots.png` — boxplots of watermarked z by (δ,γ)
- `figs/tpr_fpr_bars.png` — bar chart of TPR@z=4/5 and baseline FPR@z=4/5
- `tables/table2_z4_z5.csv` — numerical Table 2 results (CSV)
- `tables/table2_z4_z5.md` — same table rendered as Markdown

Summary of what was run to produce these files
- Model: `facebook/opt-350m` (Hugging Face)
- Decoding: `multinomial`
- Grid: δ ∈ {1.0, 2.0, 5.0}, γ ∈ {0.25, 0.5}
- n (samples per cell): 6 (small-sample experiment)
- Generation length: `--max_new_tokens 200` (to increase T)
- Generation and detection orchestrated by `experiments/run_llama_fresh.py` (fresh single-run format)
- Aggregation performed by `experiments/make_table2_figure4.py` which reads the `all_results_*.json` files and writes the figures + `table2_z4_z5.*` files

Exact commands to reproduce (from repository root)

1) Generate outputs (example single cell). Replace `--gamma`/`--delta` per grid.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 experiments/run_llama_fresh.py \
  --model facebook/opt-350m \
  --n 6 \
  --max_new_tokens 200 \
  --decoding multinomial \
  --gamma 0.25 \
  --delta 1.0 \
  --output_dir hybrid_watermark_results_opt350m_token200
```

Run that command for each of the six grid points (order doesn't matter). The script will write per-run JSONs named like `all_results_fresh_facebook_opt-350m_<timestamp>.json` into the `--output_dir`.

2) Aggregate into Table 2 + Figure 4

```bash
PYTHONPATH=. python3 experiments/make_table2_figure4.py \
  --results_dir hybrid_watermark_results_opt350m_token200 \
  --out_dir figs_tables_opt350m_token200
```

This reads all `all_results_*.json` files in `--results_dir` and writes the `figs/` and `tables/` subfolders under `--out_dir` (the current directory already contains these outputs).

How to interpret the outputs (key takeaways from the produced `table2_z4_z5.csv`)

The CSV contains the following columns: `gamma, delta, model, decoding, auc, w_mean_z, b_mean_z, w_median_z, b_median_z, w_tpr_z4, w_tpr_z5, b_fpr_z4, b_fpr_z5, n_w, n_b`.

- `w_mean_z` / `b_mean_z`: mean z-score across watermarked / baseline samples (higher is stronger signal). In these runs with `max_new_tokens=200`:
  - δ=1.0, γ=0.25 → w_mean_z ≈ 2.83, baseline ≈ 0.07
  - δ=1.0, γ=0.5  → w_mean_z ≈ 3.78
  - δ=2.0, γ=0.25 → w_mean_z ≈ 8.85 (very strong)
  - δ=2.0, γ=0.5  → w_mean_z ≈ 5.32
  - δ=5.0, γ=0.25 → w_mean_z ≈ 14.20 (very strong)
  - δ=5.0, γ=0.5  → w_mean_z ≈ 9.29

- `w_tpr_z4` / `w_tpr_z5`: fraction of watermarked samples exceeding z thresholds 4 and 5 (true positive rate at that z). For higher δ (2.0, 5.0) most TPR@4/5 are near 1.0, even with n=6.

- `b_fpr_z4` / `b_fpr_z5`: baseline false positive rates — in these runs they are 0.0 across the board (no baseline sample exceeded z thresholds), indicating a clean separation for these settings (caveat: small n).

- `auc`: computed area under ROC (based on per-sample z); many grid cells report AUC≈1.0 or high values (perfect separation in this limited sample).

Main conclusions you can draw from these outputs
- Increasing the watermark bias parameter δ strongly increases the observed z-scores for watermarked text. With δ≥2 and long generations (200 tokens) detection is near-perfect even with only 6 samples per condition.
- The baseline remains near-zero mean z and very low FPR at z thresholds 4 and 5 in these tests — the watermark appears detectable without producing many false positives in the limited experiments.
- γ (greenlist fraction) modulates the effect: lower γ (0.25) with large δ tends to produce higher z in some cells (see δ=5.0, γ=0.25 → very large mean z), but δ is the dominant lever.

Caveats and reproducibility notes
- Small sample size: these runs use n=6 per cell. That yields noisy empirical TPR estimates; do not over-interpret point estimates (use larger n for stable estimates).
- Generation length matters: `max_new_tokens=200` strongly increases T (tokens scored) and thereby the z statistic (z scales roughly with sqrt(T)). Shorter outputs will reduce z.
- Model & tokenization: results depend on the exact model checkpoint, tokenizer version, and Hugging Face transformers version. Record model id and local package versions for exact replication.
- HF gated models: for LLaMA-style models you may need HF token and model access; `opt-350m` is public.
- Detector settings: the detector used `ignore_repeated_ngrams=True` and no extra normalizers; changing detection options changes reported z.

Repro script snippets
- Generate all six multinomial points (one-liner script):

```bash
for delta in 1.0 2.0 5.0; do
  for gamma in 0.25 0.5; do
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 experiments/run_llama_fresh.py \
      --model facebook/opt-350m --n 6 --max_new_tokens 200 --decoding multinomial \
      --gamma ${gamma} --delta ${delta} --output_dir hybrid_watermark_results_opt350m_token200
  done
done

# Aggregate
PYTHONPATH=. python3 experiments/make_table2_figure4.py --results_dir hybrid_watermark_results_opt350m_token200 --out_dir figs_tables_opt350m_token200
```

If you want, I can:
- Add this README into the repository (I can commit it),
- Run the remaining grid cells (beam8 or multinomial) and regenerate figures, or
- Increase `n` and re-run to produce statistically stable estimates.

Contact / next steps
If you want a PR with this README committed and the generation/aggregation scripts combined into one runnable workflow, tell me and I will create the patch and run a verification pass.
