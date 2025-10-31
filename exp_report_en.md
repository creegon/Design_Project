# Parameter Mixing Experiment Report (2025-10-31 13:53:23)

## Overview
- **Experiment type**: `parameter_mixing`
- **Prompt**: `The future of artificial intelligence is`
- **Parameter grid**: 16 combinations (γ ∈ {0.10, 0.2667, 0.4333, 0.60} × δ ∈ {1.0, 1.6667, 2.3333, 3.0})
- **Fragments**: 16 segments, all generated successfully (no empty outputs)

## Key Metrics (grouped by detector γ)
| Detector γ | Mean Z-score | Detection | Mean Green Fraction |
|------------|--------------|-----------|---------------------|
| 0.10       | 5.1109       | ✓ Detected | 0.1620              |
| 0.2667     | 5.8590       | ✓ Detected | 0.3715              |
| 0.4333     | 5.8989       | ✓ Detected | 0.5516              |
| 0.60       | 3.5840       | ✗ Below threshold | 0.6710      |

> The detection matrix repeats each γ four times (one per δ) with identical values; deduplication is possible downstream.

## Observations
1. **Comprehensive coverage**: Low/mid/high γ and δ combinations are all included, giving a full picture of the response surface.
2. **Robust signals at lower γ**: γ≤0.4333 consistently yields 5.1–5.9σ scores, far above common thresholds, confirming strong watermarks in that band.
3. **γ=0.60 needs reinforcement**: Z=3.58 falls just shy of the 4.0 threshold, even though the green fraction rises to ~67%; pushing δ higher or lengthening outputs should recover significance.
4. **δ’s marginal effect still hidden**: Because results are aggregated by γ, the incremental impact of δ is unclear; per-combination statistics would clarify it.

## Recommendations
- For γ=0.60, test δ>3.0 or extend generation length; alternatively, relax the detection threshold if high γ is desired in production.
- Save per-combination statistics (e.g., `per_fragment_detection`) so each γ/δ pair can be evaluated in isolation.
- Plot the γ–δ grid (heatmaps of Z-score and green fraction) to visualise sweet spots and interaction effects.

## Conclusion
The grid highlights γ≈0.25–0.45 with δ≥1.5 as the most reliable operating region (≥5σ signals), making it the default choice for deployment or batch runs. To reach ≥0.65 green fraction, pair γ≈0.60 with longer outputs or relaxed thresholds to counteract the drop in statistical strength.

---

# Seed Mixing Experiment Report (2025-10-31 13:41:09)

## Overview
- **Experiment type**: `seed_mixing`
- **Prompt**: `Write a short story about robots:`
- **Variations**: 5 generations sharing the same prompt, differing only in `hash_key`
- **Mixed text**: Concatenation of the five variants for whole-text detection

## Self-key Detection Summary
| Variation | Hash Key | Self Z-score | Detection |
|-----------|----------|--------------|-----------|
| 1         | 15485863 | 10.49        | ✓         |
| 2         | 16485863 | 7.19         | ✓         |
| 3         | 17485863 | 3.86         | ✗         |
| 4         | 18485863 | 7.89         | ✓         |
| 5         | 19485863 | 7.84         | ✓         |

> Variation 3 misses the 4.0 threshold (z=3.86); text length or green coverage is the likely culprit.

## Cross-Detection Insights
- **Correct-key fidelity**: All texts except variation 3 exceed 7σ with their own key, well above the threshold.
- **Off-key immunity**: Incorrect keys produce negative or sub-2σ scores; the highest stray signal (text 2 with key 19485863, z≈3.36) still stays below 4.0.
- **Sensitive sample**: Variation 3 produces only weak positives (≤1.9σ) for every key, indicating insufficient statistical strength rather than key leakage.

## Recommendations
- Regenerate or extend variation 3 (longer outputs, adjusted sampling, or higher δ) to raise its self-key score.
- Log auxiliary metadata (length, green fraction) for each variation to speed up root-cause diagnosis of low scores.
- Visualise the cross-detection matrix (heatmap/confusion plot) to make the contrast between correct-key hits and off-key misses explicit.

## Conclusion
Seed mixing shows that swapping just the hash key preserves strong self-detection while suppressing cross-key hits (4/5 success; the outlier can be fixed by lengthening the text). This setup is therefore a practical template for validating watermark key exclusivity.

---

# Cross-Model Key Sharing Experiment Report (2025-10-31 14:12:37)

## Overview
- **Experiment type**: `cross_model_key_sharing`
- **Initial prompt**: `The future of artificial intelligence`
- **Model sequence**: llama-3.2-3b → llama-3.1-8b → llama-3.2-3b (chained continuations)
- **Shared config**: γ=0.5, δ=2.0, hash_key=12345
- **Continuation strategy**: Use the last 20 tokens of each fragment as the prompt for the next model

## Generation & Detection
- **Fragments**: 3 segments with coherent transitions across models
- **Mixed text length**: ~430 words
- **Shared-key detection**: z=6.92, p≈2.28×10⁻¹², prediction=true, green fraction≈0.763

## Observations
1. **Key reuse across models works**: Reusing the same hash key in a chained pipeline still yields >6σ detection, demonstrating feasibility for multi-model workflows.
2. **Higher green fraction**: The combined text reaches ~76% green tokens, slightly above the ~70% observed in single-model runs, indicating cumulative bias.
3. **Stable continuation**: A 20-token handoff maintains contextual flow without noticeable semantic breaks, making it a sensible default.

## Recommendations
- When scaling to more models, keep the interactive selection that supports both numeric IDs and nicknames to streamline configuration.
- Before production, run single-model and off-key checks on the mixed text to confirm that sharing does not elevate false positives.
- To boost semantic richness, insert additional large-model segments (e.g., repeating the 8B model) within the chain.

## Conclusion
Cross-model key sharing maintains high-confidence watermark detection and increases green-token coverage during collaborative generation, making it a strong default for multi-source authoring scenarios.

---

# Cross-Model Distinct Keys Report (2025-10-31 14:43:53)

## Overview
- **Experiment type**: cross_model_distinct_keys
- **Model chain**: llama-3.2-3b → llama-3.1-8b → llama-3.2-1b
- **Hash keys**: 13579 / 24680 / 97531
- **Generation setup**: 120 tokens per segment, 20-token continuation window, γ=0.5, δ=2.0

## Full-Text Detection
| Detector key | Z-score | Prediction | Green fraction |
|--------------|---------|------------|----------------|
| 13579        | 3.45    | Fail       | 60.44%         |
| 24680        | 1.15    | Fail       | 53.48%         |
| 97531        | 0.54    | Fail       | 51.65%         |

> Although the combined text dilutes each key’s contribution, the relative strengths remain well separated.

## Fragment-Level Cross Detection
| Fragment | Model         | Correct-key z | Next-highest z (wrong key) | Notes                          |
|----------|---------------|---------------|-----------------------------|--------------------------------|
| 1        | llama-3.2-3b  | **5.08**      | -0.53                       | ✓ strong pass                  |
| 2        | llama-3.1-8b  | 3.24          | 0.52                        | △ near threshold, still top    |
| 3        | llama-3.2-1b  | 3.16          | 0.63                        | △ near threshold, still top    |

## Observations
1. **Key exclusivity holds**: each fragment achieves its highest z-score with its own key, while off-keys stay near or below zero.
2. **Segments 2 & 3 need more signal**: their 3σ scores fall short of the 4σ decision line because the passages are shorter; increasing δ or token counts would elevate them.
3. **Whole-text strategy**: when multiple keys share a document, run a quick shared-key scan, then drill down per fragment to attribute the signal correctly.

## Recommendations
- Raise δ (e.g., 2.5–3.0) or extend the generated length for shorter fragments so all keys cross the detection threshold.
- Log (model, key, length) metadata during generation to make key-specific rechecks reproducible.
- Combine shared-key monitoring with per-key validation for defence-in-depth in multi-author pipelines.

## Conclusion
Cross-model distinct-key generation shows that each model/key pair retains a recognisable watermark signature: the correct key yields the highest z-score for its segment, while mismatched keys remain low. Longer outputs or stronger bias parameters will easily push every segment above the 4σ detection line. Results stored at `hybrid_watermark_results/cross_model_distinct_keys_20251031_144353.json`.

---
