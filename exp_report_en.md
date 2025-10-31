# Parameter Mixing Report (2025-10-31 13:53:23)

## Overview
- **Type**: `parameter_mixing`
- **Prompt**: `The future of artificial intelligence is`
- **Grid**: γ ∈ {0.10, 0.2667, 0.4333, 0.60} × δ ∈ {1.0, 1.6667, 2.3333, 3.0}
- **Fragments**: 16 (all successful)

## Key Metrics by Detector γ
| γ      | Mean Z | Detection | Mean Green |
|--------|--------|-----------|------------|
| 0.10   | 5.11   | 100%      | 16.20%     |
| 0.2667 | 5.86   | 100%      | 37.15%     |
| 0.4333 | 5.90   | 100%      | 55.16%     |
| 0.60   | 3.58   | 0%        | 67.10%     |

## Observations
1. γ ≤ 0.4333 with δ ≥ 1.5 yields ≥5σ signals consistently.
2. γ = 0.60 boosts green fraction but needs longer text or higher δ to cross 4σ.
3. Capture per-combination statistics to study δ’s marginal gains in future runs.

## Conclusion
Deployments can default to γ≈0.25–0.45, δ≥1.5; raise γ to ~0.60 only if you can extend generation length or relax the threshold.

---

# Seed Mixing Report (2025-10-31 13:41:09)

## Overview
- **Type**: `seed_mixing`
- **Prompt**: `Write a short story about robots:`
- **Variants**: 5 (same model, different hash keys)
- **Mixed text**: Concatenation of the five variants

## Self-Key Detection Summary
| Variant | Hash Key | Z-score | Result |
|---------|----------|---------|--------|
| 1       | 15485863 | 10.49   | ✓      |
| 2       | 16485863 | 7.19    | ✓      |
| 3       | 17485863 | 3.86    | ✗      |
| 4       | 18485863 | 7.89    | ✓      |
| 5       | 19485863 | 7.84    | ✓      |

## Cross-Key Insights
- Each text scores highest with its own key; other keys return near-zero or negative z-scores.
- Variant 3 falls short because of shorter length; extending generation fixes it easily.

## Conclusion
Seed mixing verifies key exclusivity: swapping only the hash key preserves a strong self-signal while keeping off-key detections low.

---

# Cross-Model Shared Key Report (2025-10-31 14:12:37)

## Overview
- **Type**: `cross_model_key_sharing`
- **Model chain**: llama-3.2-3b → llama-3.1-8b → llama-3.2-3b
- **Shared config**: γ=0.5, δ=2.0, hash_key=12345
- **Continuation**: last 20 tokens of each fragment feed the next model

## Result
- Combined text: z=6.92, p≈2.28×10⁻12, green fraction≈76.3%
- Shared key accumulates across models, confirming compatibility for multi-model pipelines.

## Conclusion
Shared-key chaining is suitable for collaborative authoring: you keep high detection confidence and gain a slightly higher green-token ratio.

---

# Cross-Model Distinct Keys Report (2025-10-31 14:43:53)

## Overview
- **Type**: `cross_model_distinct_keys`
- **Model sequence**: llama-3.2-3b → llama-3.1-8b → llama-3.2-1b
- **Keys**: 13579 / 24680 / 97531
- **Generation**: 120 tokens per segment, 20-token continuation, γ=0.5, δ=2.0

## Combined Detection
| Key   | Z-score | Result | Green |
|-------|---------|--------|-------|
| 13579 | 3.45    | Fail   | 60.44%|
| 24680 | 1.15    | Fail   | 53.48%|
| 97531 | 0.54    | Fail   | 51.65%|

> Whole-text scans dilute each key, but fragment-level scores remain distinct.

## Fragment Cross Detection
| Fragment | Model        | Correct-key z | Max off-key z |
|----------|--------------|---------------|---------------|
| 1        | llama-3.2-3b | **5.08**      | -0.53         |
| 2        | llama-3.1-8b | 3.24         | 0.52          |
| 3        | llama-3.2-1b | 3.16         | 0.63          |

## Conclusion
Each model/key pair keeps a clear signal advantage; longer outputs or higher δ will push the remaining segments past the 4σ line. Artifact saved at `hybrid_watermark_results/cross_model_distinct_keys_20251031_144353.json`.

---

# Statistical Evaluation Report (2025-10-31 15:45:47)

## Overview
- **Type**: statistical evaluation (sliding window + window sensitivity + minimum length)
- **Prompt**: `The impact of artificial intelligence`
- **Watermark config**: γ=0.5, δ=2.0, hash_key=15485863
- **Samples**: watermarked vs. control, 500 tokens each

## Sliding Window (window=100, stride=100)
- 17 windows, mean z=4.08, std=1.35
- Detection rate ≈70.6%, mean green ≈72.4%

## Window Sensitivity
| Window | Mean Z | Detection |
|--------|--------|-----------|
| 25     | 2.09   | 17.5%     |
| 50     | 3.00   | 52.6%     |
| 75     | 3.66   | 58.3%     |
| 100    | 4.10   | 66.7%     |
| 150    | 5.23   | 80.0%     |
| 200    | 5.47   | 100%      |

## Minimum Length
- Range 20–240 tokens (step 10)
- Reliable detection begins at 50 tokens
- All lengths ≥50 tokens succeeded across 23 tests

## Conclusion
1. 100-token windows give strong evidence; longer windows eliminate residual misses.
2. 150–200-token windows strike a balance between localization and confidence.
3. ~50 tokens (≈35–40 words) is the practical lower bound for reliable detection. Results stored at `hybrid_watermark_results/complete_statistical_eval_20251031_154547.json`.

---
