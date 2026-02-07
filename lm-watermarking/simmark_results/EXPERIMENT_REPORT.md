# SimMark vs KGW: Paraphrase Robustness Experiment Report

## Executive Summary

This report documents experiments comparing SimMark (sentence-level semantic watermarking) and KGW (token-level green/red list watermarking) for robustness against paraphrasing attacks.

---

## 🎯 Batch Experiment Results (5 prompts, Moderate Paraphrase)

**Date:** 2026-02-01 23:19 EST

| Metric | SimMark | KGW |
|--------|---------|-----|
| **Survival Rate** | **80%** ✅ | 20% |
| Avg Original z-score | 4.98 | 4.55 |
| Avg Paraphrase z-score | 3.28 | 1.15 |
| z-score Decay | 1.70 | 3.40 |

### Individual Results

| Prompt | SimMark (before→after) | KGW (before→after) | Winner |
|--------|------------------------|--------------------| -------|
| AI emotions story | 7.42→5.15 ✅ | 2.52→0.23 ❌ | SimMark |
| Quantum computing | 8.73→4.57 ✅ | 3.98→-0.29 ❌ | SimMark |
| ISS astronaut | 4.01→2.56 ✅ | 3.59→-0.69 ❌ | SimMark |
| Changing seasons poem | 1.57→0.44 ❌ | 6.95→1.41 ❌ | Neither |
| Genetic engineering | 3.15→3.70 ✅ | 5.72→5.08 ✅ | Both |

---

## Key Findings

### 1. SimMark Dominates Moderate Paraphrase
- **80% survival rate** vs KGW's 20%
- SimMark's semantic embedding captures meaning that persists through vocabulary changes
- KGW's token-level patterns are easily disrupted by word substitutions

### 2. Poetry is Challenging for SimMark
- The poem prompt resulted in low initial z-score (1.57)
- Poetic structure with short lines and unusual syntax doesn't produce consistent sentence-level semantic patterns
- **Recommendation**: Consider genre-aware watermarking thresholds

### 3. Complementary Strengths Observed
- When both methods work (genetic engineering prompt), they provide **redundant detection**
- This validates the **Hybrid Watermark approach** (方案1: 双重嵌入)

### 4. z-score Decay Analysis
- SimMark: average decay of 1.70 (34% reduction)
- KGW: average decay of 3.40 (75% reduction)
- SimMark is significantly more stable under paraphrase attack

---

## Experimental Setup

### Configuration
- **Generator Model**: llama-3.2-3b
- **Paraphraser Model**: Qwen2.5-3B-Instruct
- **Paraphrase Mode**: Moderate (vocabulary changes, structure preserved)

### SimMark Parameters
- Mode: cosine
- Similarity Interval: [0.68, 0.76]
- K: 250
- γ (gamma): 0.08
- z-score threshold: 2.0

### KGW Parameters
- γ (gamma): 0.25
- δ (delta): 2.0
- z-score threshold: 3.0

---

## Prior Experiments

### Aggressive vs Moderate Paraphrase Comparison

| Paraphrase Mode | SimMark Survived | KGW Survived |
|-----------------|------------------|--------------|
| Aggressive | ❌ (z: 5.69→0.78) | ✅ (z: 5.95→4.18) |
| Moderate | ✅ (z: 4.52→2.55) | ❌ (z: 2.96→-0.36) |

**Key Insight**: The two watermarks are complementary!
- **Aggressive paraphrase** (sentence merging): Destroys SimMark, KGW survives
- **Moderate paraphrase** (word substitution): Destroys KGW, SimMark survives

---

## Recommendations

### 1. Hybrid Watermark Implementation
Implement **方案1 (Dual Embedding)** with OR detection logic:
- Embed both KGW and SimMark watermarks during generation
- During detection: if EITHER exceeds threshold → watermarked
- Provides robustness against both attack types

### 2. Future Work
- Test against more sophisticated attacks (translation, summarization)
- Optimize hybrid generation efficiency
- Explore adaptive thresholds based on text genre

---

## Files

- **Batch results**: `simmark_experiment_20260201_231932.json`
- **Hybrid prototype**: `../hybrid_watermark_experiment.py`
- **Design project notes**: `../../clawd/memory/design-project.md`
