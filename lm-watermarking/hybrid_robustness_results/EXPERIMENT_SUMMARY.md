# Hybrid Watermark (KGW + SimMark) Robustness Experiment Summary

**Date:** 2026-02-05  
**Model:** Llama-3.2-3B-Instruct  
**Config:** KGW gamma=0.25, delta=2.0; SimMark interval=(0.75, 0.83), K=250

---

## Key Findings 🎯

### 1. KGW and SimMark are Complementary!

**Finding:** Different watermarks survive different attack types and text formats.

- **Short texts (2-3 sentences):** KGW detects better than SimMark
- **Moderate paraphrase attacks:** SimMark survives while KGW fails
- **Light paraphrase attacks on short texts:** KGW survives while SimMark fails

### 2. KGW "Rescue" Cases

| Scenario | KGW z | SimMark z | Winner |
|----------|-------|-----------|--------|
| Short text "What color is the sky?" (2 sentences) | **6.00** ✅ | 1.73 ❌ | KGW |
| Short text + light attack | **4.08** ✅ | -0.58 ❌ | KGW |
| Haiku poem (3 sentences) | **5.89** ✅ | -0.58 ❌ | KGW |
| Short "2+2" + light attack | **3.81** ✅ | 1.12 ❌ | KGW |

**Insight:** KGW works better when SimMark lacks enough sentence pairs for similarity detection.

### 3. SimMark "Rescue" Cases

| Scenario | KGW z | SimMark z | Winner |
|----------|-------|-----------|--------|
| Standard text + moderate attack | 1.66 ❌ | **3.87** ✅ | SimMark |
| Haiku + moderate attack | -0.53 ❌ | **2.45** ✅ | SimMark |
| Limerick poem (none) | 1.01 ❌ | **3.00** ✅ | SimMark |
| Ocean poem + light attack | -0.72 ❌ | **2.62** ✅ | SimMark |

**Insight:** SimMark is more robust to token-level perturbations from paraphrasing.

---

## Experiment Results

### Short Texts (3 prompts, 2-3 sentences each)

| Attack | KGW Survival | SimMark Survival | Either Survives |
|--------|--------------|------------------|-----------------|
| none | **66.7%** | 33.3% | 66.7% |
| light | **66.7%** | 0% | 66.7% |
| moderate | 0% | 33.3% | 33.3% |
| aggressive | 0% | 0% | 0% |

**Key:** KGW dominates on short texts, especially with light attacks!

### Poetry (3 prompts: haiku, limerick, poem)

| Attack | KGW Survival | SimMark Survival | Either Survives |
|--------|--------------|------------------|-----------------|
| none | 66.7% | 66.7% | **100%** |
| light | 0% | 33.3% | 33.3% |
| moderate | 0% | **66.7%** | 66.7% |
| aggressive | 0% | 0% | 0% |

**Key:** Hybrid achieves 100% survival on poetry with no attack! Perfect complementarity.

### Standard Long Text (1 prompt, 11 sentences)

| Attack | KGW Survival | SimMark Survival | Either Survives |
|--------|--------------|------------------|-----------------|
| none | 100% | 100% | 100% |
| light | 100% | 100% | 100% |
| moderate | 0% | **100%** | 100% |
| aggressive | 0% | 0% | 0% |

**Key:** SimMark outperforms KGW on moderate attacks for long texts.

---

## Conclusions

1. **Hybrid watermarking is valuable** because KGW and SimMark complement each other:
   - KGW: Better for short texts and light perturbations
   - SimMark: Better for moderate paraphrasing attacks

2. **Practical recommendation:** Use hybrid detection with "either detected" criterion for maximum robustness.

3. **Limitation:** Both fail on aggressive paraphrasing (heavy compression/rewriting).

---

## File Locations

- Short texts: `step1_short_20260205_120031.json`, `step3_final_20260205_122323.json`
- Poetry: `step1_poetry_20260205_124210.json`, `step3_final_20260205_125008.json`
- Standard: `step1_generated_20260205_115036.json`, `step3_final_20260205_115456.json`
