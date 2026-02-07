# Piggyback Watermark Attack Module

This module simulates robustness “piggyback” attacks: modifying an already watermarked text while keeping detection positive so that malicious or misleading content can be passed off as a legitimate model output.

## Layout

- `piggyback_attack.py` – core attack utilities (generation, detection, insertions, replacements, numeric heuristics, presets support)
- `piggyback_presets.py` – ready-to-use attack scenarios (e.g., Appendix F demo)
- `watermark_attack_interactive.py` – interactive CLI for running custom or preset attacks
- `__init__.py` – exports the public symbols

## Quick Start

```bash
python watermark_attack/watermark_attack_interactive.py --model llama-2-7b
```

The CLI guides you through:

1. Generating or loading a watermarked text
2. Choosing manual configuration or a preset scenario
3. Applying insertions, direct replacements, antonym/synonym flips, and optional numeric heuristics
4. Saving experiment artifacts under `watermark_attack/watermark_attack_results/`

> ⚠️ **Safety Notice:** These tools are for research/defense evaluation only. Do not use them for malicious purposes.*** End Patch
