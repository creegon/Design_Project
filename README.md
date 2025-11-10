# Watermark System – Project Navigation

This project implements a watermark generation, detection, and hybrid watermark experimental system based on large language models, supporting multiple models and API providers.

## 📁 Directory Structure

```
lm-watermarking/
├── docs_llama/               # Chinese documentation & navigation 📄
├── hybrid_watermark/         # Hybrid watermark experimental system ⭐
│   ├── hybrid_watermark_experiment.py   (Core experiment)
│   ├── hybrid_watermark_interactive.py  (⭐ Interactive experiment interface)
│   ├── hybrid_watermark_analyzer.py     (Results analysis tool)
│   ├── statistical_evaluation.py        (Statistical evaluation module)
│   └── README.md                        ⭐ Directory description
├── llama_demos/              # Basic watermark demo scripts 📄
│   ├── llama_simple_example.py          (Introductory example)
│   ├── llama_watermark_demo.py          (Full demo)
│   ├── llama_interactive_demo.py        (Interactive interface)
│   ├── llama_batch_test.py              (Batch testing)
│   ├── model_config_manager.py          (⭐ Model configuration manager)
│   ├── model_config.json                (⭐ Model configuration file)
│   └── README.md                        ⭐ Directory description
├── upstream/
│   └── lm_watermarking/      # Original lm-watermarking full source code 📦
│       ├── alternative_prf_schemes.py
│       ├── experiments/
│       ├── hf_hub_space_demo/
│       ├── homoglyph_data/
│       ├── watermark_processor.py
│       ├── demo_watermark.py
│       ├── requirements.txt / setup.cfg / pyproject.toml
│       └── watermark_reliability_release/ …
├── extended_watermark_processor.py      # Custom extended processor (626 lines)
├── REPORT_LLAMAWATERMARK_LLAMA.md       # October 24 experiment report
├── SUMMARY.md                           # Project summary
└── IMPORT_FIX.md                        # Import fix notes
```

**📄 indicates that the directory already contains a README.md documentation file**  
**⭐ indicates important files or new features**  
**📦 indicates that the full upstream project is packaged inside a single directory**

> All upstream code is now consolidated under `upstream/lm_watermarking/`.  
> You can import modules using statements like  
> `from upstream.lm_watermarking import watermark_processor`.  
> Custom modules (with Chinese comments) are kept in separate subdirectories at the repository root.

## 🚀 Quick Start

### 1. Configure the Model (Required)


First `llama_demos/model_config.json`：

```json
{
  "api_providers": {
    "openai": {
      "api_key": "your-openai-api-key-or-env:OPENAI_API_KEY"
    },
    "deepseek": {
      "api_key": "env:DEEPSEEK_API_KEY",
      "api_base": "https://api.deepseek.com/v1"
    }
  },
  "models": {
    "llama-3.2-3b": {
      "model_identifier": "meta-llama/Llama-3.2-3B-Instruct",
      "nickname": "llama-3.2-3b",
      "api_provider": "deepseek"
    }
  }
}
```

### 2. Basic Watermarking

```powershell

cd llama_demos

python llama_simple_example.py llama-3.2-3b

.\run_llama_demo.ps1
```

> Tip: `llama_simple_example.py` and `llama_batch_test.py` use the **first positional argument** to specify the model nickname; there is no `--model` option.

### 3. Hybrid Watermark Experiments

```powershell
# Enter the experiment directory
cd hybrid_watermark

# Run the interactive interface (recommended)
python hybrid_watermark_interactive.py

# Or run the full experiment script
python hybrid_watermark_experiment.py


## 📚 Experiment Types

### Hybrid Watermark Experiments (3 types)

| Experiment No. | Name | Description |
|----------------|------|-------------|
| **Experiment 1** | Hybrid Configuration Experiment | Segment-level / parameter-level hybrid watermarking |
| **Experiment 2** | Key Cross-Detection | Seed-mixing / key-sharing strategies |
| **Experiment 3** | Cross-Model Shared Key | Multi-model cooperative watermarking |

### Statistical Evaluation Experiments (4 types)

| Experiment No. | Name | Description |
|----------------|------|-------------|
| **Experiment 4** | Sliding-Window Detection | Analyze uniformity of watermark signal distribution |
| **Experiment 5** | Window Sensitivity Analysis | Determine optimal window size |
| **Experiment 6** | Minimum Detectable Length | Find minimum length required for reliable detection |
| **Experiment 7** | Full Statistical Evaluation | Perform all three statistical analyses |

## 🎯 Usage Scenarios

### Scenario 1: Quick Watermark Function Test

```powershell
cd llama_demos
python llama_simple_example.py llama-3.2-3b
```

**Best for:** First-time users to understand core features

### Scenario 2: Interactive Experiment Research

```powershell
cd hybrid_watermark
python hybrid_watermark_interactive.py --model llama-3.2-3b
```

**Best for:** Researchers comparing multiple watermark schemes  
**Features:**  
- 7 experiment types (3 hybrid + 4 statistical)  
- Real-time visualization  
- Automatic result saving  

### Scenario 3: Batch Parameter Testing

```powershell
cd llama_demos
python llama_batch_test.py llama-3.2-3b
```

**Best for:** Systematic parameter comparison studies

### Scenario 4: Result Analysis

```powershell
cd hybrid_watermark
python hybrid_watermark_analyzer.py
```

**Best for:** Analyzing saved experiment outputs

## 💡 Supported Models

### API Providers
- **OpenAI**: GPT series  
- **DeepSeek**: DeepSeek series, Llama series  
- **Local Models**: Loaded via HuggingFace Transformers  

### Recommended Model Configuration

```json
{
  "models": {
    "llama-3.2-3b": {
      "model_identifier": "meta-llama/Llama-3.2-3B-Instruct",
      "api_provider": "deepseek",
      "description": "Small and efficient; recommended for daily use"
    },
    "gpt-4o-mini": {
      "model_identifier": "gpt-4o-mini",
      "api_provider": "openai",
      "description": "High-quality outputs; good for comparison studies"
    }
  }
}
```

### Model Management

```powershell
# List all configured models
cd llama_demos
python -c "from model_config_manager import ModelConfigManager; mgr = ModelConfigManager(); print(mgr.list_model_names())"

# View model details
python -c "from model_config_manager import ModelConfigManager; mgr = ModelConfigManager(); print(mgr.get_model_info_by_nickname('llama-3.2-3b'))"
```

## 🔧 Installing Dependencies

```powershell
# Method 1: Basic dependencies
cd llama_demos
pip install -r requirements_llama.txt

# Method 2: Full dependencies (recommended)
cd ..
pip install -r requirements.txt

# Main packages:
# - torch >= 2.0.0
# - transformers >= 4.30.0
# - openai >= 1.0.0
# - scipy
# - matplotlib
# - numpy
# - tqdm
```

## ⚙️ Environment Setup

### 1. API Key Configuration (recommended: environment variables)

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:DEEPSEEK_API_KEY = "your-deepseek-api-key"

# Or configure inside model_config.json
{
  "api_providers": {
    "openai": {
      "api_key": "env:OPENAI_API_KEY"
    }
  }
}
```

### 2. GPU Configuration (optional)

```python
python hybrid_watermark_interactive.py --device cuda
```

## 🆘 Frequently Asked Questions

### Q1: How do I add a new model?

```json
{
  "models": {
    "my-model": {
      "model_identifier": "organization/model-name",
      "nickname": "my-model",
      "api_provider": "openai",
      "description": "My custom model"
    }
  }
}
```

### Q2: What if detection accuracy is low?

Try the following:

1. **Increase delta** (e.g. 2.0 → 2.5) — strengthens watermark signal  
2. **Lower gamma** (e.g. 0.5 → 0.4) — improves signal-to-noise ratio  
3. **Generate longer text** — more statistical evidence  

### Q3: What is Z-score?

Z-score measures statistical significance:

- **Z = 3.0** → 99.87% confidence (**recommended**)  
- **Z = 4.0** → 99.997% confidence (too strict; deprecated)  
- **Z = 2.5** → 99.38% confidence (lenient)

Formula:  
`Z = (observed_green - expected_green) / std_dev`

### Q4: How to choose Gamma and Delta?

| Scenario | Gamma | Delta | Notes |
|----------|--------|--------|------|
| Quality-first | 0.5 | 1.5–2.0 | More natural text |
| Balanced | 0.5 | 2.0 | **Recommended default** |
| Detection-first | 0.25 | 2.5–3.0 | Strong signal, possible text impact |

### Q5: List all models?

```powershell
python -c "from model_config_manager import ModelConfigManager; print('\n'.join(ModelConfigManager().list_model_names()))"
```

### Q6: Where are results saved?

`hybrid_watermark/hybrid_watermark_results/`

- JSON data files  
- PNG visualization files  

### Q7: How to analyze existing results?

```powershell
cd hybrid_watermark
python hybrid_watermark_analyzer.py
```

## 🎓 Core Features

### Base Functionality
- ✅ Watermark generation & detection  
- ✅ Multi-model support (local/API)  
- ✅ Model configuration system  
- ✅ Interactive UI  
- ✅ Batch testing  

### Hybrid Watermark Experiments
- ✅ Segment-level hybrid  
- ✅ Parameter-grid hybrid  
- ✅ Seed variants  
- ✅ Key sharing  
- ✅ Cross-model cooperation  

### Statistical Evaluation
- ✅ Sliding-window detection  
- ✅ Window sensitivity  
- ✅ Minimum-length analysis  
- ✅ Full statistical evaluation  


### Watermark Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|----------|--------------------|
| **gamma** | Green-list ratio | 0.5 | 0.25–0.5 |
| **delta** | Logits bias strength | 2.0 | 1.5–3.0 |
| **hash_key** | PRF seed | 15485863 | Any integer |
| **z_threshold** | Detection threshold | 3.0 | 2.5–4.0 |

**Parameter Notes**:
- **gamma**: Controls the proportion of green tokens in the vocabulary; affects expected green-token rate  
- **delta**: Controls how strongly green tokens are boosted; affects actual green-token rate  
- **z_threshold**: Statistical significance threshold (optimized to 3.0 to improve detection rate)

## 📊 Experiment Results

Results are generated in the following location:

### Results Directory
- `hybrid_watermark/hybrid_watermark_results/` — all experiment outputs

### Output File Types

**JSON Files** — full data logs

```
sliding_window_20251024_143022.json
window_sensitivity_20251024_143155.json
minimum_length_20251024_143340.json
complete_statistical_eval_20251024_143512.json
```

**PNG Format** - Charts
```
sliding_window_20251024_143022.png
window_sensitivity_20251024_143155.png
minimum_length_20251024_143340.png
```

### JSON Structure

Each experiment result contains:
- `experiment_type`: experiment type identifier  
- `prompt`: prompt used  
- `watermark_config`: watermark parameter configuration  
- `generated_texts`: generated texts with full content  
- `results`: statistical analysis results  
- `detailed_results`: detailed detection data  

### Visualization Analysis

All statistical evaluation experiments automatically generate matplotlib charts:
- Z-score distribution curve  
- Detection-rate trend plot  
- Green-token ratio analysis  
- Success/Failure scatter plot  


## ✅ Project Features

### 1. Unified Model Management
- ✅ Supports multiple API providers (OpenAI, DeepSeek, etc.)
- ✅ Model nickname system for simplified usage
- ✅ Secure API key management via environment variables
- ✅ Unified configuration file `model_config.json`

### 2. Complete Experiment Framework
- ✅ 3 hybrid watermarking schemes (configuration / key / cross-model)
- ✅ 4 statistical evaluation methods (window / sensitivity / minimum length / comprehensive)
- ✅ Interactive interface with real-time feedback
- ✅ Automatic saving of JSON + PNG results

### 3. Optimized Detection Algorithm
- ✅ Z-score threshold optimization (3.0 vs 4.0)
- ✅ Improved detection sensitivity (accuracy from 40% → nearly 100%)
- ✅ Maintains low false-positive rate (<0.13%)

### 4. Visualization & Analysis
- ✅ Automatic chart generation with matplotlib
- ✅ Z-score distribution, detection rate, green-token ratio
- ✅ Success/failure scatter plots
- ✅ Cumulative detection-rate curves

### 5. Research Tools
- ✅ Sliding-window analysis of watermark uniformity
- ✅ Window-sensitivity analysis for optimal parameters
- ✅ Minimum-length analysis for detection thresholds
- ✅ Batch-experiment support for large-scale testing


## 📖 Command Quick Reference

```powershell
# 1. Configuration check
cd llama_demos
python -c "from model_config_manager import ModelConfigManager; ModelConfigManager().validate_config()"

# 2. Quick test
python llama_simple_example.py llama-3.2-3b

# 3. Interactive experiment (recommended)
cd ../hybrid_watermark
python hybrid_watermark_interactive.py --model llama-3.2-3b

# 4. Statistical evaluation (full workflow, including sliding window, etc.)
python statistical_evaluation.py --model llama-3.2-3b

# 5. Result analysis
python hybrid_watermark_analyzer.py

# 6. View help
python hybrid_watermark_interactive.py --help

```

## 🔗 Related Resources

- **Original Project**: [lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking)
- **Paper**: [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
- **Key Files**:
  - `extended_watermark_processor.py` – Watermark processor (626 lines)
  - `hybrid_watermark_interactive.py` – Interactive interface (1558 lines)
  - `model_config_manager.py` – Model manager (443 lines)


## 📝 Changelog

### Latest Version (2025-10-24)

**New Features**:
- ✅ Statistical evaluation module (4 evaluation methods)
- ✅ Z-score threshold optimization (3.0 replacing 4.0)
- ✅ Model configuration management system
- ✅ Complete JSON output (including generated text)
- ✅ Automatic visualization chart generation

**Improvements**:
- ✅ Detection accuracy significantly improved (40% → nearly 100% @ 200 tokens)
- ✅ Experiment consolidation (5 → 3 hybrid experiments)
- ✅ Interactive UI optimization (7 experiment types)

**Bug Fixes**:
- ✅ `hash_key` parameter passing error
- ✅ Overly strict Z-score threshold
- ✅ Inconsistent visualization chart thresholds

---

**Created**: October 23, 2025  
**Last Updated**: October 24, 2025  
**Recommended Model**: Llama 3.2 3B Instruct (DeepSeek API)  
**Experiment Types**: 3 hybrid experiments + 4 statistical evaluations

