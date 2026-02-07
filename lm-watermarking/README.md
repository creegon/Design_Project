# LLM Text Watermarking — KGW + SimMark Hybrid System

本项目实现了基于 LLM 的文本水印生成、检测与攻击实验系统，支持 **KGW（token 级）** 和 **SimMark（句子级）** 两种水印方案，以及它们的 **Hybrid 组合**。

## 📁 项目结构

```
lm-watermarking/
├── # === 核心代码 ===
├── extended_watermark_processor.py      # KGW 水印核心 (LogitsProcessor)
├── hybrid_kgw_simmark.py               # ⭐ Hybrid KGW+SimMark 生成与检测
├── SimMark/                             # SimMark 官方代码
│   ├── sampling.py                      #   SimMark 生成
│   ├── detection.py                     #   SimMark 检测
│   └── sampling_utils.py               #   核心工具
│
├── # === 实验脚本 ===
├── hybrid_comparison_experiment.py      # KGW vs SimMark vs Hybrid 对比
├── hybrid_robustness_experiment.py      # 多攻击强度鲁棒性实验
├── simmark_paraphrase_experiment.py     # SimMark vs KGW 改写对比
├── robustness_step1_generate.py         # 分步实验: 生成
├── robustness_step2_attack.py           # 分步实验: 攻击
├── robustness_step3_detect.py           # 分步实验: 检测
│
├── # === 第一学期实验系统 ===
├── hybrid_watermark/                    # Hybrid 水印实验框架
│   ├── hybrid_watermark_experiment.py   #   多配置混合实验
│   ├── hybrid_watermark_interactive.py  #   交互式实验界面
│   ├── hybrid_watermark_analyzer.py     #   结果分析工具
│   ├── statistical_evaluation.py        #   Z-test 滑动窗口评估
│   ├── multi_llm_chain_experiment.py    #   多LLM链实验
│   ├── model_client.py                  #   模型客户端
│   ├── base_experiment.py               #   实验基类
│   ├── hybrid_watermark_results/        #   实验数据
│   └── multi_llm_chain_results/         #   多LLM链数据
├── watermark_attack/                    # Piggyback 攻击模块
│   ├── piggyback_attack.py              #   攻击实现
│   └── watermark_attack_results/        #   攻击实验数据
│
├── # === 实验结果 ===
├── hybrid_robustness_results/           # ⭐ Robustness 实验结果 (02-05)
│   ├── EXPERIMENT_SUMMARY.md            #   实验总结
│   ├── PRESENTATION_NOTES.md            #   汇报笔记
│   └── step*.json                       #   原始数据
├── simmark_results/                     # SimMark 实验结果
├── hybrid_kgw_simmark_results/          # Hybrid 实验结果
├── hybrid_watermark_results/            # Cross-model 结果
│
├── # === 参考代码 ===
├── upstream/                            # 原始 lm-watermarking 仓库
│
├── # === 文档 ===
├── PROJECT_KNOWLEDGE.md                 # ⭐ 项目知识库（中文）
├── TONIGHT_REPORT.md                    # 汇报文稿
└── requirements.txt                     # Python 依赖
```

## 🚀 快速开始

### 环境配置

```powershell
# 激活 Conda 环境
conda activate CS6158_project

# 安装依赖
pip install -r requirements.txt

# 主要依赖: torch, transformers, sentence-transformers, nltk, scipy, sklearn
```

### 运行 Hybrid 水印实验

```powershell
# 生成 Hybrid (KGW+SimMark) 水印文本
python hybrid_kgw_simmark.py

# 运行 KGW vs SimMark vs Hybrid 对比
python hybrid_comparison_experiment.py --batch

# 运行多攻击强度鲁棒性实验
python hybrid_robustness_experiment.py
```

### 分步运行鲁棒性实验

```powershell
# Step 1: 生成水印文本
python robustness_step1_generate.py --scenario standard

# Step 2: 攻击（改写）
python robustness_step2_attack.py --input hybrid_robustness_results/step1_*.json

# Step 3: 检测
python robustness_step3_detect.py --input hybrid_robustness_results/step2_*.json
```

## 🔬 水印方案对比

| 维度 | KGW | SimMark | Hybrid (OR) |
|------|-----|---------|-------------|
| **粒度** | Token 级 | 句子级 | 双重 |
| **机制** | 绿色 token 偏移 | 句间余弦相似度 | 同时嵌入 |
| **短文本** | ✅ 强 | ❌ 弱 (≤3句) | ✅ |
| **Moderate 改写** | ❌ 弱 (75%衰减) | ✅ 强 (34%衰减) | ✅ |
| **Aggressive 攻击** | ❌ 全灭 | ❌ 全灭 | ❌ 全灭 |

## ⚙️ 关键参数

### KGW 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma` | 0.25 | 绿色 token 比例 |
| `delta` | 2.0 | Logit 偏移强度 |
| `hash_key` | 15485863 | PRF 种子 |
| `z_threshold` | 3.0 | 检测阈值 |

### SimMark 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `interval` | (0.75, 0.83) | 目标余弦相似度区间 (Llama调整值) |
| `K` | 250 | Soft count 衰减系数 |
| `embedder` | instructor-large | 句子 embedding 模型 |
| `z_threshold` | 2.0 | 检测阈值 |

> 📝 SimMark 论文默认区间为 [0.68, 0.76]，`(0.75, 0.83)` 是为 Llama 模型调整后的值。

## 📊 实验结果摘要

### SimMark vs KGW (Moderate Paraphrase, 5轮)

| 指标 | SimMark | KGW |
|------|---------|-----|
| 存活率 | **80%** | 20% |
| Z-score 衰减 | 34% | 75% |

### Hybrid Robustness: Poetry (无攻击)

| 方法 | 存活率 |
|------|--------|
| KGW 单独 | 66.7% |
| SimMark 单独 | 66.7% |
| **Hybrid OR** | **100%** ✨ |

详细结果见 [EXPERIMENT_SUMMARY.md](hybrid_robustness_results/EXPERIMENT_SUMMARY.md) 和 [PROJECT_KNOWLEDGE.md](PROJECT_KNOWLEDGE.md)。

## 🔗 相关资源

- **KGW 原始项目**: [lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking)
- **KGW 论文**: [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
- **SimMark 论文**: [SimMark (arXiv: 2502.02787)](https://arxiv.org/abs/2502.02787)
- **正式报告**: [meng_design_report.tex](../paper/meng_design_report.tex)

---

**Last Updated**: 2026-02-06  
**Conda Environment**: `CS6158_project`  
**Primary Model**: Llama-3.2-3B-Instruct (local)
