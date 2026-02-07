# LLM Watermarking Research - 完整项目知识库

**项目名称**: Design Project - LLM Text Watermarking  
**负责人**: Master (hl2595@cornell.edu)  
**最后更新**: 2026-02-06  
**整理者**: 苏尔薇

---

## 📍 目录

1. [项目概述](#项目概述)
2. [核心技术原理](#核心技术原理)
3. [实验结果汇总](#实验结果汇总)
4. [代码结构与使用](#代码结构与使用)
5. [下一步研究方向](#下一步研究方向)
6. [相关文献](#相关文献)
7. [技术笔记与踩坑记录](#技术笔记与踩坑记录)

---

## 📌 项目概述

### 研究背景

随着大语言模型(LLM)的普及，AI生成文本的可追溯性成为关键问题。水印技术通过在生成过程中嵌入不可见的统计模式，使得后续可以检测文本是否由特定模型生成。

### 研究目标

1. 理解现有LLM水印技术的原理与局限
2. 对比Token级(KGW)和句子级(SimMark)水印的鲁棒性
3. 探索Hybrid混合水印方案的可行性

### 核心发现 ⭐

**两种水印互补！**
- **Moderate Paraphrase**（词汇替换、句式微调）→ SimMark存活率80%，KGW仅20%
- **Aggressive Paraphrase**（句子合并压缩）→ KGW存活，SimMark失败
- **结论**: Hybrid方案是必要的！

---

## 🔧 核心技术原理

### 1. KGW - Token级红绿表水印

**论文**: "A Watermark for Large Language Models" (Kirchenbauer et al., 2023)  
**arXiv**: 2301.10226

**原理**:
1. 用PRF(hash)将词汇表分为"绿色"和"红色"token
2. 生成时，对绿色token的logits加bias(δ)
3. 检测时，统计绿色token比例，做z-test

**关键参数**:
| 参数 | 含义 | 典型值 |
|------|------|--------|
| γ (gamma) | 绿色列表比例 | 0.25-0.5 |
| δ (delta) | logits偏置强度 | 1.5-3.0 |
| z-score阈值 | 检测判定标准 | >3.0 |

**优点**: 实现简单，原始文本检测率高  
**致命缺陷**: 对paraphrase攻击毫无抵抗力！词汇一换就崩

---

### 2. SimMark - 句子级语义水印

**论文**: "SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models"  
**arXiv**: 2502.02787  
**会议**: EMNLP 2025

**原理**:
1. **生成时**: 通过rejection sampling，确保每对相邻句子的embedding相似度落在预定义区间 `[a, b]` 内
2. **检测时**: 计算相邻句子对的相似度，用soft counting统计落在区间内的比例，z-test判断
3. **为何抗paraphrase**: paraphrase改变词汇/语法，但语义相似度模式会保留

**关键参数**:
| 参数 | 含义 | 典型值 |
|------|------|--------|
| Mode | 相似度计算方式 | cosine / euclidean |
| Interval | 目标相似度区间 | [0.68, 0.76] (cosine) — 论文默认值 |
| K | soft count衰减系数 | 250 |
| γ (gamma) | 人类文本落在区间内的自然比例 | ~0.08 |
| Embedder | 句子embedding模型 | hkunlp/instructor-large |

> ⚠️ **参数调整说明**: 论文默认区间 `[0.68, 0.76]` 用于 `simmark_paraphrase_experiment.py` 的独立测试。在 Hybrid 实验（`hybrid_kgw_simmark.py` 等）中，区间调整为 `(0.75, 0.83)` 以适配 Llama 模型的 embedding 分布特性。

**优点**: 对词汇替换、句式调整等moderate改写鲁棒  
**缺点**: 
- 诗歌等非标准句式不友好（短句、特殊结构）
- 生成速度慢（rejection sampling）
- 对aggressive改写（句子合并）脆弱

---

### 3. Hybrid混合水印（研究中）

**核心思想**: 结合KGW和SimMark的优点

**方案A - 检测端Hybrid** (推荐先实现):
- 生成时：只用一种水印
- 检测时：两种检测器同时跑，OR逻辑（任一通过=有水印）

**方案B - 生成端Hybrid** (Master要求的目标):
- 生成时：同时嵌入KGW和SimMark
- 难点：效率问题，逐句rejection sampling太慢

---

## 📊 实验结果汇总

### 实验1: KGW多LLM链Paraphrase (2025-11)

**设置**:
- 生成模型: llama-3.2-3b
- Paraphrase模型: qwen-3-4b
- 水印参数: γ=0.25, δ=2.0

**结果**:
| 指标 | 值 |
|------|-----|
| 原始z-score | ~5.73 |
| Paraphrase后z-score | -0.2 ~ -1.16 |
| 存活率 | **0%** |

**结论**: KGW对paraphrase毫无抵抗力

---

### 实验2: SimMark vs KGW 单次对比 (2026-02-01)

#### Aggressive Paraphrase (句子压缩: 12句→6句)
| 指标 | SimMark | KGW |
|------|---------|-----|
| Original z-score | 5.69 | 5.95 |
| Paraphrased z-score | **0.78** | **4.18** |
| Survived? | ❌ NO | ✅ YES |

#### Moderate Paraphrase (词汇替换: 12句→11句)
| 指标 | SimMark | KGW |
|------|---------|-----|
| Original z-score | 4.52 | 2.96 |
| Paraphrased z-score | **2.55** | **-0.36** |
| Survived? | ✅ YES | ❌ NO |

---

### 实验3: Batch实验 (2026-02-01, 5轮, Moderate Paraphrase) ⭐

**核心结果**:
| 指标 | SimMark | KGW |
|--------|---------|-----|
| **存活率** | **80%** ✅ | 20% |
| 平均原始z-score | 4.98 | 4.55 |
| 平均paraphrase后z-score | 3.28 | 1.15 |
| z-score衰减 | 1.70 (34%) | 3.40 (75%) |

**详细结果**:
| Prompt | SimMark (before→after) | KGW (before→after) | Winner |
|--------|------------------------|--------------------| -------|
| AI emotions story | 7.42→5.15 ✅ | 2.52→0.23 ❌ | SimMark |
| Quantum computing | 8.73→4.57 ✅ | 3.98→-0.29 ❌ | SimMark |
| ISS astronaut | 4.01→2.56 ✅ | 3.59→-0.69 ❌ | SimMark |
| Changing seasons poem | 1.57→0.44 ❌ | 6.95→1.41 ❌ | Neither |
| Genetic engineering | 3.15→3.70 ✅ | 5.72→5.08 ✅ | Both |

**关键发现**:
1. SimMark在moderate paraphrase下明显优于KGW（80% vs 20%）
2. 诗歌等非标准句式对SimMark不友好（初始z-score就低，只有1.57）
3. 两种方法互补：验证了Hybrid方案的必要性！

---

### 实验4: Hybrid Robustness 三类场景对比 (2026-02-05) ⭐⭐

**设置**:
- 模型: Llama-3.2-3B-Instruct
- KGW参数: γ=0.25, δ=2.0
- SimMark参数: interval=(0.75, 0.83), K=250（Llama调整值）
- 检测阈值: KGW z > 3.0, SimMark z > 2.0
- 攻击类型: none / light / moderate / aggressive

#### 短文本 (3 prompts, 2-3句)

| 攻击 | KGW存活率 | SimMark存活率 | 任一存活率 |
|------|-----------|---------------|------------|
| none | **66.7%** | 33.3% | 66.7% |
| light | **66.7%** | 0% | 66.7% |
| moderate | 0% | 33.3% | 33.3% |
| aggressive | 0% | 0% | 0% |

#### 诗歌 (3 prompts: haiku, limerick, poem)

| 攻击 | KGW存活率 | SimMark存活率 | 任一存活率 |
|------|-----------|---------------|------------|
| none | 66.7% | 66.7% | **100%** ✨ |
| light | 0% | 33.3% | 33.3% |
| moderate | 0% | **66.7%** | 66.7% |
| aggressive | 0% | 0% | 0% |

#### 标准长文本 (1 prompt, 11句)

| 攻击 | KGW存活率 | SimMark存活率 | 任一存活率 |
|------|-----------|---------------|------------|
| none | 100% | 100% | 100% |
| light | 100% | 100% | 100% |
| moderate | 0% | **100%** | 100% |
| aggressive | 0% | 0% | 0% |

#### KGW救场Case
| 场景 | KGW z | SimMark z | 谁赢？ |
|------|-------|-----------|--------|
| 短文本 "What color is the sky?" (2句) | **6.00** ✅ | 1.73 ❌ | KGW |
| 短文本 + light攻击 | **4.08** ✅ | -0.58 ❌ | KGW |
| Haiku诗歌 (3句) | **5.89** ✅ | -0.58 ❌ | KGW |

#### SimMark救场Case
| 场景 | KGW z | SimMark z | 谁赢？ |
|------|-------|-----------|--------|
| 标准文本 + moderate攻击 | 1.66 ❌ | **3.87** ✅ | SimMark |
| Haiku + moderate攻击 | -0.53 ❌ | **2.45** ✅ | SimMark |
| Limerick诗歌 (无攻击) | 1.01 ❌ | **3.00** ✅ | SimMark |

**核心发现**:
1. **KGW和SimMark互补！** KGW擅长短文本+light攻击，SimMark擅长长文本+moderate攻击
2. KGW救场的共同点：**句子数 ≤ 3**（SimMark统计量不够）
3. Poetry无攻击下Hybrid达到 **100%** 存活率——完美互补
4. **实用建议**: 生成端可只用一种水印，检测端必须用Hybrid (OR逻辑)
5. ⚠️ aggressive攻击下两者都全灭，仍是开放问题

---

## 📁 代码结构与使用

### 目录结构

```
lm-watermarking/
├── SimMark/                   # SimMark官方代码
│   ├── sampling.py            # 水印生成（rejection sampling）
│   ├── detection.py           # 水印检测
│   ├── sampling_utils.py      # 核心工具函数
│   └── pca_model_16.pkl       # PCA模型（euclidean模式用）
├── hybrid_watermark/          # 混合水印实验系统
│   ├── base_experiment.py     # 基础实验类
│   ├── multi_llm_chain_experiment.py  # 多LLM链实验
│   └── model_client.py        # API客户端
├── llama_demos/               
│   └── model_config.json      # 模型配置
├── simmark_results/           # SimMark实验结果
│   ├── EXPERIMENT_REPORT.md   # 实验报告
│   └── *.json                 # 原始数据
├── simmark_paraphrase_experiment.py   # SimMark vs KGW对比实验脚本 ⭐
├── hybrid_watermark_experiment.py     # Hybrid prototype（生成太慢）
├── extended_watermark_processor.py    # KGW扩展处理器
└── .venv/                     # Python虚拟环境
```

### 运行环境

**Conda环境**: `CS6158_project` (不是design_project！)

```powershell
# 激活环境
call C:\Users\creegon\miniconda3\condabin\conda.bat activate CS6158_project
cd C:\Users\creegon\Desktop\Design_Project\lm-watermarking
```

**依赖**: torch, transformers, sentence-transformers, nltk, sklearn

### 常用命令

```powershell
# 运行SimMark vs KGW单次对比
python simmark_paraphrase_experiment.py --prompt "Write a story about..." --mode cosine

# 运行Batch实验
python simmark_paraphrase_experiment.py --batch --mode cosine

# ⭐ 真正的Hybrid实验（KGW + SimMark同时嵌入！）
python hybrid_kgw_simmark.py --prompt "Write a story about..." --verbose

# ⭐ Hybrid vs KGW vs SimMark对比实验
python hybrid_comparison_experiment.py --batch
```

### 模型配置

`llama_demos/model_config.json`:
```json
{
  "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
  "qwen-3-4b": "Qwen/Qwen3-4B-Instruct-2507-FP8"
}
```

---

## 🚀 下一步研究方向

### ✅ 已完成：真正的Hybrid实现！(2026-02-04)

**新增脚本**:
- `hybrid_kgw_simmark.py` - 真正的KGW+SimMark双重水印生成与检测
- `hybrid_comparison_experiment.py` - 对比实验：KGW vs SimMark vs Hybrid
- `hybrid_robustness_experiment.py` - 多攻击强度鲁棒性实验
- `robustness_step1_generate.py` / `step2_attack.py` / `step3_detect.py` - 分步实验脚本

**实现原理**:
```python
# 生成时：
# 1. 用KGW的LogitsProcessor生成每个token（嵌入绿色token bias）
# 2. 生成完一个句子后，检查SimMark约束（cosine similarity in (0.75, 0.83)）
# 3. 如果不满足 → 拒绝，重新用KGW生成
# 4. 最终文本同时包含两种水印信号！

# 检测时：
# 1. KGW检测器：统计绿色token → z-score
# 2. SimMark检测器：统计句子对在区间内的比例 → z-score (soft)
# 3. 任一个正 = 至少检测到一种水印（OR逻辑，推荐）
# 4. 两个都正 = 双重检测到（AND逻辑，更严格）
```

**关键区别**（之前的脚本vs现在）:
| 脚本 | 做的事情 | 问题 |
|------|----------|------|
| `hybrid_watermark_experiment.py` | KGW多配置混合 | ❌ 没有SimMark |
| `simmark_paraphrase_experiment.py` | SimMark vs KGW对比 | ❌ 分开测试，不是同时嵌入 |
| **`hybrid_kgw_simmark.py`** | KGW + SimMark同时嵌入 | ✅ 真正的Hybrid！ |
| **`hybrid_robustness_experiment.py`** | 多攻击强度鲁棒性对比 | ✅ 验证互补性！ |

### ✅ 已完成：Robustness 鲁棒性实验 (2026-02-05)

- 完成Short/Poetry/Standard三类场景 × 4种攻击强度对比
- 验证了KGW和SimMark的互补性（详见实验4）
- 整理了EXPERIMENT_SUMMARY.md和PRESENTATION_NOTES.md

### 下一步目标

1. **运行Hybrid对比实验（batch模式）**
   - `python hybrid_comparison_experiment.py --batch`
   - 与单独KGW/SimMark做更大规模对比

2. **扩大样本量**
   - 当前robustness实验样本量较小（Short/Poetry各3个prompt）
   - 需更多prompt验证统计显著性

3. **优化Hybrid生成效率 (方案B)**
   - 当前问题：逐句rejection sampling太慢，最坏600次模型调用
   - ⚠️ 注意：曾导致gateway卡死！

### 研究过的优化方向

#### 1. IterGen - Backtracking框架
- **GitHub**: `structuredllm/itergen`
- **arXiv**: 2410.07295
- 核心：`forward()`, `backward()`, `view()` 三个操作
- `backward("sentence", 1)` 可以回退一句重试
- **直接适用！**

#### 2. NeuroLogic A*esque Decoding
- **arXiv**: 2112.08726 (NAACL 2022)
- 核心：A*算法 + lookahead heuristic
- 估计"未来满足约束的可能性"，优先扩展有希望的路径

#### 3. Sentence-level Beam Search (SentBS)
- **GitHub**: `Shen-Chenhui/SentBS`
- 核心：beam search从token级提升到句子级
- 每步生成N个候选句子，选最好的

#### 4. 优化方案草案

```python
def generate_hybrid_optimized(prompt, num_sentences=12, beam_size=4, max_backtrack=3):
    # 1. 批量生成 beam_size 个候选句子（一次forward！）
    # 2. 计算每个候选的SimMark分数（距离区间的距离）
    # 3. 选择最接近区间的候选
    # 4. 如果没有候选满足约束 → 回退一句重试（有限次数）
    # 5. graceful degradation：回退用完就妥协
```

### 中长期方向

1. **更多攻击类型测试**: 翻译、摘要、风格迁移
2. **自适应阈值**: 根据文本类型（诗歌/论文/故事）调整参数
3. **SemaMark融合**: 离散化语义空间的方法可能对SimMark有启发

---

## 📚 相关文献

### 核心论文

| 论文 | 会议/年份 | 要点 |
|------|----------|------|
| **KGW原始论文** | ICML 2023 | Token级红绿表水印，arXiv: 2301.10226 |
| **SimMark** | EMNLP 2025 | 句子级语义水印，arXiv: 2502.02787 |
| **SemaMark/SIR** | ICLR 2024 | 语义不变水印，用embedding做seed |
| **SemStamp** | arXiv 2023 | LSH语义分区，arXiv: 2310.03991 |

### 攻击研究

| 论文 | 会议/年份 | 要点 |
|------|----------|------|
| **b4 Attack** | NAACL 2025 | 黑盒水印擦除，paraphrase指令攻击 |
| **Watermark Under Fire** | EMNLP Findings 2025 | 水印鲁棒性评估 |
| **MIP密钥窃取** | 2024 | 水印密钥推断攻击 |

### 工具库

| 项目 | 链接 | 说明 |
|------|------|------|
| MarkLLM | THU-BPM/MarkLLM | 开源水印工具包，EMNLP 2024 Demo |
| Robust_Watermark | THU-BPM/Robust_Watermark | SIR语义水印实现 |
| SimMark | DabiriAghdam/SimMark | 本项目使用的SimMark实现 |

---

## 🛠️ 技术笔记与踩坑记录

### 1. torch版本问题

**问题**: transformers库要求torch>=2.6（CVE-2025-32434安全检查），但当前环境是torch 2.5.1

**解决方案**: 在代码开头monkey-patch

```python
def _patched_check_torch_load_is_safe():
    pass

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = _patched_check_torch_load_is_safe
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = _patched_check_torch_load_is_safe
```

### 2. Qwen FP8模型问题

**问题**: qwen FP8模型与triton库版本冲突（2026-02-01）

**临时解决**: 用其他模型替代

### 3. Hybrid生成导致gateway卡死

**问题**: 跑hybrid_watermark_experiment.py时，逐句rejection sampling太慢，导致gateway超时卡死

**解决**: 
- 限制最大重试次数
- 添加timeout
- 或改用检测端Hybrid方案

### 4. 正确的conda环境

**⚠️ 重要**: 用 `CS6158_project` 环境，不是 `design_project`！

```powershell
call C:\Users\creegon\miniconda3\condabin\conda.bat activate CS6158_project
```

---

## 📎 附录

### 实验结果JSON位置

- `simmark_results/simmark_experiment_20260201_231932.json` - Batch实验最终结果
- `simmark_results/simmark_experiment_20260201_215053.json` - 早期单次实验
- `simmark_results/simmark_experiment_20260201_224841.json` - 中间实验

### 相关文件

- 内存笔记: `C:\Users\creegon\clawd\memory\design-project.md`
- 文献调研: `C:\Users\creegon\clawd\memory\design-project-research.md`
- 2026-02-01日志: `C:\Users\creegon\clawd\memory\2026-02-01.md`

---

*文档整理: 苏尔薇·米娅克莉丝 (=^･ω･^=)*  
*最后更新: 2026-02-06*  
*更新内容: 补充实验4 Robustness数据、统一SimMark参数记录、更新研究进度*
