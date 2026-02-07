# 🎤 Design Project 进度汇报 — 2026-02-06

**汇报人**: Han Li (hl2595)  
**项目**: LLM Text Watermarking — Hybrid KGW+SimMark 方案

---

## 📋 本阶段工作概述

上次汇报以来完成了三项核心工作：

1. **整合 SimMark 句子级水印** — 引入第二种水印机制，弥补 KGW 的弱点
2. **实现真正的 Hybrid (KGW+SimMark) 方案** — 同时嵌入两种水印信号
3. **完成 Robustness 鲁棒性对比实验** — 验证两种方法的互补性

---

## 🔍 SimMark 简介

**SimMark** (Aghdam et al., 2025, arXiv: 2502.02787) 是一种**句子级语义水印**方法，与 KGW 的 token 级水印形成互补。

### 核心思想

| 维度 | KGW | SimMark |
|------|-----|---------|
| **工作粒度** | Token 级 | 句子级 |
| **嵌入机制** | 绿色 token logit 偏移 | 句间余弦相似度落入目标区间 |
| **生成方式** | LogitsProcessor 修改概率分布 | Rejection Sampling 逐句筛选 |
| **检测方式** | 统计绿色 token 比例 → z-score | 统计句对在区间内的比例 → z-score |
| **鲁棒性** | 词汇替换即失效 | 对 moderate 改写鲁棒，但怕句子合并 |

### 工作流程

```
生成阶段:
  句子₁ 生成 → 句子₂ 生成 → 计算 cos_sim(句子₁, 句子₂) 
                                  ↓
                        在目标区间 (0.75, 0.83)?
                           ✅ 接受     ❌ 拒绝重生成
```

```
检测阶段:
  文本 → 逐句分割 → 计算所有相邻句对的 cos_sim
       → 统计落在 (0.75, 0.83) 内的比例 → z-test
       → z > 2.0 → 判定有水印
```

**关键参数**:
- Embedder: `hkunlp/instructor-large`
- 论文默认区间: [0.68, 0.76]；Llama 调整后: (0.75, 0.83)
- Soft count 衰减系数 K = 250

---

## 🔧 Hybrid 方案实现

### 设计思路

KGW 和 SimMark 各有弱点——**能不能同时嵌入两种水印，检测时互相兜底？**

### 实现方式

```
Hybrid 生成 (hybrid_kgw_simmark.py):
  每个 token → 用 KGW LogitsProcessor 生成（嵌入绿色 token bias）
  每个句子完成后 → 检查 SimMark 约束（句间 cos_sim ∈ (0.75, 0.83)）
                    不满足 → 拒绝，重新用 KGW 生成该句
  最终文本 = 同时包含 KGW + SimMark 信号

Hybrid 检测:
  OR 逻辑（推荐）: KGW 或 SimMark 任一检测到 → 有水印
  AND 逻辑（严格）: 两者都检测到 → 高置信度有水印
```

---

## 📊 实验结果

### 实验1: SimMark vs KGW 单独对比（Moderate Paraphrase, 5轮）

| 指标 | SimMark | KGW |
|------|---------|-----|
| **存活率** | **80%** | 20% |
| 原始 z-score 均值 | 4.98 | 4.55 |
| 改写后 z-score 均值 | 3.28 | 1.15 |
| z-score 衰减 | 34% | **75%** |

> 💡 SimMark 对 moderate paraphrase 明显更鲁棒——因为改写保留了语义，句间相似度模式不变。

### 实验2: Hybrid Robustness 三类场景（2026-02-05）

**配置**: KGW γ=0.25, δ=2.0; SimMark interval=(0.75, 0.83), K=250

#### 短文本（3 prompts, 2-3 句）

| 攻击 | KGW | SimMark | **Hybrid OR** |
|------|-----|---------|---------------|
| 无 | **66.7%** | 33.3% | 66.7% |
| 轻微 | **66.7%** | 0% | 66.7% |
| 中等 | 0% | 33.3% | 33.3% |
| 强烈 | 0% | 0% | 0% |

> KGW 擅长短文本——SimMark 统计量不够（≤3句）

#### 诗歌（haiku / limerick / poem）

| 攻击 | KGW | SimMark | **Hybrid OR** |
|------|-----|---------|---------------|
| 无 | 66.7% | 66.7% | ✨ **100%** |
| 轻微 | 0% | 33.3% | 33.3% |
| 中等 | 0% | **66.7%** | 66.7% |
| 强烈 | 0% | 0% | 0% |

> 🌟 **无攻击下 Hybrid 达到 100%！** 两种方法完美互补。

#### 标准长文本（1 prompt, 11 句）

| 攻击 | KGW | SimMark | **Hybrid OR** |
|------|-----|---------|---------------|
| 无 | 100% | 100% | 100% |
| 轻微 | 100% | 100% | 100% |
| 中等 | 0% | **100%** | **100%** |
| 强烈 | 0% | 0% | 0% |

> SimMark 在长文本 + moderate 攻击下完胜——KGW 的绿 token 全被替换了，但句间语义模式还在。

---

## ⚡ 关键发现："救场"案例

### KGW 救场（SimMark 失败的情况）

| 场景 | KGW z | SimMark z | 原因 |
|------|-------|-----------|------|
| "What color is the sky?" (2句) | **6.00** ✅ | 1.73 ❌ | 句子太少，SimMark 无法统计 |
| 短文本 + light 攻击 | **4.08** ✅ | -0.58 ❌ | 同上 |
| Haiku 诗歌 (3句) | **5.89** ✅ | -0.58 ❌ | 诗歌结构不规则 |

### SimMark 救场（KGW 失败的情况）

| 场景 | KGW z | SimMark z | 原因 |
|------|-------|-----------|------|
| 长文本 + moderate 攻击 | 1.66 ❌ | **3.87** ✅ | 改写破坏 token 模式，但保留语义 |
| Haiku + moderate 攻击 | -0.53 ❌ | **2.45** ✅ | 同上 |
| Limerick (无攻击) | 1.01 ❌ | **3.00** ✅ | 诗歌 token 分布不规则 |

---

## 📌 结论与实用建议

### 核心结论

1. **KGW 和 SimMark 在不同维度互补** — KGW 强于短文本/light攻击，SimMark 强于长文本/moderate攻击
2. **Hybrid OR 逻辑是最优检测策略** — 最大化覆盖率
3. **Aggressive 改写仍然全灭** — 两种方法都扛不住句子合并+大幅压缩

### 实用建议

```
生成端: 可以只嵌入一种水印（节省计算）
检测端: 必须用 Hybrid OR 逻辑！
        if kgw_z > 3.0 or simmark_z > 2.0:
            → 判定有水印
```

---

## 🚧 局限性与未来工作

| 局限 | 说明 |
|------|------|
| 样本量小 | Short/Poetry 各 3 个 prompt，缺少置信区间 |
| Aggressive 攻击全灭 | 需要更高语义层次的水印（如 SemaMark） |
| Hybrid 生成慢 | Rejection sampling 最坏 50 次重试/句 |
| 只测了 Llama | 应扩展到更多模型 |

### 下一步计划

1. 扩大样本量，计算置信区间
2. 跑 batch 模式 Hybrid 对比实验
3. 优化 Hybrid 生成效率（SentBS / IterGen 方案）
4. 撰写正式报告（已完成初稿更新）

---

*准备者: Han Li | 2026-02-06*
