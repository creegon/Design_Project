# 混合水印实验系统使用指南

## 📚 概述

混合水印实验系统是一个用于研究多种水印组合方案的工具集，基于 Llama 2 7B 模型实现。系统提供了四种主要的混合水印实验类型。

## 🎯 实验类型

### 1. 片段级混合水印 (Fragment Mixing)

**原理**: 在同一段落中，不同片段使用不同的水印配置（gamma, delta, hash_key）

**应用场景**:
- 研究混合水印的可检测性
- 探索不同水印强度的组合效果
- 分析部分水印文本的检测特性

**示例**:
```python
fragment_configs = [
    {'gamma': 0.25, 'delta': 2.0, 'hash_key': 15485863},  # 片段1: 标准配置
    {'gamma': 0.5, 'delta': 2.0, 'hash_key': 15485863},   # 片段2: 高gamma
    {'gamma': 0.25, 'delta': 3.0, 'hash_key': 15485863},  # 片段3: 强水印
]
```

**研究问题**:
- 混合水印文本的整体检测率如何？
- 哪种配置在混合文本中更容易被检测？
- 不同片段配置如何相互影响？

### 2. 种子混合 (Seed Mixing)

**原理**: 同一模型使用不同的 hash_key 生成多个文本变体

**应用场景**:
- 研究不同种子之间的相互检测能力
- 探索水印的唯一性和区分度
- 分析密钥泄露的风险

**示例**:
```python
# 生成3个使用不同hash_key的变体
num_variations = 3
# 自动生成: 15485863, 16485863, 17485863
```

**研究问题**:
- 使用密钥A的检测器能检测到密钥B的水印吗？
- 不同种子生成的文本有多大区别？
- 交叉检测矩阵的对角线特性如何？

### 3. 参数混合 (Parameter Mixing)

**原理**: 使用不同的 gamma 和 delta 组合生成文本片段

**应用场景**:
- 研究参数配置对检测的影响
- 探索最优参数组合
- 分析参数鲁棒性

**示例**:
```python
gamma_values = [0.25, 0.5]        # 2个gamma值
delta_values = [1.0, 2.0, 3.0]    # 3个delta值
# 总共生成 2×3=6 个参数组合
```

**研究问题**:
- 不同 gamma 值的检测器对混合文本的检测效果？
- delta 值如何影响混合水印的强度？
- 参数不匹配时的检测性能如何？

### 4. 密钥共享混合 (Key Sharing)

**原理**: 部分文本使用共享密钥，部分使用独立密钥

**应用场景**:
- 研究密钥共享策略
- 探索协作水印方案
- 分析多源文本的溯源能力

**示例**:
```python
# 4个文本，奇数索引用共享密钥，偶数索引用独立密钥
prompts = [
    "Prompt 1",  # 独立密钥
    "Prompt 2",  # 共享密钥
    "Prompt 3",  # 独立密钥
    "Prompt 4",  # 共享密钥
]
```

**研究问题**:
- 共享密钥能否检测到所有使用该密钥的文本？
- 独立密钥文本会被共享密钥误检吗？
- 密钥共享策略的安全性如何？

## 🚀 快速开始

### 方式一：运行完整实验套件

```powershell
# 运行所有四种实验
python hybrid_watermark_experiment.py
```

这将依次运行所有实验并保存结果到 `hybrid_watermark_results/` 目录。

### 方式二：交互式实验

```powershell
# 启动交互式界面
python hybrid_watermark_interactive.py
```

交互式界面允许你：
- 选择实验类型
- 自定义参数
- 实时查看结果
- 保存实验数据

### 方式三：分析已有结果

```powershell
# 分析实验结果
python hybrid_watermark_analyzer.py
```

## 📊 文件说明

| 文件 | 功能 | 使用场景 |
|------|------|----------|
| `hybrid_watermark_experiment.py` | 核心实验类 | 批量运行实验 |
| `hybrid_watermark_interactive.py` | 交互式界面 | 自定义实验参数 |
| `hybrid_watermark_analyzer.py` | 结果分析工具 | 分析和比较结果 |

## 💡 使用示例

### 示例1：片段级混合实验

```python
from hybrid_watermark_experiment import HybridWatermarkExperiment

# 初始化
exp = HybridWatermarkExperiment(model_name="meta-llama/Llama-2-7b-hf")

# 定义片段配置
configs = [
    {'gamma': 0.25, 'delta': 2.0, 'hash_key': 15485863},
    {'gamma': 0.5, 'delta': 2.0, 'hash_key': 15485863},
    {'gamma': 0.25, 'delta': 3.0, 'hash_key': 15485863},
]

# 运行实验
result = exp.experiment_fragment_mixing(
    base_prompt="The future of AI is",
    fragment_configs=configs,
    tokens_per_fragment=50
)

# 保存和打印结果
exp.print_summary(result)
exp.save_results(result)
```

### 示例2：种子混合实验

```python
# 生成3个不同种子的变体
result = exp.experiment_seed_mixing(
    prompt="Write a story about robots:",
    num_variations=3,
    max_new_tokens=100
)

# 查看交叉检测结果
for cd in result['cross_detection']:
    print(f"文本 {cd['text_id']} (key={cd['text_key']}):")
    for det in cd['detections']:
        print(f"  检测器 {det['detector_key']}: {det['prediction']}")
```

### 示例3：分析实验结果

```python
from hybrid_watermark_analyzer import HybridWatermarkAnalyzer

analyzer = HybridWatermarkAnalyzer()

# 加载结果
result = analyzer.load_result("fragment_mixing_20251023_120000.json")

# 生成分析报告
analyzer.generate_report(result)
```

## 📈 结果解读

### 片段混合结果

```
检测结果示例:
  配置1 (γ=0.25, δ=2.0): z=5.23, 检测=True
  配置2 (γ=0.5, δ=2.0):  z=6.45, 检测=True
  配置3 (γ=0.25, δ=3.0): z=7.89, 检测=True

检测成功率: 100%
```

**解释**: 所有配置都成功检测到，说明混合水印保持了检测能力。

### 种子混合结果

```
交叉检测矩阵:
文本/检测器 | Key1 | Key2 | Key3 |
   文本1    |  ✓   |  ✗   |  ✗   |
   文本2    |  ✗   |  ✓   |  ✗   |
   文本3    |  ✗   |  ✗   |  ✓   |

正确检测: 3/3
误检: 0/6
```

**解释**: 对角线全为✓，说明各密钥互不干扰，水印唯一性好。

### 参数混合结果

```
不同Gamma值的平均Z分数:
  γ=0.25: 5.67
  γ=0.5:  6.89
```

**解释**: 较高的gamma值产生更高的Z分数，检测更容易。

### 密钥共享结果

```
共享密钥文本:
  正确密钥检测率: 100%
  共享密钥检测率: 100%

独立密钥文本:
  正确密钥检测率: 100%
  被共享密钥误检率: 0%
```

**解释**: 共享密钥能准确识别自己的文本，不会误检其他文本。

## 🔧 高级配置

### 自定义水印处理器

```python
processor = exp.create_watermark_processor(
    gamma=0.3,           # 自定义gamma
    delta=2.5,           # 自定义delta
    seeding_scheme="minhash",  # 使用minhash方案
    hash_key=12345678    # 自定义密钥
)
```

### 自定义检测器

```python
detector = exp.create_watermark_detector(
    gamma=0.3,
    seeding_scheme="minhash",
    hash_key=12345678
)
```

### 修改生成参数

```python
text = exp.generate_with_watermark(
    prompt="Your prompt",
    watermark_processor=processor,
    max_new_tokens=150,   # 生成更多token
    temperature=0.8       # 更高的温度
)
```

## 📊 实验数据格式

所有实验结果保存为JSON格式，包含以下字段：

```json
{
  "experiment_type": "fragment_mixing",
  "base_prompt": "...",
  "fragments": [...],
  "combined_text": "...",
  "detection_results": [...],
  "timestamp": "..."
}
```

## 🎓 研究建议

### 实验设计建议

1. **片段混合**: 
   - 建议使用3-5个不同配置的片段
   - 每个片段至少50 tokens
   - 测试极端参数组合

2. **种子混合**:
   - 建议使用3-5个不同种子
   - 种子间隔足够大（如相差100万）
   - 记录交叉检测完整矩阵

3. **参数混合**:
   - gamma范围: [0.2, 0.3, 0.4, 0.5]
   - delta范围: [1.0, 1.5, 2.0, 2.5, 3.0]
   - 注意参数组合数量指数增长

4. **密钥共享**:
   - 建议至少4个文本
   - 共享/独立比例约为1:1
   - 测试多层次的密钥共享

### 性能优化

1. **批量生成**: 使用较小的 `max_new_tokens` 加快实验
2. **GPU使用**: 确保CUDA可用以加速推理
3. **并行实验**: 可以在不同GPU上运行不同实验

### 结果验证

1. **重复实验**: 运行多次以验证结果稳定性
2. **对照组**: 始终包含不带水印的对照文本
3. **统计检验**: 使用t检验等方法验证显著性

## ⚠️ 注意事项

1. **内存管理**: 大规模实验可能占用大量内存
2. **随机性**: 由于采样，相同配置可能产生不同文本
3. **检测阈值**: 默认z_threshold=4.0，可根据需要调整
4. **文本长度**: 太短的文本可能检测不准确

## 🔍 常见问题

**Q: 为什么有些配置检测失败？**

A: 可能原因：
- 文本太短（建议至少50 tokens）
- delta值太小（建议>=1.5）
- 参数不匹配（生成和检测使用不同参数）

**Q: 如何提高检测率？**

A: 
- 增加delta值
- 使用更大的gamma值
- 生成更长的文本
- 确保参数完全匹配

**Q: 交叉检测矩阵应该是什么样的？**

A: 理想情况下，对角线应该全部为True，其他位置为False，表示每个密钥只能检测到自己的水印。

## 📚 相关文档

- 主README: `LLAMA_DEMO_README.md`
- 快速参考: `QUICK_REFERENCE.md`
- 模型配置: `llama_model_config.py`

## 🤝 贡献

如果您有新的混合水印方案或改进建议，欢迎提交！

---

**创建日期**: 2025年10月23日  
**版本**: 1.0  
**基础模型**: Llama 2 7B
