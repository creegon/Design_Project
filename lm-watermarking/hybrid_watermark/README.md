# 混合水印实验系统

本目录包含混合水印实验的核心代码，实现了4种不同的混合水印方案。

## 📁 文件列表

| 文件 | 功能 | 代码行数 |
|------|------|----------|
| `hybrid_watermark_experiment.py` | 核心实验类 | ~700行 |
| `hybrid_watermark_interactive.py` | 交互式实验界面 | ~350行 |
| `hybrid_watermark_analyzer.py` | 结果分析工具 | ~400行 |

## 🎯 四种实验方案

### 1. 片段级混合 (Fragment Mixing)

在同一段落中，不同片段使用不同的水印配置。

```python
fragment_configs = [
    {'gamma': 0.25, 'delta': 2.0, 'hash_key': 15485863},
    {'gamma': 0.5, 'delta': 2.0, 'hash_key': 15485863},
    {'gamma': 0.25, 'delta': 3.0, 'hash_key': 15485863},
]
```

### 2. 种子混合 (Seed Mixing)

同一模型使用不同的hash_key生成多个变体。

```python
num_variations = 3  # 生成3个不同种子的变体
```

### 3. 参数混合 (Parameter Mixing)

使用不同的gamma和delta组合。

```python
gamma_values = [0.25, 0.5]
delta_values = [1.0, 2.0, 3.0]
```

### 4. 密钥共享 (Key Sharing)

部分文本使用共享密钥，部分使用独立密钥。

```python
shared_key = 15485863
# 奇数索引用共享密钥，偶数索引用独立密钥
```

## 🚀 快速开始

### 方式1: 运行完整实验

```powershell
python hybrid_watermark_experiment.py
```

这将依次运行所有4种实验，并保存结果到 `hybrid_watermark_results/` 目录。

### 方式2: 交互式实验

```powershell
python hybrid_watermark_interactive.py
```

提供菜单界面，可以：
- 选择实验类型
- 自定义参数
- 实时查看结果

### 方式3: 分析结果

```powershell
python hybrid_watermark_analyzer.py
```

加载已有的实验结果并生成分析报告。

## 📊 使用示例

### 示例1: 片段级混合

```python
from hybrid_watermark_experiment import HybridWatermarkExperiment

exp = HybridWatermarkExperiment()

configs = [
    {'gamma': 0.25, 'delta': 2.0, 'hash_key': 15485863},
    {'gamma': 0.5, 'delta': 2.0, 'hash_key': 15485863},
]

result = exp.experiment_fragment_mixing(
    base_prompt="The future of AI is",
    fragment_configs=configs,
    tokens_per_fragment=50
)

exp.save_results(result)
```

### 示例2: 种子混合

```python
result = exp.experiment_seed_mixing(
    prompt="Write a story:",
    num_variations=3,
    max_new_tokens=100
)
```

### 示例3: 参数混合

```python
result = exp.experiment_parameter_mixing(
    prompt="Explain quantum computing:",
    gamma_values=[0.25, 0.5],
    delta_values=[1.5, 2.5],
    tokens_per_config=40
)
```

### 示例4: 密钥共享

```python
result = exp.experiment_key_sharing(
    prompts=[
        "Prompt 1",
        "Prompt 2",
        "Prompt 3",
        "Prompt 4"
    ],
    max_new_tokens=60
)
```

## 📈 实验结果

### 结果格式

所有实验结果保存为JSON格式：

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

### 结果目录

- `hybrid_watermark_results/` - 所有实验结果
  - `fragment_mixing_*.json` - 片段混合结果
  - `seed_mixing_*.json` - 种子混合结果
  - `parameter_mixing_*.json` - 参数混合结果
  - `key_sharing_*.json` - 密钥共享结果
  - `summary_*.txt` - 统计摘要

## 🔬 研究价值

### 可探索的问题

1. **检测鲁棒性**
   - 混合水印文本的检测率？
   - 哪种混合方式更难检测？

2. **水印唯一性**
   - 不同种子的区分度？
   - 交叉检测误报率？

3. **参数优化**
   - 最佳参数组合？
   - gamma和delta的相互影响？

4. **密钥管理**
   - 密钥共享策略的安全性？
   - 多密钥系统的可行性？

## 🔧 API参考

### HybridWatermarkExperiment 类

```python
class HybridWatermarkExperiment:
    def __init__(self, model_name, device)
    
    def experiment_fragment_mixing(...)
    def experiment_seed_mixing(...)
    def experiment_parameter_mixing(...)
    def experiment_key_sharing(...)
    
    def save_results(result, output_dir)
    def print_summary(result)
```

### 关键方法

- `create_watermark_processor()` - 创建水印处理器
- `create_watermark_detector()` - 创建检测器
- `generate_with_watermark()` - 生成带水印文本

## 📊 结果分析

### 分析工具

```powershell
python hybrid_watermark_analyzer.py
```

功能：
- 加载JSON结果
- 生成统计分析
- 显示交叉检测矩阵
- 对比多个实验

### 分析示例

```python
from hybrid_watermark_analyzer import HybridWatermarkAnalyzer

analyzer = HybridWatermarkAnalyzer()

# 加载结果
result = analyzer.load_result("fragment_mixing_20251023_120000.json")

# 生成报告
analyzer.generate_report(result)
```

## 💡 实验建议

### 片段混合
- 使用3-5个不同配置
- 每个片段至少50 tokens
- 测试极端参数组合

### 种子混合
- 使用3-5个不同种子
- 种子间隔足够大（如相差100万）
- 记录完整交叉检测矩阵

### 参数混合
- gamma范围: [0.2, 0.3, 0.4, 0.5]
- delta范围: [1.0, 1.5, 2.0, 2.5, 3.0]
- 注意组合数量

### 密钥共享
- 至少4个文本
- 共享/独立比例约1:1
- 测试多层次共享

## 🎓 高级用法

### 自定义实验

```python
# 创建自定义配置
custom_configs = [
    {'gamma': 0.3, 'delta': 2.5, 'hash_key': 12345678},
    {'gamma': 0.4, 'delta': 1.8, 'hash_key': 87654321},
]

result = exp.experiment_fragment_mixing(
    base_prompt="Custom prompt",
    fragment_configs=custom_configs,
    tokens_per_fragment=60
)
```

### 批量实验

```python
# 运行多个实验
for prompt in prompts:
    for config in configs:
        result = exp.experiment_fragment_mixing(...)
        exp.save_results(result)
```

## 📚 相关文档

- **详细指南**: `../docs_llama/HYBRID_WATERMARK_README.md`
- **快速参考**: `../docs_llama/QUICK_REFERENCE.md`
- **项目总结**: `../docs_llama/PROJECT_SUMMARY.md`

## ⚠️ 注意事项

1. 实验可能需要较长时间（10-30分钟）
2. 确保有足够的磁盘空间保存结果
3. GPU推荐（至少14GB显存）
4. 可以随时中断并查看已保存的结果

## 🤝 扩展建议

- 添加新的混合方案
- 实现可视化工具
- 集成到现有项目
- 进行大规模实验

---

**基础模型**: Llama 2 7B (meta-llama/Llama-2-7b-hf)  
**实验类型**: 4种混合水印方案  
**输出格式**: JSON + 文本摘要
