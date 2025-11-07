# 水印系统 - 项目导航

本项目实现了基于大语言模型的水印生成、检测和混合水印实验系统，支持多种模型和API提供商。

## 📁 目录结构

```
lm-watermarking/
├── docs_llama/               # 项目中文文档与导航 📄
├── hybrid_watermark/         # 混合水印实验系统 ⭐
│   ├── hybrid_watermark_experiment.py   (核心实验)
│   ├── hybrid_watermark_interactive.py  (⭐ 交互式实验界面)
│   ├── hybrid_watermark_analyzer.py     (结果分析工具)
│   ├── statistical_evaluation.py        (统计评估模块)
│   └── README.md                        ⭐ 目录说明
├── llama_demos/              # 基础水印演示脚本 📄
│   ├── llama_simple_example.py          (入门示例)
│   ├── llama_watermark_demo.py          (完整演示)
│   ├── llama_interactive_demo.py        (交互界面)
│   ├── llama_batch_test.py              (批量测试)
│   ├── model_config_manager.py          (⭐ 模型配置管理器)
│   ├── model_config.json                (⭐ 模型配置文件)
│   └── README.md                        ⭐ 目录说明
├── upstream/
│   └── lm_watermarking/      # 原始 lm-watermarking 源码全集 📦
│       ├── alternative_prf_schemes.py
│       ├── experiments/
│       ├── hf_hub_space_demo/
│       ├── homoglyph_data/
│       ├── watermark_processor.py
│       ├── demo_watermark.py
│       ├── requirements.txt / setup.cfg / pyproject.toml
│       └── watermark_reliability_release/ …
├── extended_watermark_processor.py      # 自定义扩展处理器 (626行)
├── REPORT_LLAMAWATERMARK_LLAMA.md       # 10 月 24 日实验报告
├── SUMMARY.md                           # 项目摘要
└── IMPORT_FIX.md                        # 导入修复备忘
```

**📄 表示目录已包含 README.md 说明文档**  
**⭐ 表示重要文件或新功能**  
**📦 表示完整的上游项目打包在单一目录中**

> 现在所有上游代码都集中在 `upstream/lm_watermarking/` 内，可通过
> `from upstream.lm_watermarking import watermark_processor` 等方式导入；
> 自定义模块（含中文注释）保持在仓库根目录下的独立子目录中。

## 🚀 快速开始

### 1. 配置模型 (必需)

首先配置 `llama_demos/model_config.json`：

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

### 2. 基础水印演示

```powershell
# 进入演示目录
cd llama_demos

# 运行简单示例
python llama_simple_example.py llama-3.2-3b

# 或使用启动脚本
.\run_llama_demo.ps1
```

> 提示：`llama_simple_example.py` 和 `llama_batch_test.py` 使用**第一个位置参数**指定模型昵称，无 `--model` 选项。

### 3. 混合水印实验

```powershell
# 进入实验目录
cd hybrid_watermark

# 运行交互式界面（推荐）
python hybrid_watermark_interactive.py

# 或运行完整实验脚本
python hybrid_watermark_experiment.py
```

> `hybrid_watermark_interactive.py` 支持 `--model` 选项；`hybrid_watermark_experiment.py` 同样可接受一个可选的模型昵称位置参数（默认使用 `llama-2-7b`）。

## 📚 实验类型

### 混合水印实验 (3种)

| 实验编号 | 实验名称 | 说明 |
|---------|---------|------|
| **实验1** | 混合配置实验 | 片段级/参数级混合水印 |
| **实验2** | 密钥交叉检测 | 种子混合/密钥共享策略 |
| **实验3** | 跨模型共享密钥 | 多模型协作水印 |

### 统计评估实验 (4种)

| 实验编号 | 实验名称 | 说明 |
|---------|---------|------|
| **实验4** | 滑动窗口检测 | 分析水印信号分布均匀性 |
| **实验5** | 窗口敏感性分析 | 确定最优检测窗口大小 |
| **实验6** | 最小可检测长度 | 找出可靠检测所需最小长度 |
| **实验7** | 完整统计评估 | 执行全部三项统计分析 |

### 鲁棒性测试实验 (1种)

| 实验编号 | 实验名称 | 说明 |
|---------|---------|------|
| **实验8** | 多模型改写鲁棒性 | 测试水印在跨模型改写后的存活率 |

## 🎯 使用场景

### 场景1: 快速测试水印功能

```powershell
cd llama_demos
python llama_simple_example.py llama-3.2-3b
```

**适合**: 初次使用，了解基本功能

### 场景2: 交互式实验研究

```powershell
cd hybrid_watermark
python hybrid_watermark_interactive.py --model llama-3.2-3b
```

**适合**: 研究人员进行多种水印方案对比
**功能**: 
- 8种实验类型（3种混合+4种统计+1种鲁棒性）
- 实时可视化
- 自动保存结果

### 场景3: 批量参数测试

```powershell
cd llama_demos
python llama_batch_test.py llama-3.2-3b
```

**适合**: 系统性参数对比研究

### 场景4: 结果分析

```powershell
cd hybrid_watermark
python hybrid_watermark_analyzer.py
```

**适合**: 分析已保存的实验结果

## 💡 支持的模型

### API提供商
- **OpenAI**: GPT系列模型
- **DeepSeek**: DeepSeek系列、Llama系列
- **本地模型**: 通过Transformers库加载

### 推荐模型配置

```json
{
  "models": {
    "llama-3.2-3b": {
      "model_identifier": "meta-llama/Llama-3.2-3B-Instruct",
      "api_provider": "deepseek",
      "description": "小型高效模型，推荐日常使用"
    },
    "gpt-4o-mini": {
      "model_identifier": "gpt-4o-mini",
      "api_provider": "openai",
      "description": "高质量生成，适合对比实验"
    }
  }
}
```

### 模型管理

```powershell
# 列出所有配置的模型
cd llama_demos
python -c "from model_config_manager import ModelConfigManager; mgr = ModelConfigManager(); print(mgr.list_model_names())"

# 查看模型详情
python -c "from model_config_manager import ModelConfigManager; mgr = ModelConfigManager(); print(mgr.get_model_info_by_nickname('llama-3.2-3b'))"
```

## 🔧 安装依赖

```powershell
# 方法1: 安装基础依赖
cd llama_demos
pip install -r requirements_llama.txt

# 方法2: 安装完整依赖（推荐）
cd ..
pip install -r requirements.txt

# 主要依赖包
# - torch >= 2.0.0
# - transformers >= 4.30.0
# - openai >= 1.0.0
# - scipy
# - matplotlib
# - numpy
# - tqdm
```

## ⚙️ 环境配置

### 1. API密钥配置（推荐使用环境变量）

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:DEEPSEEK_API_KEY = "your-deepseek-api-key"

# 或在 model_config.json 中直接配置
{
  "api_providers": {
    "openai": {
      "api_key": "env:OPENAI_API_KEY"  # 推荐：使用环境变量
    }
  }
}
```

### 2. GPU配置（可选）

```python
# 系统自动选择：cuda（GPU）或 cpu
# 可在运行时指定：
python hybrid_watermark_interactive.py --device cuda
```

## 🆘 常见问题

### Q1: 如何添加新模型？

编辑 `llama_demos/model_config.json`：
```json
{
  "models": {
    "my-model": {
      "model_identifier": "organization/model-name",
      "nickname": "my-model",
      "api_provider": "openai",
      "description": "我的自定义模型"
    }
  }
}
```

### Q2: 检测率低怎么办？

当前系统已优化 `z_threshold = 3.0`（从4.0降低），显著提升检测率。

如果仍然检测率低，可以尝试：
1. **增加 delta**（如从2.0提高到2.5）- 增强水印信号
2. **降低 gamma**（如从0.5降到0.4）- 提高信噪比
3. **增加生成长度** - 更长文本提供更多统计证据

### Q3: 如何理解 Z-score？

Z-score 是统计显著性指标：
- **Z = 3.0**: 99.87%置信度，检测阈值（推荐）
- **Z = 4.0**: 99.997%置信度（过于严格，已弃用）
- **Z = 2.5**: 99.38%置信度（较宽松）

公式: `Z = (observed_green - expected_green) / std_dev`

### Q4: Gamma 和 Delta 如何选择？

| 场景 | Gamma | Delta | 说明 |
|------|-------|-------|------|
| **质量优先** | 0.5 | 1.5-2.0 | 文本自然，水印中等 |
| **平衡配置** | 0.5 | 2.0 | **推荐默认** |
| **检测优先** | 0.25 | 2.5-3.0 | 水印强，可能影响质量 |

### Q5: 如何查看所有可用模型？

```powershell
cd llama_demos
python -c "from model_config_manager import ModelConfigManager; mgr = ModelConfigManager(); print('\n'.join(mgr.list_model_names()))"
```

### Q6: 实验结果保存在哪里？

所有结果保存在 `hybrid_watermark/hybrid_watermark_results/`，包括：
- JSON数据文件（完整实验数据）
- PNG图表文件（可视化结果）

### Q7: 如何分析已有结果？

```powershell
cd hybrid_watermark
python hybrid_watermark_analyzer.py
```

## 🎓 核心功能

### 基础功能 (`llama_demos/`)
- ✅ 水印生成和检测
- ✅ 多模型支持（本地/API）
- ✅ 模型配置管理系统
- ✅ 交互式界面
- ✅ 批量测试

### 混合水印实验 (`hybrid_watermark/`)

**混合方案 (3种)**
- ✅ 片段级混合 - 不同片段用不同配置
- ✅ 参数网格混合 - gamma×delta组合扫描
- ✅ 种子变体 - 不同hash_key生成变体
- ✅ 密钥共享 - 共享密钥vs独立密钥
- ✅ 跨模型协作 - 多模型共享密钥

**统计评估 (4种)**
- ✅ 滑动窗口检测 - Z-score分布分析
- ✅ 窗口敏感性 - 最优窗口大小
- ✅ 最小长度分析 - 可靠检测阈值
- ✅ 完整统计评估 - 综合性能评估

### 水印参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|---------|
| **gamma** | 绿名单比例 | 0.5 | 0.25-0.5 |
| **delta** | Logits偏置强度 | 2.0 | 1.5-3.0 |
| **hash_key** | PRF种子 | 15485863 | 任意整数 |
| **z_threshold** | 检测阈值 | 3.0 | 2.5-4.0 |

**参数说明**:
- **gamma**: 控制词汇表中绿色token的比例，影响期望绿色率
- **delta**: 控制对绿色token的推动强度，影响实际绿色率
- **z_threshold**: 统计显著性阈值（已优化为3.0，提升检测率）

## 📊 实验结果

运行后会在以下位置生成结果：

### 结果目录
- `hybrid_watermark/hybrid_watermark_results/` - 所有实验结果

### 输出文件类型

**JSON格式** - 完整数据记录
```
sliding_window_20251024_143022.json
window_sensitivity_20251024_143155.json
minimum_length_20251024_143340.json
complete_statistical_eval_20251024_143512.json
```

**PNG格式** - 可视化图表
```
sliding_window_20251024_143022.png
window_sensitivity_20251024_143155.png
minimum_length_20251024_143340.png
```

### JSON结构

每个实验结果包含：
- `experiment_type`: 实验类型标识
- `prompt`: 使用的提示词
- `watermark_config`: 水印参数配置
- `generated_texts`: 生成的文本及完整内容
- `results`: 统计分析结果
- `detailed_results`: 详细检测数据

### 可视化分析

所有统计评估实验自动生成matplotlib图表：
- Z-score分布曲线
- 检测率趋势图
- 绿色token比例分析
- 成功/失败散点图

## ✅ 项目特色

### 1. 统一模型管理
- ✅ 支持多API提供商（OpenAI, DeepSeek等）
- ✅ 模型昵称系统，简化调用
- ✅ 环境变量安全管理API密钥
- ✅ 统一配置文件 `model_config.json`

### 2. 完整实验体系
- ✅ 3种混合水印方案（配置/密钥/跨模型）
- ✅ 4种统计评估方法（窗口/敏感性/最小长度/综合）
- ✅ 交互式界面，实时反馈
- ✅ 自动保存JSON+PNG结果

### 3. 优化的检测算法
- ✅ Z-score阈值优化（3.0 vs 4.0）
- ✅ 提升检测灵敏度（检测率从40%→近100%）
- ✅ 保持低假阳性率（<0.13%）

### 4. 可视化分析
- ✅ matplotlib自动生成图表
- ✅ Z-score分布、检测率、绿色比例
- ✅ 成功/失败散点图
- ✅ 累积检测率曲线

### 5. 研究工具
- ✅ 滑动窗口分析水印均匀性
- ✅ 窗口敏感性确定最优参数
- ✅ 最小长度分析找检测阈值
- ✅ 批量实验支持大规模测试

## 📖 命令速查

```powershell
# 1. 配置检查
cd llama_demos
python -c "from model_config_manager import ModelConfigManager; ModelConfigManager().validate_config()"

# 2. 快速测试
python llama_simple_example.py llama-3.2-3b

# 3. 交互式实验（推荐）
cd ../hybrid_watermark
python hybrid_watermark_interactive.py --model llama-3.2-3b

# 4. 统计评估（完整流程，含滑动窗口等）
python statistical_evaluation.py --model llama-3.2-3b

# 5. 结果分析
python hybrid_watermark_analyzer.py

# 6. 查看帮助
python hybrid_watermark_interactive.py --help
```

## 🔗 相关资源

- **原始项目**: [lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking)
- **论文**: [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
- **核心文件**: 
  - `extended_watermark_processor.py` - 水印处理器（626行）
  - `hybrid_watermark_interactive.py` - 交互界面（1558行）
  - `model_config_manager.py` - 模型管理（443行）

## 📝 更新日志

### 最新版本 (2025-10-24)

**新增功能**:
- ✅ 统计评估实验模块（4种评估方法）
- ✅ 多模型鲁棒性测试（跨模型改写存活率）
- ✅ Z-score阈值优化（3.0替代4.0）
- ✅ 模型配置管理系统
- ✅ 完整JSON输出（包含生成文本）
- ✅ 自动可视化图表生成

**优化改进**:
- ✅ 检测率显著提升（40%→近100%@200tokens）
- ✅ 实验整合（5个→3个混合实验）
- ✅ 交互界面优化（8种实验类型）
- ✅ 支持跨模型水印鲁棒性评估

**修复问题**:
- ✅ hash_key参数传递错误
- ✅ Z-score阈值过严格问题
- ✅ 可视化图表阈值不一致

---

**创建日期**: 2025年10月23日  
**最后更新**: 2025年11月6日  
**推荐模型**: Llama 3.2 3B Instruct (HuggingFace API)  
**实验类型**: 3种混合实验 + 4种统计评估 + 1种鲁棒性测试
