# Llama 3.2 3B 水印Demo - 文件清单

本文档列出了为使用Llama 3.2 3B模型进行水印演示而创建的所有文件。

## 📋 创建的文件清单

### 1. 核心演示脚本 (5个)

#### `llama_simple_example.py`
- **类型**: 简单示例
- **行数**: ~150行
- **用途**: 最基础的入门示例，展示水印生成和检测的完整流程
- **适合**: 初学者快速了解功能
- **运行**: `python llama_simple_example.py`

#### `llama_watermark_demo.py`
- **类型**: 完整演示类
- **行数**: ~300行
- **用途**: 提供完整的类封装，包含多个测试案例
- **特点**: 
  - LlamaWatermarkDemo类，易于集成
  - 支持带/不带水印的文本生成
  - 自动检测和对比
  - 可作为模块导入
- **运行**: `python llama_watermark_demo.py`

#### `llama_interactive_demo.py`
- **类型**: 交互式界面
- **行数**: ~350行
- **用途**: 命令行交互式界面，支持多次测试
- **特点**:
  - 菜单驱动界面
  - 支持动态参数修改
  - 实时检测功能
  - 配置查看
- **运行**: `python llama_interactive_demo.py [参数]`
- **参数示例**: 
  ```powershell
  python llama_interactive_demo.py --gamma 0.25 --delta 2.0 --temperature 0.7
  ```

#### `llama_batch_test.py`
- **类型**: 批量测试工具
- **行数**: ~400行
- **用途**: 系统性测试不同参数组合
- **特点**:
  - 批量测试多个提示词
  - 参数网格搜索
  - 自动生成JSON结果
  - 生成统计摘要
  - 结果保存到`llama_test_results/`目录
- **运行**: `python llama_batch_test.py`

#### `llama_model_config.py`
- **类型**: 模型配置工具
- **行数**: ~150行
- **用途**: 管理支持的模型和水印配置
- **特点**:
  - 预定义支持的模型列表
  - 水印配置预设
  - 模型信息查询
- **运行**: `python llama_model_config.py --list-models`

### 2. 混合水印实验系统 (3个) 🆕

#### `hybrid_watermark_experiment.py`
- **类型**: 混合水印核心实验类
- **行数**: ~700行
- **用途**: 实现四种混合水印实验方案
- **特点**:
  - 片段级混合水印
  - 种子混合实验
  - 参数混合实验
  - 密钥共享实验
  - 自动保存实验结果
- **运行**: `python hybrid_watermark_experiment.py`

#### `hybrid_watermark_interactive.py`
- **类型**: 交互式实验界面
- **行数**: ~350行
- **用途**: 提供用户友好的交互式实验界面
- **特点**:
  - 菜单驱动
  - 自定义实验参数
  - 实时结果展示
  - 实验说明
- **运行**: `python hybrid_watermark_interactive.py`

#### `hybrid_watermark_analyzer.py`
- **类型**: 结果分析工具
- **行数**: ~400行
- **用途**: 分析和可视化实验结果
- **特点**:
  - 加载和解析JSON结果
  - 生成统计分析
  - 交叉检测矩阵
  - 实验对比
- **运行**: `python hybrid_watermark_analyzer.py`

### 3. 文档文件 (4个)

#### `LLAMA_DEMO_README.md`
- **类型**: 使用指南
- **内容**:
  - 4个脚本的详细说明
  - 安装依赖说明
  - 参数配置建议
  - 使用示例
  - 故障排除
  - 性能优化建议

#### `requirements_llama.txt`
- **类型**: 依赖清单
- **内容**:
  - PyTorch相关包
  - Transformers库
  - 科学计算包
  - 安装说明和注意事项

#### `HYBRID_WATERMARK_README.md` 🆕
- **类型**: 混合水印实验指南
- **内容**:
  - 四种实验类型详解
  - 使用示例
  - 结果解读
  - 研究建议
  - 常见问题

#### `QUICK_REFERENCE.md` 🆕
- **类型**: 快速参考卡
- **内容**:
  - 快速命令速查
  - 模型选择表格
  - 参数配置速查
  - 性能对比

### 4. 配置文件 (1个)

#### `llama_config_example.json`
- **类型**: 配置示例
- **内容**:
  - 模型配置
  - 水印参数配置
  - 生成参数配置
  - 检测参数配置
  - 测试提示词
  - 参数扫描配置

### 5. 启动脚本 (1个)

#### `run_llama_demo.ps1`
- **类型**: PowerShell启动脚本
- **用途**: 简化demo启动过程
- **功能**:
  - 检查Python环境
  - 交互式菜单选择
  - 依赖安装向导
  - 打开文档
- **运行**: `.\run_llama_demo.ps1`

## 📊 文件结构

```
lm-watermarking/
├── # 基础演示脚本
├── llama_simple_example.py           # 简单示例
├── llama_watermark_demo.py           # 完整演示
├── llama_interactive_demo.py         # 交互式界面
├── llama_batch_test.py               # 批量测试
├── llama_model_config.py             # 模型配置工具
│
├── # 混合水印实验系统 🆕
├── hybrid_watermark_experiment.py    # 核心实验类
├── hybrid_watermark_interactive.py   # 交互式实验
├── hybrid_watermark_analyzer.py      # 结果分析工具
│
├── # 文档
├── LLAMA_DEMO_README.md              # 基础使用指南
├── HYBRID_WATERMARK_README.md        # 混合水印实验指南 🆕
├── QUICK_REFERENCE.md                # 快速参考 🆕
├── LLAMA_FILES_SUMMARY.md            # 文件清单(本文件)
│
├── # 配置和脚本
├── requirements_llama.txt            # 依赖清单
├── llama_config_example.json         # 配置示例
├── run_llama_demo.ps1                # 启动脚本
│
└── # 结果目录(运行后生成)
    ├── llama_test_results/           # 批量测试结果
    └── hybrid_watermark_results/     # 混合水印实验结果 🆕
```

## 🚀 快速开始

### 方式一: 使用启动脚本 (推荐)

```powershell
.\run_llama_demo.ps1
```

然后按照菜单提示操作。

### 方式二: 直接运行

1. **安装依赖**:
```powershell
# 安装PyTorch (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements_llama.txt
```

2. **运行简单示例**:
```powershell
python llama_simple_example.py
```

3. **运行交互式Demo**:
```powershell
python llama_interactive_demo.py
```

## 📝 功能对比

### 基础演示脚本对比

| 功能 | simple | demo | interactive | batch_test | model_config |
|------|--------|------|-------------|------------|--------------|
| 基础生成 | ✓ | ✓ | ✓ | ✓ | ✗ |
| 水印检测 | ✓ | ✓ | ✓ | ✓ | ✗ |
| 多个测试 | ✗ | ✓ | ✓ | ✓ | ✗ |
| 交互式界面 | ✗ | ✗ | ✓ | ✗ | ✗ |
| 参数修改 | ✗ | ✗ | ✓ | ✗ | ✗ |
| 批量测试 | ✗ | ✗ | ✗ | ✓ | ✗ |
| 结果保存 | ✗ | ✗ | ✗ | ✓ | ✗ |
| 模型管理 | ✗ | ✗ | ✗ | ✗ | ✓ |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 功能丰富度 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### 混合水印实验系统对比 🆕

| 功能 | experiment | interactive | analyzer |
|------|------------|-------------|----------|
| 片段混合 | ✓ | ✓ | ✓ |
| 种子混合 | ✓ | ✓ | ✓ |
| 参数混合 | ✓ | ✓ | ✓ |
| 密钥共享 | ✓ | ✓ | ✓ |
| 自动运行 | ✓ | ✗ | ✗ |
| 交互界面 | ✗ | ✓ | ✓ |
| 结果分析 | ✗ | ✗ | ✓ |
| 统计报告 | ✗ | ✗ | ✓ |
| 易用性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 功能深度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 💡 使用建议

### 初次使用
1. 先运行 `llama_simple_example.py` 了解基本概念
2. 阅读 `LLAMA_DEMO_README.md` 了解详细用法
3. 使用 `llama_interactive_demo.py` 进行多次实验

### 日常使用
- 推荐使用 `llama_interactive_demo.py`
- 或导入 `LlamaWatermarkDemo` 类到自己的代码中

### 研究测试
- 使用 `llama_batch_test.py` 进行参数对比
- 结果会保存在 `llama_test_results/` 目录

### 混合水印研究 🆕
1. **快速入门**: 运行 `hybrid_watermark_experiment.py` 查看所有实验
2. **自定义实验**: 使用 `hybrid_watermark_interactive.py` 交互式配置
3. **结果分析**: 使用 `hybrid_watermark_analyzer.py` 分析结果
4. **阅读指南**: 参考 `HYBRID_WATERMARK_README.md` 了解实验原理

## 🔧 依赖关系

所有脚本都依赖于项目原有的文件：
- `extended_watermark_processor.py` (必需，位于项目根目录)
- `upstream/lm_watermarking/normalizers.py` (必需，原始归档)
- `upstream/lm_watermarking/alternative_prf_schemes.py` (必需，原始归档)
- `upstream/lm_watermarking/homoglyph_data/` (必需，原始归档)

## 📦 输出文件

运行批量测试后会生成：
- `llama_test_results/batch_test_results_YYYYMMDD_HHMMSS.json`
- `llama_test_results/summary_YYYYMMDD_HHMMSS.txt`

## ⚙️ 系统要求

- **Python**: 3.8+
- **内存**: 16GB+ RAM推荐
- **GPU**: NVIDIA GPU (8GB+ VRAM) 推荐，CPU也可用但较慢
- **磁盘**: 至少10GB空闲空间（用于模型缓存）

## 🎯 核心特性

### 基础功能
1. **完整的水印流程**: 从生成到检测一站式完成
2. **灵活的参数配置**: 支持多种水印参数组合
3. **易于使用**: 从简单到复杂，满足不同需求
4. **结果可视化**: 清晰的检测结果展示
5. **批量测试**: 支持系统性的参数对比研究
6. **多模型支持**: 支持Llama 2/3.2等多种模型

### 混合水印特性 🆕
1. **四种混合方案**: 片段、种子、参数、密钥混合
2. **交叉检测分析**: 完整的检测矩阵和统计分析
3. **实验可重复**: 结果自动保存为JSON格式
4. **灵活配置**: 支持自定义各种实验参数
5. **深度分析**: 提供统计分析和可视化工具

## 📖 相关文档

- 主项目README: `README.md`
- Llama Demo指南: `LLAMA_DEMO_README.md`
- 配置示例: `llama_config_example.json`

## 🤝 贡献

如有问题或改进建议，欢迎提交Issue或Pull Request。

---

**创建日期**: 2025年10月23日  
**版本**: 1.0  
**适用模型**: Llama 3.2 3B (及其他兼容的Llama系列模型)
