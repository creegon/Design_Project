# Llama 水印系统 - 项目导航

本目录包含基于 Llama 模型的水印生成、检测和混合水印实验系统。

## 📁 目录结构

```
lm-watermarking/
│
├── llama_demos/              # 基础水印演示脚本 ✓
│   ├── llama_simple_example.py         (入门示例)
│   ├── llama_watermark_demo.py         (完整演示)
│   ├── llama_interactive_demo.py       (交互界面)
│   ├── llama_batch_test.py             (批量测试)
│   ├── llama_model_config.py           (模型配置)
│   ├── run_llama_demo.ps1              (启动脚本)
│   ├── requirements_llama.txt          (依赖列表)
│   ├── llama_config_example.json       (配置示例)
│   └── README.md                       ⭐ 目录说明
│
├── hybrid_watermark/         # 混合水印实验系统 ✓
│   ├── hybrid_watermark_experiment.py  (核心实验)
│   ├── hybrid_watermark_interactive.py (交互界面)
│   ├── hybrid_watermark_analyzer.py    (结果分析)
│   └── README.md                       ⭐ 目录说明
│
└── docs_llama/               # 完整文档 ✓
    ├── LLAMA_DEMO_README.md            (演示文档)
    ├── HYBRID_WATERMARK_README.md      (实验文档)
    ├── QUICK_REFERENCE.md              (快速参考)
    ├── LLAMA_FILES_SUMMARY.md          (文件清单)
    └── PROJECT_SUMMARY.md              (项目总结)
```

**✓ 表示目录已包含 README.md 说明文档**  
**⭐ 表示新创建的目录级文档，建议先阅读**

## 🚀 快速开始

### 1. 基础水印演示

```powershell
# 进入演示目录
cd llama_demos

# 运行简单示例
python llama_simple_example.py

# 或使用启动脚本
.\run_llama_demo.ps1
```

### 2. 混合水印实验

```powershell
# 进入实验目录
cd hybrid_watermark

# 运行完整实验
python hybrid_watermark_experiment.py

# 或使用交互式界面
python hybrid_watermark_interactive.py
```

## 📚 文档索引

| 文档 | 位置 | 说明 |
|------|------|------|
| 快速参考 | `docs_llama/QUICK_REFERENCE.md` | 5分钟快速上手 |
| 基础使用 | `docs_llama/LLAMA_DEMO_README.md` | 基础水印功能详解 |
| 混合实验 | `docs_llama/HYBRID_WATERMARK_README.md` | 4种混合水印方案 |
| 文件清单 | `docs_llama/LLAMA_FILES_SUMMARY.md` | 完整文件索引 |
| 项目总结 | `docs_llama/PROJECT_SUMMARY.md` | 项目概述 |

## 🎯 使用场景

### 场景1: 快速测试水印功能

```powershell
cd llama_demos
python llama_simple_example.py
```

**适合**: 初次使用，了解基本功能

### 场景2: 交互式使用

```powershell
cd llama_demos
python llama_interactive_demo.py
```

**适合**: 需要多次测试不同输入

### 场景3: 混合水印研究

```powershell
cd hybrid_watermark
python hybrid_watermark_experiment.py
```

**适合**: 研究人员探索混合水印方案

### 场景4: 批量参数测试

```powershell
cd llama_demos
python llama_batch_test.py
```

**适合**: 系统性参数对比研究

## 💡 默认模型

所有脚本默认使用 **Llama 2 7B** (`meta-llama/Llama-2-7b-hf`)

- ✅ 公开可用，无需特殊权限
- ✅ 性能良好（14GB显存）
- ✅ 可自由切换到其他模型

## 🔧 安装依赖

```powershell
# 进入演示目录
cd llama_demos

# 安装依赖
pip install -r requirements_llama.txt
```

## 📖 推荐阅读顺序

1. **快速了解**: `docs_llama/QUICK_REFERENCE.md` (5分钟)
2. **基础使用**: `docs_llama/LLAMA_DEMO_README.md` (15分钟)
3. **深入研究**: `docs_llama/HYBRID_WATERMARK_README.md` (30分钟)
4. **完整索引**: `docs_llama/LLAMA_FILES_SUMMARY.md` (参考)

## 🎓 核心功能

### 基础功能 (`llama_demos/`)
- ✅ 水印生成和检测
- ✅ 多模型支持
- ✅ 交互式界面
- ✅ 批量测试

### 混合水印 (`hybrid_watermark/`)
- ✅ 片段级混合
- ✅ 种子混合
- ✅ 参数混合
- ✅ 密钥共享

## 📊 实验结果

运行后会在以下位置生成结果：

- `llama_demos/llama_test_results/` - 批量测试结果
- `hybrid_watermark/hybrid_watermark_results/` - 混合实验结果

## ✅ 导入路径修复

所有子目录中的脚本已经修复导入路径问题，可以正常访问项目根目录下的 `extended_watermark_processor.py`。

**验证修复**: 运行 `python verify_imports.py` 检查所有导入路径

**详细说明**: 参见 `IMPORT_FIX.md` 文档

## 🆘 获取帮助

```powershell
# 查看模型列表
cd llama_demos
python llama_model_config.py --list-models

# 查看水印配置
python llama_model_config.py --list-configs

# 验证导入路径
cd ..
python verify_imports.py

# 阅读文档
cd docs_llama
notepad QUICK_REFERENCE.md
```

## 🔗 相关链接

- 原始项目: [lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking)
- 论文: [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
- 导入修复说明: `IMPORT_FIX.md`

---

**创建日期**: 2025年10月23日  
**默认模型**: Llama 2 7B  
**实验类型**: 基础水印 + 4种混合方案
