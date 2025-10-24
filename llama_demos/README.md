# Llama 水印演示脚本

本目录包含基础的Llama水印生成和检测演示脚本。

## 📁 文件列表

| 文件 | 功能 | 使用难度 |
|------|------|----------|
| `llama_simple_example.py` | 最简单的入门示例 | ⭐ |
| `llama_watermark_demo.py` | 完整功能演示 | ⭐⭐ |
| `llama_interactive_demo.py` | 交互式命令行界面 | ⭐⭐⭐ |
| `llama_batch_test.py` | 批量参数测试 | ⭐⭐⭐⭐ |
| `model_config.json` | 模型和API配置文件 | ⭐⭐ |
| `model_config_manager.py` | 配置管理工具 | ⭐⭐ |

## ⚙️ 模型配置

所有模型和API提供商配置都在 `model_config.json` 中管理。

### 查看配置

```powershell
# 查看所有配置摘要
python model_config_manager.py --summary

# 列出所有API提供商
python model_config_manager.py --list-providers

# 列出所有模型
python model_config_manager.py --list-models

# 查看特定模型详情
python model_config_manager.py --model deepseek-v3

# 列出所有水印配置
python model_config_manager.py --list-watermark

# 列出所有生成配置
python model_config_manager.py --list-generation
```

### 配置文件结构

`model_config.json` 包含：
- **api_providers**: API服务提供商配置（DeepSeek, SiliconFlow, HuggingFace等）
- **models**: 模型配置（包括标识符、价格、上下文长度等）
- **watermark_configs**: 水印参数预设（default, strong, weak, balanced）
- **generation_configs**: 生成参数预设（default, creative, precise）

## 🚀 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements_llama.txt
```

### 2. 运行简单示例

```powershell
# 使用默认模型 (Llama 2 7B)
python llama_simple_example.py

# 指定其他模型
python llama_simple_example.py meta-llama/Llama-2-13b-hf
```

### 3. 使用启动脚本（推荐）

```powershell
.\run_llama_demo.ps1
```

## 📖 使用说明

### llama_simple_example.py

最简单的入门脚本，展示完整的水印流程。

```powershell
python llama_simple_example.py [模型名称]
```

**输出**:
- 生成带水印的文本
- 检测结果（z-score, p-value等）
- 对比不带水印的文本

### llama_watermark_demo.py

包含多个测试案例的完整演示。

```powershell
python llama_watermark_demo.py [模型名称]
```

**特点**:
- 提供 `LlamaWatermarkDemo` 类
- 可导入到其他脚本
- 包含多个测试提示词

### llama_interactive_demo.py

交互式命令行界面。

```powershell
python llama_interactive_demo.py --model_name meta-llama/Llama-2-7b-hf --gamma 0.25 --delta 2.0
```

**功能**:
- 菜单驱动界面
- 实时生成和检测
- 动态修改参数

### llama_batch_test.py

批量测试不同参数组合。

```powershell
python llama_batch_test.py [模型名称]
```

**特点**:
- 自动测试多个提示词
- 参数网格搜索
- 生成JSON结果和统计报告

### llama_model_config.py

模型配置管理工具。

```powershell
# 列出所有支持的模型
python llama_model_config.py --list-models

# 列出水印配置预设
python llama_model_config.py --list-configs
```

## ⚙️ 配置文件

### requirements_llama.txt

Python依赖包列表。

### llama_config_example.json

配置示例文件，包含：
- 模型配置
- 水印参数
- 生成参数
- 检测参数

### run_llama_demo.ps1

PowerShell启动脚本，提供：
- 模型选择菜单
- 功能选择
- 依赖安装向导

## 🎯 使用场景

| 需求 | 推荐脚本 |
|------|----------|
| 快速测试 | `llama_simple_example.py` |
| 多次实验 | `llama_interactive_demo.py` |
| 代码集成 | `llama_watermark_demo.py` (导入类) |
| 参数对比 | `llama_batch_test.py` |
| 查看配置 | `llama_model_config.py` |

## 📊 输出结果

- **llama_test_results/**: 批量测试结果目录
  - `batch_test_results_*.json`: 完整实验结果
  - `summary_*.txt`: 统计摘要

## 💡 示例用法

### 导入类到自己的代码

```python
from llama_watermark_demo import LlamaWatermarkDemo

# 初始化
demo = LlamaWatermarkDemo(
    model_name="meta-llama/Llama-2-7b-hf",
    gamma=0.25,
    delta=2.0
)

# 生成
text = demo.generate_with_watermark("Your prompt")

# 检测
result = demo.detect_watermark(text)
```

## 🔧 参数说明

### 水印参数
- `gamma`: 绿名单比例 (推荐: 0.25)
- `delta`: 水印强度 (推荐: 2.0)
- `seeding_scheme`: 种子方案 (推荐: "selfhash")

### 生成参数
- `max_new_tokens`: 最大生成token数 (默认: 200)
- `temperature`: 采样温度 (默认: 0.7)
- `top_p`: nucleus采样 (默认: 0.9)

## 📚 相关文档

- **详细指南**: `../docs_llama/LLAMA_DEMO_README.md`
- **快速参考**: `../docs_llama/QUICK_REFERENCE.md`
- **项目总结**: `../docs_llama/PROJECT_SUMMARY.md`

## ⚠️ 注意事项

1. 首次运行会自动下载模型（约13GB）
2. 推荐使用GPU（至少14GB显存）
3. CPU模式较慢但可用
4. 确保参数在生成和检测时一致

---

**默认模型**: Llama 2 7B (meta-llama/Llama-2-7b-hf)  
**推荐显存**: 14GB+
