# 导入路径修复说明

## 问题描述

当我们将演示脚本移动到 `llama_demos/` 和 `hybrid_watermark/` 子目录后，这些脚本无法直接导入项目根目录下的 `extended_watermark_processor.py` 模块。

## 解决方案

在每个需要导入 `extended_watermark_processor` 的脚本开头添加了以下代码：

```python
import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

### 工作原理

1. `os.path.dirname(__file__)` - 获取当前脚本所在的目录
2. `os.path.join(..., '..')` - 获取父目录（即项目根目录）
3. `os.path.abspath(...)` - 转换为绝对路径
4. `sys.path.insert(0, ...)` - 将项目根目录添加到 Python 搜索路径的最前面

## 已修复的文件

### llama_demos/ 目录

✓ `llama_simple_example.py` - 已添加路径修复  
✓ `llama_watermark_demo.py` - 已添加路径修复  
✓ `llama_interactive_demo.py` - 已添加路径修复  
✓ `llama_batch_test.py` - 已添加路径修复  
○ `llama_model_config.py` - 不需要导入（纯配置文件）

### hybrid_watermark/ 目录

✓ `hybrid_watermark_experiment.py` - 已添加路径修复  
○ `hybrid_watermark_interactive.py` - 不需要导入（使用相对导入）  
○ `hybrid_watermark_analyzer.py` - 不需要导入（仅分析结果）

## 验证测试

运行以下命令验证导入路径是否正确：

```powershell
python verify_imports.py
```

### 预期输出

```
[测试 2] 验证所有 llama_demos 下的 Python 文件
------------------------------------------------------------
✓ llama_simple_example.py
✓ llama_watermark_demo.py
✓ llama_interactive_demo.py
✓ llama_batch_test.py

[测试 3] 验证 hybrid_watermark 下的 Python 文件
------------------------------------------------------------
✓ hybrid_watermark_experiment.py
○ hybrid_watermark_interactive.py - 不需要导入 extended_watermark_processor
○ hybrid_watermark_analyzer.py - 不需要导入 extended_watermark_processor
```

## 使用说明

现在所有脚本都可以从各自的目录中直接运行：

```powershell
# 从 llama_demos 目录运行
cd llama_demos
python llama_simple_example.py

# 从 hybrid_watermark 目录运行
cd hybrid_watermark
python hybrid_watermark_experiment.py

# 或从项目根目录运行
python llama_demos/llama_simple_example.py
python hybrid_watermark/hybrid_watermark_experiment.py
```

## 注意事项

1. **不要删除路径修复代码** - 这些代码是脚本正常运行的必要条件
2. **保持目录结构** - 脚本假设 `extended_watermark_processor.py` 在项目根目录
3. **相对路径** - 代码使用相对路径，即使移动整个项目文件夹也能正常工作

## 技术细节

### 为什么不使用相对导入？

```python
# 相对导入需要包结构
from ..extended_watermark_processor import WatermarkLogitsProcessor
```

这种方式需要：
- 添加 `__init__.py` 文件
- 将项目作为包安装
- 修改所有脚本的运行方式

### 为什么不修改 PYTHONPATH？

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
```

这种方式：
- 需要用户手动配置环境变量
- 不同系统配置方式不同
- 降低了脚本的可移植性

### 为什么使用 sys.path.insert(0, ...)?

- `insert(0, ...)` 将路径添加到最前面，确保优先搜索
- `append(...)` 会添加到末尾，可能被其他路径覆盖
- 这是 Python 社区推荐的做法

## 相关文件

- `verify_imports.py` - 验证所有导入路径配置
- `test_imports.py` - 测试实际导入功能
- `llama_demos/README.md` - llama_demos 目录说明
- `hybrid_watermark/README.md` - hybrid_watermark 目录说明

## 修复日期

2025年10月23日

---

**状态**: ✅ 所有导入路径已修复并验证通过
