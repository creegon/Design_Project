# Llama 水印Demo - 快速参考

## 🚀 快速开始

```powershell
# 最简单的方式
.\run_llama_demo.ps1

# 或直接运行
python llama_simple_example.py
```

## 📋 模型选择速查表

| 模型 | 命令 | 显存需求 | 访问权限 | 推荐 |
|------|------|----------|----------|------|
| Llama-2-7b-hf | `python llama_simple_example.py` | 14GB | ✓ 公开 | ⭐⭐⭐⭐⭐ |
| Llama-2-13b-hf | `python llama_simple_example.py meta-llama/Llama-2-13b-hf` | 26GB | ✓ 公开 | ⭐⭐⭐⭐ |
| Llama-2-7b-chat | `python llama_simple_example.py meta-llama/Llama-2-7b-chat-hf` | 14GB | ✓ 公开 | ⭐⭐⭐ |
| Llama-3.2-1B | `python llama_simple_example.py meta-llama/Llama-3.2-1B` | 4GB | ✗ 需申请 | ⭐⭐ |
| Llama-3.2-3B | `python llama_simple_example.py meta-llama/Llama-3.2-3B` | 8GB | ✗ 需申请 | ⭐⭐⭐ |

## 🎯 常用命令

### 查看支持的模型
```powershell
python llama_model_config.py --list-models
```

### 查看水印配置
```powershell
python llama_model_config.py --list-configs
```

### 简单测试
```powershell
# 使用默认模型 (Llama-2-7b-hf)
python llama_simple_example.py

# 使用其他模型
python llama_simple_example.py meta-llama/Llama-2-13b-hf
```

### 交互式使用
```powershell
# 使用默认模型
python llama_interactive_demo.py

# 指定模型
python llama_interactive_demo.py --model_name meta-llama/Llama-2-7b-chat-hf

# 完整参数
python llama_interactive_demo.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --gamma 0.25 \
  --delta 2.0 \
  --temperature 0.7 \
  --max_new_tokens 200
```

### 完整演示
```powershell
# 使用默认模型
python llama_watermark_demo.py

# 使用其他模型
python llama_watermark_demo.py meta-llama/Llama-2-13b-hf
```

### 批量测试
```powershell
# 使用默认模型
python llama_batch_test.py

# 使用其他模型
python llama_batch_test.py meta-llama/Llama-2-7b-hf
```

## ⚙️ 水印参数速查

### 推荐配置 (默认)
```python
gamma = 0.25           # 绿名单比例
delta = 2.0            # 水印强度
seeding_scheme = "selfhash"  # 种子方案
```

### 其他配置

| 场景 | Gamma | Delta | Seeding | 说明 |
|------|-------|-------|---------|------|
| 默认 | 0.25 | 2.0 | selfhash | 适用于大多数场景 |
| 强水印 | 0.25 | 3.0 | selfhash | 更强检测，可能影响质量 |
| 弱水印 | 0.25 | 1.0 | selfhash | 对质量影响小，检测较弱 |
| 鲁棒 | 0.25 | 2.0 | minhash | 抵抗编辑 |

## 🔍 检测结果解读

| 指标 | 说明 | 阈值 |
|------|------|------|
| z_score | Z分数 | > 4.0 表示含水印 |
| p_value | 统计显著性 | < 0.001 很显著 |
| prediction | 检测结论 | True/False |
| green_fraction | 绿色token比例 | 应 > gamma (0.25) |

## 📊 性能对比

| 模型 | 生成速度 | 质量 | 显存 | 推荐用途 |
|------|----------|------|------|----------|
| Llama-2-7b-hf | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 14GB | 日常使用 |
| Llama-2-13b-hf | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 26GB | 高质量生成 |
| Llama-2-7b-chat | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 14GB | 对话场景 |
| Llama-3.2-1B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 4GB | 资源受限 |
| Llama-3.2-3B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8GB | 平衡性能 |

## 🛠️ 故障排除

### CUDA out of memory
```powershell
# 使用更小的模型
python llama_simple_example.py meta-llama/Llama-3.2-1B

# 或使用CPU (慢)
python llama_interactive_demo.py --device cpu
```

### 模型下载失败
```powershell
# 设置镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"
python llama_simple_example.py
```

### 检测不到水印
- 确保生成和检测使用相同参数
- 生成文本长度至少50+ tokens
- 检查是否正确使用watermark_processor

## 📁 文件对应关系

| 需求 | 使用文件 |
|------|----------|
| 快速入门 | `llama_simple_example.py` |
| 完整功能 | `llama_watermark_demo.py` |
| 交互测试 | `llama_interactive_demo.py` |
| 参数对比 | `llama_batch_test.py` |
| 查看配置 | `llama_model_config.py` |
| 一键启动 | `run_llama_demo.ps1` |

## 💡 使用建议

1. **首次使用**: `python llama_simple_example.py`
2. **日常使用**: `python llama_interactive_demo.py`
3. **研究测试**: `python llama_batch_test.py`
4. **代码集成**: 导入 `LlamaWatermarkDemo` 类

## 📚 相关文档

- 详细指南: `LLAMA_DEMO_README.md`
- 文件清单: `LLAMA_FILES_SUMMARY.md`
- 配置示例: `llama_config_example.json`
- 项目主页: `README.md`

---

**默认模型**: Llama-2-7b-hf  
**推荐显存**: 14GB+  
**最低要求**: 8GB VRAM 或 16GB RAM (CPU模式)
