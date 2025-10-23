# Llama 水印Demo使用指南

本指南介绍如何使用多种Llama模型进行文本水印的生成和检测。

## 🎯 支持的模型

### Llama 2 系列 (推荐，无需特殊权限)
- **Llama-2-7b-hf** ⭐ (默认) - 推荐使用，性能良好，约14GB显存
- **Llama-2-13b-hf** - 更强性能，约26GB显存
- **Llama-2-7b-chat-hf** - 对话优化版本，约14GB显存

### Llama 3.2 系列 (需要HuggingFace访问权限)
- **Llama-3.2-1B** - 轻量级，约4GB显存
- **Llama-3.2-3B** - 中等大小，约8GB显存

### 其他兼容模型
- **facebook/opt-6.7b** - 约13GB显存
- 任何支持HuggingFace transformers的因果语言模型

## 文件说明

我们创建了5个Python文件，从简单到复杂，满足不同的使用需求：

### 1. `llama_simple_example.py` - 最简单的入门示例
- **适合**: 快速了解水印功能的基本使用
- **功能**: 生成一个带水印的文本并检测，同时生成不带水印的文本进行对比
- **使用方式**: 

```powershell
# 使用默认模型 (Llama-2-7b-hf)
python llama_simple_example.py

# 指定其他模型
python llama_simple_example.py meta-llama/Llama-2-13b-hf
python llama_simple_example.py facebook/opt-6.7b
```

### 2. `llama_watermark_demo.py` - 完整功能演示
- **适合**: 需要了解完整API和多个测试案例的用户
- **功能**: 
  - 提供完整的类封装
  - 包含多个测试提示词
  - 自动进行带水印和不带水印的对比测试
- **使用方式**: 

```powershell
# 使用默认模型
python llama_watermark_demo.py

# 指定其他模型
python llama_watermark_demo.py meta-llama/Llama-2-13b-hf
```

### 3. `llama_interactive_demo.py` - 交互式命令行界面
- **适合**: 需要多次测试不同输入的用户
- **功能**:
  - 交互式菜单界面
  - 支持生成带/不带水印的文本
  - 支持检测任意文本的水印
  - 可以动态修改生成参数
- **使用方式**: 

```powershell
# 使用默认模型 (Llama-2-7b-hf)
python llama_interactive_demo.py

# 指定其他模型
python llama_interactive_demo.py --model_name meta-llama/Llama-2-13b-hf

# 自定义所有参数
python llama_interactive_demo.py --model_name meta-llama/Llama-2-7b-hf --gamma 0.25 --delta 2.0 --max_new_tokens 200 --temperature 0.7
```

**命令行参数**:
- `--model_name`: 模型名称 (默认: meta-llama/Llama-2-7b-hf)
- `--device`: 运行设备 (默认: cuda 如果可用，否则 cpu)
- `--gamma`: 水印参数gamma (默认: 0.25)
- `--delta`: 水印参数delta (默认: 2.0)
- `--seeding_scheme`: 种子方案 (默认: selfhash)
- `--max_new_tokens`: 最大生成token数 (默认: 200)
- `--temperature`: 采样温度 (默认: 0.7)
- `--top_p`: nucleus采样参数 (默认: 0.9)

### 4. `llama_batch_test.py` - 批量测试脚本
- **适合**: 需要系统性测试不同参数组合效果的研究人员
- **功能**:
  - 批量测试多个提示词
  - 测试不同的gamma、delta、seeding_scheme组合
  - 自动生成测试报告和统计摘要
  - 结果保存为JSON格式
- **使用方式**:

```powershell
# 使用默认模型
python llama_batch_test.py

# 指定其他模型
python llama_batch_test.py meta-llama/Llama-2-13b-hf
```

### 5. `llama_model_config.py` - 模型配置工具
- **适合**: 查看支持的模型和配置选项
- **功能**:
  - 列出所有支持的模型
  - 显示水印配置预设
  - 提供配置管理功能
- **使用方式**:

```powershell
# 列出所有支持的模型
python llama_model_config.py --list-models

# 列出水印配置预设
python llama_model_config.py --list-configs
```

## 快速启动 🚀

### 方式一: 使用启动脚本 (最简单，推荐)

```powershell
.\run_llama_demo.ps1
```

脚本会：
1. 让你选择要使用的模型 (默认 Llama-2-7b-hf)
2. 显示功能菜单
3. 引导你完成操作

### 方式二: 直接运行

```powershell
# 1. 快速测试（使用默认Llama-2-7b-hf模型）
python llama_simple_example.py

# 2. 交互式使用
python llama_interactive_demo.py

# 3. 使用其他模型
python llama_simple_example.py meta-llama/Llama-2-13b-hf
python llama_interactive_demo.py --model_name facebook/opt-6.7b
```

## 模型选择指南

### 推荐: Llama 2 7B (默认)
```powershell
# 最推荐，无需特殊权限，性能良好
python llama_simple_example.py
# 或明确指定
python llama_simple_example.py meta-llama/Llama-2-7b-hf
```

### 如果你有更多显存: Llama 2 13B
```powershell
python llama_interactive_demo.py --model_name meta-llama/Llama-2-13b-hf
```

### 对话场景: Llama 2 Chat
```powershell
python llama_interactive_demo.py --model_name meta-llama/Llama-2-7b-chat-hf
```

### 如果你有Llama 3.2访问权限
```powershell
python llama_simple_example.py meta-llama/Llama-3.2-3B
```

### 查看所有支持的模型
```powershell
python llama_model_config.py --list-models
```

```powershell
# 安装PyTorch (根据你的CUDA版本选择)
# 对于CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 对于CPU版本:
pip install torch torchvision torchaudio

# 安装transformers和其他依赖
pip install transformers accelerate scipy
```

## 安装依赖

在运行这些脚本之前，请确保安装了所需的依赖：

根据项目README的建议，以下是推荐的参数配置：

### 关键参数

1. **gamma** (默认: 0.25)
   - 绿名单token的比例
   - 范围: 0.25 - 0.75
   - 推荐值: 0.25

2. **delta** (默认: 2.0)
   - 水印强度
   - 范围: 0.5 - 2.0 (适中), 更大值用于instruction-tuned模型
   - 推荐值: 2.0

3. **seeding_scheme** (默认: "selfhash")
   - 种子生成方案
   - 可选值: "selfhash", "minhash", "simple_1"
   - 推荐值: "selfhash"

4. **context_width** (h)
   - 上下文宽度
   - selfhash默认为4
   - 更长的上下文提高隐蔽性，但降低鲁棒性

5. **ignore_repeated_ngrams** (检测时)
   - 检测时是否忽略重复的n-gram
   - 推荐: True

### 最佳实践配置

```python
# 推荐的基准配置
watermark_processor = WatermarkLogitsProcessor(
    vocab=list(tokenizer.get_vocab().values()),
    gamma=0.25,           # 绿名单比例
    delta=2.0,            # 水印强度
    seeding_scheme="selfhash"  # 种子方案
)

watermark_detector = WatermarkDetector(
    vocab=list(tokenizer.get_vocab().values()),
    gamma=0.25,
    seeding_scheme="selfhash",
    device=device,
    tokenizer=tokenizer,
    z_threshold=4.0,      # Z分数阈值
    normalizers=[],       # 文本归一化器
    ignore_repeated_ngrams=True  # 忽略重复n-gram
)
```

## 使用示例

### 基本使用流程

```python
from llama_watermark_demo import LlamaWatermarkDemo

# 1. 初始化 (使用默认Llama-2-7b-hf模型)
demo = LlamaWatermarkDemo(
    model_name="meta-llama/Llama-2-7b-hf",  # 可以改为其他模型
    gamma=0.25,
    delta=2.0,
    seeding_scheme="selfhash"
)

# 2. 生成带水印的文本
watermarked_text = demo.generate_with_watermark(
    prompt="The future of AI is",
    max_new_tokens=100
)

# 3. 检测水印
detection_results = demo.detect_watermark(watermarked_text)

# 4. 查看结果
print(f"Z分数: {detection_results['z_score']}")
print(f"是否包含水印: {detection_results['prediction']}")
```

### 交互式使用

运行交互式demo后，你会看到以下菜单：

```
命令说明:
  1 - 生成带水印的文本
  2 - 生成不带水印的文本
  3 - 检测文本水印
  4 - 显示当前配置
  5 - 修改生成参数
  q - 退出
```

## 注意事项

### 模型下载

首次运行时，程序会自动下载所选的模型。对于Llama-2-7b-hf（约13GB）：
- 确保有足够的磁盘空间
- 网络连接稳定
- Llama 2系列模型公开可用，无需特殊权限
- Llama 3.2系列可能需要在HuggingFace上申请访问权限

### 硬件要求

**Llama-2-7b-hf (默认模型)**:
- **GPU推荐**: NVIDIA GPU (至少14GB显存)
- **CPU可用**: 但速度较慢，需要至少16GB RAM
- **内存**: 至少16GB RAM

**其他模型**:
- **Llama-2-13b-hf**: 需要约26GB显存
- **Llama-3.2-1B**: 需要约4GB显存
- **Llama-3.2-3B**: 需要约8GB显存

### 性能优化

1. **使用GPU**: 自动检测，优先使用CUDA
2. **批量处理**: 使用batch_test脚本进行批量测试
3. **半精度**: GPU上自动使用float16

## 检测结果解读

水印检测会返回以下关键指标：

- **z_score**: Z分数，越高表示越可能含有水印
  - z > 4.0: 强烈表明含有水印
  - z < 4.0: 可能不含水印
  
- **p_value**: p值，统计显著性
  - p < 0.001: 非常显著
  
- **prediction**: 布尔值，是否检测到水印
  
- **green_fraction**: 绿色token的比例
  - 应该显著高于gamma值（如0.25）

## 故障排除

### 常见问题

1. **CUDA out of memory**
   - 减少max_new_tokens
   - 使用CPU模式
   - 使用更小的batch size

2. **模型下载失败**
   - 检查网络连接
   - 设置HF_ENDPOINT环境变量
   - 手动下载模型

3. **检测不到水印**
   - 确保生成和检测使用相同的gamma、delta和seeding_scheme
   - 生成的文本长度足够（至少50+ tokens）
   - 检查是否正确使用watermark_processor

## 进一步学习

- 阅读原始论文: [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
- 查看项目主README: [lm-watermarking](../README.md)
- 尝试不同的参数组合，观察效果变化

## 贡献

如有问题或建议，请提交Issue或Pull Request。
