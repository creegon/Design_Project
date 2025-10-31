# Piggyback Watermark Attack 模块

该模块用于模拟鲁棒性“搭便车”（Piggyback）水印攻击：在保持文本仍被检测为带水印的前提下，通过轻量语义扰动或插入高风险句子，从而伪造特定厂商/模型的输出。

## 目录结构

- `piggyback_attack.py`：核心攻击逻辑，封装了生成、检测、定向插入、近义词扰动等能力。
- `watermark_attack_interactive.py`：交互式 CLI 脚本，方便按步骤完成模拟实验。
- `__init__.py`：导出常用类。

运行交互式脚本的示例：

```bash
python watermark_attack/watermark_attack_interactive.py --model llama-2-7b
```

脚本会引导用户：

1. 生成或载入带水印文本；
2. 指定需要插入的“敏感/非法”句子；
3. 自动尝试插入并保持水印检测结果为阳性；
4. 可选保存实验记录至 `watermark_attack/watermark_attack_results/`。
