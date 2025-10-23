"""
Llama 3.2 3B 简单示例
最简单的使用示例，快速上手
"""

import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector


def simple_example(model_name="meta-llama/Llama-2-7b-hf"):
    """简单示例：生成并检测水印文本
    
    Args:
        model_name: 模型名称，支持的模型包括：
                   - meta-llama/Llama-2-7b-hf (默认)
                   - meta-llama/Llama-2-13b-hf
                   - meta-llama/Llama-3.2-1B
                   - meta-llama/Llama-3.2-3B
                   - 或任何兼容的Llama模型
    """
    
    print("\n" + "="*80)
    print("Llama 水印简单示例")
    print("="*80 + "\n")
    
    # 1. 设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"设备: {device}")
    print(f"模型: {model_name}\n")
    
    # 2. 加载模型和tokenizer
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    print("✓ 模型加载完成\n")
    
    # 3. 创建水印处理器
    print("创建水印处理器...")
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        delta=2.0,
        seeding_scheme="selfhash"
    )
    print("✓ 水印处理器创建完成\n")
    
    # 4. 创建水印检测器
    print("创建水印检测器...")
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        seeding_scheme="selfhash",
        device=device,
        tokenizer=tokenizer,
        z_threshold=4.0,
        normalizers=[],
        ignore_repeated_ngrams=True
    )
    print("✓ 水印检测器创建完成\n")
    
    # 5. 生成带水印的文本
    prompt = "The future of artificial intelligence is"
    print(f"{'='*80}")
    print(f"提示词: {prompt}")
    print(f"{'='*80}\n")
    
    print("生成带水印的文本...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            logits_processor=LogitsProcessorList([watermark_processor]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 只保留新生成的token
    generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1]:]
    watermarked_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    print(f"\n生成的文本 (带水印):")
    print(f"{'-'*80}")
    print(watermarked_text)
    print(f"{'-'*80}\n")
    
    # 6. 检测水印
    print("检测水印...")
    detection_results = watermark_detector.detect(watermarked_text)
    
    print(f"\n检测结果:")
    print(f"{'-'*80}")
    print(f"Z分数:          {detection_results['z_score']:.4f}")
    print(f"P值:            {detection_results['p_value']:.6f}")
    print(f"检测结论:       {'✓ 包含水印' if detection_results['prediction'] else '✗ 不包含水印'}")
    print(f"绿色token比例:  {detection_results['green_fraction']:.4f}")
    print(f"{'-'*80}\n")
    
    # 7. 对比：生成不带水印的文本
    print("生成不带水印的文本 (对比)...")
    with torch.no_grad():
        output_tokens_no_wm = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_tokens_no_wm = output_tokens_no_wm[:, inputs["input_ids"].shape[-1]:]
    normal_text = tokenizer.batch_decode(generated_tokens_no_wm, skip_special_tokens=True)[0]
    
    print(f"\n生成的文本 (不带水印):")
    print(f"{'-'*80}")
    print(normal_text)
    print(f"{'-'*80}\n")
    
    # 8. 检测不带水印的文本
    print("检测不带水印的文本...")
    detection_results_no_wm = watermark_detector.detect(normal_text)
    
    print(f"\n检测结果:")
    print(f"{'-'*80}")
    print(f"Z分数:          {detection_results_no_wm['z_score']:.4f}")
    print(f"P值:            {detection_results_no_wm['p_value']:.6f}")
    print(f"检测结论:       {'✓ 包含水印' if detection_results_no_wm['prediction'] else '✗ 不包含水印'}")
    print(f"绿色token比例:  {detection_results_no_wm['green_fraction']:.4f}")
    print(f"{'-'*80}\n")
    
    print("="*80)
    print("示例完成!")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数指定模型
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"使用指定模型: {model_name}\n")
        simple_example(model_name)
    else:
        # 默认使用 Llama 2 7B
        simple_example()
