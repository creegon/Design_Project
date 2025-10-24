"""
Llama Watermarking Demo
支持多种Llama模型的水印生成和检测
支持通过模型昵称（nickname）指定模型
"""

import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList
)
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from model_config_manager import ModelConfigManager


class LlamaWatermarkDemo:
    def __init__(
        self,
        model_nickname: str = "llama-2-7b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gamma: float = 0.25,
        delta: float = 2.0,
        seeding_scheme: str = "selfhash"
    ):
        """
        初始化Llama水印demo
        
        Args:
            model_nickname: 模型昵称（在 model_config.json 中配置），支持的模型包括：
                           - llama-2-7b (默认)
                           - llama-2-13b
                           - llama-2-7b-chat
                           - llama-3.2-1b
                           - llama-3.2-3b
                           - deepseek-v3
                           - deepseek-chat
                           等等
            device: 运行设备 (cuda/cpu)
            gamma: 绿名单比例 (推荐 0.25)
            delta: 水印强度 (推荐 2.0)
            seeding_scheme: 种子方案 (推荐 selfhash)
        """
        # 通过配置管理器解析模型
        config_manager = ModelConfigManager()
        model_info = config_manager.get_model_info_by_nickname(model_nickname)
        
        if not model_info:
            available_models = config_manager.list_model_names()
            raise ValueError(
                f"找不到模型 '{model_nickname}'。\n"
                f"可用的模型: {', '.join(available_models)}"
            )
        
        model_name = model_info["model_identifier"]
        
        print(f"模型昵称: {model_nickname}")
        print(f"模型标识: {model_name}")
        print(f"API提供商: {model_info['model_config'].get('api_provider')}")
        print(f"设备: {device}")
        
        self.model_nickname = model_nickname
        self.model_name = model_name
        self.model_info = model_info
        self.device = device
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 设置pad_token（如果没有的话）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        
        # 初始化水印处理器
        self.watermark_processor = WatermarkLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme
        )
        
        # 初始化水印检测器
        self.watermark_detector = WatermarkDetector(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=gamma,
            seeding_scheme=seeding_scheme,
            device=self.device,
            tokenizer=self.tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_ngrams=True
        )
        
        print("Model and watermark processors initialized successfully!")
    
    def generate_with_watermark(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        生成带水印的文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            do_sample: 是否使用采样
        
        Returns:
            生成的文本（不包含prompt）
        """
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}\n")
        
        # Tokenize输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 使用水印处理器生成
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                logits_processor=LogitsProcessorList([self.watermark_processor]),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 只保留新生成的token（去掉prompt部分）
        generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1]:]
        
        # 解码
        generated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        print(f"Generated text (with watermark):\n{generated_text}\n")
        
        return generated_text
    
    def generate_without_watermark(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        生成不带水印的文本（用于对比）
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            do_sample: 是否使用采样
        
        Returns:
            生成的文本（不包含prompt）
        """
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}\n")
        
        # Tokenize输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 不使用水印处理器生成
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 只保留新生成的token
        generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1]:]
        
        # 解码
        generated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        print(f"Generated text (without watermark):\n{generated_text}\n")
        
        return generated_text
    
    def detect_watermark(self, text: str) -> dict:
        """
        检测文本中的水印
        
        Args:
            text: 要检测的文本
        
        Returns:
            检测结果字典，包含z分数、p值等信息
        """
        print(f"\n{'='*60}")
        print(f"Detecting watermark in text...")
        print(f"{'='*60}\n")
        
        score_dict = self.watermark_detector.detect(text)
        
        # 打印检测结果
        print("Detection Results:")
        print(f"  - z-score: {score_dict['z_score']:.4f}")
        print(f"  - p-value: {score_dict['p_value']:.6f}")
        print(f"  - Prediction: {'WATERMARKED' if score_dict['prediction'] else 'NOT WATERMARKED'}")
        print(f"  - Green tokens: {score_dict['num_tokens_scored']}")
        print(f"  - Green token ratio: {score_dict['green_fraction']:.4f}")
        print()
        
        return score_dict


def main(demo):
    """主函数：演示水印生成和检测
    
    Args:
        demo: LlamaWatermarkDemo 实例
    """
    print("=" * 80)
    print("Llama Watermarking Demo")
    print("=" * 80)
    print()
    
    # 测试提示
    test_prompts = [
        "The future of artificial intelligence is",
        "Write a short story about a robot:",
        "Explain quantum computing in simple terms:",
    ]
    
    # 对每个提示进行测试
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'#' * 80}")
        print(f"Test {i}/{len(test_prompts)}")
        print(f"{'#' * 80}\n")
        
        # 生成带水印的文本
        watermarked_text = demo.generate_with_watermark(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        
        # 检测水印
        watermark_results = demo.detect_watermark(watermarked_text)
        
        # 生成不带水印的文本（对比）
        normal_text = demo.generate_without_watermark(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        
        # 检测不带水印的文本（应该检测不到）
        normal_results = demo.detect_watermark(normal_text)
        
        print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数指定模型（使用昵称）
    if len(sys.argv) > 1:
        model_nickname = sys.argv[1]
        print(f"使用指定模型: {model_nickname}\n")
        
        # 创建demo实例（使用指定模型）
        demo = LlamaWatermarkDemo(model_nickname=model_nickname)
    else:
        print("提示: 可以通过命令行参数指定模型，例如:")
        print("  python llama_watermark_demo.py deepseek-v3")
        print("  python llama_watermark_demo.py llama-2-13b\n")
        
        # 使用默认模型
        demo = LlamaWatermarkDemo()
    
    # 运行测试
    main()
