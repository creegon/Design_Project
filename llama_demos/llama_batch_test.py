"""
Llama 批量测试脚本
用于批量测试不同参数配置下的水印效果
支持多种Llama模型，默认使用 Llama 2 7B
"""

import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
from pathlib import Path
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList
)
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector


class LlamaBatchTester:
    """批量测试水印效果"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        print(f"\n{'='*80}")
        print("初始化批量测试器...")
        print(f"{'='*80}\n")
        
        self.device = device
        self.model_name = model_name
        
        # 加载tokenizer
        print(f"加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型
        print(f"加载模型到 {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        
        print("✓ 初始化完成!\n")
    
    def test_single_config(
        self,
        prompt: str,
        gamma: float,
        delta: float,
        seeding_scheme: str,
        max_new_tokens: int = 150
    ) -> dict:
        """测试单个配置"""
        
        # 创建水印处理器
        watermark_processor = WatermarkLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme
        )
        
        # 创建检测器
        watermark_detector = WatermarkDetector(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=gamma,
            seeding_scheme=seeding_scheme,
            device=self.device,
            tokenizer=self.tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_ngrams=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 生成带水印的文本
        start_time = time.time()
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                logits_processor=LogitsProcessorList([watermark_processor]),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generation_time = time.time() - start_time
        
        # 解码
        generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1]:]
        watermarked_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        # 检测水印
        detection_results = watermark_detector.detect(watermarked_text)
        
        # 生成不带水印的文本（对比）
        with torch.no_grad():
            output_tokens_no_wm = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_tokens_no_wm = output_tokens_no_wm[:, inputs["input_ids"].shape[-1]:]
        normal_text = self.tokenizer.batch_decode(
            generated_tokens_no_wm,
            skip_special_tokens=True
        )[0]
        
        # 检测不带水印的文本
        detection_results_no_wm = watermark_detector.detect(normal_text)
        
        return {
            "config": {
                "gamma": gamma,
                "delta": delta,
                "seeding_scheme": seeding_scheme
            },
            "prompt": prompt,
            "watermarked_text": watermarked_text,
            "normal_text": normal_text,
            "watermarked_detection": {
                "z_score": float(detection_results['z_score']),
                "p_value": float(detection_results['p_value']),
                "prediction": bool(detection_results['prediction']),
                "green_fraction": float(detection_results['green_fraction']),
                "num_tokens": int(detection_results['num_tokens_scored'])
            },
            "normal_detection": {
                "z_score": float(detection_results_no_wm['z_score']),
                "p_value": float(detection_results_no_wm['p_value']),
                "prediction": bool(detection_results_no_wm['prediction']),
                "green_fraction": float(detection_results_no_wm['green_fraction']),
                "num_tokens": int(detection_results_no_wm['num_tokens_scored'])
            },
            "generation_time": generation_time
        }
    
    def run_batch_test(
        self,
        prompts: list,
        gamma_values: list = [0.25, 0.5],
        delta_values: list = [1.0, 2.0, 3.0],
        seeding_schemes: list = ["selfhash", "minhash"],
        output_dir: str = "test_results"
    ):
        """运行批量测试"""
        
        print(f"\n{'='*80}")
        print("开始批量测试")
        print(f"{'='*80}\n")
        print(f"提示词数量: {len(prompts)}")
        print(f"Gamma值: {gamma_values}")
        print(f"Delta值: {delta_values}")
        print(f"种子方案: {seeding_schemes}")
        print(f"\n总测试数: {len(prompts) * len(gamma_values) * len(delta_values) * len(seeding_schemes)}\n")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 存储所有结果
        all_results = []
        
        total_tests = len(prompts) * len(gamma_values) * len(delta_values) * len(seeding_schemes)
        current_test = 0
        
        # 遍历所有配置
        for prompt in prompts:
            for gamma in gamma_values:
                for delta in delta_values:
                    for scheme in seeding_schemes:
                        current_test += 1
                        
                        print(f"\n[{current_test}/{total_tests}] 测试配置:")
                        print(f"  Prompt: {prompt[:50]}...")
                        print(f"  Gamma={gamma}, Delta={delta}, Scheme={scheme}")
                        
                        try:
                            result = self.test_single_config(
                                prompt=prompt,
                                gamma=gamma,
                                delta=delta,
                                seeding_scheme=scheme
                            )
                            all_results.append(result)
                            
                            # 显示简要结果
                            wm_det = result['watermarked_detection']
                            normal_det = result['normal_detection']
                            print(f"  带水印: z={wm_det['z_score']:.2f}, 检测={wm_det['prediction']}")
                            print(f"  不带水印: z={normal_det['z_score']:.2f}, 检测={normal_det['prediction']}")
                            
                        except Exception as e:
                            print(f"  ✗ 错误: {str(e)}")
                            continue
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"batch_test_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model": self.model_name,
                "device": self.device,
                "timestamp": timestamp,
                "total_tests": len(all_results),
                "results": all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"批量测试完成!")
        print(f"结果已保存到: {output_file}")
        print(f"{'='*80}\n")
        
        # 生成统计摘要
        self.generate_summary(all_results, output_path / f"summary_{timestamp}.txt")
        
        return all_results
    
    def generate_summary(self, results: list, output_file: Path):
        """生成测试摘要"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("批量测试摘要\n")
            f.write("="*80 + "\n\n")
            
            # 统计信息
            f.write(f"总测试数: {len(results)}\n\n")
            
            # 按配置分组统计
            configs = {}
            for result in results:
                config_key = (
                    result['config']['gamma'],
                    result['config']['delta'],
                    result['config']['seeding_scheme']
                )
                if config_key not in configs:
                    configs[config_key] = {
                        'watermarked_detected': 0,
                        'normal_detected': 0,
                        'total': 0,
                        'avg_wm_z_score': 0,
                        'avg_normal_z_score': 0
                    }
                
                configs[config_key]['total'] += 1
                if result['watermarked_detection']['prediction']:
                    configs[config_key]['watermarked_detected'] += 1
                if result['normal_detection']['prediction']:
                    configs[config_key]['normal_detected'] += 1
                configs[config_key]['avg_wm_z_score'] += result['watermarked_detection']['z_score']
                configs[config_key]['avg_normal_z_score'] += result['normal_detection']['z_score']
            
            # 输出每个配置的统计
            f.write("各配置性能:\n")
            f.write("-"*80 + "\n\n")
            
            for config_key, stats in configs.items():
                gamma, delta, scheme = config_key
                total = stats['total']
                
                f.write(f"配置: Gamma={gamma}, Delta={delta}, Scheme={scheme}\n")
                f.write(f"  测试数量: {total}\n")
                f.write(f"  带水印文本检测率: {stats['watermarked_detected']/total*100:.1f}%\n")
                f.write(f"  不带水印文本误检率: {stats['normal_detected']/total*100:.1f}%\n")
                f.write(f"  平均Z分数 (带水印): {stats['avg_wm_z_score']/total:.2f}\n")
                f.write(f"  平均Z分数 (不带水印): {stats['avg_normal_z_score']/total:.2f}\n")
                f.write("\n")
        
        print(f"摘要已保存到: {output_file}")


def main():
    """主函数"""
    
    import sys
    
    # 支持命令行参数指定模型
    model_name = "meta-llama/Llama-2-7b-hf"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"使用指定模型: {model_name}\n")
    
    # 测试提示词
    test_prompts = [
        "The future of artificial intelligence is",
        "Write a short story about a robot:",
        "Explain quantum computing in simple terms:",
        "The benefits of renewable energy include",
        "In the year 2050, technology will",
    ]
    
    # 初始化测试器
    tester = LlamaBatchTester(
        model_name=model_name
    )
    
    # 运行批量测试
    results = tester.run_batch_test(
        prompts=test_prompts,
        gamma_values=[0.25, 0.5],
        delta_values=[1.0, 2.0],
        seeding_schemes=["selfhash"],
        output_dir="llama_test_results"
    )
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
