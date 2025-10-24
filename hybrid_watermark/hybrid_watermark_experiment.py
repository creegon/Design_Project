"""
混合水印实验系统
实现多种混合水印方案的生成和检测

实验类型：
1. 片段级混合：段落中不同片段使用不同水印方案
2. 种子混合：同一模型使用不同种子
3. 密钥混合：不同模型共享或混合密钥
4. 参数混合：不同gamma/delta组合
"""

import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

# 动态导入 model_config_manager
llama_demos_path = os.path.join(os.path.dirname(__file__), '..', 'llama_demos')
sys.path.insert(0, os.path.abspath(llama_demos_path))
from model_config_manager import ModelConfigManager

from typing import List, Dict, Tuple
import json
from datetime import datetime
from pathlib import Path


class HybridWatermarkExperiment:
    """混合水印实验类"""
    
    def __init__(
        self,
        model_nickname: str = "llama-2-7b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化实验环境
        
        Args:
            model_nickname: 模型昵称（在 model_config.json 中配置）
            device: 运行设备
        """
        print(f"\n{'='*80}")
        print("混合水印实验系统初始化")
        print(f"{'='*80}\n")
        
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
        print(f"设备: {device}\n")
        
        self.device = device
        self.model_nickname = model_nickname
        self.model_name = model_name
        self.model_info = model_info
        
        # 加载模型
        print(f"加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        print("✓ 模型加载完成\n")
    
    def create_watermark_processor(
        self,
        gamma: float = 0.25,
        delta: float = 2.0,
        seeding_scheme: str = "selfhash",
        hash_key: int = 15485863
    ) -> WatermarkLogitsProcessor:
        """创建水印处理器"""
        
        # 如果需要自定义hash_key，修改seeding_scheme
        if hash_key != 15485863:
            # 解析seeding_scheme并替换hash_key
            parts = seeding_scheme.split('-')
            if len(parts) >= 6:
                parts[-1] = str(hash_key)
                seeding_scheme = '-'.join(parts)
            else:
                # 对于简单的scheme名称，添加完整参数
                seeding_scheme = f"ff-anchored_minhash_prf-4-True-{hash_key}"
        
        return WatermarkLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme
        )
    
    def create_watermark_detector(
        self,
        gamma: float = 0.25,
        seeding_scheme: str = "selfhash",
        hash_key: int = 15485863
    ) -> WatermarkDetector:
        """创建水印检测器"""
        
        # 同样处理hash_key
        if hash_key != 15485863:
            parts = seeding_scheme.split('-')
            if len(parts) >= 6:
                parts[-1] = str(hash_key)
                seeding_scheme = '-'.join(parts)
            else:
                seeding_scheme = f"ff-anchored_minhash_prf-4-True-{hash_key}"
        
        return WatermarkDetector(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=gamma,
            seeding_scheme=seeding_scheme,
            device=self.device,
            tokenizer=self.tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_ngrams=True
        )
    
    def generate_with_watermark(
        self,
        prompt: str,
        watermark_processor: WatermarkLogitsProcessor,
        max_new_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """使用指定水印处理器生成文本"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                logits_processor=LogitsProcessorList([watermark_processor]),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1]:]
        generated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        return generated_text
    
    # ========== 实验1: 片段级混合水印 ==========
    
    def experiment_fragment_mixing(
        self,
        base_prompt: str,
        fragment_configs: List[Dict],
        tokens_per_fragment: int = 50
    ) -> Dict:
        """
        实验1：片段级混合水印
        在同一段落中，不同片段使用不同的水印配置
        
        Args:
            base_prompt: 初始提示词
            fragment_configs: 每个片段的水印配置列表
            tokens_per_fragment: 每个片段的token数
        
        Returns:
            实验结果字典
        """
        print(f"\n{'='*80}")
        print("实验1: 片段级混合水印")
        print(f"{'='*80}\n")
        print(f"基础提示: {base_prompt}")
        print(f"片段数量: {len(fragment_configs)}")
        print(f"每片段token数: {tokens_per_fragment}\n")
        
        fragments = []
        current_prompt = base_prompt
        
        for i, config in enumerate(fragment_configs, 1):
            print(f"生成片段 {i}/{len(fragment_configs)}...")
            print(f"  配置: gamma={config.get('gamma', 0.25)}, "
                  f"delta={config.get('delta', 2.0)}, "
                  f"key={config.get('hash_key', 15485863)}")
            
            # 创建该片段的水印处理器
            processor = self.create_watermark_processor(
                gamma=config.get('gamma', 0.25),
                delta=config.get('delta', 2.0),
                seeding_scheme=config.get('seeding_scheme', 'selfhash'),
                hash_key=config.get('hash_key', 15485863)
            )
            
            # 生成片段
            fragment = self.generate_with_watermark(
                prompt=current_prompt,
                watermark_processor=processor,
                max_new_tokens=tokens_per_fragment,
                temperature=0.7
            )
            
            fragments.append({
                'text': fragment,
                'config': config,
                'fragment_id': i
            })
            
            # 更新提示词（累积生成）
            current_prompt = current_prompt + " " + fragment
            print(f"  ✓ 片段生成完成\n")
        
        # 组合所有片段
        combined_text = " ".join([f['text'] for f in fragments])
        
        # 使用每个配置的检测器检测整体文本
        detection_results = []
        
        for i, config in enumerate(fragment_configs, 1):
            detector = self.create_watermark_detector(
                gamma=config.get('gamma', 0.25),
                seeding_scheme=config.get('seeding_scheme', 'selfhash'),
                hash_key=config.get('hash_key', 15485863)
            )
            
            result = detector.detect(combined_text)
            detection_results.append({
                'config_id': i,
                'config': config,
                'z_score': float(result['z_score']),
                'p_value': float(result['p_value']),
                'prediction': bool(result['prediction']),
                'green_fraction': float(result['green_fraction'])
            })
        
        return {
            'experiment_type': 'fragment_mixing',
            'base_prompt': base_prompt,
            'fragments': fragments,
            'combined_text': combined_text,
            'detection_results': detection_results,
            'fragment_configs': fragment_configs
        }
    
    # ========== 实验2: 种子混合 ==========
    
    def experiment_seed_mixing(
        self,
        prompt: str,
        num_variations: int = 3,
        base_gamma: float = 0.25,
        base_delta: float = 2.0,
        max_new_tokens: int = 100
    ) -> Dict:
        """
        实验2：种子混合
        同一模型使用不同的水印种子（hash_key）
        
        Args:
            prompt: 输入提示
            num_variations: 生成变体数量
            base_gamma: 基础gamma值
            base_delta: 基础delta值
            max_new_tokens: 每个变体的最大token数
        
        Returns:
            实验结果字典
        """
        print(f"\n{'='*80}")
        print("实验2: 种子混合 (不同Hash Key)")
        print(f"{'='*80}\n")
        print(f"提示: {prompt}")
        print(f"变体数量: {num_variations}\n")
        
        # 生成不同的hash keys
        base_key = 15485863
        hash_keys = [base_key + i * 1000000 for i in range(num_variations)]
        
        variations = []
        
        for i, hash_key in enumerate(hash_keys, 1):
            print(f"生成变体 {i}/{num_variations} (hash_key={hash_key})...")
            
            # 创建带有特定hash_key的处理器
            processor = self.create_watermark_processor(
                gamma=base_gamma,
                delta=base_delta,
                seeding_scheme='selfhash',
                hash_key=hash_key
            )
            
            # 生成文本
            text = self.generate_with_watermark(
                prompt=prompt,
                watermark_processor=processor,
                max_new_tokens=max_new_tokens
            )
            
            variations.append({
                'variation_id': i,
                'hash_key': hash_key,
                'text': text
            })
            
            print(f"  ✓ 变体生成完成\n")
        
        # 交叉检测：每个文本用所有检测器检测
        cross_detection = []
        
        for var in variations:
            var_detections = []
            
            for key in hash_keys:
                detector = self.create_watermark_detector(
                    gamma=base_gamma,
                    seeding_scheme='selfhash',
                    hash_key=key
                )
                
                result = detector.detect(var['text'])
                var_detections.append({
                    'detector_key': key,
                    'z_score': float(result['z_score']),
                    'prediction': bool(result['prediction'])
                })
            
            cross_detection.append({
                'text_id': var['variation_id'],
                'text_key': var['hash_key'],
                'detections': var_detections
            })
        
        # 创建混合文本（组合所有变体）
        mixed_text = " ".join([v['text'] for v in variations])
        
        return {
            'experiment_type': 'seed_mixing',
            'prompt': prompt,
            'variations': variations,
            'mixed_text': mixed_text,
            'cross_detection': cross_detection,
            'hash_keys': hash_keys
        }
    
    # ========== 实验3: 参数混合 ==========
    
    def experiment_parameter_mixing(
        self,
        prompt: str,
        gamma_values: List[float] = [0.25, 0.5],
        delta_values: List[float] = [1.0, 2.0, 3.0],
        tokens_per_config: int = 50
    ) -> Dict:
        """
        实验3：参数混合
        使用不同的gamma和delta组合生成文本片段
        
        Args:
            prompt: 输入提示
            gamma_values: gamma值列表
            delta_values: delta值列表
            tokens_per_config: 每个配置的token数
        
        Returns:
            实验结果字典
        """
        print(f"\n{'='*80}")
        print("实验3: 参数混合 (Gamma/Delta组合)")
        print(f"{'='*80}\n")
        print(f"提示: {prompt}")
        print(f"Gamma值: {gamma_values}")
        print(f"Delta值: {delta_values}\n")
        
        # 创建所有参数组合
        param_combinations = []
        for gamma in gamma_values:
            for delta in delta_values:
                param_combinations.append({'gamma': gamma, 'delta': delta})
        
        fragments = []
        current_prompt = prompt
        
        for i, params in enumerate(param_combinations, 1):
            print(f"生成片段 {i}/{len(param_combinations)} "
                  f"(gamma={params['gamma']}, delta={params['delta']})...")
            
            processor = self.create_watermark_processor(
                gamma=params['gamma'],
                delta=params['delta'],
                seeding_scheme='selfhash'
            )
            
            fragment = self.generate_with_watermark(
                prompt=current_prompt,
                watermark_processor=processor,
                max_new_tokens=tokens_per_config
            )
            
            fragments.append({
                'text': fragment,
                'gamma': params['gamma'],
                'delta': params['delta'],
                'fragment_id': i
            })
            
            current_prompt = current_prompt + " " + fragment
            print(f"  ✓ 片段生成完成\n")
        
        # 组合文本
        combined_text = " ".join([f['text'] for f in fragments])
        
        # 使用不同参数的检测器检测
        detection_matrix = []
        
        for detector_params in param_combinations:
            detector = self.create_watermark_detector(
                gamma=detector_params['gamma'],
                seeding_scheme='selfhash'
            )
            
            result = detector.detect(combined_text)
            detection_matrix.append({
                'detector_gamma': detector_params['gamma'],
                'z_score': float(result['z_score']),
                'prediction': bool(result['prediction']),
                'green_fraction': float(result['green_fraction'])
            })
        
        return {
            'experiment_type': 'parameter_mixing',
            'prompt': prompt,
            'param_combinations': param_combinations,
            'fragments': fragments,
            'combined_text': combined_text,
            'detection_matrix': detection_matrix
        }
    
    # ========== 实验4: 密钥共享混合 ==========
    
    def experiment_key_sharing(
        self,
        prompts: List[str],
        shared_key: int = 15485863,
        individual_keys: List[int] = None,
        max_new_tokens: int = 100
    ) -> Dict:
        """
        实验4：密钥共享混合
        部分文本使用共享密钥，部分使用独立密钥
        
        Args:
            prompts: 提示词列表
            shared_key: 共享的hash key
            individual_keys: 个别密钥列表（如果为None则自动生成）
            max_new_tokens: 每个文本的最大token数
        
        Returns:
            实验结果字典
        """
        print(f"\n{'='*80}")
        print("实验4: 密钥共享混合")
        print(f"{'='*80}\n")
        print(f"文本数量: {len(prompts)}")
        print(f"共享密钥: {shared_key}\n")
        
        if individual_keys is None:
            individual_keys = [shared_key + (i+1) * 500000 for i in range(len(prompts))]
        
        # 生成文本：一半使用共享密钥，一半使用独立密钥
        texts = []
        
        for i, prompt in enumerate(prompts):
            use_shared = i % 2 == 0  # 偶数索引使用共享密钥
            key = shared_key if use_shared else individual_keys[i]
            
            print(f"生成文本 {i+1}/{len(prompts)} "
                  f"({'共享密钥' if use_shared else '独立密钥'}: {key})...")
            
            processor = self.create_watermark_processor(
                gamma=0.25,
                delta=2.0,
                seeding_scheme='selfhash',
                hash_key=key
            )
            
            text = self.generate_with_watermark(
                prompt=prompt,
                watermark_processor=processor,
                max_new_tokens=max_new_tokens
            )
            
            texts.append({
                'text_id': i + 1,
                'prompt': prompt,
                'text': text,
                'key_type': 'shared' if use_shared else 'individual',
                'hash_key': key
            })
            
            print(f"  ✓ 文本生成完成\n")
        
        # 组合所有文本
        combined_text = " ".join([t['text'] for t in texts])
        
        # 使用共享密钥检测器检测
        shared_detector = self.create_watermark_detector(
            gamma=0.25,
            seeding_scheme='selfhash',
            hash_key=shared_key
        )
        
        shared_detection = shared_detector.detect(combined_text)
        
        # 对每个独立文本进行检测
        individual_detections = []
        
        for text_info in texts:
            # 用正确的密钥检测
            correct_detector = self.create_watermark_detector(
                gamma=0.25,
                seeding_scheme='selfhash',
                hash_key=text_info['hash_key']
            )
            
            # 用共享密钥检测
            result_correct = correct_detector.detect(text_info['text'])
            result_shared = shared_detector.detect(text_info['text'])
            
            individual_detections.append({
                'text_id': text_info['text_id'],
                'key_type': text_info['key_type'],
                'correct_key_detection': {
                    'z_score': float(result_correct['z_score']),
                    'prediction': bool(result_correct['prediction'])
                },
                'shared_key_detection': {
                    'z_score': float(result_shared['z_score']),
                    'prediction': bool(result_shared['prediction'])
                }
            })
        
        return {
            'experiment_type': 'key_sharing',
            'prompts': prompts,
            'shared_key': shared_key,
            'individual_keys': individual_keys,
            'texts': texts,
            'combined_text': combined_text,
            'shared_key_detection': {
                'z_score': float(shared_detection['z_score']),
                'prediction': bool(shared_detection['prediction']),
                'green_fraction': float(shared_detection['green_fraction'])
            },
            'individual_detections': individual_detections
        }
    
    # ========== 保存和报告 ==========
    
    def save_results(self, results: Dict, output_dir: str = "hybrid_watermark_results"):
        """保存实验结果"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_type = results['experiment_type']
        
        filename = output_path / f"{exp_type}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 结果已保存到: {filename}")
        
        return filename
    
    def print_summary(self, results: Dict):
        """打印实验摘要"""
        
        print(f"\n{'='*80}")
        print(f"实验摘要: {results['experiment_type']}")
        print(f"{'='*80}\n")
        
        if results['experiment_type'] == 'fragment_mixing':
            print(f"片段数量: {len(results['fragments'])}")
            print(f"组合文本长度: {len(results['combined_text'])} 字符\n")
            print("各配置检测结果:")
            for det in results['detection_results']:
                print(f"  配置{det['config_id']}: z={det['z_score']:.2f}, "
                      f"检测={det['prediction']}")
        
        elif results['experiment_type'] == 'seed_mixing':
            print(f"变体数量: {len(results['variations'])}")
            print(f"使用的Hash Keys: {results['hash_keys']}\n")
            print("交叉检测摘要（对角线应为True）:")
            for cd in results['cross_detection']:
                matches = [d for d in cd['detections'] if d['prediction']]
                print(f"  文本{cd['text_id']} (key={cd['text_key']}): "
                      f"检测到 {len(matches)}/{len(cd['detections'])} 个匹配")
        
        elif results['experiment_type'] == 'parameter_mixing':
            print(f"参数组合数: {len(results['param_combinations'])}")
            print(f"片段数量: {len(results['fragments'])}\n")
            print("检测矩阵:")
            for det in results['detection_matrix']:
                print(f"  Gamma={det['detector_gamma']}: z={det['z_score']:.2f}, "
                      f"检测={det['prediction']}")
        
        elif results['experiment_type'] == 'key_sharing':
            shared_count = sum(1 for t in results['texts'] if t['key_type'] == 'shared')
            print(f"共享密钥文本: {shared_count}/{len(results['texts'])}")
            print(f"共享密钥检测: z={results['shared_key_detection']['z_score']:.2f}, "
                  f"检测={results['shared_key_detection']['prediction']}\n")
            print("个别检测:")
            for det in results['individual_detections']:
                print(f"  文本{det['text_id']} ({det['key_type']}): "
                      f"正确密钥={det['correct_key_detection']['prediction']}, "
                      f"共享密钥={det['shared_key_detection']['prediction']}")
        
        print(f"\n{'='*80}\n")


def main():
    """运行所有混合水印实验"""
    
    import sys
    
    # 支持命令行参数指定模型（使用昵称）
    model_nickname = "llama-2-7b"
    if len(sys.argv) > 1:
        model_nickname = sys.argv[1]
        print(f"使用指定模型: {model_nickname}\n")
    else:
        print("提示: 可以通过命令行参数指定模型，例如:")
        print("  python hybrid_watermark_experiment.py deepseek-v3")
        print("  python hybrid_watermark_experiment.py llama-2-13b\n")
    
    print("\n" + "="*80)
    print("混合水印实验系统")
    print(f"模型: {model_nickname}")
    print("="*80 + "\n")
    
    # 初始化实验环境
    experiment = HybridWatermarkExperiment(
        model_nickname=model_nickname
    )
    
    # ========== 实验1: 片段级混合 ==========
    print("\n开始实验1...")
    fragment_configs = [
        {'gamma': 0.25, 'delta': 2.0, 'hash_key': 15485863},
        {'gamma': 0.5, 'delta': 2.0, 'hash_key': 15485863},
        {'gamma': 0.25, 'delta': 3.0, 'hash_key': 15485863},
    ]
    
    result1 = experiment.experiment_fragment_mixing(
        base_prompt="The future of artificial intelligence is",
        fragment_configs=fragment_configs,
        tokens_per_fragment=50
    )
    
    experiment.print_summary(result1)
    experiment.save_results(result1)
    
    # ========== 实验2: 种子混合 ==========
    print("\n开始实验2...")
    result2 = experiment.experiment_seed_mixing(
        prompt="Write a short story about robots:",
        num_variations=3,
        max_new_tokens=80
    )
    
    experiment.print_summary(result2)
    experiment.save_results(result2)
    
    # ========== 实验3: 参数混合 ==========
    print("\n开始实验3...")
    result3 = experiment.experiment_parameter_mixing(
        prompt="Explain quantum computing:",
        gamma_values=[0.25, 0.5],
        delta_values=[1.5, 2.5],
        tokens_per_config=40
    )
    
    experiment.print_summary(result3)
    experiment.save_results(result3)
    
    # ========== 实验4: 密钥共享 ==========
    print("\n开始实验4...")
    result4 = experiment.experiment_key_sharing(
        prompts=[
            "The benefits of renewable energy include",
            "In the year 2050, technology will",
            "Climate change is affecting",
            "Space exploration has led to"
        ],
        max_new_tokens=60
    )
    
    experiment.print_summary(result4)
    experiment.save_results(result4)
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
