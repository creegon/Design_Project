"""
混合水印交互式实验界面
允许用户自定义实验参数并实时查看结果
支持通过模型昵称（nickname）指定模型

包含功能：
1. 混合水印实验 (片段混合、种子混合、参数混合、密钥共享)
2. 跨模型共享密钥实验
3. 统计评估实验 (滑动窗口、窗口敏感性、最小长度分析)
"""

import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import json

from hybrid_watermark_experiment import HybridWatermarkExperiment
from extended_watermark_processor import WatermarkDetector

# 动态导入 model_config_manager
llama_demos_path = os.path.join(os.path.dirname(__file__), '..', 'llama_demos')
sys.path.insert(0, os.path.abspath(llama_demos_path))
from model_config_manager import ModelConfigManager

import json


def parse_args():
    # 加载可用模型列表
    config_manager = ModelConfigManager()
    available_models = config_manager.list_model_names()
    
    parser = argparse.ArgumentParser(
        description="混合水印交互式实验 - 支持模型昵称"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.2-3b",
        help=f"模型昵称 (默认: llama-3.2-3b)\n"
             f"可用模型: {', '.join(available_models[:5])}..."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="运行设备"
    )
    
    return parser.parse_args()


class InteractiveHybridExperiment:
    """交互式混合水印实验"""
    
    def __init__(self, args):
        self.args = args
        
        # 通过配置管理器解析模型
        config_manager = ModelConfigManager()
        model_info = config_manager.get_model_info_by_nickname(args.model)
        
        if not model_info:
            available_models = config_manager.list_model_names()
            raise ValueError(
                f"找不到模型 '{args.model}'。\n"
                f"可用的模型: {', '.join(available_models)}"
            )
        
        model_name = model_info["model_identifier"]
        
        print(f"\n模型昵称: {args.model}")
        print(f"模型标识: {model_name}")
        print(f"API提供商: {model_info['model_config'].get('api_provider')}\n")
        
        self.model_nickname = args.model
        self.model_name = model_name
        self.model_info = model_info
        
        # 初始化实验
        self.experiment = HybridWatermarkExperiment(
            model_nickname=args.model,
            device=args.device
        )
    
    def run(self):
        """Main interactive loop for launching experiments."""
        print("\n" + "="*80)
        print("Hybrid Watermark Interactive Suite")
        print(f"Active model: {self.model_nickname} ({self.model_name})")
        print("="*80)

        while True:
            print("\n" + "="*80)
            print("Experiment Menu:")
            print("="*80)
            print("\n[Hybrid Watermark Experiments]")
            print("  1 - Fragment / Parameter Mixing")
            print("  2 - Seed / Key Cross Detection")
            print("  3 - Cross-Model Key Experiments")
            print("\n[Statistical Evaluations]")
            print("  4 - Sliding Window Detection")
            print("  5 - Window Sensitivity Analysis")
            print("  6 - Minimum Detectable Length")
            print("  7 - Complete Statistical Suite")
            print("\n[Robustness Testing]")
            print("  8 - Multi-LLM Chain Robustness")
            print("  9 - Watermarked Paraphrase Chain (Different Green/Red Lists)")
            print("\n[Other]")
            print("  h - Help / Show Experiment Info")
            print("  q - Quit")
            print("="*80)

            choice = input("\nSelect an option (1-9/h/q): ").strip().lower()

            if choice == 'q':
                print("\nExiting interactive suite.\n")
                break
            elif choice == '1':
                self.run_hybrid_config_experiment()
            elif choice == '2':
                self.run_key_cross_detection_experiment()
            elif choice == '3':
                self.run_cross_model_key_experiments()
            elif choice == '4':
                self.run_sliding_window_detection()
            elif choice == '5':
                self.run_window_sensitivity_analysis()
            elif choice == '6':
                self.run_minimum_length_analysis()
            elif choice == '7':
                self.run_complete_statistical_evaluation()
            elif choice == '8':
                self.run_multi_llm_robustness_test()
            elif choice == '9':
                self.run_watermarked_paraphrase_chain()
            elif choice == 'h':
                self.show_experiment_info()
            else:
                print("\nInvalid option, please try again.\n")

    def run_hybrid_config_experiment(self):
        """运行混合配置实验（合并片段混合和参数混合）"""
        print("\n" + "-"*80)
        print("实验1: 混合配置实验")
        print("-"*80 + "\n")
        print("此实验支持两种模式：")
        print("  a) 片段级混合 - 手动为每个片段指定不同配置")
        print("  b) 参数网格混合 - 自动生成gamma×delta的所有组合\n")
        
        mode = input("选择模式 (a=片段级, b=参数网格, 默认a): ").strip().lower()
        mode = mode if mode in ['a', 'b'] else 'a'
        
        prompt = input("\n输入基础提示词 (回车使用默认): ").strip()
        if not prompt:
            prompt = "The future of artificial intelligence is"
        
        if mode == 'a':
            # 片段级混合模式
            num_fragments = input("输入片段数量 (默认3): ").strip()
            num_fragments = int(num_fragments) if num_fragments else 3
            
            fragment_configs = []
            print(f"\n配置 {num_fragments} 个片段:")
            
            default_params = [
                {'gamma': 0.25, 'delta': 2.0, 'hash_key': 15485863},
                {'gamma': 0.5, 'delta': 1.5, 'hash_key': 32452843},
                {'gamma': 0.1, 'delta': 3.0, 'hash_key': 49979687}
            ]
            
            for i in range(num_fragments):
                print(f"\n片段 {i+1}:")
                gamma_input = input(f"  Gamma (默认{default_params[i % 3]['gamma']}): ").strip()
                gamma = float(gamma_input) if gamma_input else default_params[i % 3]['gamma']
                
                delta_input = input(f"  Delta (默认{default_params[i % 3]['delta']}): ").strip()
                delta = float(delta_input) if delta_input else default_params[i % 3]['delta']
                
                hash_key_input = input(f"  Hash Key (默认{default_params[i % 3]['hash_key']}): ").strip()
                hash_key = int(hash_key_input) if hash_key_input else default_params[i % 3]['hash_key']
                
                fragment_configs.append({
                    'gamma': gamma,
                    'delta': delta,
                    'hash_key': hash_key
                })
            
            print("\n开始实验...")
            result = self.experiment.experiment_fragment_mixing(
                base_prompt=prompt,
                fragment_configs=fragment_configs,
                tokens_per_fragment=50
            )
        
        else:
            # 参数网格模式
            print("\n参数配置方式：")
            print("  1. 手动输入列表 (默认)")
            print("  2. 自动生成等间距网格")
            param_mode = input("请选择 (1/2): ").strip()
            param_mode = param_mode if param_mode in ['1', '2'] else '1'

            if param_mode == '2':
                def parse_range(prompt_text: str, default_range: Tuple[float, float, int]):
                    user_input = input(prompt_text).strip()
                    if not user_input:
                        return default_range

                    parts = [p.strip() for p in user_input.split(',')]
                    if len(parts) != 3:
                        print("格式错误，使用默认设置。")
                        return default_range
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        count = int(parts[2])
                        if count < 2:
                            raise ValueError
                        return (start, end, count)
                    except ValueError:
                        print("解析失败，使用默认设置。")
                        return default_range

                gamma_range = parse_range(
                    "\n请输入Gamma范围和数量 (格式: 起始,结束,数量；默认0.1,0.6,4): ",
                    (0.1, 0.6, 4)
                )
                delta_range = parse_range(
                    "请输入Delta范围和数量 (格式: 起始,结束,数量；默认1.0,3.0,4): ",
                    (1.0, 3.0, 4)
                )

                gamma_values = [round(x, 4) for x in np.linspace(*gamma_range)]
                delta_values = [round(x, 4) for x in np.linspace(*delta_range)]

                print(f"\n自动生成Gamma列表: {gamma_values}")
                print(f"自动生成Delta列表: {delta_values}")
            else:
                gamma_input = input("\n输入Gamma值列表 (逗号分隔, 默认0.25,0.5): ").strip()
                gamma_values = [float(x.strip()) for x in gamma_input.split(',')] if gamma_input else [0.25, 0.5]
                
                delta_input = input("输入Delta值列表 (逗号分隔, 默认1.5,2.5): ").strip()
                delta_values = [float(x.strip()) for x in delta_input.split(',')] if delta_input else [1.5, 2.5]
            
            print(f"\n将生成 {len(gamma_values) * len(delta_values)} 个参数组合")
            
            print("\n开始实验...")
            result = self.experiment.experiment_parameter_mixing(
                prompt=prompt,
                gamma_values=gamma_values,
                delta_values=delta_values,
                tokens_per_config=40
            )
        
        # 显示结果
        self.experiment.print_summary(result)
        
        save = input("\n是否保存结果? (y/n): ").strip().lower()
        if save == 'y':
            self.experiment.save_results(result)
    
    def run_key_cross_detection_experiment(self):
        """运行密钥交叉检测实验（合并种子混合和密钥共享）"""
        print("\n" + "-"*80)
        print("实验2: 密钥交叉检测实验")
        print("-"*80 + "\n")
        print("此实验支持两种模式：")
        print("  a) 种子变体 - 同一提示词用不同hash_key生成多个变体")
        print("  b) 密钥共享 - 不同提示词，部分用共享密钥，部分用独立密钥\n")
        
        mode = input("选择模式 (a=种子变体, b=密钥共享, 默认a): ").strip().lower()
        mode = mode if mode in ['a', 'b'] else 'a'
        
        if mode == 'a':
            # 种子变体模式
            prompt = input("\n输入提示词 (回车使用默认): ").strip()
            if not prompt:
                prompt = "Write a short story about robots:"
            
            num_variations = input("输入变体数量 (默认3): ").strip()
            num_variations = int(num_variations) if num_variations else 3
            
            print("\n开始实验...")
            result = self.experiment.experiment_seed_mixing(
                prompt=prompt,
                num_variations=num_variations,
                max_new_tokens=100
            )
            
            self.experiment.print_summary(result)
            
            # 显示交叉检测详情
            print("\n交叉检测矩阵:")
            print("-"*80)
            for cd in result['cross_detection']:
                print(f"\n文本 {cd['text_id']} (Key: {cd['text_key']}):")
                for det in cd['detections']:
                    status = 'OK' if det['prediction'] else 'FAIL'
                    print(f"  {status} 检测器 (Key: {det['detector_key']}): z={det['z_score']:.2f}")
        
        else:
            # 密钥共享模式
            num_prompts = input("\n输入文本数量 (默认4): ").strip()
            num_prompts = int(num_prompts) if num_prompts else 4
            
            prompts = []
            print(f"\n输入 {num_prompts} 个提示词:")
            default_prompts = [
                "The benefits of renewable energy include",
                "In the year 2050, technology will",
                "Climate change is affecting",
                "Space exploration has led to"
            ]
            
            for i in range(num_prompts):
                prompt = input(f"  提示词 {i+1} (回车使用默认): ").strip()
                prompts.append(prompt if prompt else default_prompts[i % 4])
            
            shared_key = input("\n输入共享Hash Key (默认15485863): ").strip()
            shared_key = int(shared_key) if shared_key else 15485863
            
            print("\n开始实验...")
            result = self.experiment.experiment_key_sharing(
                prompts=prompts,
                shared_key=shared_key,
                max_new_tokens=60
            )
            
            self.experiment.print_summary(result)
            
            # 显示详细结果
            print("\n详细检测结果:")
            print("-"*80)
            for text_info, det_info in zip(result['texts'], result['individual_detections']):
                print(f"\n文本 {text_info['text_id']} ({text_info['key_type']}, Key={text_info['hash_key']}):")
                print(f"  提示: {text_info['prompt'][:50]}...")
                print(f"  正确密钥: z={det_info['correct_key_detection']['z_score']:.2f}, "
                      f"检测={'✓' if det_info['correct_key_detection']['prediction'] else '✗'}")
                print(f"  共享密钥: z={det_info['shared_key_detection']['z_score']:.2f}, "
                      f"检测={'✓' if det_info['shared_key_detection']['prediction'] else '✗'}")
        
        save = input("\n是否保存结果? (y/n): ").strip().lower()
        if save == 'y':
            self.experiment.save_results(result)
    
    def run_cross_model_key_experiments(self):
        """Sub-menu for cross-model key experiments."""
        print("\n" + "-"*80)
        print("Cross-Model Key Experiments")
        print("-"*80 + "\n")
        print("  1 - Shared key (all models reuse the same hash)")
        print("  2 - Distinct keys (each model uses its own hash)")
        choice = input("Select mode (1/2, default 1): ").strip()
        if choice == '2':
            self.run_cross_model_mixed_keys()
        else:
            self.run_cross_model_key_sharing()

    def run_cross_model_key_sharing(self):
        """运行跨模型共享密钥实验"""
        print("\n" + "-"*80)
        print("实验5: 跨模型共享密钥")
        print("-"*80 + "\n")
        print("此实验使用多个模型，用相同的密钥生成文本，然后检测混合效果\n")
        
        # 列出可用模型
        config_manager = ModelConfigManager()
        available_models = config_manager.list_model_names()
        
        print("可用模型:")
        for i, model in enumerate(available_models, 1):
            print(f"  {i}. {model}")
        
        # 选择模型
        num_models = input(f"\n选择模型数量 (默认2，当前已加载1个): ").strip()
        num_models = int(num_models) if num_models else 2
        
        selected_models = [self.model_nickname]  # 已加载的模型
        print(f"\n已加载模型: {self.model_nickname}")
        
        # 构建编号映射，方便用户直接输入数字
        model_index_map = {str(i): model for i, model in enumerate(available_models, 1)}

        # 选择额外的模型
        for i in range(1, num_models):
            model_input = input(f"输入第{i+1}个模型 (编号或昵称): ").strip()
            if not model_input:
                print("  警告: 未输入内容，跳过该模型")
                continue

            # 先尝试按编号匹配
            if model_input in model_index_map:
                selected_models.append(model_index_map[model_input])
            elif model_input in available_models:
                selected_models.append(model_input)
            else:
                print(f"  警告: 模型'{model_input}'不可用，跳过")
        
        if len(selected_models) < 2:
            print("\n需要至少2个模型，实验取消")
            return
        
        print(f"\n将使用 {len(selected_models)} 个模型: {', '.join(selected_models)}")
        
        # 共享密钥配置
        shared_key = input("\n输入共享Hash Key (默认12345): ").strip()
        shared_key = int(shared_key) if shared_key else 12345
        
        gamma = input("输入Gamma值 (默认0.5): ").strip()
        gamma = float(gamma) if gamma else 0.5
        
        delta = input("输入Delta值 (默认2.0): ").strip()
        delta = float(delta) if delta else 2.0
        
        shared_config = {
            'gamma': gamma,
            'delta': delta,
            'hash_key': shared_key
        }
        
        print(f"\n共享配置: gamma={gamma}, delta={delta}, key={shared_key}")
        
        # 获取初始提示词
        default_prompt = "The future of artificial intelligence"
        initial_prompt = input(f"\n输入初始提示词 (回车使用默认): ").strip()
        initial_prompt = initial_prompt if initial_prompt else default_prompt
        
        # 链接长度配置
        continuation_tokens = input("每个模型使用前序文本的最后多少个token作为提示 (默认20): ").strip()
        continuation_tokens = int(continuation_tokens) if continuation_tokens else 20
        
        # 加载额外的模型并生成
        print("\n开始实验...")
        fragments = []
        experiments = {}
        cumulative_text = initial_prompt  # 累积文本，从初始prompt开始
        
        # 使用已加载的模型生成第一个片段
        print(f"\n[1/{len(selected_models)}] 使用 {selected_models[0]} 生成...")
        print(f"  提示词: {initial_prompt}")
        processor = self.experiment.create_watermark_processor(**shared_config)
        new_tokens = self.experiment.generate_with_watermark(
            prompt=initial_prompt,
            watermark_processor=processor,
            max_new_tokens=60
        )
        fragments.append({
            'model': selected_models[0],
            'prompt': initial_prompt,
            'new_tokens': new_tokens
        })
        cumulative_text = cumulative_text + " " + new_tokens  # 追加新生成的tokens
        print(f"  ✓ 生成完成: {new_tokens[:60]}...")
        
        # 加载并使用其他模型
        for i, model_nick in enumerate(selected_models[1:], 2):
            print(f"\n[{i}/{len(selected_models)}] 加载模型 {model_nick}...")
            
            try:
                exp = HybridWatermarkExperiment(
                    model_nickname=model_nick,
                    device=self.args.device
                )
                experiments[model_nick] = exp
                
                # 从前一个文本的末尾提取continuation prompt
                tokens = cumulative_text.split()
                continuation_prompt = ' '.join(tokens[-continuation_tokens:])
                print(f"  提示词 (续接): ...{continuation_prompt}")
                
                processor = exp.create_watermark_processor(**shared_config)
                new_tokens = exp.generate_with_watermark(
                    prompt=continuation_prompt,
                    watermark_processor=processor,
                    max_new_tokens=60
                )
                
                fragments.append({
                    'model': model_nick,
                    'prompt': continuation_prompt,
                    'new_tokens': new_tokens
                })
                
                # 追加新生成的tokens到累积文本
                cumulative_text = cumulative_text + " " + new_tokens
                print(f"  ✓ 生成完成: {new_tokens[:60]}...")
                print(f"  ✓ 混合文本片段: {cumulative_text[-60:] if len(cumulative_text) > 60 else cumulative_text}...")
            except Exception as e:
                print(f"  ✗ 错误: {e}")
        
        # 混合文本就是最终的累积文本
        mixed_text = cumulative_text
        
        # 使用共享密钥检测
        print("\n使用共享密钥检测混合文本...")
        detector = self.experiment.create_watermark_detector(
            gamma=shared_config['gamma'],
            seeding_scheme='selfhash',
            hash_key=shared_config['hash_key']
        )
        result = detector.detect(mixed_text)
        
        # 显示结果
        print("\n" + "="*80)
        print("实验结果")
        print("="*80)
        
        print(f"\n初始提示词: {initial_prompt}")
        print(f"使用的模型: {', '.join(selected_models)}")
        print(f"共享配置: γ={gamma}, δ={delta}, key={shared_key}")
        print(f"续接token数: {continuation_tokens}")
        
        print("\n各模型生成过程:")
        for i, frag in enumerate(fragments, 1):
            print(f"\n片段 {i} - 模型: {frag['model']}")
            print(f"  提示: {frag['prompt'][:80]}...")
            print(f"  新生成: {frag['new_tokens'][:100]}...")
        
        print(f"\n最终混合文本 (长度: {len(mixed_text)} 字符):")
        print(f"  开头: {mixed_text[:150]}...")
        if len(mixed_text) > 300:
            print(f"  结尾: ...{mixed_text[-150:]}")
        
        print(f"\n混合文本检测结果:")
        print(f"  Z-score: {result['z_score']:.4f}")
        print(f"  P-value: {result['p_value']:.6f}")
        print(f"  检测结果: {'✓ 检测到水印' if result['prediction'] else '✗ 未检测到水印'}")
        print(f"  绿色token比例: {result['green_fraction']:.4f}")
        
        # 清理额外加载的模型
        for exp in experiments.values():
            del exp
        
        save = input("\n是否保存结果? (y/n): ").strip().lower()
        if save == 'y':
            result_data = {
                'experiment_type': 'cross_model_key_sharing',
                'initial_prompt': initial_prompt,
                'continuation_tokens': continuation_tokens,
                'models': selected_models,
                'shared_config': shared_config,
                'fragments': fragments,
                'mixed_text': mixed_text,
                'detection': {
                    'z_score': float(result['z_score']),
                    'p_value': float(result['p_value']),
                    'prediction': bool(result['prediction']),
                    'green_fraction': float(result['green_fraction'])
                }
            }
            self.experiment.save_results(result_data)
    
    
    def run_cross_model_mixed_keys(self):
        """Run cross-model experiment using distinct watermark keys."""
        print("\n" + "-"*80)
        print("Experiment 4: Cross-Model Distinct Keys")
        print("-"*80 + "\n")
        print("This experiment chains multiple models, each with its own hash key,")
        print("shows combined detections, and reports a cross-key matrix.\n")

        config_manager = ModelConfigManager()
        available_models = config_manager.list_model_names()
        print("Available models:")
        for idx, model in enumerate(available_models, 1):
            print(f"  {idx}. {model}")

        num_models = input("\nNumber of models to use (default 2, current session already loaded 1): ").strip()
        num_models = int(num_models) if num_models else 2

        selected_models = [self.model_nickname]
        print(f"\nPrimary model: {self.model_nickname}")

        model_index_map = {str(i): model for i, model in enumerate(available_models, 1)}
        for i in range(1, num_models):
            model_input = input(f"Select model #{i+1} (index or nickname): ").strip()
            if not model_input:
                print("  Warning: empty input, skipping this slot.")
                continue
            if model_input in model_index_map:
                selected_models.append(model_index_map[model_input])
            elif model_input in available_models:
                selected_models.append(model_input)
            else:
                print(f"  Warning: model '{model_input}' not found, skipping.")

        if len(selected_models) < 2:
            print("\nAt least two models are required. Experiment cancelled.")
            return

        print(f"\nUsing {len(selected_models)} models: {', '.join(selected_models)}")

        default_base_key = 12345
        model_keys = []
        for idx, model in enumerate(selected_models, 1):
            suggested_key = default_base_key + idx * 50000
            key_input = input(f"Hash key for {model} (default {suggested_key}): ").strip()
            model_key = int(key_input) if key_input else suggested_key
            model_keys.append(model_key)

        gamma_input = input("\nGamma (default 0.5): ").strip()
        gamma = float(gamma_input) if gamma_input else 0.5
        delta_input = input("Delta (default 2.0): ").strip()
        delta = float(delta_input) if delta_input else 2.0

        initial_prompt = input("\nInitial prompt (default 'The future of artificial intelligence'): ").strip()
        if not initial_prompt:
            initial_prompt = "The future of artificial intelligence"

        continuation_tokens_input = input("Continuation tokens between models (default 20): ").strip()
        continuation_tokens = int(continuation_tokens_input) if continuation_tokens_input else 20

        max_new_tokens_input = input("Max new tokens per model (default 60): ").strip()
        max_new_tokens = int(max_new_tokens_input) if max_new_tokens_input else 60

        fragments = []
        experiments = {self.model_nickname: self.experiment}
        cumulative_text = initial_prompt

        for idx, (model, key) in enumerate(zip(selected_models, model_keys), start=1):
            if model in experiments:
                exp = experiments[model]
            else:
                exp = HybridWatermarkExperiment(model_nickname=model, device=self.args.device)
                experiments[model] = exp

            if idx == 1:
                prompt = initial_prompt
            else:
                tokens = cumulative_text.split()
                if tokens and len(tokens) >= continuation_tokens:
                    prompt = " ".join(tokens[-continuation_tokens:])
                else:
                    prompt = cumulative_text

            print(f"\n[{idx}/{len(selected_models)}] Using {model} (key={key})")
            print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

            processor = exp.create_watermark_processor(
                gamma=gamma,
                delta=delta,
                seeding_scheme='selfhash',
                hash_key=key
            )
            new_tokens = exp.generate_with_watermark(
                prompt=prompt,
                watermark_processor=processor,
                max_new_tokens=max_new_tokens
            )

            fragments.append({
                'fragment_id': idx,
                'model': model,
                'hash_key': key,
                'prompt': prompt,
                'new_tokens': new_tokens
            })

            cumulative_text = (cumulative_text + " " + new_tokens).strip()
            print(f"  [OK] generated: {new_tokens[:60]}{'...' if len(new_tokens) > 60 else ''}")

        combined_text = cumulative_text

        detectors = {key: self.experiment.create_watermark_detector(
            gamma=gamma,
            seeding_scheme='selfhash',
            hash_key=key
        ) for key in model_keys}

        combined_detections = []
        for key, detector in detectors.items():
            det = detector.detect(combined_text)
            combined_detections.append({
                'hash_key': key,
                'z_score': float(det['z_score']),
                'prediction': bool(det['prediction']),
                'green_fraction': float(det.get('green_fraction', 0.0))
            })

        cross_detections = []
        for fragment in fragments:
            detections = []
            for key, detector in detectors.items():
                det = detector.detect(fragment['new_tokens'])
                detections.append({
                    'hash_key': key,
                    'z_score': float(det['z_score']),
                    'prediction': bool(det['prediction'])
                })
            cross_detections.append({
                'fragment_id': fragment['fragment_id'],
                'model': fragment['model'],
                'hash_key': fragment['hash_key'],
                'text_length': len(fragment['new_tokens']),
                'detections': detections
            })

        print("\nCombined text detections:")
        for det in combined_detections:
            status = 'OK' if det['prediction'] else 'FAIL'
            frac = det['green_fraction']
            print(f"  Key {det['hash_key']}: {status} (z={det['z_score']:.2f}, green={frac:.2%})")

        print("\nPer-fragment cross detection:")
        for row in cross_detections:
            print(f"Fragment {row['fragment_id']} [{row['model']}] (key={row['hash_key']}):")
            for det in row['detections']:
                status = 'OK' if det['prediction'] else 'FAIL'
                print(f"    Detector key {det['hash_key']}: {status} (z={det['z_score']:.2f})")

        result = {
            'experiment_type': 'cross_model_distinct_keys',
            'initial_prompt': initial_prompt,
            'continuation_tokens': continuation_tokens,
            'max_new_tokens': max_new_tokens,
            'models': selected_models,
            'model_keys': model_keys,
            'gamma': gamma,
            'delta': delta,
            'fragments': fragments,
            'combined_text': combined_text,
            'combined_detections': combined_detections,
            'cross_detections': cross_detections
        }

        save = input("\nSave results? (y/n): ").strip().lower()
        if save == 'y':
            self.experiment.save_results(result)

    # ========== Statistical evaluation experiments ==========


    def _create_detector(self, watermark_config: Dict) -> WatermarkDetector:
        """创建水印检测器（共用方法）"""
        # 构造包含hash_key的seeding_scheme
        hash_key = watermark_config.get('hash_key', 15485863)
        seeding_scheme = watermark_config.get('seeding_scheme', 'selfhash')
        
        # 如果使用的是预定义scheme但需要自定义hash_key，使用freeform格式
        if seeding_scheme == 'selfhash' and hash_key != 15485863:
            # selfhash = anchored_minhash_prf-4-True-hash_key
            seeding_scheme = f"ff-anchored_minhash_prf-4-True-{hash_key}"
        elif seeding_scheme == 'lefthash' and hash_key != 15485863:
            # lefthash = additive_prf-1-False-hash_key
            seeding_scheme = f"ff-additive_prf-1-False-{hash_key}"
        
        return WatermarkDetector(
            vocab=list(self.experiment.tokenizer.get_vocab().values()),
            gamma=watermark_config.get('gamma', 0.5),
            seeding_scheme=seeding_scheme,
            device=self.args.device,
            tokenizer=self.experiment.tokenizer,
            z_threshold=3.0  # 降低阈值从4.0到3.0，提高检测灵敏度
        )
    
    def _get_watermark_config(self) -> Dict:
        """获取水印配置（共用方法）"""
        print("\n水印配置:")
        
        gamma = input("  Gamma值 (默认0.5): ").strip()
        gamma = float(gamma) if gamma else 0.5
        
        delta = input("  Delta值 (默认2.0): ").strip()
        delta = float(delta) if delta else 2.0
        
        hash_key = input("  Hash Key (默认15485863): ").strip()
        hash_key = int(hash_key) if hash_key else 15485863
        
        return {
            'gamma': gamma,
            'delta': delta,
            'seeding_scheme': 'selfhash',
            'hash_key': hash_key
        }
    
    def _generate_text_pair(
        self,
        prompt: str,
        watermark_config: Dict,
        max_tokens: int,
        generate_unwatermarked: bool = True
    ):
        """生成带水印和无水印文本对（共用方法）"""
        # 创建水印处理器
        processor = self.experiment.create_watermark_processor(**watermark_config)
        
        # 生成带水印文本
        print("  生成带水印文本...")
        watermarked_text = self.experiment.generate_with_watermark(
            prompt, processor, max_tokens
        )
        print(f"  ✓ 完成 ({len(watermarked_text)} 字符)")
        
        unwatermarked_text = None
        if generate_unwatermarked:
            gen_unwm = input("\n是否生成无水印对照文本? (y/n, 默认y): ").strip().lower()
            if gen_unwm != 'n':
                print("  生成无水印文本...")
                # 生成无水印文本
                inputs = self.experiment.tokenizer(prompt, return_tensors="pt").to(self.args.device)
                with torch.no_grad():
                    output_tokens = self.experiment.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.experiment.tokenizer.pad_token_id,
                        eos_token_id=self.experiment.tokenizer.eos_token_id
                    )
                generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1]:]
                unwatermarked_text = self.experiment.tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True
                )[0]
                print(f"  ✓ 完成 ({len(unwatermarked_text)} 字符)")
        
        return watermarked_text, unwatermarked_text
    
    def run_sliding_window_detection(self):
        """滑动窗口检测"""
        print("\n" + "-"*80)
        print("实验6: 滑动窗口检测")
        print("-"*80 + "\n")
        print("此实验在文本上滑动固定大小的窗口，计算每个窗口的z-score。\n")
        
        # 获取参数
        prompt = input("输入生成提示词 (默认: The impact of AI): ").strip()
        prompt = prompt if prompt else "The impact of artificial intelligence"
        
        max_tokens = input("生成token数 (默认200): ").strip()
        max_tokens = int(max_tokens) if max_tokens else 200
        
        watermark_config = self._get_watermark_config()
        
        # 生成文本
        print("\n生成文本...")
        watermarked_text, _ = self._generate_text_pair(
            prompt, watermark_config, max_tokens, generate_unwatermarked=False
        )
        
        # 窗口参数
        print("\n滑动窗口参数:")
        window_size = input("  窗口大小 (tokens, 默认100): ").strip()
        window_size = int(window_size) if window_size else 100
        
        stride = input("  滑动步长 (tokens, 默认25): ").strip()
        stride = int(stride) if stride else 25
        
        # 创建检测器
        detector = self._create_detector(watermark_config)
        
        # 执行滑动窗口检测
        print("\n执行滑动窗口检测...")
        tokens = self.experiment.tokenizer.encode(watermarked_text, add_special_tokens=False)
        text_length = len(tokens)
        
        window_positions = []
        z_scores = []
        predictions = []
        green_fractions = []
        
        for start_pos in tqdm(range(0, text_length - window_size + 1, stride),
                             desc="滑动窗口"):
            end_pos = start_pos + window_size
            window_tokens = tokens[start_pos:end_pos]
            window_text = self.experiment.tokenizer.decode(window_tokens, skip_special_tokens=True)
            
            result = detector.detect(window_text)
            
            window_positions.append(start_pos)
            z_scores.append(result['z_score'])
            predictions.append(result['prediction'])
            green_fractions.append(result['green_fraction'])
        
        # 显示结果
        print("\n" + "="*80)
        print("检测结果")
        print("="*80)
        print(f"\n窗口大小: {window_size} tokens")
        print(f"窗口数量: {len(window_positions)}")
        print(f"平均 Z-score: {np.mean(z_scores):.4f} ± {np.std(z_scores):.4f}")
        print(f"检测率: {np.mean(predictions)*100:.1f}%")
        print(f"平均绿色token比例: {np.mean(green_fractions):.4f}")
        
        # 保存结果
        save = input("\n是否保存结果? (y/n): ").strip().lower()
        if save == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = "hybrid_watermark_results"
            os.makedirs(results_dir, exist_ok=True)
            
            result_file = os.path.join(results_dir, f"sliding_window_{timestamp}.json")
            
            # 整体检测结果
            full_detection = detector.detect(watermarked_text)
            
            results = {
                'experiment_type': 'sliding_window_detection',
                'prompt': prompt,
                'watermark_config': watermark_config,
                'generated_text': {
                    'text': watermarked_text,
                    'length_chars': len(watermarked_text),
                    'length_tokens': len(tokens),
                    'full_detection': {
                        'z_score': float(full_detection['z_score']),
                        'p_value': float(full_detection['p_value']),
                        'prediction': bool(full_detection['prediction']),
                        'green_fraction': float(full_detection['green_fraction'])
                    }
                },
                'sliding_window_params': {
                    'window_size': window_size,
                    'stride': stride,
                    'num_windows': len(window_positions)
                },
                'results': {
                    'avg_z_score': float(np.mean(z_scores)),
                    'std_z_score': float(np.std(z_scores)),
                    'detection_rate': float(np.mean(predictions)),
                    'avg_green_fraction': float(np.mean(green_fractions))
                },
                'detailed_results': [
                    {
                        'window_id': i,
                        'start_position': int(pos),
                        'z_score': float(z),
                        'prediction': bool(p),
                        'green_fraction': float(gf)
                    }
                    for i, (pos, z, p, gf) in enumerate(zip(window_positions, z_scores, predictions, green_fractions))
                ]
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✓ 结果已保存: {result_file}")
        
        # 绘图
        plot = input("\n是否生成可视化图表? (y/n): ").strip().lower()
        if plot == 'y':
            self._plot_sliding_window(
                window_positions, z_scores, predictions, green_fractions, window_size
            )
    
    def run_window_sensitivity_analysis(self):
        """窗口敏感性分析"""
        print("\n" + "-"*80)
        print("实验7: 窗口敏感性分析")
        print("-"*80 + "\n")
        print("此实验测试不同窗口大小对检测效果的影响。\n")
        
        # 获取参数
        prompt = input("输入生成提示词 (默认: The impact of AI): ").strip()
        prompt = prompt if prompt else "The impact of artificial intelligence"
        
        max_tokens = input("生成token数 (默认300): ").strip()
        max_tokens = int(max_tokens) if max_tokens else 300
        
        watermark_config = self._get_watermark_config()
        
        # 生成文本
        print("\n生成文本...")
        watermarked_text, unwatermarked_text = self._generate_text_pair(
            prompt, watermark_config, max_tokens, generate_unwatermarked=True
        )
        
        # 窗口大小范围
        print("\n窗口大小范围:")
        sizes_input = input("  输入窗口大小列表 (逗号分隔, 默认: 25,50,75,100,150,200): ").strip()
        if sizes_input:
            window_sizes = [int(s.strip()) for s in sizes_input.split(',')]
        else:
            window_sizes = [25, 50, 75, 100, 150, 200]
        
        # 创建检测器
        detector = self._create_detector(watermark_config)
        
        # 执行分析
        print("\n执行窗口敏感性分析...")
        avg_z_scores = []
        std_z_scores = []
        detection_rates = []
        false_positive_rates = []
        
        for window_size in tqdm(window_sizes, desc="窗口敏感性分析"):
            stride = max(1, int(window_size * 0.5))
            
            # 检测带水印文本
            tokens = self.experiment.tokenizer.encode(watermarked_text, add_special_tokens=False)
            z_scores_wm = []
            predictions_wm = []
            
            for start_pos in range(0, len(tokens) - window_size + 1, stride):
                window_tokens = tokens[start_pos:start_pos + window_size]
                window_text = self.experiment.tokenizer.decode(window_tokens, skip_special_tokens=True)
                result = detector.detect(window_text)
                z_scores_wm.append(result['z_score'])
                predictions_wm.append(result['prediction'])
            
            avg_z_scores.append(np.mean(z_scores_wm))
            std_z_scores.append(np.std(z_scores_wm))
            detection_rates.append(np.mean(predictions_wm))
            
            # 如果有无水印文本，计算假阳性率
            if unwatermarked_text:
                tokens_unwm = self.experiment.tokenizer.encode(unwatermarked_text, add_special_tokens=False)
                predictions_unwm = []
                
                for start_pos in range(0, len(tokens_unwm) - window_size + 1, stride):
                    window_tokens = tokens_unwm[start_pos:start_pos + window_size]
                    window_text = self.experiment.tokenizer.decode(window_tokens, skip_special_tokens=True)
                    result = detector.detect(window_text)
                    predictions_unwm.append(result['prediction'])
                
                false_positive_rates.append(np.mean(predictions_unwm))
            else:
                false_positive_rates.append(0.0)
        
        # 显示结果
        print("\n" + "="*80)
        print("分析结果")
        print("="*80)
        print(f"\n测试窗口范围: {min(window_sizes)} - {max(window_sizes)} tokens")
        
        optimal_idx = avg_z_scores.index(max(avg_z_scores))
        print(f"最优窗口大小: {window_sizes[optimal_idx]} tokens")
        print(f"最高平均 Z-score: {avg_z_scores[optimal_idx]:.4f}")
        print(f"最高检测率: {max(detection_rates)*100:.1f}%")
        
        print("\n各窗口大小详情:")
        for i, size in enumerate(window_sizes):
            print(f"  {size:3d} tokens: Z={avg_z_scores[i]:6.3f} ± {std_z_scores[i]:5.3f}, "
                  f"检测率={detection_rates[i]*100:5.1f}%, "
                  f"假阳性={false_positive_rates[i]*100:5.1f}%")
        
        # 保存结果
        save = input("\n是否保存结果? (y/n): ").strip().lower()
        if save == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = "hybrid_watermark_results"
            os.makedirs(results_dir, exist_ok=True)
            
            result_file = os.path.join(results_dir, f"window_sensitivity_{timestamp}.json")
            
            # 整体检测结果
            watermarked_detection = detector.detect(watermarked_text)
            unwatermarked_detection = detector.detect(unwatermarked_text) if unwatermarked_text else None
            
            results = {
                'experiment_type': 'window_sensitivity_analysis',
                'prompt': prompt,
                'watermark_config': watermark_config,
                'generated_texts': {
                    'watermarked': {
                        'text': watermarked_text,
                        'length_chars': len(watermarked_text),
                        'length_tokens': len(self.experiment.tokenizer.encode(watermarked_text, add_special_tokens=False)),
                        'full_detection': {
                            'z_score': float(watermarked_detection['z_score']),
                            'p_value': float(watermarked_detection['p_value']),
                            'prediction': bool(watermarked_detection['prediction']),
                            'green_fraction': float(watermarked_detection['green_fraction'])
                        }
                    },
                    'unwatermarked': {
                        'text': unwatermarked_text,
                        'length_chars': len(unwatermarked_text) if unwatermarked_text else None,
                        'length_tokens': len(self.experiment.tokenizer.encode(unwatermarked_text, add_special_tokens=False)) if unwatermarked_text else None,
                        'full_detection': {
                            'z_score': float(unwatermarked_detection['z_score']) if unwatermarked_detection else None,
                            'p_value': float(unwatermarked_detection['p_value']) if unwatermarked_detection else None,
                            'prediction': bool(unwatermarked_detection['prediction']) if unwatermarked_detection else None,
                            'green_fraction': float(unwatermarked_detection['green_fraction']) if unwatermarked_detection else None
                        } if unwatermarked_detection else None
                    } if unwatermarked_text else None
                },
                'tested_window_sizes': window_sizes,
                'optimal_window': {
                    'size': window_sizes[optimal_idx],
                    'avg_z_score': float(avg_z_scores[optimal_idx]),
                    'detection_rate': float(detection_rates[optimal_idx])
                },
                'detailed_results': [
                    {
                        'window_size': size,
                        'avg_z_score': float(avg_z),
                        'std_z_score': float(std_z),
                        'detection_rate': float(det_rate),
                        'false_positive_rate': float(fp_rate)
                    }
                    for size, avg_z, std_z, det_rate, fp_rate in zip(
                        window_sizes, avg_z_scores, std_z_scores, detection_rates, false_positive_rates
                    )
                ]
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✓ 结果已保存: {result_file}")
        
        # 绘图
        plot = input("\n是否生成可视化图表? (y/n): ").strip().lower()
        if plot == 'y':
            self._plot_window_sensitivity(
                window_sizes, avg_z_scores, std_z_scores, 
                detection_rates, false_positive_rates
            )
    
    def run_minimum_length_analysis(self):
        """最小可检测长度分析"""
        print("\n" + "-"*80)
        print("实验8: 最小可检测长度分析")
        print("-"*80 + "\n")
        print("此实验找出可靠检测水印所需的最小文本长度。\n")
        
        # 获取参数
        prompt = input("输入生成提示词 (默认: The impact of AI): ").strip()
        prompt = prompt if prompt else "The impact of artificial intelligence"
        
        max_tokens = input("生成token数 (默认300): ").strip()
        max_tokens = int(max_tokens) if max_tokens else 300
        
        watermark_config = self._get_watermark_config()
        
        # 生成文本
        print("\n生成文本...")
        watermarked_text, _ = self._generate_text_pair(
            prompt, watermark_config, max_tokens, generate_unwatermarked=False
        )
        
        # 长度范围
        print("\n测试长度范围:")
        min_len = input("  最小长度 (tokens, 默认20): ").strip()
        min_len = int(min_len) if min_len else 20
        
        max_len = input("  最大长度 (tokens, 默认250): ").strip()
        max_len = int(max_len) if max_len else 250
        
        step = input("  步长 (tokens, 默认10): ").strip()
        step = int(step) if step else 10
        
        length_range = list(range(min_len, max_len + 1, step))
        
        num_samples = input("  每个长度采样次数 (默认3): ").strip()
        num_samples = int(num_samples) if num_samples else 3
        
        # 创建检测器
        detector = self._create_detector(watermark_config)
        
        # 执行分析
        print("\n执行最小长度分析...")
        tokens = self.experiment.tokenizer.encode(watermarked_text, add_special_tokens=False)
        
        text_lengths = []
        z_scores = []
        predictions = []
        green_fractions = []
        
        for length in tqdm(length_range, desc="最小长度分析"):
            if length > len(tokens):
                continue
            
            length_z_scores = []
            length_predictions = []
            length_green_fractions = []
            
            for _ in range(num_samples):
                if len(tokens) - length > 0:
                    start_pos = np.random.randint(0, len(tokens) - length + 1)
                else:
                    start_pos = 0
                
                sample_tokens = tokens[start_pos:start_pos + length]
                sample_text = self.experiment.tokenizer.decode(sample_tokens, skip_special_tokens=True)
                
                result = detector.detect(sample_text)
                
                length_z_scores.append(result['z_score'])
                length_predictions.append(result['prediction'])
                length_green_fractions.append(result['green_fraction'])
            
            text_lengths.append(length)
            z_scores.append(np.mean(length_z_scores))
            predictions.append(np.mean(length_predictions) >= 0.5)
            green_fractions.append(np.mean(length_green_fractions))
        
        # 找出最小可靠检测长度
        min_reliable_length = None
        for i, (length, pred) in enumerate(zip(text_lengths, predictions)):
            if pred and i + 2 < len(predictions) and all(predictions[i:i+3]):
                min_reliable_length = length
                break
        
        # 显示结果
        print("\n" + "="*80)
        print("分析结果")
        print("="*80)
        
        if min_reliable_length:
            print(f"\n✓ 最小可靠检测长度: {min_reliable_length} tokens")
        else:
            print(f"\n✗ 未找到可靠检测长度")
        
        print(f"\n测试长度范围: {min(length_range)} - {max(length_range)} tokens")
        
        success_lengths = [l for l, p in zip(text_lengths, predictions) if p]
        if success_lengths:
            print(f"最短成功检测: {min(success_lengths)} tokens")
        
        print("\n部分长度详情:")
        step_show = max(1, len(text_lengths) // 10)
        for i in range(0, len(text_lengths), step_show):
            length = text_lengths[i]
            z = z_scores[i]
            p = predictions[i]
            status = "✓" if p else "✗"
            print(f"  {status} {length:3d} tokens: Z={z:6.3f}, Green={green_fractions[i]:.3f}")
        
        # 保存结果
        save = input("\n是否保存结果? (y/n): ").strip().lower()
        if save == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = "hybrid_watermark_results"
            os.makedirs(results_dir, exist_ok=True)
            
            result_file = os.path.join(results_dir, f"minimum_length_{timestamp}.json")
            
            # 整体检测结果
            full_detection = detector.detect(watermarked_text)
            
            results = {
                'experiment_type': 'minimum_length_analysis',
                'prompt': prompt,
                'watermark_config': watermark_config,
                'generated_text': {
                    'text': watermarked_text,
                    'length_chars': len(watermarked_text),
                    'length_tokens': len(tokens),
                    'full_detection': {
                        'z_score': float(full_detection['z_score']),
                        'p_value': float(full_detection['p_value']),
                        'prediction': bool(full_detection['prediction']),
                        'green_fraction': float(full_detection['green_fraction'])
                    }
                },
                'test_parameters': {
                    'min_length': min_len,
                    'max_length': max_len,
                    'step_size': step,
                    'num_samples_per_length': num_samples
                },
                'summary': {
                    'min_reliable_length': min_reliable_length,
                    'min_success_length': min(success_lengths) if success_lengths else None,
                    'num_success_lengths': len(success_lengths),
                    'success_rate': float(np.mean(predictions))
                },
                'detailed_results': [
                    {
                        'length': int(length),
                        'avg_z_score': float(z),
                        'prediction': bool(p),
                        'avg_green_fraction': float(gf)
                    }
                    for length, z, p, gf in zip(text_lengths, z_scores, predictions, green_fractions)
                ]
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✓ 结果已保存: {result_file}")
        
        # 绘图
        plot = input("\n是否生成可视化图表? (y/n): ").strip().lower()
        if plot == 'y':
            self._plot_minimum_length(
                text_lengths, z_scores, predictions, 
                green_fractions, min_reliable_length
            )
    
    def run_complete_statistical_evaluation(self):
        """运行完整统计评估"""
        print("\n" + "-"*80)
        print("实验9: 完整统计评估")
        print("-"*80 + "\n")
        print("此实验将执行全部三项统计分析。\n")
        
        # 获取参数
        prompt = input("输入生成提示词 (默认: The impact of AI): ").strip()
        prompt = prompt if prompt else "The impact of artificial intelligence"
        
        max_tokens = input("生成token数 (默认300): ").strip()
        max_tokens = int(max_tokens) if max_tokens else 300
        
        watermark_config = self._get_watermark_config()
        
        # 生成文本
        print("\n生成文本...")
        watermarked_text, unwatermarked_text = self._generate_text_pair(
            prompt, watermark_config, max_tokens, generate_unwatermarked=True
        )
        
        detector = self._create_detector(watermark_config)
        tokens = self.experiment.tokenizer.encode(watermarked_text, add_special_tokens=False)
        
        # 1. 滑动窗口检测
        print("\n[1/3] 执行滑动窗口检测...")
        window_size = 100
        stride = 25
        
        window_positions = []
        z_scores_sw = []
        predictions_sw = []
        green_fractions_sw = []
        
        for start_pos in tqdm(range(0, len(tokens) - window_size + 1, stride),
                             desc="滑动窗口", leave=False):
            window_tokens = tokens[start_pos:start_pos + window_size]
            window_text = self.experiment.tokenizer.decode(window_tokens, skip_special_tokens=True)
            result = detector.detect(window_text)
            
            window_positions.append(start_pos)
            z_scores_sw.append(result['z_score'])
            predictions_sw.append(result['prediction'])
            green_fractions_sw.append(result['green_fraction'])
        
        print(f"  ✓ 完成 ({len(window_positions)} 个窗口)")
        
        # 2. 窗口敏感性分析
        print("\n[2/3] 执行窗口敏感性分析...")
        window_sizes = [25, 50, 75, 100, 150, 200]
        avg_z_scores = []
        std_z_scores = []
        detection_rates = []
        
        for ws in tqdm(window_sizes, desc="窗口敏感性", leave=False):
            stride = max(1, int(ws * 0.5))
            z_scores_ws = []
            predictions_ws = []
            
            for start_pos in range(0, len(tokens) - ws + 1, stride):
                window_tokens = tokens[start_pos:start_pos + ws]
                window_text = self.experiment.tokenizer.decode(window_tokens, skip_special_tokens=True)
                result = detector.detect(window_text)
                z_scores_ws.append(result['z_score'])
                predictions_ws.append(result['prediction'])
            
            avg_z_scores.append(np.mean(z_scores_ws))
            std_z_scores.append(np.std(z_scores_ws))
            detection_rates.append(np.mean(predictions_ws))
        
        print(f"  ✓ 完成 (测试了 {len(window_sizes)} 个窗口大小)")
        
        # 3. 最小长度分析
        print("\n[3/3] 执行最小长度分析...")
        length_range = list(range(20, min(len(tokens), 250), 10))
        text_lengths = []
        z_scores_ml = []
        predictions_ml = []
        
        for length in tqdm(length_range, desc="最小长度", leave=False):
            if length > len(tokens):
                continue
            
            start_pos = 0 if len(tokens) - length <= 0 else np.random.randint(0, len(tokens) - length + 1)
            sample_tokens = tokens[start_pos:start_pos + length]
            sample_text = self.experiment.tokenizer.decode(sample_tokens, skip_special_tokens=True)
            
            result = detector.detect(sample_text)
            
            text_lengths.append(length)
            z_scores_ml.append(result['z_score'])
            predictions_ml.append(result['prediction'])
        
        print(f"  ✓ 完成 (测试了 {len(length_range)} 个长度)")
        
        # 显示摘要
        print("\n" + "="*80)
        print("完整统计评估摘要")
        print("="*80)
        
        print("\n1. 滑动窗口检测:")
        print(f"   平均 Z-score: {np.mean(z_scores_sw):.4f} ± {np.std(z_scores_sw):.4f}")
        print(f"   检测率: {np.mean(predictions_sw)*100:.1f}%")
        
        print("\n2. 窗口敏感性分析:")
        optimal_idx = avg_z_scores.index(max(avg_z_scores))
        print(f"   最优窗口大小: {window_sizes[optimal_idx]} tokens")
        print(f"   最高平均 Z-score: {avg_z_scores[optimal_idx]:.4f}")
        
        print("\n3. 最小长度分析:")
        success_lengths = [l for l, p in zip(text_lengths, predictions_ml) if p]
        if success_lengths:
            print(f"   最短成功检测: {min(success_lengths)} tokens")
        
        # 保存结果
        save = input("\n是否保存结果? (y/n): ").strip().lower()
        if save == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = "hybrid_watermark_results"
            os.makedirs(results_dir, exist_ok=True)
            
            result_file = os.path.join(results_dir, f"complete_statistical_eval_{timestamp}.json")
            
            # 计算带水印文本的检测结果
            watermarked_detection = detector.detect(watermarked_text)
            
            # 计算不带水印文本的检测结果（如果存在）
            unwatermarked_detection = None
            if unwatermarked_text:
                unwatermarked_detection = detector.detect(unwatermarked_text)
            
            results = {
                'prompt': prompt,
                'watermark_config': watermark_config,
                'generated_texts': {
                    'watermarked': {
                        'text': watermarked_text,
                        'length_chars': len(watermarked_text),
                        'length_tokens': len(tokens),
                        'detection': {
                            'z_score': float(watermarked_detection['z_score']),
                            'p_value': float(watermarked_detection['p_value']),
                            'prediction': bool(watermarked_detection['prediction']),
                            'green_fraction': float(watermarked_detection['green_fraction'])
                        }
                    },
                    'unwatermarked': {
                        'text': unwatermarked_text if unwatermarked_text else None,
                        'length_chars': len(unwatermarked_text) if unwatermarked_text else None,
                        'length_tokens': len(self.experiment.tokenizer.encode(unwatermarked_text, add_special_tokens=False)) if unwatermarked_text else None,
                        'detection': {
                            'z_score': float(unwatermarked_detection['z_score']) if unwatermarked_detection else None,
                            'p_value': float(unwatermarked_detection['p_value']) if unwatermarked_detection else None,
                            'prediction': bool(unwatermarked_detection['prediction']) if unwatermarked_detection else None,
                            'green_fraction': float(unwatermarked_detection['green_fraction']) if unwatermarked_detection else None
                        } if unwatermarked_detection else None
                    }
                },
                'sliding_window': {
                    'window_size': window_size,
                    'stride': stride,
                    'num_windows': len(window_positions),
                    'avg_z_score': float(np.mean(z_scores_sw)),
                    'std_z_score': float(np.std(z_scores_sw)),
                    'detection_rate': float(np.mean(predictions_sw)),
                    'avg_green_fraction': float(np.mean(green_fractions_sw))
                },
                'window_sensitivity': {
                    'window_sizes': window_sizes,
                    'avg_z_scores': [float(x) for x in avg_z_scores],
                    'std_z_scores': [float(x) for x in std_z_scores],
                    'detection_rates': [float(x) for x in detection_rates],
                    'optimal_window_size': window_sizes[optimal_idx],
                    'optimal_z_score': float(avg_z_scores[optimal_idx])
                },
                'minimum_length': {
                    'test_range': [min(length_range), max(length_range)],
                    'step_size': 10,
                    'num_tests': len(text_lengths),
                    'min_success_length': min(success_lengths) if success_lengths else None,
                    'all_success_lengths': success_lengths if success_lengths else []
                }
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✓ 结果已保存: {result_file}")
        
        # 生成所有图表
        plot = input("\n是否生成所有可视化图表? (y/n): ").strip().lower()
        if plot == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = "hybrid_watermark_results"
            
            print("\n生成图表...")
            
            self._plot_sliding_window(
                window_positions, z_scores_sw, predictions_sw, 
                green_fractions_sw, window_size,
                save_path=f"{results_dir}/sliding_window_{timestamp}.png"
            )
            
            self._plot_window_sensitivity(
                window_sizes, avg_z_scores, std_z_scores, 
                detection_rates, [0.0] * len(window_sizes),
                save_path=f"{results_dir}/window_sensitivity_{timestamp}.png"
            )
            
            # 找最小可靠长度
            min_reliable = None
            for i, (length, pred) in enumerate(zip(text_lengths, predictions_ml)):
                if pred and i + 2 < len(predictions_ml) and all(predictions_ml[i:i+3]):
                    min_reliable = length
                    break
            
            self._plot_minimum_length(
                text_lengths, z_scores_ml, predictions_ml,
                [0.5] * len(text_lengths), min_reliable,
                save_path=f"{results_dir}/minimum_length_{timestamp}.png"
            )
            
            print("✓ 所有图表已生成")
    
    # ========== 可视化辅助方法 ==========
    
    def _plot_sliding_window(
        self,
        window_positions: List[int],
        z_scores: List[float],
        predictions: List[bool],
        green_fractions: List[float],
        window_size: int,
        save_path: Optional[str] = None
    ):
        """绘制滑动窗口结果"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Z-score曲线
        axes[0].plot(window_positions, z_scores, marker='o', linestyle='-', linewidth=2, markersize=4)
        axes[0].axhline(y=3.0, color='r', linestyle='--', label='Threshold (z=3)')
        axes[0].set_xlabel('Window Start Position (tokens)')
        axes[0].set_ylabel('Z-score')
        axes[0].set_title('Z-score Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 绿色token比例
        axes[1].plot(window_positions, green_fractions, marker='s', linestyle='-', 
                    linewidth=2, markersize=4, color='green')
        axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Expected (γ=0.5)')
        axes[1].set_xlabel('Window Start Position (tokens)')
        axes[1].set_ylabel('Green Token Fraction')
        axes[1].set_title('Green Token Fraction Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 检测结果
        detection_map = [1 if pred else 0 for pred in predictions]
        axes[2].bar(window_positions, detection_map, width=window_size * 0.8, 
                   color='blue', alpha=0.6)
        axes[2].set_xlabel('Window Start Position (tokens)')
        axes[2].set_ylabel('Detection (1=Yes, 0=No)')
        axes[2].set_title('Detection Results')
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"hybrid_watermark_results/sliding_window_{timestamp}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        
        plt.close()
    
    def _plot_window_sensitivity(
        self,
        window_sizes: List[int],
        avg_z_scores: List[float],
        std_z_scores: List[float],
        detection_rates: List[float],
        false_positive_rates: List[float],
        save_path: Optional[str] = None
    ):
        """绘制窗口敏感性结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 平均Z-score
        axes[0, 0].errorbar(window_sizes, avg_z_scores, yerr=std_z_scores, 
                           marker='o', capsize=5, linewidth=2, markersize=8)
        axes[0, 0].axhline(y=3.0, color='r', linestyle='--', label='Threshold (z=3)')
        axes[0, 0].set_xlabel('Window Size (tokens)')
        axes[0, 0].set_ylabel('Average Z-score')
        axes[0, 0].set_title('Average Z-score vs Window Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 检测率
        axes[0, 1].plot(window_sizes, detection_rates, marker='s', 
                       linewidth=2, markersize=8, color='green')
        axes[0, 1].axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
        axes[0, 1].set_xlabel('Window Size (tokens)')
        axes[0, 1].set_ylabel('Detection Rate')
        axes[0, 1].set_title('Detection Rate vs Window Size')
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Z-score标准差
        axes[1, 0].plot(window_sizes, std_z_scores, marker='^', 
                       linewidth=2, markersize=8, color='orange')
        axes[1, 0].set_xlabel('Window Size (tokens)')
        axes[1, 0].set_ylabel('Z-score Std Dev')
        axes[1, 0].set_title('Z-score Stability vs Window Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 假阳性率
        if any(false_positive_rates):
            axes[1, 1].plot(window_sizes, false_positive_rates, marker='D', 
                           linewidth=2, markersize=8, color='red')
            axes[1, 1].axhline(y=0.05, color='gray', linestyle='--', label='5% threshold')
            axes[1, 1].set_xlabel('Window Size (tokens)')
            axes[1, 1].set_ylabel('False Positive Rate')
            axes[1, 1].set_title('False Positive Rate vs Window Size')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No unwatermarked text',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"hybrid_watermark_results/window_sensitivity_{timestamp}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        
        plt.close()
    
    def _plot_minimum_length(
        self,
        text_lengths: List[int],
        z_scores: List[float],
        predictions: List[bool],
        green_fractions: List[float],
        min_reliable_length: Optional[int],
        save_path: Optional[str] = None
    ):
        """绘制最小长度分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Z-score vs 长度
        axes[0, 0].plot(text_lengths, z_scores, marker='o', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=3.0, color='r', linestyle='--', label='Threshold (z=3)')
        if min_reliable_length:
            axes[0, 0].axvline(x=min_reliable_length, color='g', linestyle='--',
                              label=f'Min: {min_reliable_length}')
        axes[0, 0].set_xlabel('Text Length (tokens)')
        axes[0, 0].set_ylabel('Z-score')
        axes[0, 0].set_title('Z-score vs Text Length')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 绿色token比例
        axes[0, 1].plot(text_lengths, green_fractions, marker='s', 
                       linewidth=2, markersize=6, color='green')
        axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', label='Expected (γ=0.5)')
        if min_reliable_length:
            axes[0, 1].axvline(x=min_reliable_length, color='g', linestyle='--')
        axes[0, 1].set_xlabel('Text Length (tokens)')
        axes[0, 1].set_ylabel('Green Token Fraction')
        axes[0, 1].set_title('Green Token Fraction vs Text Length')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 检测成功/失败
        colors = ['green' if pred else 'red' for pred in predictions]
        axes[1, 0].scatter(text_lengths, z_scores, c=colors, s=100, 
                          alpha=0.6, edgecolors='black')
        axes[1, 0].axhline(y=3.0, color='r', linestyle='--', linewidth=2)
        if min_reliable_length:
            axes[1, 0].axvline(x=min_reliable_length, color='g', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Text Length (tokens)')
        axes[1, 0].set_ylabel('Z-score')
        axes[1, 0].set_title('Detection Success (Green) / Failure (Red)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 检测率累积
        detection_rate_cumulative = []
        for i in range(len(predictions)):
            detection_rate_cumulative.append(np.mean(predictions[:i+1]))
        
        axes[1, 1].plot(text_lengths, detection_rate_cumulative, marker='^',
                       linewidth=2, markersize=6, color='purple')
        axes[1, 1].axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
        if min_reliable_length:
            axes[1, 1].axvline(x=min_reliable_length, color='g', linestyle='--')
        axes[1, 1].set_xlabel('Text Length (tokens)')
        axes[1, 1].set_ylabel('Cumulative Detection Rate')
        axes[1, 1].set_title('Cumulative Detection Rate vs Length')
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"hybrid_watermark_results/minimum_length_{timestamp}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        
        plt.close()
    
    def show_experiment_info(self):
        """Display reference information for each experiment."""
        print("\n" + "="*80)
        print("Experiment Reference")
        print("="*80 + "\n")

        print("[Hybrid Watermark Experiments]")
        print("-"*80)

        print("Experiment 1: Fragment / Parameter Mixing")
        print("  - Mode A (Fragment level): assign distinct gamma/delta/hash per segment.")
        print("  - Mode B (Grid scan): sweep gamma/delta pairs to study detection trends.")

        print("Experiment 2: Seed & Key Cross Detection")
        print("  - Mode A (Seed variants): same model, different hash keys; evaluate cross-key confusion.")
        print("  - Mode B (Shared vs. individual): mix shared and private keys within a batch.")

        print("Experiment 3: Cross-Model Key Experiments")
        print("  - Shared-key mode: chain multiple models while reusing one hash key to validate signal compatibility.")
        print("  - Distinct-key mode: assign unique hashes per model and report a cross-key detection matrix to prove exclusivity.")

        print("\n[Statistical Evaluations]")
        print("-"*80)

        print("Experiment 4: Sliding Window Detection")
        print("  - Score contiguous windows to observe spatial consistency of the watermark.")
        print("  - Outputs a z-score curve along the text.")

        print("Experiment 5: Window Sensitivity Analysis")
        print("  - Sweep window sizes (e.g., 25/50/100/200) to balance accuracy and recall.")
        print("  - Produces curves of window size vs. mean z-score / detection rate.")

        print("Experiment 6: Minimum Detectable Length")
        print("  - Identify the shortest text length that achieves reliable detection (target FP/FN < 5%).")
        print("  - Visualises z-score growth against length.")

        print("Experiment 7: Complete Statistical Suite")
        print("  - Runs Experiments 5–7 sequentially and aggregates the plots/results.")

        print("\n[Robustness Testing]")
        print("-"*80)

        print("Experiment 8: Multi-LLM Paraphrase Robustness")
        print("  - Model A generates watermarked text; Model B paraphrases it.")
        print("  - Measures watermark detectability before/after paraphrasing.")
        print("  - Evaluates semantic similarity and watermark decay/survival rate.")

        print("Experiment 9: Watermarked Paraphrase Chain (Different Green/Red Lists)")
        print("  - Generator embeds watermark with one hash key; paraphraser embeds watermark with a different hash key.")
        print("  - Tests 'same LLM, different green/red list' sequential generation/refinement.")
        print("  - Detects text with both generator and paraphraser keys to measure dual watermark survival.")

        print("="*80 + "\n")
    
    def run_multi_llm_robustness_test(self):
        """运行多模型链路水印实验"""
        print("\n" + "-" * 80)
        print("Experiment 8: Multi-LLM Chain Watermark Robustness")
        print("-" * 80 + "\n")

        try:
            from multi_llm_chain_experiment import MultiLLMChainExperiment
        except ImportError as exc:
            print(f"❌ Cannot import multi_llm_chain_experiment: {exc}")
            return

        config_manager = ModelConfigManager()
        available_models = config_manager.list_model_names()

        prompt = input("Enter prompt (default: Write a short story about artificial intelligence.): ").strip()
        if not prompt:
            prompt = "Write a short story about artificial intelligence."

        paraphraser_input = input("Paraphraser models (comma, default: qwen-3-8b): ").strip()
        paraphraser_models = [m.strip() for m in paraphraser_input.split(',') if m.strip()] if paraphraser_input else ["qwen-3-8b"]

        missing = [m for m in paraphraser_models if m not in available_models]
        if missing:
            print(f"❌ 未找到改写模型: {', '.join(missing)}")
            return

        paraphrase_presets = {
            "1": (
                "Standard paraphrase",
                "Paraphrase the following text while preserving its meaning:",
            ),
            "2": (
                "Structure shuffle",
                "Rewrite the passage below using different sentence structures and synonyms while keeping every fact accurate:",
            ),
            "3": (
                "Creative retelling",
                "Retell the passage in a fresh narrative voice using varied vocabulary but keep the chronology and key details unchanged:",
            ),
            "4": (
                "Summary-expand",
                "Summarize the core ideas of the passage in your own words, then expand each point into full sentences to produce a cohesive paraphrase:",
            ),
        }

        print("\nParaphrase instruction presets:")
        for key, (label, _) in paraphrase_presets.items():
            print(f"  {key} - {label}")
        print("  0 - Run all presets sequentially")
        print("  c - Custom instruction")

        paraphrase_choice = input("Select paraphrase preset (default: 1): ").strip().lower() or "1"
        preset_sequence = []
        if paraphrase_choice == "c":
            custom_instruction = input("Enter custom paraphrase instruction: ").strip()
            instruction = custom_instruction or paraphrase_presets["1"][1]
            preset_sequence.append(("Custom", instruction))
        elif paraphrase_choice == "0":
            preset_sequence.extend(paraphrase_presets.values())
        elif paraphrase_choice in paraphrase_presets:
            preset_sequence.append(paraphrase_presets[paraphrase_choice])
        else:
            preset_sequence.append(paraphrase_presets["1"])

        print("Selected paraphrase methods:")
        for label, _ in preset_sequence:
            print(f"  - {label}")

        compare_mode = input("Compare multiple generator models? (y/N): ").strip().lower() == 'y'
        if compare_mode:
            generator_input = input("Generator models (comma, default includes current model): ").strip()
            generator_models = [m.strip() for m in generator_input.split(',') if m.strip()]
            if not generator_models:
                generator_models = [self.model_nickname]
        else:
            generator_models = [self.model_nickname]

        missing_gens = [m for m in generator_models if m not in available_models]
        if missing_gens:
            print(f"❌ 未找到生成模型: {', '.join(missing_gens)}")
            return

        z_threshold_input = input("Z-score threshold (default 4.0): ").strip()
        z_threshold = float(z_threshold_input) if z_threshold_input else 4.0

        experiment = MultiLLMChainExperiment(
            generator_default=generator_models[0],
            paraphraser_defaults=paraphraser_models,
            device=self.experiment.device,
            config_manager=config_manager,
        )

        prefetched_generations: Optional[Dict[str, Dict[str, object]]] = None
        if len(preset_sequence) > 1:
            prefetched_generations = {}
            print("\nPreparing shared watermarked text for selected paraphrase presets...")
            for generator in generator_models:
                try:
                    watermarked_text, generation_meta, tokenizer = experiment.generate_with_watermark(
                        prompt,
                        generator_nickname=generator,
                    )
                except Exception as exc:
                    print(f"❌ Failed to generate base text with {generator}: {exc}")
                    return
                prefetched_generations[generator] = {
                    "watermarked_text": watermarked_text,
                    "generation_metadata": generation_meta,
                    "tokenizer": tokenizer,
                }

        try:
            for label, paraphrase_instruction in preset_sequence:
                print(f"\n>>> Running paraphrase preset: {label}")
                if compare_mode and len(generator_models) > 1:
                    result = experiment.compare_across_models(
                        prompt=prompt,
                        generator_models=generator_models,
                        paraphraser_models=paraphraser_models,
                        z_threshold=z_threshold,
                        paraphrase_instruction=paraphrase_instruction,
                        prefetched_generations=prefetched_generations,
                    )
                    result["paraphrase_mode"] = label
                    for individual in result.get("individual_results", []):
                        experiment.print_summary(individual)
                    summary = result.get("summary", {})
                    print("\nOverall survival comparison:")
                    print(f"  Generators: {', '.join(summary.get('generator_models', []))}")
                    print(f"  Avg survival: {summary.get('average_survival_rate', 0.0):.2%}")
                    print(f"  Best survival: {summary.get('highest_survival', 0.0):.2%}")
                    print(f"  Worst survival: {summary.get('lowest_survival', 0.0):.2%}")
                else:
                    result = experiment.run_chain(
                        prompt=prompt,
                        generator_model=generator_models[0],
                        paraphraser_models=paraphraser_models,
                        z_threshold=z_threshold,
                        paraphrase_instruction=paraphrase_instruction,
                        prefetched=(prefetched_generations or {}).get(generator_models[0]),
                    )
                    result["paraphrase_mode"] = label
                    experiment.print_summary(result)

                slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"multi_llm_chain_{slug}_{timestamp}.json"
                experiment.save_results(result, filename=filename)
            print("\n✅ Experiment completed!")
        except Exception as exc:
            print(f"❌ Experiment failed: {exc}")

    def run_watermarked_paraphrase_chain(self):
        """运行带水印改写链路实验（同一模型用不同 green/red list）"""
        print("\n" + "-" * 80)
        print("Experiment 9: Watermarked Paraphrase Chain (Different Green/Red Lists)")
        print("-" * 80 + "\n")
        print("This experiment embeds different watermarks during paraphrasing,")
        print("allowing you to test watermark survival with distinct green/red lists.\n")

        try:
            from multi_llm_chain_experiment import MultiLLMChainExperiment
        except ImportError as exc:
            print(f"❌ Cannot import multi_llm_chain_experiment: {exc}")
            return

        config_manager = ModelConfigManager()
        available_models = config_manager.list_model_names()

        prompt = input("Enter prompt (default: Write a short story about artificial intelligence.): ").strip()
        if not prompt:
            prompt = "Write a short story about artificial intelligence."

        paraphraser_input = input("Paraphraser models (comma, default: qwen-3-8b): ").strip()
        paraphraser_models = [m.strip() for m in paraphraser_input.split(',') if m.strip()] if paraphraser_input else ["qwen-3-8b"]

        missing = [m for m in paraphraser_models if m not in available_models]
        if missing:
            print(f"❌ 未找到改写模型: {', '.join(missing)}")
            return

        z_threshold_input = input("Z-score threshold (default 3.0): ").strip()
        z_threshold = float(z_threshold_input) if z_threshold_input else 3.0

        # 配置生成器水印
        print("\n=== Generator Watermark Configuration ===")
        gen_gamma_input = input("Generator gamma (default 0.25): ").strip()
        gen_gamma = float(gen_gamma_input) if gen_gamma_input else 0.25
        gen_delta_input = input("Generator delta (default 2.0): ").strip()
        gen_delta = float(gen_delta_input) if gen_delta_input else 2.0
        gen_hash_key_input = input("Generator hash_key (default 15485863): ").strip()
        gen_hash_key = int(gen_hash_key_input) if gen_hash_key_input else 15485863

        generator_watermark_config = {
            "gamma": gen_gamma,
            "delta": gen_delta,
            "seeding_scheme": "selfhash",
            "hash_key": gen_hash_key,
        }

        # 配置改写器水印
        print("\n=== Paraphraser Watermark Configurations ===")
        enable_paraphrase_wm = input("Embed watermark during paraphrasing? (y/N): ").strip().lower() == 'y'
        
        paraphraser_watermark_configs = None
        if enable_paraphrase_wm:
            paraphraser_watermark_configs = []
            base_hash_key = 50000000
            for idx, paraphraser in enumerate(paraphraser_models):
                print(f"\n--- Paraphraser {idx+1}: {paraphraser} ---")
                auto_config = input(f"  Auto-configure (different hash_key from generator)? (Y/n): ").strip().lower()
                if auto_config != 'n':
                    para_hash_key = base_hash_key + idx * 10000
                    para_config = {
                        "gamma": gen_gamma,
                        "delta": gen_delta,
                        "seeding_scheme": "selfhash",
                        "hash_key": para_hash_key,
                    }
                    print(f"  Auto: gamma={gen_gamma}, delta={gen_delta}, hash_key={para_hash_key}")
                else:
                    para_gamma_input = input(f"  Gamma (default {gen_gamma}): ").strip()
                    para_gamma = float(para_gamma_input) if para_gamma_input else gen_gamma
                    para_delta_input = input(f"  Delta (default {gen_delta}): ").strip()
                    para_delta = float(para_delta_input) if para_delta_input else gen_delta
                    para_hash_key_input = input(f"  Hash_key (default {base_hash_key + idx * 10000}): ").strip()
                    para_hash_key = int(para_hash_key_input) if para_hash_key_input else (base_hash_key + idx * 10000)
                    para_config = {
                        "gamma": para_gamma,
                        "delta": para_delta,
                        "seeding_scheme": "selfhash",
                        "hash_key": para_hash_key,
                    }
                paraphraser_watermark_configs.append(para_config)

        experiment = MultiLLMChainExperiment(
            generator_default=self.model_nickname,
            paraphraser_defaults=paraphraser_models,
            device=self.experiment.device,
            config_manager=config_manager,
        )

        try:
            result = experiment.run_chain_with_watermarked_paraphrase(
                prompt=prompt,
                generator_model=self.model_nickname,
                paraphraser_models=paraphraser_models,
                generator_watermark_config=generator_watermark_config,
                paraphraser_watermark_configs=paraphraser_watermark_configs,
                z_threshold=z_threshold,
            )

            # 显示结果
            print("\n" + "=" * 80)
            print("Experiment Results")
            print("=" * 80)
            print(f"\nPrompt: {result['prompt']}")
            print(f"Generator: {result['generator_model']}")
            print(f"Paraphrasers: {', '.join(result['paraphraser_models'])}")
            print(f"\nGenerator watermark config: gamma={generator_watermark_config['gamma']}, delta={generator_watermark_config['delta']}, hash_key={generator_watermark_config['hash_key']}")
            
            print(f"\nOriginal z-score (generator key): {result['original_detection']['z_score']:.4f}")
            print(f"Original detection: {'✓ PASS' if result['original_detection']['prediction'] else '✗ FAIL'}")

            summary = result['summary']
            print(f"\n=== Summary ===")
            print(f"Generator watermark survival rate: {summary['generator_watermark_survival_rate']:.2%}")
            print(f"Generator watermark survived: {summary['generator_watermark_survived_count']}/{len(paraphraser_models)}")
            if enable_paraphrase_wm:
                print(f"Paraphraser watermark detection rate: {summary['paraphraser_watermark_detection_rate']:.2%}")
                print(f"Paraphraser watermark detected: {summary['paraphraser_watermark_detected_count']}/{len(paraphraser_models)}")
            print(f"Average semantic similarity: {summary['average_similarity']:.4f}")
            print(f"Average generator watermark decay: {summary['average_generator_decay']:.4f}")

            print("\n=== Per-Paraphraser Details ===")
            for idx, para_result in enumerate(result['paraphraser_results']):
                print(f"\n[{idx+1}] {para_result['paraphraser']}")
                print(f"  Paraphrased text: {para_result['paraphrased_text'][:100]}...")
                print(f"  Detection (generator key): z={para_result['detection_with_generator_key']['z_score']:.4f}, "
                      f"{'✓ PASS' if para_result['detection_with_generator_key']['prediction'] else '✗ FAIL'}")
                if para_result['detection_with_paraphraser_key']:
                    print(f"  Detection (paraphraser key): z={para_result['detection_with_paraphraser_key']['z_score']:.4f}, "
                          f"{'✓ PASS' if para_result['detection_with_paraphraser_key']['prediction'] else '✗ FAIL'}")
                print(f"  Semantic similarity: {para_result['semantic_similarity']:.4f}")
                print(f"  Generator watermark retention: {para_result['generator_z_score_retention']:.2%}")

            # 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"watermarked_paraphrase_chain_{timestamp}.json"
            experiment.save_results(result, filename=filename)
            print(f"\n✅ Results saved to: {filename}")

        except Exception as exc:
            print(f"❌ Experiment failed: {exc}")
            import traceback
            traceback.print_exc()


def main():
    args = parse_args()
    
    interactive = InteractiveHybridExperiment(args)
    interactive.run()


if __name__ == "__main__":
    main()
