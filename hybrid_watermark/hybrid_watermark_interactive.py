"""
混合水印交互式实验界面
允许用户自定义实验参数并实时查看结果
支持通过模型昵称（nickname）指定模型
"""

import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from hybrid_watermark_experiment import HybridWatermarkExperiment

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
        """运行交互式界面"""
        
        print("\n" + "="*80)
        print("混合水印交互式实验系统")
        print(f"模型: {self.model_nickname} ({self.model_name})")
        print("="*80)
        
        while True:
            print("\n请选择实验类型:")
            print("  1 - 片段级混合水印")
            print("  2 - 种子混合实验")
            print("  3 - 参数混合实验")
            print("  4 - 密钥共享实验 (单模型)")
            print("  5 - 跨模型共享密钥实验 (多模型)")
            print("  6 - 自定义实验")
            print("  7 - 查看实验说明")
            print("  q - 退出")
            print()
            
            choice = input("请输入选择 (1-7/q): ").strip().lower()
            
            if choice == 'q':
                print("\n退出实验系统。\n")
                break
            
            elif choice == '1':
                self.run_fragment_mixing()
            
            elif choice == '2':
                self.run_seed_mixing()
            
            elif choice == '3':
                self.run_parameter_mixing()
            
            elif choice == '4':
                self.run_key_sharing()
            
            elif choice == '5':
                self.run_cross_model_key_sharing()
            
            elif choice == '6':
                self.run_custom_experiment()
            
            elif choice == '7':
                self.show_experiment_info()
            
            else:
                print("\n无效的选择，请重试。\n")
    
    def run_fragment_mixing(self):
        """运行片段级混合实验"""
        print("\n" + "-"*80)
        print("实验1: 片段级混合水印")
        print("-"*80 + "\n")
        
        # 获取提示词
        prompt = input("输入基础提示词 (回车使用默认): ").strip()
        if not prompt:
            prompt = "The future of artificial intelligence is"
        
        # 获取片段数量
        num_fragments = input("输入片段数量 (默认3): ").strip()
        num_fragments = int(num_fragments) if num_fragments else 3
        
        # 配置每个片段
        fragment_configs = []
        print(f"\n配置 {num_fragments} 个片段:")
        
        # 设定好三组默认参数以供参考
        default_params = [
            {'gamma': 0.25, 'delta': 2.0, 'hash_key': 15485863},
            {'gamma': 0.5, 'delta': 1.5, 'hash_key': 32452843},
            {'gamma': 0.1, 'delta': 3.0, 'hash_key': 49979687}
        ]
        
        for i in range(num_fragments):
            print(f"\n片段 {i+1}:")
            gamma_input = input(f"  输入Gamma值 (默认{default_params[i % len(default_params)]['gamma']}): ").strip()
            gamma = float(gamma_input) if gamma_input else default_params[i % len(default_params)]['gamma']
            
            delta_input = input(f"  输入Delta值 (默认{default_params[i % len(default_params)]['delta']}): ").strip()
            delta = float(delta_input) if delta_input else default_params[i % len(default_params)]['delta']
            
            hash_key_input = input(f"  输入Hash Key (默认{default_params[i % len(default_params)]['hash_key']}): ").strip()
            hash_key = int(hash_key_input) if hash_key_input else default_params[i % len(default_params)]['hash_key']
            
            fragment_configs.append({
                'gamma': gamma,
                'delta': delta,
                'hash_key': hash_key
            })
            
        
        # 运行实验
        print("\n开始实验...")
        result = self.experiment.experiment_fragment_mixing(
            base_prompt=prompt,
            fragment_configs=fragment_configs,
            tokens_per_fragment=50
        )
        
        # 显示结果
        self.experiment.print_summary(result)
        
        # 询问是否保存
        save = input("\n是否保存结果? (y/n): ").strip().lower()
        if save == 'y':
            filename = self.experiment.save_results(result)
            print(f"结果已保存到: {filename}")
    
    def run_seed_mixing(self):
        """运行种子混合实验"""
        print("\n" + "-"*80)
        print("实验2: 种子混合")
        print("-"*80 + "\n")
        
        prompt = input("输入提示词 (回车使用默认): ").strip()
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
        print("\n交叉检测详细结果:")
        print("-"*80)
        for cd in result['cross_detection']:
            print(f"\n文本 {cd['text_id']} (Hash Key: {cd['text_key']}):")
            for det in cd['detections']:
                status = "✓" if det['prediction'] else "✗"
                print(f"  {status} 检测器 (Key: {det['detector_key']}): z={det['z_score']:.2f}")
        
        save = input("\n是否保存结果? (y/n): ").strip().lower()
        if save == 'y':
            self.experiment.save_results(result)
    
    def run_parameter_mixing(self):
        """运行参数混合实验"""
        print("\n" + "-"*80)
        print("实验3: 参数混合")
        print("-"*80 + "\n")
        
        prompt = input("输入提示词 (回车使用默认): ").strip()
        if not prompt:
            prompt = "Explain quantum computing:"
        
        # 获取gamma值
        gamma_input = input("输入Gamma值列表，逗号分隔 (默认0.25,0.5): ").strip()
        if gamma_input:
            gamma_values = [float(x.strip()) for x in gamma_input.split(',')]
        else:
            gamma_values = [0.25, 0.5]
        
        # 获取delta值
        delta_input = input("输入Delta值列表，逗号分隔 (默认1.5,2.5): ").strip()
        if delta_input:
            delta_values = [float(x.strip()) for x in delta_input.split(',')]
        else:
            delta_values = [1.5, 2.5]
        
        print(f"\n将生成 {len(gamma_values) * len(delta_values)} 个参数组合")
        
        print("\n开始实验...")
        result = self.experiment.experiment_parameter_mixing(
            prompt=prompt,
            gamma_values=gamma_values,
            delta_values=delta_values,
            tokens_per_config=40
        )
        
        self.experiment.print_summary(result)
        
        # 显示生成的文本片段
        print("\n生成的文本片段:")
        print("-"*80)
        for frag in result['fragments']:
            print(f"\n片段 {frag['fragment_id']} (γ={frag['gamma']}, δ={frag['delta']}):")
            print(f"  {frag['text'][:100]}...")
        
        save = input("\n是否保存结果? (y/n): ").strip().lower()
        if save == 'y':
            self.experiment.save_results(result)
    
    def run_key_sharing(self):
        """运行密钥共享实验"""
        print("\n" + "-"*80)
        print("实验4: 密钥共享")
        print("-"*80 + "\n")
        
        # 获取提示词数量
        num_prompts = input("输入文本数量 (默认4): ").strip()
        num_prompts = int(num_prompts) if num_prompts else 4
        
        # 获取每个提示词
        prompts = []
        print(f"\n输入 {num_prompts} 个提示词:")
        for i in range(num_prompts):
            prompt = input(f"  提示词 {i+1}: ").strip()
            if not prompt:
                # 使用默认提示词
                default_prompts = [
                    "The benefits of renewable energy include",
                    "In the year 2050, technology will",
                    "Climate change is affecting",
                    "Space exploration has led to"
                ]
                prompt = default_prompts[i % len(default_prompts)]
            prompts.append(prompt)
        
        # 获取共享密钥
        shared_key = input("\n输入共享Hash Key (默认15485863): ").strip()
        shared_key = int(shared_key) if shared_key else 15485863
        
        print("\n开始实验...")
        result = self.experiment.experiment_key_sharing(
            prompts=prompts,
            shared_key=shared_key,
            max_new_tokens=60
        )
        
        self.experiment.print_summary(result)
        
        # 显示详细的文本和检测结果
        print("\n详细结果:")
        print("-"*80)
        for text_info, det_info in zip(result['texts'], result['individual_detections']):
            print(f"\n文本 {text_info['text_id']} ({text_info['key_type']}, Key={text_info['hash_key']}):")
            print(f"  提示: {text_info['prompt']}")
            print(f"  文本: {text_info['text'][:80]}...")
            print(f"  正确密钥检测: z={det_info['correct_key_detection']['z_score']:.2f}, "
                  f"结果={det_info['correct_key_detection']['prediction']}")
            print(f"  共享密钥检测: z={det_info['shared_key_detection']['z_score']:.2f}, "
                  f"结果={det_info['shared_key_detection']['prediction']}")
        
        save = input("\n是否保存结果? (y/n): ").strip().lower()
        if save == 'y':
            self.experiment.save_results(result)
    
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
        
        # 选择额外的模型
        for i in range(1, num_models):
            model_input = input(f"输入第{i+1}个模型昵称: ").strip()
            if model_input and model_input in available_models:
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
                print(f"  ✓ 生成完成: {generated_text[-60:] if len(generated_text) > 60 else generated_text}...")
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
    
    def run_custom_experiment(self):
        """运行自定义实验"""
        print("\n" + "-"*80)
        print("自定义实验")
        print("-"*80 + "\n")
        
        print("请选择基础实验类型:")
        print("  1 - 基于片段混合")
        print("  2 - 基于种子混合")
        print("  3 - 基于参数混合")
        print("  4 - 基于密钥共享")
        
        exp_type = input("\n选择 (1-4): ").strip()
        
        if exp_type == '1':
            self.run_fragment_mixing()
        elif exp_type == '2':
            self.run_seed_mixing()
        elif exp_type == '3':
            self.run_parameter_mixing()
        elif exp_type == '4':
            self.run_key_sharing()
        else:
            print("无效选择")
    
    def show_experiment_info(self):
        """显示实验说明"""
        print("\n" + "="*80)
        print("实验说明")
        print("="*80 + "\n")
        
        print("实验1: 片段级混合水印")
        print("-"*40)
        print("在同一段落中，不同片段使用不同的水印配置（gamma, delta, hash_key）")
        print("目的: 研究混合水印的检测特性和鲁棒性")
        print()
        
        print("实验2: 种子混合")
        print("-"*40)
        print("同一模型使用不同的hash_key生成多个文本变体")
        print("目的: 研究不同种子之间的相互检测能力")
        print()
        
        print("实验3: 参数混合")
        print("-"*40)
        print("使用不同的gamma和delta组合生成文本片段")
        print("目的: 研究不同参数配置的混合效果")
        print()
        
        print("实验4: 密钥共享")
        print("-"*40)
        print("部分文本使用共享密钥，部分使用独立密钥")
        print("目的: 研究密钥共享对检测的影响")
        print()
        
        print("="*80 + "\n")


def main():
    args = parse_args()
    
    interactive = InteractiveHybridExperiment(args)
    interactive.run()


if __name__ == "__main__":
    main()
