"""
混合水印交互式实验界面
允许用户自定义实验参数并实时查看结果
"""

import argparse
from hybrid_watermark_experiment import HybridWatermarkExperiment
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="混合水印交互式实验"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="模型名称"
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
        self.experiment = HybridWatermarkExperiment(
            model_name=args.model_name,
            device=args.device
        )
    
    def run(self):
        """运行交互式界面"""
        
        print("\n" + "="*80)
        print("混合水印交互式实验系统")
        print(f"模型: {self.args.model_name}")
        print("="*80)
        
        while True:
            print("\n请选择实验类型:")
            print("  1 - 片段级混合水印")
            print("  2 - 种子混合实验")
            print("  3 - 参数混合实验")
            print("  4 - 密钥共享实验")
            print("  5 - 自定义实验")
            print("  6 - 查看实验说明")
            print("  q - 退出")
            print()
            
            choice = input("请输入选择 (1-6/q): ").strip().lower()
            
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
                self.run_custom_experiment()
            
            elif choice == '6':
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
        
        for i in range(num_fragments):
            print(f"\n片段 {i+1}:")
            gamma = input(f"  Gamma (默认0.25): ").strip()
            gamma = float(gamma) if gamma else 0.25
            
            delta = input(f"  Delta (默认2.0): ").strip()
            delta = float(delta) if delta else 2.0
            
            hash_key = input(f"  Hash Key (默认15485863): ").strip()
            hash_key = int(hash_key) if hash_key else 15485863
            
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
