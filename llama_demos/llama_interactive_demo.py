"""
Llama 交互式水印Demo
支持用户通过命令行界面进行交互式文本生成和水印检测
支持多种Llama模型，默认使用 Llama 2 7B
"""

import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList
)
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Llama Interactive Watermarking Demo - 支持多种Llama模型"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Llama模型名称或路径 (默认: Llama-2-7b-hf)\n"
             "支持的模型: Llama-2-7b-hf, Llama-2-13b-hf, Llama-2-7b-chat-hf, "
             "Llama-3.2-1B, Llama-3.2-3B 等"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="运行设备"
    )
    
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="水印参数gamma（绿名单比例），推荐值: 0.25"
    )
    
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="水印参数delta（水印强度），推荐值: 2.0"
    )
    
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="selfhash",
        help="种子方案，推荐值: selfhash"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="最大生成token数"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="nucleus采样参数"
    )
    
    return parser.parse_args()


class InteractiveLlamaWatermark:
    """交互式Llama水印系统"""
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        print("\n" + "="*80)
        print("正在加载模型...")
        print("="*80 + "\n")
        
        # 加载tokenizer
        print(f"加载tokenizer: {args.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型
        print(f"加载模型到设备: {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # 初始化水印处理器
        print("初始化水印处理器...")
        self.watermark_processor = WatermarkLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=args.gamma,
            delta=args.delta,
            seeding_scheme=args.seeding_scheme
        )
        
        # 初始化水印检测器
        print("初始化水印检测器...")
        self.watermark_detector = WatermarkDetector(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=args.gamma,
            seeding_scheme=args.seeding_scheme,
            device=self.device,
            tokenizer=self.tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_ngrams=True
        )
        
        print("\n✓ 所有组件加载完成!\n")
    
    def generate_text(self, prompt: str, use_watermark: bool = True):
        """生成文本"""
        print(f"\n{'='*80}")
        print(f"提示词: {prompt}")
        print(f"使用水印: {'是' if use_watermark else '否'}")
        print(f"{'='*80}\n")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # 生成
        print("正在生成文本...\n")
        
        logits_processor = LogitsProcessorList([self.watermark_processor]) if use_watermark else None
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=True,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码（只保留新生成的部分）
        generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1]:]
        generated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        print(f"生成的文本:\n{'-'*80}")
        print(generated_text)
        print(f"{'-'*80}\n")
        
        return generated_text
    
    def detect_text(self, text: str):
        """检测文本水印"""
        print(f"\n{'='*80}")
        print("检测水印...")
        print(f"{'='*80}\n")
        
        score_dict = self.watermark_detector.detect(text)
        
        # 显示结果
        print("检测结果:")
        print(f"{'-'*80}")
        print(f"  Z分数:          {score_dict['z_score']:.4f}")
        print(f"  P值:            {score_dict['p_value']:.6f}")
        print(f"  检测结论:       {'✓ 包含水印' if score_dict['prediction'] else '✗ 不包含水印'}")
        print(f"  绿色token数:    {score_dict['num_tokens_scored']}")
        print(f"  绿色token比例:  {score_dict['green_fraction']:.4f}")
        print(f"{'-'*80}\n")
        
        return score_dict
    
    def run(self):
        """运行交互式demo"""
        print("\n" + "="*80)
        print("Llama 交互式水印Demo")
        print(f"当前模型: {self.args.model_name}")
        print("="*80)
        print("\n命令说明:")
        print("  1 - 生成带水印的文本")
        print("  2 - 生成不带水印的文本")
        print("  3 - 检测文本水印")
        print("  4 - 显示当前配置")
        print("  5 - 修改生成参数")
        print("  q - 退出")
        print("="*80 + "\n")
        
        while True:
            try:
                command = input("请输入命令 (1-5/q): ").strip().lower()
                
                if command == 'q':
                    print("\n感谢使用! 再见!\n")
                    break
                
                elif command == '1':
                    # 生成带水印的文本
                    prompt = input("\n请输入提示词: ").strip()
                    if prompt:
                        generated_text = self.generate_text(prompt, use_watermark=True)
                        
                        # 询问是否检测
                        detect = input("\n是否立即检测这段文本的水印? (y/n): ").strip().lower()
                        if detect == 'y':
                            self.detect_text(generated_text)
                
                elif command == '2':
                    # 生成不带水印的文本
                    prompt = input("\n请输入提示词: ").strip()
                    if prompt:
                        generated_text = self.generate_text(prompt, use_watermark=False)
                        
                        # 询问是否检测
                        detect = input("\n是否检测这段文本? (y/n): ").strip().lower()
                        if detect == 'y':
                            self.detect_text(generated_text)
                
                elif command == '3':
                    # 检测文本
                    print("\n请输入要检测的文本 (输入空行结束):")
                    lines = []
                    while True:
                        line = input()
                        if line.strip() == "":
                            break
                        lines.append(line)
                    
                    text = "\n".join(lines)
                    if text.strip():
                        self.detect_text(text)
                    else:
                        print("未输入文本。\n")
                
                elif command == '4':
                    # 显示配置
                    print(f"\n{'='*80}")
                    print("当前配置:")
                    print(f"{'='*80}")
                    print(f"  模型:           {self.args.model_name}")
                    print(f"  设备:           {self.args.device}")
                    print(f"  Gamma:          {self.args.gamma}")
                    print(f"  Delta:          {self.args.delta}")
                    print(f"  种子方案:       {self.args.seeding_scheme}")
                    print(f"  最大token数:    {self.args.max_new_tokens}")
                    print(f"  温度:           {self.args.temperature}")
                    print(f"  Top-p:          {self.args.top_p}")
                    print(f"{'='*80}\n")
                
                elif command == '5':
                    # 修改参数
                    print("\n修改生成参数:")
                    try:
                        max_tokens = input(f"最大token数 (当前: {self.args.max_new_tokens}): ").strip()
                        if max_tokens:
                            self.args.max_new_tokens = int(max_tokens)
                        
                        temp = input(f"温度 (当前: {self.args.temperature}): ").strip()
                        if temp:
                            self.args.temperature = float(temp)
                        
                        top_p = input(f"Top-p (当前: {self.args.top_p}): ").strip()
                        if top_p:
                            self.args.top_p = float(top_p)
                        
                        print("\n✓ 参数已更新!\n")
                    except ValueError:
                        print("\n✗ 输入格式错误，参数未更改。\n")
                
                else:
                    print("\n无效的命令，请重新输入。\n")
            
            except KeyboardInterrupt:
                print("\n\n检测到中断，退出程序。\n")
                break
            except Exception as e:
                print(f"\n错误: {str(e)}\n")


def main():
    args = parse_args()
    demo = InteractiveLlamaWatermark(args)
    demo.run()


if __name__ == "__main__":
    main()
