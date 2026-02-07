"""
Statistical Evaluation Using Z-Test with Sliding Window

实现三个核心功能：
1. 滑动窗口检测 (Sliding Window Detection)
2. 窗口敏感性分析 (Window Sensitivity Analysis)
3. 最小可检测文本长度分析 (Minimum Detectable Length Analysis)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
from tqdm import tqdm

from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# 导入模型配置管理器
sys.path.insert(0, os.path.abspath('../llama_demos'))
from model_config_manager import ModelConfigManager


@dataclass
class SlidingWindowResult:
    """滑动窗口检测结果"""
    window_positions: List[int]  # 窗口起始位置
    z_scores: List[float]  # 每个窗口的z-score
    p_values: List[float]  # 每个窗口的p-value
    predictions: List[bool]  # 每个窗口的检测结果
    green_fractions: List[float]  # 每个窗口的绿色token比例
    window_size: int  # 窗口大小
    text_length: int  # 文本总长度


@dataclass
class WindowSensitivityResult:
    """窗口敏感性分析结果"""
    window_sizes: List[int]  # 测试的窗口大小
    avg_z_scores: List[float]  # 每个窗口大小的平均z-score
    std_z_scores: List[float]  # 每个窗口大小的z-score标准差
    detection_rates: List[float]  # 检测成功率
    false_positive_rates: List[float]  # 假阳性率（如果有无水印文本）


@dataclass
class MinimumLengthResult:
    """最小可检测长度分析结果"""
    text_lengths: List[int]  # 测试的文本长度
    z_scores: List[float]  # 每个长度的z-score
    p_values: List[float]  # 每个长度的p-value
    predictions: List[bool]  # 检测结果
    green_fractions: List[float]  # 绿色token比例
    min_reliable_length: Optional[int]  # 最小可靠检测长度


class StatisticalEvaluator:
    """统计评估器"""
    
    def __init__(
        self,
        model_nickname: str = "llama-2-7b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化统计评估器
        
        Args:
            model_nickname: 模型昵称
            device: 设备
        """
        self.device = device
        self.model_nickname = model_nickname
        
        # 解析模型配置
        config_manager = ModelConfigManager()
        model_identifier = config_manager.resolve_model_nickname(model_nickname)
        
        print(f"加载模型: {model_identifier}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_identifier,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print("✓ 模型加载完成\n")
    
    def generate_with_watermark(
        self,
        prompt: str,
        watermark_processor: WatermarkLogitsProcessor,
        max_new_tokens: int = 200
    ) -> str:
        """生成带水印的文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
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
    
    def generate_without_watermark(
        self,
        prompt: str,
        max_new_tokens: int = 200
    ) -> str:
        """生成无水印的文本（用于对照）"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1]:]
        generated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        return generated_text
    
    # ========== 功能1: 滑动窗口检测 ==========
    
    def sliding_window_detection(
        self,
        text: str,
        window_size: int,
        stride: int,
        detector: WatermarkDetector
    ) -> SlidingWindowResult:
        """
        滑动窗口检测
        
        Args:
            text: 待检测文本
            window_size: 窗口大小（tokens数）
            stride: 滑动步长
            detector: 水印检测器
        
        Returns:
            SlidingWindowResult
        """
        # 将文本分词
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        text_length = len(tokens)
        
        window_positions = []
        z_scores = []
        p_values = []
        predictions = []
        green_fractions = []
        
        # 滑动窗口
        for start_pos in range(0, text_length - window_size + 1, stride):
            end_pos = start_pos + window_size
            
            # 提取窗口内的tokens
            window_tokens = tokens[start_pos:end_pos]
            window_text = self.tokenizer.decode(window_tokens, skip_special_tokens=True)
            
            # 检测该窗口
            result = detector.detect(window_text)
            
            window_positions.append(start_pos)
            z_scores.append(result['z_score'])
            p_values.append(result['p_value'])
            predictions.append(result['prediction'])
            green_fractions.append(result['green_fraction'])
        
        return SlidingWindowResult(
            window_positions=window_positions,
            z_scores=z_scores,
            p_values=p_values,
            predictions=predictions,
            green_fractions=green_fractions,
            window_size=window_size,
            text_length=text_length
        )
    
    def plot_sliding_window_results(
        self,
        result: SlidingWindowResult,
        save_path: Optional[str] = None,
        title: str = "Sliding Window Detection Results"
    ):
        """
        绘制滑动窗口检测结果
        
        Args:
            result: 滑动窗口检测结果
            save_path: 保存路径
            title: 图表标题
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 图1: Z-score曲线
        axes[0].plot(result.window_positions, result.z_scores, 
                     marker='o', linestyle='-', linewidth=2, markersize=4)
        axes[0].axhline(y=4.0, color='r', linestyle='--', label='Detection threshold (z=4)')
        axes[0].set_xlabel('Window Start Position (tokens)')
        axes[0].set_ylabel('Z-score')
        axes[0].set_title(f'{title} - Z-score Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 图2: 绿色token比例
        axes[1].plot(result.window_positions, result.green_fractions,
                     marker='s', linestyle='-', linewidth=2, markersize=4, color='green')
        axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Expected (γ=0.5)')
        axes[1].set_xlabel('Window Start Position (tokens)')
        axes[1].set_ylabel('Green Token Fraction')
        axes[1].set_title('Green Token Fraction Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 图3: 检测结果热图
        detection_map = [1 if pred else 0 for pred in result.predictions]
        axes[2].bar(result.window_positions, detection_map, 
                    width=result.window_size * 0.8, color='blue', alpha=0.6)
        axes[2].set_xlabel('Window Start Position (tokens)')
        axes[2].set_ylabel('Detection (1=Detected, 0=Not)')
        axes[2].set_title('Detection Results')
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        
        plt.show()
    
    # ========== 功能2: 窗口敏感性分析 ==========
    
    def window_sensitivity_analysis(
        self,
        watermarked_text: str,
        unwatermarked_text: Optional[str],
        window_sizes: List[int],
        detector: WatermarkDetector,
        stride_ratio: float = 0.5
    ) -> WindowSensitivityResult:
        """
        窗口敏感性分析
        
        Args:
            watermarked_text: 带水印文本
            unwatermarked_text: 无水印文本（用于计算假阳性率）
            window_sizes: 要测试的窗口大小列表
            detector: 水印检测器
            stride_ratio: 步长比例（相对于窗口大小）
        
        Returns:
            WindowSensitivityResult
        """
        avg_z_scores = []
        std_z_scores = []
        detection_rates = []
        false_positive_rates = []
        
        for window_size in tqdm(window_sizes, desc="窗口敏感性分析"):
            stride = max(1, int(window_size * stride_ratio))
            
            # 检测带水印文本
            wm_result = self.sliding_window_detection(
                watermarked_text, window_size, stride, detector
            )
            
            avg_z_scores.append(np.mean(wm_result.z_scores))
            std_z_scores.append(np.std(wm_result.z_scores))
            detection_rates.append(np.mean(wm_result.predictions))
            
            # 如果有无水印文本，计算假阳性率
            if unwatermarked_text:
                unwm_result = self.sliding_window_detection(
                    unwatermarked_text, window_size, stride, detector
                )
                false_positive_rates.append(np.mean(unwm_result.predictions))
            else:
                false_positive_rates.append(0.0)
        
        return WindowSensitivityResult(
            window_sizes=window_sizes,
            avg_z_scores=avg_z_scores,
            std_z_scores=std_z_scores,
            detection_rates=detection_rates,
            false_positive_rates=false_positive_rates
        )
    
    def plot_window_sensitivity(
        self,
        result: WindowSensitivityResult,
        save_path: Optional[str] = None
    ):
        """
        绘制窗口敏感性分析结果
        
        Args:
            result: 窗口敏感性分析结果
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 图1: 平均Z-score vs 窗口大小
        axes[0, 0].errorbar(result.window_sizes, result.avg_z_scores,
                            yerr=result.std_z_scores, marker='o', capsize=5,
                            linewidth=2, markersize=8)
        axes[0, 0].axhline(y=4.0, color='r', linestyle='--', label='Threshold (z=4)')
        axes[0, 0].set_xlabel('Window Size (tokens)')
        axes[0, 0].set_ylabel('Average Z-score')
        axes[0, 0].set_title('Average Z-score vs Window Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 图2: 检测率 vs 窗口大小
        axes[0, 1].plot(result.window_sizes, result.detection_rates,
                        marker='s', linewidth=2, markersize=8, color='green')
        axes[0, 1].axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
        axes[0, 1].set_xlabel('Window Size (tokens)')
        axes[0, 1].set_ylabel('Detection Rate')
        axes[0, 1].set_title('Detection Rate vs Window Size')
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 图3: Z-score标准差 vs 窗口大小
        axes[1, 0].plot(result.window_sizes, result.std_z_scores,
                        marker='^', linewidth=2, markersize=8, color='orange')
        axes[1, 0].set_xlabel('Window Size (tokens)')
        axes[1, 0].set_ylabel('Z-score Std Dev')
        axes[1, 0].set_title('Z-score Stability vs Window Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 图4: 假阳性率 vs 窗口大小（如果有数据）
        if any(result.false_positive_rates):
            axes[1, 1].plot(result.window_sizes, result.false_positive_rates,
                            marker='D', linewidth=2, markersize=8, color='red')
            axes[1, 1].axhline(y=0.05, color='gray', linestyle='--', label='5% threshold')
            axes[1, 1].set_xlabel('Window Size (tokens)')
            axes[1, 1].set_ylabel('False Positive Rate')
            axes[1, 1].set_title('False Positive Rate vs Window Size')
            axes[1, 1].set_ylim(0, max(result.false_positive_rates) * 1.2 if result.false_positive_rates else 0.1)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No unwatermarked text provided',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('False Positive Rate (N/A)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        
        plt.show()
    
    # ========== 功能3: 最小可检测长度分析 ==========
    
    def minimum_length_analysis(
        self,
        full_text: str,
        length_range: List[int],
        detector: WatermarkDetector,
        num_samples: int = 1
    ) -> MinimumLengthResult:
        """
        最小可检测长度分析
        
        Args:
            full_text: 完整文本
            length_range: 要测试的长度范围
            detector: 水印检测器
            num_samples: 每个长度采样次数（随机位置）
        
        Returns:
            MinimumLengthResult
        """
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        text_lengths = []
        z_scores = []
        p_values = []
        predictions = []
        green_fractions = []
        
        for length in tqdm(length_range, desc="最小长度分析"):
            if length > len(tokens):
                continue
            
            # 对每个长度进行多次采样
            length_z_scores = []
            length_predictions = []
            length_green_fractions = []
            length_p_values = []
            
            for _ in range(num_samples):
                # 随机选择起始位置
                if len(tokens) - length > 0:
                    start_pos = np.random.randint(0, len(tokens) - length + 1)
                else:
                    start_pos = 0
                
                sample_tokens = tokens[start_pos:start_pos + length]
                sample_text = self.tokenizer.decode(sample_tokens, skip_special_tokens=True)
                
                # 检测
                result = detector.detect(sample_text)
                
                length_z_scores.append(result['z_score'])
                length_p_values.append(result['p_value'])
                length_predictions.append(result['prediction'])
                length_green_fractions.append(result['green_fraction'])
            
            # 计算平均值
            text_lengths.append(length)
            z_scores.append(np.mean(length_z_scores))
            p_values.append(np.mean(length_p_values))
            predictions.append(np.mean(length_predictions) >= 0.5)  # 多数投票
            green_fractions.append(np.mean(length_green_fractions))
        
        # 找出最小可靠检测长度
        min_reliable_length = None
        for i, (length, pred) in enumerate(zip(text_lengths, predictions)):
            if pred and i + 2 < len(predictions) and all(predictions[i:i+3]):
                # 连续3个长度都能检测到
                min_reliable_length = length
                break
        
        return MinimumLengthResult(
            text_lengths=text_lengths,
            z_scores=z_scores,
            p_values=p_values,
            predictions=predictions,
            green_fractions=green_fractions,
            min_reliable_length=min_reliable_length
        )
    
    def plot_minimum_length_analysis(
        self,
        result: MinimumLengthResult,
        save_path: Optional[str] = None
    ):
        """
        绘制最小长度分析结果
        
        Args:
            result: 最小长度分析结果
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 图1: Z-score vs 文本长度
        axes[0, 0].plot(result.text_lengths, result.z_scores,
                        marker='o', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=4.0, color='r', linestyle='--', label='Threshold (z=4)')
        if result.min_reliable_length:
            axes[0, 0].axvline(x=result.min_reliable_length, color='g',
                              linestyle='--', label=f'Min length: {result.min_reliable_length}')
        axes[0, 0].set_xlabel('Text Length (tokens)')
        axes[0, 0].set_ylabel('Z-score')
        axes[0, 0].set_title('Z-score vs Text Length')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 图2: P-value vs 文本长度（对数刻度）
        axes[0, 1].semilogy(result.text_lengths, result.p_values,
                            marker='s', linewidth=2, markersize=6, color='orange')
        axes[0, 1].axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
        axes[0, 1].axhline(y=0.01, color='purple', linestyle='--', label='p=0.01')
        if result.min_reliable_length:
            axes[0, 1].axvline(x=result.min_reliable_length, color='g',
                              linestyle='--', label=f'Min length: {result.min_reliable_length}')
        axes[0, 1].set_xlabel('Text Length (tokens)')
        axes[0, 1].set_ylabel('P-value (log scale)')
        axes[0, 1].set_title('P-value vs Text Length')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 图3: 绿色token比例 vs 文本长度
        axes[1, 0].plot(result.text_lengths, result.green_fractions,
                        marker='^', linewidth=2, markersize=6, color='green')
        axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', label='Expected (γ=0.5)')
        if result.min_reliable_length:
            axes[1, 0].axvline(x=result.min_reliable_length, color='g',
                              linestyle='--', label=f'Min length: {result.min_reliable_length}')
        axes[1, 0].set_xlabel('Text Length (tokens)')
        axes[1, 0].set_ylabel('Green Token Fraction')
        axes[1, 0].set_title('Green Token Fraction vs Text Length')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 图4: 检测成功/失败可视化
        colors = ['green' if pred else 'red' for pred in result.predictions]
        axes[1, 1].scatter(result.text_lengths, result.z_scores,
                          c=colors, s=100, alpha=0.6, edgecolors='black')
        axes[1, 1].axhline(y=4.0, color='r', linestyle='--', linewidth=2)
        if result.min_reliable_length:
            axes[1, 1].axvline(x=result.min_reliable_length, color='g',
                              linestyle='--', linewidth=2, label=f'Min length: {result.min_reliable_length}')
        axes[1, 1].set_xlabel('Text Length (tokens)')
        axes[1, 1].set_ylabel('Z-score')
        axes[1, 1].set_title('Detection Success (Green) / Failure (Red)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        
        plt.show()
    
    # ========== 综合实验 ==========
    
    def run_complete_evaluation(
        self,
        prompt: str,
        watermark_config: Dict,
        max_new_tokens: int = 300,
        window_sizes: List[int] = [25, 50, 75, 100, 150, 200],
        length_range: List[int] = None,
        generate_unwatermarked: bool = True
    ) -> Dict:
        """
        运行完整的统计评估
        
        Args:
            prompt: 生成提示词
            watermark_config: 水印配置
            max_new_tokens: 生成token数
            window_sizes: 窗口大小列表
            length_range: 长度范围（None则自动生成）
            generate_unwatermarked: 是否生成无水印对照
        
        Returns:
            包含所有结果的字典
        """
        print("="*80)
        print("统计评估实验")
        print("="*80 + "\n")
        
        # 创建水印处理器和检测器
        watermark_processor = WatermarkLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=watermark_config.get('gamma', 0.5),
            delta=watermark_config.get('delta', 2.0),
            seeding_scheme=watermark_config.get('seeding_scheme', 'selfhash'),
            hash_key=watermark_config.get('hash_key', 15485863)
        )
        
        detector = WatermarkDetector(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=watermark_config.get('gamma', 0.5),
            seeding_scheme=watermark_config.get('seeding_scheme', 'selfhash'),
            device=self.device,
            tokenizer=self.tokenizer,
            z_threshold=4.0,
            hash_key=watermark_config.get('hash_key', 15485863)
        )
        
        # 生成带水印文本
        print("生成带水印文本...")
        watermarked_text = self.generate_with_watermark(
            prompt, watermark_processor, max_new_tokens
        )
        print(f"✓ 生成完成 ({len(watermarked_text)} 字符)\n")
        
        # 生成无水印文本（可选）
        unwatermarked_text = None
        if generate_unwatermarked:
            print("生成无水印对照文本...")
            unwatermarked_text = self.generate_without_watermark(
                prompt, max_new_tokens
            )
            print(f"✓ 生成完成 ({len(unwatermarked_text)} 字符)\n")
        
        # 1. 滑动窗口检测
        print("执行滑动窗口检测...")
        sliding_result = self.sliding_window_detection(
            watermarked_text,
            window_size=100,
            stride=25,
            detector=detector
        )
        print(f"✓ 完成 ({len(sliding_result.window_positions)} 个窗口)\n")
        
        # 2. 窗口敏感性分析
        print("执行窗口敏感性分析...")
        sensitivity_result = self.window_sensitivity_analysis(
            watermarked_text,
            unwatermarked_text,
            window_sizes,
            detector
        )
        print(f"✓ 完成 (测试了 {len(window_sizes)} 个窗口大小)\n")
        
        # 3. 最小长度分析
        if length_range is None:
            # 自动生成长度范围
            tokens = self.tokenizer.encode(watermarked_text, add_special_tokens=False)
            length_range = list(range(20, min(len(tokens), 250), 10))
        
        print("执行最小可检测长度分析...")
        min_length_result = self.minimum_length_analysis(
            watermarked_text,
            length_range,
            detector,
            num_samples=3
        )
        print(f"✓ 完成 (测试了 {len(length_range)} 个长度)\n")
        
        # 汇总结果
        results = {
            'prompt': prompt,
            'watermark_config': watermark_config,
            'watermarked_text': watermarked_text,
            'unwatermarked_text': unwatermarked_text,
            'sliding_window': {
                'window_size': sliding_result.window_size,
                'num_windows': len(sliding_result.window_positions),
                'avg_z_score': np.mean(sliding_result.z_scores),
                'std_z_score': np.std(sliding_result.z_scores),
                'detection_rate': np.mean(sliding_result.predictions)
            },
            'window_sensitivity': {
                'window_sizes': sensitivity_result.window_sizes,
                'avg_z_scores': sensitivity_result.avg_z_scores,
                'detection_rates': sensitivity_result.detection_rates,
                'optimal_window_size': sensitivity_result.window_sizes[
                    np.argmax(sensitivity_result.avg_z_scores)
                ]
            },
            'minimum_length': {
                'min_reliable_length': min_length_result.min_reliable_length,
                'test_range': [min(length_range), max(length_range)],
                'num_tests': len(length_range)
            }
        }
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "hybrid_watermark_results"
        os.makedirs(results_dir, exist_ok=True)
        
        result_file = os.path.join(
            results_dir,
            f"statistical_evaluation_{timestamp}.json"
        )
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 结果已保存: {result_file}\n")
        
        # 打印摘要
        self.print_summary(results, sliding_result, sensitivity_result, min_length_result)
        
        return {
            'results': results,
            'sliding_result': sliding_result,
            'sensitivity_result': sensitivity_result,
            'min_length_result': min_length_result
        }
    
    def print_summary(
        self,
        results: Dict,
        sliding_result: SlidingWindowResult,
        sensitivity_result: WindowSensitivityResult,
        min_length_result: MinimumLengthResult
    ):
        """打印结果摘要"""
        print("="*80)
        print("实验结果摘要")
        print("="*80 + "\n")
        
        print("1. 滑动窗口检测:")
        print(f"   窗口大小: {sliding_result.window_size} tokens")
        print(f"   窗口数量: {len(sliding_result.window_positions)}")
        print(f"   平均 Z-score: {np.mean(sliding_result.z_scores):.4f} ± {np.std(sliding_result.z_scores):.4f}")
        print(f"   检测率: {np.mean(sliding_result.predictions)*100:.1f}%")
        print(f"   平均绿色token比例: {np.mean(sliding_result.green_fractions):.4f}\n")
        
        print("2. 窗口敏感性分析:")
        print(f"   测试窗口范围: {min(sensitivity_result.window_sizes)} - {max(sensitivity_result.window_sizes)} tokens")
        optimal_idx = np.argmax(sensitivity_result.avg_z_scores)
        print(f"   最优窗口大小: {sensitivity_result.window_sizes[optimal_idx]} tokens")
        print(f"   最高平均 Z-score: {sensitivity_result.avg_z_scores[optimal_idx]:.4f}")
        print(f"   最高检测率: {max(sensitivity_result.detection_rates)*100:.1f}%\n")
        
        print("3. 最小可检测长度分析:")
        if min_length_result.min_reliable_length:
            print(f"   最小可靠检测长度: {min_length_result.min_reliable_length} tokens")
        else:
            print(f"   未找到可靠检测长度（测试范围: {min(min_length_result.text_lengths)}-{max(min_length_result.text_lengths)} tokens）")
        print(f"   测试长度范围: {min(min_length_result.text_lengths)} - {max(min_length_result.text_lengths)} tokens")
        print(f"   最短成功检测: {min([l for l, p in zip(min_length_result.text_lengths, min_length_result.predictions) if p], default='N/A')} tokens")
        print()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical Evaluation with Sliding Window")
    parser.add_argument("--model", type=str, default="llama-2-7b",
                       help="模型昵称")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="设备")
    parser.add_argument("--prompt", type=str,
                       default="The impact of artificial intelligence on modern society",
                       help="生成提示词")
    parser.add_argument("--max_tokens", type=int, default=300,
                       help="生成token数")
    parser.add_argument("--gamma", type=float, default=0.5,
                       help="Gamma值")
    parser.add_argument("--delta", type=float, default=2.0,
                       help="Delta值")
    parser.add_argument("--hash_key", type=int, default=15485863,
                       help="Hash key")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = StatisticalEvaluator(
        model_nickname=args.model,
        device=args.device
    )
    
    # 水印配置
    watermark_config = {
        'gamma': args.gamma,
        'delta': args.delta,
        'seeding_scheme': 'selfhash',
        'hash_key': args.hash_key
    }
    
    # 运行完整评估
    all_results = evaluator.run_complete_evaluation(
        prompt=args.prompt,
        watermark_config=watermark_config,
        max_new_tokens=args.max_tokens,
        window_sizes=[25, 50, 75, 100, 150, 200],
        generate_unwatermarked=True
    )
    
    # 绘制图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "hybrid_watermark_results"
    
    print("生成可视化图表...")
    
    # 滑动窗口结果
    evaluator.plot_sliding_window_results(
        all_results['sliding_result'],
        save_path=os.path.join(results_dir, f"sliding_window_{timestamp}.png")
    )
    
    # 窗口敏感性结果
    evaluator.plot_window_sensitivity(
        all_results['sensitivity_result'],
        save_path=os.path.join(results_dir, f"window_sensitivity_{timestamp}.png")
    )
    
    # 最小长度结果
    evaluator.plot_minimum_length_analysis(
        all_results['min_length_result'],
        save_path=os.path.join(results_dir, f"minimum_length_{timestamp}.png")
    )
    
    print("\n✓ 所有实验完成！")


if __name__ == "__main__":
    main()
