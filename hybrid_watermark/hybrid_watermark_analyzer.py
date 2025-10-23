"""
混合水印结果分析和可视化工具
分析实验结果并生成报告
"""

import json
from pathlib import Path
from typing import Dict, List
import statistics


class HybridWatermarkAnalyzer:
    """混合水印结果分析器"""
    
    def __init__(self, results_dir: str = "hybrid_watermark_results"):
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            print(f"警告: 结果目录 {results_dir} 不存在")
    
    def load_result(self, filename: str) -> Dict:
        """加载单个结果文件"""
        filepath = self.results_dir / filename
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_results(self) -> List[str]:
        """列出所有结果文件"""
        if not self.results_dir.exists():
            return []
        
        return sorted([f.name for f in self.results_dir.glob("*.json")])
    
    def analyze_fragment_mixing(self, result: Dict):
        """分析片段混合实验"""
        print("\n" + "="*80)
        print("片段混合实验分析")
        print("="*80 + "\n")
        
        fragments = result['fragments']
        detections = result['detection_results']
        
        print(f"总片段数: {len(fragments)}")
        print(f"组合文本长度: {len(result['combined_text'])} 字符\n")
        
        # 分析每个配置的检测效果
        print("各配置检测性能:")
        print("-"*80)
        for i, det in enumerate(detections, 1):
            config = det['config']
            print(f"\n配置 {i}:")
            print(f"  参数: γ={config.get('gamma', 'N/A')}, "
                  f"δ={config.get('delta', 'N/A')}, "
                  f"key={config.get('hash_key', 'N/A')}")
            print(f"  Z分数: {det['z_score']:.4f}")
            print(f"  P值: {det['p_value']:.6f}")
            print(f"  检测结果: {'✓ 检测到' if det['prediction'] else '✗ 未检测到'}")
            print(f"  绿色比例: {det['green_fraction']:.4f}")
        
        # 统计信息
        z_scores = [d['z_score'] for d in detections]
        print(f"\nZ分数统计:")
        print(f"  平均值: {statistics.mean(z_scores):.4f}")
        print(f"  最大值: {max(z_scores):.4f}")
        print(f"  最小值: {min(z_scores):.4f}")
        if len(z_scores) > 1:
            print(f"  标准差: {statistics.stdev(z_scores):.4f}")
        
        detection_rate = sum(1 for d in detections if d['prediction']) / len(detections)
        print(f"\n检测成功率: {detection_rate*100:.1f}%")
    
    def analyze_seed_mixing(self, result: Dict):
        """分析种子混合实验"""
        print("\n" + "="*80)
        print("种子混合实验分析")
        print("="*80 + "\n")
        
        variations = result['variations']
        cross_detection = result['cross_detection']
        hash_keys = result['hash_keys']
        
        print(f"变体数量: {len(variations)}")
        print(f"Hash Keys: {hash_keys}\n")
        
        # 创建交叉检测矩阵
        print("交叉检测矩阵 (行=文本, 列=检测器):")
        print("-"*80)
        
        # 表头
        header = "文本/检测器 |"
        for key in hash_keys:
            header += f" Key{hash_keys.index(key)+1:2d} |"
        print(header)
        print("-" * len(header))
        
        # 矩阵内容
        for cd in cross_detection:
            row = f"   文本{cd['text_id']:2d}    |"
            for det in cd['detections']:
                symbol = " ✓  |" if det['prediction'] else " ✗  |"
                row += symbol
            print(row)
        
        # 统计正确检测和误检
        correct_detections = 0
        false_positives = 0
        total = len(cross_detection) * len(hash_keys)
        
        for cd in cross_detection:
            for det in cd['detections']:
                if det['detector_key'] == cd['text_key']:
                    if det['prediction']:
                        correct_detections += 1
                else:
                    if det['prediction']:
                        false_positives += 1
        
        print(f"\n检测统计:")
        print(f"  正确检测: {correct_detections}/{len(cross_detection)}")
        print(f"  误检: {false_positives}/{total - len(cross_detection)}")
        print(f"  正确率: {correct_detections/len(cross_detection)*100:.1f}%")
        print(f"  误检率: {false_positives/(total - len(cross_detection))*100:.1f}%")
    
    def analyze_parameter_mixing(self, result: Dict):
        """分析参数混合实验"""
        print("\n" + "="*80)
        print("参数混合实验分析")
        print("="*80 + "\n")
        
        fragments = result['fragments']
        param_combinations = result['param_combinations']
        detection_matrix = result['detection_matrix']
        
        print(f"参数组合数: {len(param_combinations)}")
        print(f"片段数: {len(fragments)}\n")
        
        # 显示参数组合
        print("参数组合:")
        print("-"*80)
        for i, params in enumerate(param_combinations, 1):
            print(f"  组合{i}: γ={params['gamma']}, δ={params['delta']}")
        
        # 检测性能
        print("\n检测性能:")
        print("-"*80)
        for det in detection_matrix:
            print(f"\n检测器 γ={det['detector_gamma']}:")
            print(f"  Z分数: {det['z_score']:.4f}")
            print(f"  检测结果: {'✓' if det['prediction'] else '✗'}")
            print(f"  绿色比例: {det['green_fraction']:.4f}")
        
        # 比较不同gamma值的检测效果
        gamma_groups = {}
        for det in detection_matrix:
            gamma = det['detector_gamma']
            if gamma not in gamma_groups:
                gamma_groups[gamma] = []
            gamma_groups[gamma].append(det['z_score'])
        
        print("\n不同Gamma值的平均Z分数:")
        for gamma, scores in gamma_groups.items():
            print(f"  γ={gamma}: {statistics.mean(scores):.4f}")
    
    def analyze_key_sharing(self, result: Dict):
        """分析密钥共享实验"""
        print("\n" + "="*80)
        print("密钥共享实验分析")
        print("="*80 + "\n")
        
        texts = result['texts']
        individual_detections = result['individual_detections']
        shared_detection = result['shared_key_detection']
        
        shared_texts = [t for t in texts if t['key_type'] == 'shared']
        individual_texts = [t for t in texts if t['key_type'] == 'individual']
        
        print(f"总文本数: {len(texts)}")
        print(f"共享密钥文本: {len(shared_texts)}")
        print(f"独立密钥文本: {len(individual_texts)}\n")
        
        print("共享密钥整体检测:")
        print("-"*80)
        print(f"  Z分数: {shared_detection['z_score']:.4f}")
        print(f"  检测结果: {'✓' if shared_detection['prediction'] else '✗'}")
        print(f"  绿色比例: {shared_detection['green_fraction']:.4f}")
        
        # 个别检测分析
        print("\n个别文本检测:")
        print("-"*80)
        
        shared_correct = 0
        shared_by_shared_key = 0
        individual_correct = 0
        individual_by_shared_key = 0
        
        for det in individual_detections:
            key_type = det['key_type']
            correct = det['correct_key_detection']['prediction']
            by_shared = det['shared_key_detection']['prediction']
            
            print(f"\n文本 {det['text_id']} ({key_type}):")
            print(f"  正确密钥检测: {'✓' if correct else '✗'} "
                  f"(z={det['correct_key_detection']['z_score']:.2f})")
            print(f"  共享密钥检测: {'✓' if by_shared else '✗'} "
                  f"(z={det['shared_key_detection']['z_score']:.2f})")
            
            if key_type == 'shared':
                if correct:
                    shared_correct += 1
                if by_shared:
                    shared_by_shared_key += 1
            else:
                if correct:
                    individual_correct += 1
                if by_shared:
                    individual_by_shared_key += 1
        
        # 统计
        print("\n统计摘要:")
        print("-"*80)
        if len(shared_texts) > 0:
            print(f"共享密钥文本:")
            print(f"  正确密钥检测率: {shared_correct/len(shared_texts)*100:.1f}%")
            print(f"  共享密钥检测率: {shared_by_shared_key/len(shared_texts)*100:.1f}%")
        
        if len(individual_texts) > 0:
            print(f"独立密钥文本:")
            print(f"  正确密钥检测率: {individual_correct/len(individual_texts)*100:.1f}%")
            print(f"  被共享密钥误检率: {individual_by_shared_key/len(individual_texts)*100:.1f}%")
    
    def generate_report(self, result: Dict, output_file: str = None):
        """生成完整的分析报告"""
        
        exp_type = result['experiment_type']
        
        # 根据实验类型选择分析方法
        if exp_type == 'fragment_mixing':
            self.analyze_fragment_mixing(result)
        elif exp_type == 'seed_mixing':
            self.analyze_seed_mixing(result)
        elif exp_type == 'parameter_mixing':
            self.analyze_parameter_mixing(result)
        elif exp_type == 'key_sharing':
            self.analyze_key_sharing(result)
        
        # 如果指定了输出文件，保存报告
        if output_file:
            # 这里可以添加保存文本报告的逻辑
            pass
    
    def compare_experiments(self, filenames: List[str]):
        """比较多个实验结果"""
        print("\n" + "="*80)
        print("实验对比")
        print("="*80 + "\n")
        
        results = [self.load_result(f) for f in filenames]
        
        # 按实验类型分组
        by_type = {}
        for r in results:
            exp_type = r['experiment_type']
            if exp_type not in by_type:
                by_type[exp_type] = []
            by_type[exp_type].append(r)
        
        print(f"加载了 {len(results)} 个实验结果")
        print(f"实验类型分布:")
        for exp_type, exps in by_type.items():
            print(f"  {exp_type}: {len(exps)} 个")
        
        # 针对每种类型进行比较
        for exp_type, exps in by_type.items():
            print(f"\n{'-'*80}")
            print(f"{exp_type} 类型实验对比:")
            print(f"{'-'*80}")
            
            # 这里可以添加更详细的对比逻辑
            for i, exp in enumerate(exps, 1):
                print(f"\n实验 {i}:")
                # 提取关键指标
                # ...


def main():
    """主函数：分析实验结果"""
    
    analyzer = HybridWatermarkAnalyzer()
    
    # 列出所有结果
    results = analyzer.list_results()
    
    if not results:
        print("未找到实验结果文件。")
        print("请先运行 hybrid_watermark_experiment.py 生成实验结果。")
        return
    
    print("\n" + "="*80)
    print("混合水印结果分析工具")
    print("="*80 + "\n")
    
    print("可用的实验结果:")
    for i, filename in enumerate(results, 1):
        print(f"  {i}. {filename}")
    
    print("\n选项:")
    print("  输入数字分析单个结果")
    print("  输入 'all' 分析所有结果")
    print("  输入 'q' 退出")
    
    choice = input("\n请选择: ").strip().lower()
    
    if choice == 'q':
        return
    elif choice == 'all':
        for filename in results:
            print(f"\n{'='*80}")
            print(f"分析: {filename}")
            print(f"{'='*80}")
            result = analyzer.load_result(filename)
            analyzer.generate_report(result)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                filename = results[idx]
                result = analyzer.load_result(filename)
                analyzer.generate_report(result)
            else:
                print("无效的选择")
        except ValueError:
            print("无效的输入")


if __name__ == "__main__":
    main()
