"""
混合水印实验系统
实现多种混合水印方案的生成和检测

实验类型：
1. 片段级混合：段落中不同片段使用不同水印方案
2. 种子混合：同一模型使用不同种子
3. 参数混合：不同gamma/delta组合
4. 密钥共享混合：不同模型共享或混合密钥

重构版本：继承 BaseExperiment，去除重复代码。
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from base_experiment import BaseExperiment


class HybridWatermarkExperiment(BaseExperiment):
    """混合水印实验类"""

    def __init__(
        self,
        model_nickname: str = "llama-3.2-3b",
        device: str = None,
    ):
        """
        初始化混合水印实验

        Args:
            model_nickname: 模型昵称
            device: 运行设备
        """
        super().__init__(
            default_model=model_nickname,
            device=device,
            results_subdir="hybrid_watermark_results",
        )
        self.model_nickname = model_nickname

        # 预加载默认模型
        print(f"\n{'='*80}")
        print("混合水印实验系统初始化")
        print(f"{'='*80}\n")

        self.tokenizer, self.model, self.model_info = self.load_model(model_nickname)

    # ========== 实验1: 片段级混合水印 ==========

    def experiment_fragment_mixing(
        self,
        base_prompt: str,
        fragment_configs: List[Dict],
        tokens_per_fragment: int = 50,
    ) -> Dict:
        """
        实验1：片段级混合水印
        在同一段落中，不同片段使用不同的水印配置
        """
        print(f"\n{'='*80}")
        print("实验1: 片段级混合水印")
        print(f"{'='*80}\n")
        print(f"基础提示: {base_prompt}")
        print(f"片段数量: {len(fragment_configs)}\n")

        fragments = []
        current_prompt = base_prompt

        for i, config in enumerate(fragment_configs, 1):
            print(f"生成片段 {i}/{len(fragment_configs)} (gamma={config.get('gamma', 0.25)}, delta={config.get('delta', 2.0)})...")

            processor = self.create_watermark_processor(self.tokenizer, config)
            prompt_prepared = self.ensure_min_context(current_prompt, self.tokenizer, processor)

            text, _, _ = self.generate_with_watermark(
                prompt=current_prompt,
                model_nickname=self.model_nickname,
                watermark_config=config,
                max_new_tokens=tokens_per_fragment,
            )

            fragments.append({
                "text": text,
                "config": config,
                "fragment_id": i,
            })
            current_prompt = current_prompt + " " + text
            print(f"  [OK] 片段生成完成\n")

        combined_text = " ".join([f["text"] for f in fragments])

        # 检测
        detection_results = []
        for i, config in enumerate(fragment_configs, 1):
            detection = self.detect_watermark(combined_text, self.tokenizer, config)
            detection_results.append({
                "config_id": i,
                "config": config,
                "z_score": detection.z_score,
                "prediction": detection.prediction,
                "green_fraction": detection.green_fraction,
            })

        return {
            "experiment_type": "fragment_mixing",
            "base_prompt": base_prompt,
            "fragments": fragments,
            "combined_text": combined_text,
            "detection_results": detection_results,
        }

    # ========== 实验2: 种子混合 ==========

    def experiment_seed_mixing(
        self,
        prompt: str,
        num_variations: int = 3,
        base_gamma: float = 0.25,
        base_delta: float = 2.0,
        max_new_tokens: int = 100,
    ) -> Dict:
        """
        实验2：种子混合
        同一模型使用不同的水印种子（hash_key）
        """
        print(f"\n{'='*80}")
        print("实验2: 种子混合 (不同Hash Key)")
        print(f"{'='*80}\n")
        print(f"提示: {prompt}")
        print(f"变体数量: {num_variations}\n")

        base_key = 15485863
        hash_keys = [base_key + i * 1000000 for i in range(num_variations)]
        variations = []

        for i, hash_key in enumerate(hash_keys, 1):
            print(f"生成变体 {i}/{num_variations} (hash_key={hash_key})...")

            config = {
                "gamma": base_gamma,
                "delta": base_delta,
                "seeding_scheme": "selfhash",
                "hash_key": hash_key,
            }
            text, _, _ = self.generate_with_watermark(
                prompt=prompt,
                watermark_config=config,
                max_new_tokens=max_new_tokens,
            )
            variations.append({"variation_id": i, "hash_key": hash_key, "text": text})
            print(f"  [OK] 变体生成完成\n")

        # 交叉检测
        cross_detection = []
        for var in variations:
            var_detections = []
            for key in hash_keys:
                config = {"gamma": base_gamma, "seeding_scheme": "selfhash", "hash_key": key}
                detection = self.detect_watermark(var["text"], self.tokenizer, config)
                var_detections.append({
                    "detector_key": key,
                    "z_score": detection.z_score,
                    "prediction": detection.prediction,
                })
            cross_detection.append({
                "text_id": var["variation_id"],
                "text_key": var["hash_key"],
                "detections": var_detections,
            })

        mixed_text = " ".join([v["text"] for v in variations])

        return {
            "experiment_type": "seed_mixing",
            "prompt": prompt,
            "variations": variations,
            "mixed_text": mixed_text,
            "cross_detection": cross_detection,
            "hash_keys": hash_keys,
        }

    # ========== 实验3: 参数混合 ==========

    def experiment_parameter_mixing(
        self,
        prompt: str,
        gamma_values: List[float] = None,
        delta_values: List[float] = None,
        tokens_per_config: int = 50,
    ) -> Dict:
        """
        实验3：参数混合
        使用不同的gamma和delta组合生成文本片段
        """
        gamma_values = gamma_values or [0.25, 0.5]
        delta_values = delta_values or [1.0, 2.0, 3.0]

        print(f"\n{'='*80}")
        print("实验3: 参数混合 (Gamma/Delta组合)")
        print(f"{'='*80}\n")
        print(f"提示: {prompt}")
        print(f"Gamma值: {gamma_values}, Delta值: {delta_values}\n")

        param_combinations = [{"gamma": g, "delta": d} for g in gamma_values for d in delta_values]
        fragments = []
        current_prompt = prompt

        for i, params in enumerate(param_combinations, 1):
            print(f"生成片段 {i}/{len(param_combinations)} (gamma={params['gamma']}, delta={params['delta']})...")

            config = {"gamma": params["gamma"], "delta": params["delta"], "seeding_scheme": "selfhash"}
            text, _, _ = self.generate_with_watermark(
                prompt=current_prompt,
                watermark_config=config,
                max_new_tokens=tokens_per_config,
            )
            fragments.append({"text": text, "gamma": params["gamma"], "delta": params["delta"], "fragment_id": i})
            current_prompt = current_prompt + " " + text
            print(f"  [OK] 片段生成完成\n")

        combined_text = " ".join([f["text"] for f in fragments])

        detection_matrix = []
        for params in param_combinations:
            config = {"gamma": params["gamma"], "seeding_scheme": "selfhash"}
            detection = self.detect_watermark(combined_text, self.tokenizer, config)
            detection_matrix.append({
                "detector_gamma": params["gamma"],
                "z_score": detection.z_score,
                "prediction": detection.prediction,
                "green_fraction": detection.green_fraction,
            })

        return {
            "experiment_type": "parameter_mixing",
            "prompt": prompt,
            "param_combinations": param_combinations,
            "fragments": fragments,
            "combined_text": combined_text,
            "detection_matrix": detection_matrix,
        }

    # ========== 实验4: 密钥共享混合 ==========

    def experiment_key_sharing(
        self,
        prompts: List[str],
        shared_key: int = 15485863,
        individual_keys: List[int] = None,
        max_new_tokens: int = 100,
    ) -> Dict:
        """
        实验4：密钥共享混合
        部分文本使用共享密钥，部分使用独立密钥
        """
        print(f"\n{'='*80}")
        print("实验4: 密钥共享混合")
        print(f"{'='*80}\n")
        print(f"文本数量: {len(prompts)}, 共享密钥: {shared_key}\n")

        if individual_keys is None:
            individual_keys = [shared_key + (i + 1) * 500000 for i in range(len(prompts))]

        texts = []
        for i, prompt in enumerate(prompts):
            use_shared = i % 2 == 0
            key = shared_key if use_shared else individual_keys[i]
            print(f"生成文本 {i+1}/{len(prompts)} ({'共享' if use_shared else '独立'}密钥: {key})...")

            config = {"gamma": 0.25, "delta": 2.0, "seeding_scheme": "selfhash", "hash_key": key}
            text, _, _ = self.generate_with_watermark(
                prompt=prompt,
                watermark_config=config,
                max_new_tokens=max_new_tokens,
            )
            texts.append({
                "text_id": i + 1,
                "prompt": prompt,
                "text": text,
                "key_type": "shared" if use_shared else "individual",
                "hash_key": key,
            })
            print(f"  [OK] 文本生成完成\n")

        combined_text = " ".join([t["text"] for t in texts])

        # 共享密钥检测
        shared_config = {"gamma": 0.25, "seeding_scheme": "selfhash", "hash_key": shared_key}
        shared_detection = self.detect_watermark(combined_text, self.tokenizer, shared_config)

        # 个别检测
        individual_detections = []
        for text_info in texts:
            correct_config = {"gamma": 0.25, "seeding_scheme": "selfhash", "hash_key": text_info["hash_key"]}
            result_correct = self.detect_watermark(text_info["text"], self.tokenizer, correct_config)
            result_shared = self.detect_watermark(text_info["text"], self.tokenizer, shared_config)
            individual_detections.append({
                "text_id": text_info["text_id"],
                "key_type": text_info["key_type"],
                "correct_key_detection": {"z_score": result_correct.z_score, "prediction": result_correct.prediction},
                "shared_key_detection": {"z_score": result_shared.z_score, "prediction": result_shared.prediction},
            })

        return {
            "experiment_type": "key_sharing",
            "prompts": prompts,
            "shared_key": shared_key,
            "individual_keys": individual_keys,
            "texts": texts,
            "combined_text": combined_text,
            "shared_key_detection": {
                "z_score": shared_detection.z_score,
                "prediction": shared_detection.prediction,
                "green_fraction": shared_detection.green_fraction,
            },
            "individual_detections": individual_detections,
        }

    # ========== 保存和报告 ==========

    def save_results(self, results: Dict, filename: str = None) -> str:
        """保存实验结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{results['experiment_type']}_{timestamp}.json"
        return super().save_results(results, filename)

    def print_summary(self, results: Dict) -> None:
        """打印实验摘要"""
        print(f"\n{'='*80}")
        print(f"实验摘要: {results['experiment_type']}")
        print(f"{'='*80}\n")

        exp_type = results["experiment_type"]

        if exp_type == "fragment_mixing":
            print(f"片段数量: {len(results['fragments'])}")
            for det in results["detection_results"]:
                print(f"  配置{det['config_id']}: z={det['z_score']:.2f}, 检测={det['prediction']}")

        elif exp_type == "seed_mixing":
            print(f"变体数量: {len(results['variations'])}")
            for cd in results["cross_detection"]:
                matches = sum(1 for d in cd["detections"] if d["prediction"])
                print(f"  文本{cd['text_id']}: 检测到 {matches}/{len(cd['detections'])} 个匹配")

        elif exp_type == "parameter_mixing":
            print(f"参数组合数: {len(results['param_combinations'])}")
            for det in results["detection_matrix"]:
                print(f"  Gamma={det['detector_gamma']}: z={det['z_score']:.2f}, 检测={det['prediction']}")

        elif exp_type == "key_sharing":
            shared_count = sum(1 for t in results["texts"] if t["key_type"] == "shared")
            print(f"共享密钥文本: {shared_count}/{len(results['texts'])}")
            print(f"共享密钥检测: z={results['shared_key_detection']['z_score']:.2f}")
            for det in results["individual_detections"]:
                print(f"  文本{det['text_id']}: 正确={det['correct_key_detection']['prediction']}, 共享={det['shared_key_detection']['prediction']}")

        print(f"\n{'='*80}\n")


def main():
    """运行所有混合水印实验"""
    model_nickname = "llama-3.2-3b"
    if len(sys.argv) > 1:
        model_nickname = sys.argv[1]
        print(f"使用指定模型: {model_nickname}\n")

    print("\n" + "=" * 80)
    print("混合水印实验系统")
    print(f"模型: {model_nickname}")
    print("=" * 80 + "\n")

    experiment = HybridWatermarkExperiment(model_nickname=model_nickname)

    # 实验1: 片段级混合
    print("\n开始实验1...")
    result1 = experiment.experiment_fragment_mixing(
        base_prompt="The future of artificial intelligence is",
        fragment_configs=[
            {"gamma": 0.25, "delta": 2.0, "hash_key": 15485863},
            {"gamma": 0.5, "delta": 2.0, "hash_key": 15485863},
            {"gamma": 0.25, "delta": 3.0, "hash_key": 15485863},
        ],
        tokens_per_fragment=50,
    )
    experiment.print_summary(result1)
    experiment.save_results(result1)

    # 实验2: 种子混合
    print("\n开始实验2...")
    result2 = experiment.experiment_seed_mixing(
        prompt="Write a short story about robots:",
        num_variations=3,
        max_new_tokens=80,
    )
    experiment.print_summary(result2)
    experiment.save_results(result2)

    # 实验3: 参数混合
    print("\n开始实验3...")
    result3 = experiment.experiment_parameter_mixing(
        prompt="Explain quantum computing:",
        gamma_values=[0.25, 0.5],
        delta_values=[1.5, 2.5],
        tokens_per_config=40,
    )
    experiment.print_summary(result3)
    experiment.save_results(result3)

    # 实验4: 密钥共享
    print("\n开始实验4...")
    result4 = experiment.experiment_key_sharing(
        prompts=[
            "The benefits of renewable energy include",
            "In the year 2050, technology will",
            "Climate change is affecting",
            "Space exploration has led to",
        ],
        max_new_tokens=60,
    )
    experiment.print_summary(result4)
    experiment.save_results(result4)

    print("\n" + "=" * 80)
    print("所有实验完成！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
