"""
Hybrid Watermark Comparison Experiment

Compares three watermarking approaches under paraphrase attack:
1. KGW only (token-level)
2. SimMark only (sentence-level) 
3. Hybrid (KGW + SimMark combined)

The hypothesis:
- KGW: Good detection, but vulnerable to paraphrase (tokens change)
- SimMark: Robust to paraphrase (semantics preserved), but weaker signal per sentence
- Hybrid: Best of both worlds - dual detection, one survives paraphrase

Experiment Flow:
1. Generate text with each method
2. Detect watermark (should be strong)
3. Paraphrase text
4. Detect again (see which survives)
5. Compare survival rates
"""

from __future__ import annotations

import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SimMark"))

from hybrid_kgw_simmark import (
    HybridWatermarkGenerator,
    HybridWatermarkDetector,
    HybridDetectionResult,
)
from simmark_paraphrase_experiment import SimMarkParaphraseExperiment
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector


@dataclass
class ComparisonResult:
    """Result for one prompt across all three methods"""
    prompt: str
    
    # KGW only
    kgw_original_z: float
    kgw_paraphrase_z: float
    kgw_survived: bool
    
    # SimMark only
    simmark_original_z: float
    simmark_paraphrase_z: float
    simmark_survived: bool
    
    # Hybrid
    hybrid_kgw_original_z: float
    hybrid_kgw_paraphrase_z: float
    hybrid_simmark_original_z: float
    hybrid_simmark_paraphrase_z: float
    hybrid_kgw_survived: bool
    hybrid_simmark_survived: bool
    hybrid_any_survived: bool  # At least one detector still positive
    hybrid_both_survived: bool  # Both detectors still positive


class HybridComparisonExperiment:
    """Compare KGW vs SimMark vs Hybrid under paraphrase attack."""
    
    PARAPHRASE_INSTRUCTION = """/no_think
Paraphrase the following text using different words and phrasing while preserving BOTH meaning AND structure.
CRITICAL RULES:
1. KEEP THE SAME NUMBER OF SENTENCES - do NOT merge or split sentences.
2. Maintain the same paragraph structure and sentence order.
3. Change vocabulary and sentence phrasing, but preserve the logical flow.
4. Output ONLY the rewritten text, nothing else.
5. Do NOT include any notes, explanations, labels, or prefixes.
6. Start your response directly with the paraphrased content."""

    def __init__(
        self,
        generator_model: str = "llama-3.2-3b",
        paraphraser_model: str = "qwen-3-4b",
        device: Optional[str] = None,
        kgw_gamma: float = 0.25,
        kgw_delta: float = 2.0,
        simmark_interval: Tuple[float, float] = (0.75, 0.83),
        simmark_K: int = 250,
        simmark_max_trials: int = 50,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        self.generator_model = generator_model
        self.paraphraser_model = paraphraser_model
        
        self.kgw_gamma = kgw_gamma
        self.kgw_delta = kgw_delta
        self.simmark_interval = simmark_interval
        self.simmark_K = simmark_K
        self.simmark_max_trials = simmark_max_trials
        
        # Lazy-loaded components
        self._hybrid_gen = None
        self._simmark_exp = None
        self._paraphraser_model = None
        self._paraphraser_tokenizer = None
        
        # Config manager
        from llama_demos.model_config_manager import ModelConfigManager
        config_path = os.path.join(os.path.dirname(__file__), "llama_demos", "model_config.json")
        self.config_manager = ModelConfigManager(config_path)
        
        # Results directory
        self.results_dir = os.path.join(os.path.dirname(__file__), "hybrid_comparison_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("=" * 70)
        print("HYBRID COMPARISON EXPERIMENT")
        print("Comparing: KGW only vs SimMark only vs Hybrid (KGW+SimMark)")
        print("=" * 70)
    
    def _get_hybrid_generator(self) -> HybridWatermarkGenerator:
        if self._hybrid_gen is None:
            self._hybrid_gen = HybridWatermarkGenerator(
                model_nickname=self.generator_model,
                kgw_gamma=self.kgw_gamma,
                kgw_delta=self.kgw_delta,
                simmark_interval=self.simmark_interval,
                simmark_K=self.simmark_K,
                simmark_max_trials=self.simmark_max_trials,
            )
        return self._hybrid_gen
    
    def _get_simmark_experiment(self) -> SimMarkParaphraseExperiment:
        if self._simmark_exp is None:
            self._simmark_exp = SimMarkParaphraseExperiment(
                generator_model=self.generator_model,
                paraphraser_model=self.paraphraser_model,
                simmark_mode="cosine",
                simmark_interval=self.simmark_interval,
                simmark_K=self.simmark_K,
                max_trials=self.simmark_max_trials,
                paraphrase_mode="moderate",
            )
        return self._simmark_exp
    
    def _load_paraphraser(self):
        if self._paraphraser_model is None:
            info = self.config_manager.get_model_info_by_nickname(self.paraphraser_model)
            if not info:
                raise ValueError(f"Paraphraser not found: {self.paraphraser_model}")
            
            print(f"\n📦 Loading paraphraser: {self.paraphraser_model}")
            
            self._paraphraser_tokenizer = AutoTokenizer.from_pretrained(
                info["model_identifier"], trust_remote_code=True
            )
            if self._paraphraser_tokenizer.pad_token is None:
                self._paraphraser_tokenizer.pad_token = self._paraphraser_tokenizer.eos_token
            
            self._paraphraser_model = AutoModelForCausalLM.from_pretrained(
                info["model_identifier"],
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                self._paraphraser_model = self._paraphraser_model.to(self.device)
            self._paraphraser_model.eval()
            
            print("�?Paraphraser loaded")
        
        return self._paraphraser_tokenizer, self._paraphraser_model
    
    def paraphrase(self, text: str) -> str:
        """Paraphrase text using local model."""
        tokenizer, model = self._load_paraphraser()
        
        prompt = f"""{self.PARAPHRASE_INSTRUCTION}

Text:
{text}

Output:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=int(len(text.split()) * 1.5) + 50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = output[:, inputs["input_ids"].shape[-1]:]
        result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        
        # Clean
        for prefix in ["Output:", "output:", "Here is", "Paraphrased"]:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].strip()
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1].strip()
        
        return result
    
    def run_single_comparison(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        verbose: bool = False,
    ) -> ComparisonResult:
        """
        Run comparison for a single prompt.
        
        Generates text with all three methods, paraphrases, and detects.
        """
        print(f"\n{'='*70}")
        print(f"Prompt: {prompt[:60]}...")
        print('='*70)
        
        hybrid_gen = self._get_hybrid_generator()
        simmark_exp = self._get_simmark_experiment()
        
        # ======== 1. KGW Only ========
        print("\n🔴🟢 Generating with KGW only...")
        kgw_text, _ = simmark_exp.generate_with_kgw(
            prompt, gamma=self.kgw_gamma, delta=self.kgw_delta,
            max_new_tokens=max_new_tokens
        )
        kgw_original = simmark_exp.detect_kgw(kgw_text, gamma=self.kgw_gamma)
        print(f"   Original z-score: {kgw_original.z_score:.4f}")
        
        print("   Paraphrasing...")
        kgw_para = self.paraphrase(kgw_text)
        kgw_para_det = simmark_exp.detect_kgw(kgw_para, gamma=self.kgw_gamma)
        print(f"   Paraphrased z-score: {kgw_para_det.z_score:.4f}")
        
        # ======== 2. SimMark Only ========
        print("\n🌊 Generating with SimMark only...")
        simmark_text, _ = simmark_exp.generate_with_simmark(prompt, max_new_tokens=max_new_tokens)
        simmark_original = simmark_exp.detect_simmark(simmark_text)
        print(f"   Original z-score: {simmark_original.z_score:.4f}")
        
        print("   Paraphrasing...")
        simmark_para = self.paraphrase(simmark_text)
        simmark_para_det = simmark_exp.detect_simmark(simmark_para)
        print(f"   Paraphrased z-score: {simmark_para_det.z_score:.4f}")
        
        # ======== 3. Hybrid (KGW + SimMark) ========
        print("\n🎯 Generating with Hybrid (KGW + SimMark)...")
        hybrid_text, stats, _ = hybrid_gen.generate(
            prompt, max_new_tokens=max_new_tokens, verbose=verbose
        )
        
        # Get tokenizer for detector
        tokenizer, _ = hybrid_gen._load_model()
        hybrid_detector = HybridWatermarkDetector(
            tokenizer=tokenizer,
            kgw_gamma=self.kgw_gamma,
            simmark_interval=self.simmark_interval,
            simmark_K=self.simmark_K,
        )
        
        hybrid_original = hybrid_detector.detect(hybrid_text)
        print(f"   Original KGW z: {hybrid_original.kgw_z_score:.4f}, SimMark z: {hybrid_original.simmark_z_score:.4f}")
        
        print("   Paraphrasing...")
        hybrid_para = self.paraphrase(hybrid_text)
        hybrid_para_det = hybrid_detector.detect(hybrid_para)
        print(f"   Paraphrased KGW z: {hybrid_para_det.kgw_z_score:.4f}, SimMark z: {hybrid_para_det.simmark_z_score:.4f}")
        
        # ======== Results ========
        result = ComparisonResult(
            prompt=prompt,
            
            kgw_original_z=kgw_original.z_score,
            kgw_paraphrase_z=kgw_para_det.z_score,
            kgw_survived=kgw_para_det.prediction,
            
            simmark_original_z=simmark_original.z_score,
            simmark_paraphrase_z=simmark_para_det.z_score,
            simmark_survived=simmark_para_det.prediction,
            
            hybrid_kgw_original_z=hybrid_original.kgw_z_score,
            hybrid_kgw_paraphrase_z=hybrid_para_det.kgw_z_score,
            hybrid_simmark_original_z=hybrid_original.simmark_z_score,
            hybrid_simmark_paraphrase_z=hybrid_para_det.simmark_z_score,
            hybrid_kgw_survived=hybrid_para_det.kgw_prediction,
            hybrid_simmark_survived=hybrid_para_det.simmark_prediction,
            hybrid_any_survived=hybrid_para_det.kgw_prediction or hybrid_para_det.simmark_prediction,
            hybrid_both_survived=hybrid_para_det.kgw_prediction and hybrid_para_det.simmark_prediction,
        )
        
        self._print_comparison(result)
        return result
    
    def _print_comparison(self, result: ComparisonResult):
        """Print comparison table."""
        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)
        
        print("\n┌────────────────┬─────────────────┬─────────────────┬───────────�?)
        print("�?    Method     �? Original z     �? Paraphrase z   �? Survived �?)
        print("├────────────────┼─────────────────┼─────────────────┼───────────�?)
        
        kgw_surv = "�?YES" if result.kgw_survived else "�?NO"
        print(f"�?KGW only       �?   {result.kgw_original_z:>8.4f}     �?   {result.kgw_paraphrase_z:>8.4f}     �? {kgw_surv:>6}  �?)
        
        sim_surv = "�?YES" if result.simmark_survived else "�?NO"
        print(f"�?SimMark only   �?   {result.simmark_original_z:>8.4f}     �?   {result.simmark_paraphrase_z:>8.4f}     �? {sim_surv:>6}  �?)
        
        print("├────────────────┼─────────────────┼─────────────────┼───────────�?)
        
        hyb_kgw_surv = "�?YES" if result.hybrid_kgw_survived else "�?NO"
        print(f"�?Hybrid (KGW)   �?   {result.hybrid_kgw_original_z:>8.4f}     �?   {result.hybrid_kgw_paraphrase_z:>8.4f}     �? {hyb_kgw_surv:>6}  �?)
        
        hyb_sim_surv = "�?YES" if result.hybrid_simmark_survived else "�?NO"
        print(f"�?Hybrid (SimM)  �?   {result.hybrid_simmark_original_z:>8.4f}     �?   {result.hybrid_simmark_paraphrase_z:>8.4f}     �? {hyb_sim_surv:>6}  �?)
        
        print("└────────────────┴─────────────────┴─────────────────┴───────────�?)
        
        any_surv = "�?YES" if result.hybrid_any_survived else "�?NO"
        both_surv = "�?YES" if result.hybrid_both_survived else "�?NO"
        print(f"\n🎯 Hybrid ANY detector survived: {any_surv}")
        print(f"🎯 Hybrid BOTH detectors survived: {both_surv}")
    
    def run_batch_comparison(
        self,
        prompts: List[str],
        max_new_tokens: int = 200,
    ) -> Dict:
        """Run comparison on multiple prompts and aggregate results."""
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n\n{'#'*70}")
            print(f"Experiment {i+1}/{len(prompts)}")
            print('#'*70)
            
            result = self.run_single_comparison(prompt, max_new_tokens)
            results.append(asdict(result))
        
        # Aggregate
        n = len(prompts)
        summary = {
            "total_experiments": n,
            "kgw_survival_rate": sum(1 for r in results if r["kgw_survived"]) / n,
            "simmark_survival_rate": sum(1 for r in results if r["simmark_survived"]) / n,
            "hybrid_any_survival_rate": sum(1 for r in results if r["hybrid_any_survived"]) / n,
            "hybrid_both_survival_rate": sum(1 for r in results if r["hybrid_both_survived"]) / n,
            "kgw_avg_decay": np.mean([r["kgw_original_z"] - r["kgw_paraphrase_z"] for r in results]),
            "simmark_avg_decay": np.mean([r["simmark_original_z"] - r["simmark_paraphrase_z"] for r in results]),
        }
        
        output = {
            "summary": summary,
            "results": results,
            "config": {
                "generator": self.generator_model,
                "paraphraser": self.paraphraser_model,
                "kgw_gamma": self.kgw_gamma,
                "kgw_delta": self.kgw_delta,
                "simmark_interval": self.simmark_interval,
                "simmark_K": self.simmark_K,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        # Print summary
        print("\n\n" + "=" * 70)
        print("BATCH EXPERIMENT SUMMARY")
        print("=" * 70)
        print(f"\nTotal experiments: {n}")
        print(f"\n📊 Survival Rates After Paraphrase:")
        print(f"   KGW only:          {summary['kgw_survival_rate']:.1%}")
        print(f"   SimMark only:      {summary['simmark_survival_rate']:.1%}")
        print(f"   Hybrid (ANY):      {summary['hybrid_any_survival_rate']:.1%}")
        print(f"   Hybrid (BOTH):     {summary['hybrid_both_survival_rate']:.1%}")
        
        # Save
        filepath = os.path.join(
            self.results_dir,
            f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved: {filepath}")
        
        return output


def main():
    parser = argparse.ArgumentParser(description="Hybrid Watermark Comparison")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--batch", action="store_true", help="Run batch experiment")
    parser.add_argument("--model", type=str, default="llama-3.2-3b")
    parser.add_argument("--paraphraser", type=str, default="qwen-3-4b")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()
    
    experiment = HybridComparisonExperiment(
        generator_model=args.model,
        paraphraser_model=args.paraphraser,
    )
    
    if args.batch:
        prompts = [
            "Write a short story about artificial intelligence discovering emotions.",
            "Explain the concept of quantum computing to a high school student.",
            "Describe a day in the life of an astronaut on the International Space Station.",
            "Discuss the ethical implications of genetic engineering.",
            "Write an essay about the impact of social media on modern communication.",
        ]
        experiment.run_batch_comparison(prompts, max_new_tokens=args.max_tokens)
    else:
        prompt = args.prompt or "Write a short story about a robot learning to paint."
        experiment.run_single_comparison(prompt, max_new_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
