# -*- coding: utf-8 -*-
"""
Hybrid Robustness Experiment: Compare KGW vs SimMark vs Hybrid under different attack intensities.

Hypothesis:
- Light paraphrase: SimMark should be strong
- Aggressive rewrite (12->6 sentences): Traditional KGW might be stronger
- Hybrid: Should provide best of both worlds?

This experiment tests watermark survival under:
1. No attack (baseline)
2. Light paraphrase (word substitution, minor rewording)
3. Moderate paraphrase (sentence restructuring)  
4. Aggressive paraphrase (significant compression/rewrite)
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
import time

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Project imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from hybrid_kgw_simmark import HybridWatermarkGenerator, HybridWatermarkDetector
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector


@dataclass
class AttackResult:
    """Result of applying an attack to watermarked text"""
    original_text: str
    attacked_text: str
    attack_type: str
    original_sentences: int
    attacked_sentences: int
    compression_ratio: float


@dataclass  
class DetectionResult:
    """Detection results for a single watermark type"""
    z_score: float
    detected: bool
    threshold: float


@dataclass
class ExperimentResult:
    """Full experiment result for one sample"""
    prompt: str
    original_text: str
    attacked_text: str
    attack_type: str
    
    # Detection results
    kgw_before: DetectionResult
    kgw_after: DetectionResult
    simmark_before: DetectionResult
    simmark_after: DetectionResult
    hybrid_before: bool  # Both detected
    hybrid_after: bool   # Both detected
    
    # Survival
    kgw_survived: bool
    simmark_survived: bool
    hybrid_survived: bool


class ParaphraseAttacker:
    """Applies different levels of paraphrase attacks using an LLM."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        
    def _load_model(self):
        if self._model is not None:
            return
        
        print(f"Loading paraphrase model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self._model = self._model.to(self.device)
        self._model.eval()
        print("Paraphrase model loaded")
    
    def attack(self, text: str, attack_type: str) -> AttackResult:
        """Apply a paraphrase attack of the specified type."""
        from nltk.tokenize import sent_tokenize
        
        original_sentences = len(sent_tokenize(text))
        
        if attack_type == "none":
            return AttackResult(
                original_text=text,
                attacked_text=text,
                attack_type=attack_type,
                original_sentences=original_sentences,
                attacked_sentences=original_sentences,
                compression_ratio=1.0
            )
        
        self._load_model()
        
        # Define attack prompts
        attack_prompts = {
            "light": """Paraphrase the following text with minor word substitutions. Keep the same structure and number of sentences. Only change individual words to synonyms.

Text: {text}

Paraphrased version:""",
            
            "moderate": """Rewrite the following text in your own words. You may restructure sentences but keep the main ideas and similar length.

Text: {text}

Rewritten version:""",
            
            "aggressive": """Summarize and condense the following text to about half its length. Combine sentences and remove redundancy while keeping the core meaning.

Text: {text}

Condensed version:"""
        }
        
        prompt = attack_prompts[attack_type].format(text=text)
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=len(text.split()) + 50,  # Allow some buffer
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        
        full_output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the paraphrased part (after the prompt)
        if "Paraphrased version:" in full_output:
            attacked_text = full_output.split("Paraphrased version:")[-1].strip()
        elif "Rewritten version:" in full_output:
            attacked_text = full_output.split("Rewritten version:")[-1].strip()
        elif "Condensed version:" in full_output:
            attacked_text = full_output.split("Condensed version:")[-1].strip()
        else:
            # Fallback: take everything after the original text
            attacked_text = full_output[len(prompt):].strip()
        
        attacked_sentences = len(sent_tokenize(attacked_text))
        
        return AttackResult(
            original_text=text,
            attacked_text=attacked_text,
            attack_type=attack_type,
            original_sentences=original_sentences,
            attacked_sentences=attacked_sentences,
            compression_ratio=attacked_sentences / max(original_sentences, 1)
        )


class RobustnessExperiment:
    """Run the full robustness comparison experiment."""
    
    def __init__(
        self,
        model_nickname: str = "llama-3.2-3b",
        kgw_gamma: float = 0.25,
        kgw_delta: float = 2.0,
        simmark_interval: Tuple[float, float] = (0.75, 0.83),
        kgw_z_threshold: float = 3.0,
        simmark_z_threshold: float = 2.0,
    ):
        self.model_nickname = model_nickname
        self.kgw_gamma = kgw_gamma
        self.kgw_delta = kgw_delta
        self.simmark_interval = simmark_interval
        self.kgw_z_threshold = kgw_z_threshold
        self.simmark_z_threshold = simmark_z_threshold
        
        self.generator = None
        self.hybrid_detector = None
        self.kgw_detector = None
        self.attacker = None
        
        self.results_dir = os.path.join(
            os.path.dirname(__file__), 
            "hybrid_robustness_results"
        )
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _init_components(self):
        """Initialize all components."""
        if self.generator is not None:
            return
            
        print("=" * 70)
        print("HYBRID ROBUSTNESS EXPERIMENT")
        print("=" * 70)
        
        # Initialize generator
        self.generator = HybridWatermarkGenerator(
            model_nickname=self.model_nickname,
            kgw_gamma=self.kgw_gamma,
            kgw_delta=self.kgw_delta,
            simmark_interval=self.simmark_interval,
        )
        
        # Load model to get tokenizer
        tokenizer, _ = self.generator._load_model()
        
        # Initialize hybrid detector
        self.hybrid_detector = HybridWatermarkDetector(
            tokenizer=tokenizer,
            kgw_gamma=self.kgw_gamma,
            simmark_interval=self.simmark_interval,
            kgw_z_threshold=self.kgw_z_threshold,
            simmark_z_threshold=self.simmark_z_threshold,
        )
        
        # Initialize standalone KGW detector for comparison
        self.kgw_detector = WatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=self.kgw_gamma,
            seeding_scheme="selfhash",
            device=self.generator.device,
            tokenizer=tokenizer,
            z_threshold=self.kgw_z_threshold,
        )
        
        # Initialize attacker
        self.attacker = ParaphraseAttacker(device=self.generator.device)
        
        print("All components initialized")
    
    def run_single_experiment(
        self,
        prompt: str,
        attack_type: str,
        verbose: bool = False,
    ) -> ExperimentResult:
        """Run a single experiment with one prompt and one attack type."""
        self._init_components()
        
        print(f"\n--- Experiment: {attack_type} attack ---")
        
        # Generate watermarked text
        text, stats, metadata = self.generator.generate(
            prompt=prompt,
            max_new_tokens=200,
            verbose=verbose,
        )
        
        # Detect before attack
        before_result = self.hybrid_detector.detect(text)
        
        kgw_before = DetectionResult(
            z_score=before_result.kgw_z_score,
            detected=before_result.kgw_prediction,
            threshold=self.kgw_z_threshold,
        )
        
        simmark_before = DetectionResult(
            z_score=before_result.simmark_z_score,
            detected=before_result.simmark_prediction,
            threshold=self.simmark_z_threshold,
        )
        
        # Apply attack
        attack_result = self.attacker.attack(text, attack_type)
        attacked_text = attack_result.attacked_text
        
        print(f"   Original: {attack_result.original_sentences} sentences")
        print(f"   After attack: {attack_result.attacked_sentences} sentences")
        print(f"   Compression: {attack_result.compression_ratio:.2f}")
        
        # Detect after attack
        after_result = self.hybrid_detector.detect(attacked_text)
        
        kgw_after = DetectionResult(
            z_score=after_result.kgw_z_score,
            detected=after_result.kgw_prediction,
            threshold=self.kgw_z_threshold,
        )
        
        simmark_after = DetectionResult(
            z_score=after_result.simmark_z_score,
            detected=after_result.simmark_prediction,
            threshold=self.simmark_z_threshold,
        )
        
        # Compile results
        result = ExperimentResult(
            prompt=prompt,
            original_text=text,
            attacked_text=attacked_text,
            attack_type=attack_type,
            kgw_before=kgw_before,
            kgw_after=kgw_after,
            simmark_before=simmark_before,
            simmark_after=simmark_after,
            hybrid_before=before_result.both_detected,
            hybrid_after=after_result.both_detected,
            kgw_survived=kgw_after.detected,
            simmark_survived=simmark_after.detected,
            hybrid_survived=after_result.both_detected,
        )
        
        # Print summary
        print(f"\n   Results:")
        print(f"   KGW:     z={kgw_before.z_score:.2f} -> {kgw_after.z_score:.2f} | Survived: {result.kgw_survived}")
        print(f"   SimMark: z={simmark_before.z_score:.2f} -> {simmark_after.z_score:.2f} | Survived: {result.simmark_survived}")
        print(f"   Hybrid:  {result.hybrid_before} -> {result.hybrid_after} | Survived: {result.hybrid_survived}")
        
        return result
    
    def run_full_experiment(
        self,
        prompts: List[str] = None,
        attack_types: List[str] = None,
        verbose: bool = False,
    ) -> Dict:
        """Run the full experiment with multiple prompts and attack types."""
        
        if prompts is None:
            prompts = [
                "Write a short story about a robot learning to paint.",
                "Explain how machine learning works to a high school student.",
                "Describe the process of making homemade bread from scratch.",
                "Write about the importance of exercise for mental health.",
                "Explain the water cycle in nature.",
            ]
        
        if attack_types is None:
            attack_types = ["none", "light", "moderate", "aggressive"]
        
        all_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*70}")
            print(f"PROMPT {i+1}/{len(prompts)}: {prompt[:50]}...")
            print("=" * 70)
            
            for attack_type in attack_types:
                try:
                    result = self.run_single_experiment(
                        prompt=prompt,
                        attack_type=attack_type,
                        verbose=verbose,
                    )
                    all_results.append(asdict(result))
                except Exception as e:
                    print(f"   ERROR: {e}")
                    continue
        
        # Compute aggregate statistics
        summary = self._compute_summary(all_results)
        
        # Save results
        output = {
            "experiment_config": {
                "model": self.model_nickname,
                "kgw_gamma": self.kgw_gamma,
                "kgw_delta": self.kgw_delta,
                "simmark_interval": self.simmark_interval,
                "kgw_z_threshold": self.kgw_z_threshold,
                "simmark_z_threshold": self.simmark_z_threshold,
            },
            "summary": summary,
            "results": all_results,
            "timestamp": datetime.now().isoformat(),
        }
        
        filepath = os.path.join(
            self.results_dir,
            f"robustness_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        self._print_summary(summary)
        print(f"\nResults saved: {filepath}")
        
        return output
    
    def _compute_summary(self, results: List[Dict]) -> Dict:
        """Compute aggregate statistics from results."""
        summary = {}
        
        attack_types = set(r["attack_type"] for r in results)
        
        for attack_type in attack_types:
            attack_results = [r for r in results if r["attack_type"] == attack_type]
            n = len(attack_results)
            
            if n == 0:
                continue
            
            kgw_survived = sum(1 for r in attack_results if r["kgw_survived"])
            simmark_survived = sum(1 for r in attack_results if r["simmark_survived"])
            hybrid_survived = sum(1 for r in attack_results if r["hybrid_survived"])
            
            # At least one survived
            any_survived = sum(
                1 for r in attack_results 
                if r["kgw_survived"] or r["simmark_survived"]
            )
            
            summary[attack_type] = {
                "total": n,
                "kgw_survival_rate": kgw_survived / n,
                "simmark_survival_rate": simmark_survived / n,
                "hybrid_survival_rate": hybrid_survived / n,
                "any_survival_rate": any_survived / n,
                "kgw_avg_z_after": np.mean([r["kgw_after"]["z_score"] for r in attack_results]),
                "simmark_avg_z_after": np.mean([r["simmark_after"]["z_score"] for r in attack_results]),
            }
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print a nice summary table."""
        print("\nSURVIVAL RATES BY ATTACK TYPE:")
        print("-" * 70)
        print(f"{'Attack':<15} {'KGW':<12} {'SimMark':<12} {'Hybrid':<12} {'Any':<12}")
        print("-" * 70)
        
        for attack_type in ["none", "light", "moderate", "aggressive"]:
            if attack_type not in summary:
                continue
            s = summary[attack_type]
            print(f"{attack_type:<15} {s['kgw_survival_rate']*100:>6.1f}%     {s['simmark_survival_rate']*100:>6.1f}%     {s['hybrid_survival_rate']*100:>6.1f}%     {s['any_survival_rate']*100:>6.1f}%")
        
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Robustness Experiment")
    parser.add_argument("--prompts", type=int, default=3, help="Number of prompts to test")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Quick test with 1 prompt")
    args = parser.parse_args()
    
    experiment = RobustnessExperiment()
    
    if args.quick:
        # Quick test
        result = experiment.run_single_experiment(
            prompt="Write a short story about a robot learning to paint.",
            attack_type="light",
            verbose=True,
        )
        print("\nQuick test complete!")
    else:
        # Full experiment
        prompts = [
            "Write a short story about a robot learning to paint.",
            "Explain how machine learning works to a high school student.",
            "Describe the process of making homemade bread from scratch.",
        ][:args.prompts]
        
        experiment.run_full_experiment(
            prompts=prompts,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
