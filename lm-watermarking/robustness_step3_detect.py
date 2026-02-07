# -*- coding: utf-8 -*-
"""
Step 3: Detect watermarks in attacked texts.
Loads attacked texts from Step 2, runs detection, computes survival rates.
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
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

# Project imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from hybrid_kgw_simmark import HybridWatermarkDetector
from llama_demos.model_config_manager import ModelConfigManager


def detect_watermarks(
    input_file: str,
    output_file: str,
    model_nickname: str = "llama-3.2-3b",
):
    """Detect watermarks in all attacked texts."""
    
    print("=" * 70)
    print("STEP 3: DETECT WATERMARKS")
    print("=" * 70)
    
    # Load attacked texts
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    config = data["config"]
    attack_types = data["attack_types"]
    results = data["results"]
    
    print(f"Loaded {len(results)} texts with {len(attack_types)} attack types each")
    
    # Load tokenizer for detection
    config_path = os.path.join(
        os.path.dirname(__file__), "llama_demos", "model_config.json"
    )
    config_manager = ModelConfigManager(config_path)
    info = config_manager.get_model_info_by_nickname(model_nickname)
    
    print(f"Loading tokenizer: {model_nickname}")
    tokenizer = AutoTokenizer.from_pretrained(info["model_identifier"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create detector
    detector = HybridWatermarkDetector(
        tokenizer=tokenizer,
        kgw_gamma=config["kgw_gamma"],
        simmark_interval=tuple(config["simmark_interval"]),
        simmark_K=250,
        kgw_z_threshold=3.0,
        simmark_z_threshold=2.0,
    )
    
    all_detection_results = []
    
    for i, item in enumerate(results):
        print(f"\n--- Text {i+1}/{len(results)} ---")
        print(f"Prompt: {item['prompt'][:50]}...")
        
        text_result = {
            "prompt": item["prompt"],
            "baseline_detection": item["baseline_detection"],
            "attack_results": [],
        }
        
        for attack in item["attacks"]:
            attack_type = attack["attack_type"]
            attacked_text = attack["attacked_text"]
            
            # Detect watermarks
            detection = detector.detect(attacked_text)
            
            attack_result = {
                "attack_type": attack_type,
                "original_sentences": attack["original_sentences"],
                "attacked_sentences": attack["attacked_sentences"],
                "kgw_z_score": detection.kgw_z_score,
                "kgw_detected": detection.kgw_prediction,
                "simmark_z_score": detection.simmark_z_score,
                "simmark_detected": detection.simmark_prediction,
                "both_detected": detection.both_detected,
            }
            text_result["attack_results"].append(attack_result)
            
            # Calculate survival
            baseline = item["baseline_detection"]
            kgw_survived = detection.kgw_prediction
            simmark_survived = detection.simmark_prediction
            
            print(f"  {attack_type}: KGW z={detection.kgw_z_score:.2f} ({'✓' if kgw_survived else '✗'}), "
                  f"SimMark z={detection.simmark_z_score:.2f} ({'✓' if simmark_survived else '✗'})")
        
        all_detection_results.append(text_result)
    
    # Compute summary statistics
    summary = compute_summary(all_detection_results, attack_types)
    
    # Save results
    output = {
        "config": config,
        "attack_types": attack_types,
        "detection_results": all_detection_results,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 70}")
    print("STEP 3 COMPLETE - FINAL RESULTS")
    print("=" * 70)
    print_summary(summary)
    print(f"\nSaved to: {output_file}")
    
    return output


def compute_summary(results: List[Dict], attack_types: List[str]) -> Dict:
    """Compute aggregate statistics."""
    summary = {}
    
    for attack_type in attack_types:
        kgw_survived = 0
        simmark_survived = 0
        both_survived = 0
        any_survived = 0
        kgw_z_scores = []
        simmark_z_scores = []
        total = 0
        
        for text_result in results:
            for attack_result in text_result["attack_results"]:
                if attack_result["attack_type"] == attack_type:
                    total += 1
                    if attack_result["kgw_detected"]:
                        kgw_survived += 1
                    if attack_result["simmark_detected"]:
                        simmark_survived += 1
                    if attack_result["both_detected"]:
                        both_survived += 1
                    if attack_result["kgw_detected"] or attack_result["simmark_detected"]:
                        any_survived += 1
                    kgw_z_scores.append(attack_result["kgw_z_score"])
                    simmark_z_scores.append(attack_result["simmark_z_score"])
        
        if total > 0:
            summary[attack_type] = {
                "total": total,
                "kgw_survival_rate": kgw_survived / total,
                "simmark_survival_rate": simmark_survived / total,
                "both_survival_rate": both_survived / total,
                "any_survival_rate": any_survived / total,
                "kgw_avg_z": np.mean(kgw_z_scores),
                "simmark_avg_z": np.mean(simmark_z_scores),
            }
    
    return summary


def print_summary(summary: Dict):
    """Print a nice summary table."""
    print("\nSURVIVAL RATES BY ATTACK TYPE:")
    print("-" * 80)
    print(f"{'Attack':<12} {'KGW':<15} {'SimMark':<15} {'Both':<15} {'Any':<15}")
    print("-" * 80)
    
    for attack_type in ["none", "light", "moderate", "aggressive"]:
        if attack_type not in summary:
            continue
        s = summary[attack_type]
        print(f"{attack_type:<12} "
              f"{s['kgw_survival_rate']*100:>6.1f}% (z={s['kgw_avg_z']:.1f})  "
              f"{s['simmark_survival_rate']*100:>6.1f}% (z={s['simmark_avg_z']:.1f})  "
              f"{s['both_survival_rate']*100:>6.1f}%         "
              f"{s['any_survival_rate']*100:>6.1f}%")
    
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Step 3: Detect watermarks")
    parser.add_argument("--input", type=str, required=True, help="Input file from Step 2")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()
    
    if args.output is None:
        output_dir = os.path.dirname(args.input)
        args.output = os.path.join(
            output_dir, 
            f"step3_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    
    detect_watermarks(args.input, args.output)


if __name__ == "__main__":
    main()
