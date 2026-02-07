# -*- coding: utf-8 -*-
"""
Step 1: Generate watermarked texts only (no attack).
Saves watermarked texts to a JSON file for later attack testing.
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
from typing import List

import torch

# Project imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from hybrid_kgw_simmark import HybridWatermarkGenerator, HybridWatermarkDetector


# Different prompt categories to test edge cases
PROMPT_CATEGORIES = {
    "standard": [
        "Write a short story about a robot learning to paint.",
        "Explain how machine learning works to a high school student.",
        "Describe the process of making homemade bread from scratch.",
        "Write about the importance of exercise for mental health.",
        "Explain the water cycle in nature.",
    ],
    "short": [
        "What is 2+2? Explain briefly in one or two sentences.",
        "Define artificial intelligence in one sentence.",
        "What color is the sky? Answer briefly.",
        "Name the capital of France and explain why it's important.",
        "What is photosynthesis? Give a very short explanation.",
    ],
    "code": [
        "Write a Python function to calculate factorial.",
        "Write a simple JavaScript function to reverse a string.",
        "Create a Python function that checks if a number is prime.",
        "Write a SQL query to select all users older than 18.",
        "Write a bash script to count files in a directory.",
    ],
    "list": [
        "List 5 benefits of regular exercise.",
        "Name 5 famous scientists and their contributions.",
        "List the steps to make a cup of tea.",
        "What are 5 programming languages and their uses?",
        "List 5 tips for better sleep.",
    ],
    "poetry": [
        "Write a haiku about autumn leaves.",
        "Write a short limerick about a cat.",
        "Create a four-line poem about the ocean.",
        "Write a brief rhyming verse about friendship.",
        "Compose a short poem about morning coffee.",
    ],
    "dialogue": [
        "Write a short dialogue between a teacher and student about homework.",
        "Create a brief conversation between two friends planning a trip.",
        "Write a dialogue between a customer and waiter at a restaurant.",
        "Create a short exchange between a doctor and patient.",
        "Write a dialogue between siblings arguing about chores.",
    ],
}


def generate_watermarked_texts(
    prompts: List[str],
    output_file: str,
    model_nickname: str = "llama-3.2-3b",
    max_new_tokens: int = 200,
    category: str = "standard",
):
    """Generate watermarked texts and save to file."""
    
    print("=" * 70)
    print("STEP 1: GENERATE WATERMARKED TEXTS")
    print(f"Category: {category}")
    print("=" * 70)
    
    generator = HybridWatermarkGenerator(
        model_nickname=model_nickname,
        kgw_gamma=0.25,
        kgw_delta=2.0,
        simmark_interval=(0.75, 0.83),
        simmark_K=250,
        simmark_max_trials=50,
    )
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}/{len(prompts)} ---")
        print(f"Prompt: {prompt[:60]}...")
        
        # Adjust max_new_tokens for short prompts
        tokens = max_new_tokens
        if category == "short":
            tokens = 50
        elif category == "poetry":
            tokens = 80
        
        text, stats, metadata = generator.generate(
            prompt=prompt,
            max_new_tokens=tokens,
            verbose=False,
        )
        
        # Get tokenizer for detection
        tokenizer, _ = generator._load_model()
        
        # Create detector and get baseline scores
        detector = HybridWatermarkDetector(
            tokenizer=tokenizer,
            kgw_gamma=0.25,
            simmark_interval=(0.75, 0.83),
            simmark_K=250,
        )
        
        detection = detector.detect(text)
        
        result = {
            "prompt": prompt,
            "category": category,
            "watermarked_text": text,
            "stats": {
                "total_sentences": stats.total_sentences,
                "total_trials": stats.total_trials,
                "avg_trials_per_sentence": stats.avg_trials_per_sentence,
                "maxed_out_sentences": stats.maxed_out_sentences,
            },
            "baseline_detection": {
                "kgw_z_score": detection.kgw_z_score,
                "kgw_detected": detection.kgw_prediction,
                "simmark_z_score": detection.simmark_z_score,
                "simmark_detected": detection.simmark_prediction,
                "both_detected": detection.both_detected,
            },
        }
        results.append(result)
        
        print(f"Generated {stats.total_sentences} sentences")
        print(f"Baseline: KGW z={detection.kgw_z_score:.2f}, SimMark z={detection.simmark_z_score:.2f}")
    
    # Save results
    output = {
        "config": {
            "model": model_nickname,
            "kgw_gamma": 0.25,
            "kgw_delta": 2.0,
            "simmark_interval": [0.75, 0.83],
        },
        "category": category,
        "generated_texts": results,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 70}")
    print(f"STEP 1 COMPLETE")
    print(f"{'=' * 70}")
    print(f"Generated {len(results)} watermarked texts")
    print(f"Saved to: {output_file}")
    
    # Clear GPU memory
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Step 1: Generate watermarked texts")
    parser.add_argument("--prompts", type=int, default=5, help="Number of prompts")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--category", type=str, default="standard",
                       choices=list(PROMPT_CATEGORIES.keys()),
                       help="Prompt category to test")
    args = parser.parse_args()
    
    prompts = PROMPT_CATEGORIES.get(args.category, PROMPT_CATEGORIES["standard"])[:args.prompts]
    
    if args.output is None:
        output_dir = os.path.join(os.path.dirname(__file__), "hybrid_robustness_results")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(
            output_dir, 
            f"step1_{args.category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    
    generate_watermarked_texts(prompts, args.output, category=args.category)


if __name__ == "__main__":
    main()
