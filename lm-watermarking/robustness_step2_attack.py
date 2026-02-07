# -*- coding: utf-8 -*-
"""
Step 2: Apply paraphrase attacks to watermarked texts.
Loads texts from Step 1, applies attacks, saves attacked texts.
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.tokenize import sent_tokenize


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
    
    def attack(self, text: str, attack_type: str) -> dict:
        """Apply a paraphrase attack of the specified type."""
        
        original_sentences = len(sent_tokenize(text))
        
        if attack_type == "none":
            return {
                "original_text": text,
                "attacked_text": text,
                "attack_type": attack_type,
                "original_sentences": original_sentences,
                "attacked_sentences": original_sentences,
            }
        
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
                max_new_tokens=len(text.split()) + 100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        
        full_output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the paraphrased part
        if "Paraphrased version:" in full_output:
            attacked_text = full_output.split("Paraphrased version:")[-1].strip()
        elif "Rewritten version:" in full_output:
            attacked_text = full_output.split("Rewritten version:")[-1].strip()
        elif "Condensed version:" in full_output:
            attacked_text = full_output.split("Condensed version:")[-1].strip()
        else:
            attacked_text = full_output[len(prompt):].strip()
        
        # Clean up: remove any trailing prompts or artifacts
        for marker in ["Text:", "Paraphrase", "Rewrite", "Summarize"]:
            if marker in attacked_text:
                attacked_text = attacked_text.split(marker)[0].strip()
        
        attacked_sentences = len(sent_tokenize(attacked_text))
        
        return {
            "original_text": text,
            "attacked_text": attacked_text,
            "attack_type": attack_type,
            "original_sentences": original_sentences,
            "attacked_sentences": attacked_sentences,
        }


def apply_attacks(
    input_file: str,
    output_file: str,
    attack_types: List[str] = None,
):
    """Apply paraphrase attacks to all watermarked texts."""
    
    if attack_types is None:
        attack_types = ["none", "light", "moderate", "aggressive"]
    
    print("=" * 70)
    print("STEP 2: APPLY PARAPHRASE ATTACKS")
    print("=" * 70)
    
    # Load generated texts
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    generated_texts = data["generated_texts"]
    print(f"Loaded {len(generated_texts)} watermarked texts")
    
    attacker = ParaphraseAttacker()
    
    all_results = []
    
    for i, item in enumerate(generated_texts):
        prompt = item["prompt"]
        watermarked_text = item["watermarked_text"]
        baseline = item["baseline_detection"]
        
        print(f"\n--- Text {i+1}/{len(generated_texts)} ---")
        print(f"Prompt: {prompt[:50]}...")
        
        text_results = {
            "prompt": prompt,
            "watermarked_text": watermarked_text,
            "baseline_detection": baseline,
            "attacks": [],
        }
        
        for attack_type in attack_types:
            print(f"  Applying {attack_type} attack...")
            
            attack_result = attacker.attack(watermarked_text, attack_type)
            
            text_results["attacks"].append({
                "attack_type": attack_type,
                "attacked_text": attack_result["attacked_text"],
                "original_sentences": attack_result["original_sentences"],
                "attacked_sentences": attack_result["attacked_sentences"],
            })
            
            print(f"    {attack_result['original_sentences']} -> {attack_result['attacked_sentences']} sentences")
        
        all_results.append(text_results)
    
    # Save results
    output = {
        "config": data["config"],
        "attack_types": attack_types,
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 70}")
    print(f"STEP 2 COMPLETE")
    print(f"{'=' * 70}")
    print(f"Applied {len(attack_types)} attack types to {len(all_results)} texts")
    print(f"Saved to: {output_file}")
    
    # Clear GPU memory
    del attacker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Step 2: Apply paraphrase attacks")
    parser.add_argument("--input", type=str, required=True, help="Input file from Step 1")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--attacks", type=str, nargs="+", 
                       default=["none", "light", "moderate", "aggressive"],
                       help="Attack types to apply")
    args = parser.parse_args()
    
    if args.output is None:
        output_dir = os.path.dirname(args.input)
        args.output = os.path.join(
            output_dir, 
            f"step2_attacked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    
    apply_attacks(args.input, args.output, args.attacks)


if __name__ == "__main__":
    main()
