# -*- coding: utf-8 -*-
"""
Hybrid Watermark: KGW + SimMark Combined

This implements TRUE hybrid watermarking that embeds BOTH:
1. KGW (token-level): Green/red list bias in logits during generation
2. SimMark (sentence-level): Cosine similarity constraint via rejection sampling

The key insight:
- KGW modifies the generation process at token level (via LogitsProcessor)
- SimMark filters at sentence level (via rejection sampling)
- By combining them: each generated sentence has KGW tokens, AND
  consecutive sentences satisfy SimMark's similarity constraint
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
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Project imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SimMark"))

from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from SimMark.sampling_utils import (
    SentenceEndCriteria,
    discard_final_token_in_outputs,
    DummyPCA,
)
import SimMark.sampling_utils as sampling_utils
from llama_demos.model_config_manager import ModelConfigManager


@dataclass
class HybridDetectionResult:
    """Combined detection result for hybrid watermark"""
    kgw_z_score: float
    kgw_prediction: bool
    kgw_green_fraction: float
    kgw_num_tokens: int
    simmark_z_score: float
    simmark_prediction: bool
    simmark_n_watermark: float
    simmark_n_sentences: int
    both_detected: bool


@dataclass
class GenerationStats:
    """Statistics from hybrid generation"""
    total_sentences: int
    total_trials: int
    avg_trials_per_sentence: float
    maxed_out_sentences: int


class HybridWatermarkGenerator:
    """
    Generates text with BOTH KGW and SimMark watermarks simultaneously.
    """
    
    def __init__(
        self,
        model_nickname: str = "llama-3.2-3b",
        device: Optional[str] = None,
        kgw_gamma: float = 0.25,
        kgw_delta: float = 2.0,
        kgw_seeding_scheme: str = "selfhash",
        kgw_hash_key: int = 15485863,
        simmark_interval: Tuple[float, float] = (0.75, 0.83),  # Adjusted for Llama
        simmark_K: int = 250,
        simmark_max_trials: int = 50,
        use_pca: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        config_path = os.path.join(
            os.path.dirname(__file__), "llama_demos", "model_config.json"
        )
        self.config_manager = ModelConfigManager(config_path)
        self.model_nickname = model_nickname
        
        self.kgw_gamma = kgw_gamma
        self.kgw_delta = kgw_delta
        self.kgw_seeding_scheme = kgw_seeding_scheme
        self.kgw_hash_key = kgw_hash_key
        
        self.simmark_interval = simmark_interval
        self.simmark_K = simmark_K
        self.simmark_max_trials = simmark_max_trials
        sampling_utils.MAX_TRIALS = simmark_max_trials
        self.use_pca = use_pca
        
        self._model = None
        self._tokenizer = None
        self._embedder = None
        self._pca_model = None
        self._kgw_processor = None
        
        self.results_dir = os.path.join(os.path.dirname(__file__), "hybrid_kgw_simmark_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("=" * 70)
        print("Hybrid Watermark Generator: KGW + SimMark")
        print("=" * 70)
        print(f"Model: {model_nickname}")
        print(f"KGW: gamma={kgw_gamma}, delta={kgw_delta}")
        print(f"SimMark: interval={simmark_interval}, K={simmark_K}, max_trials={simmark_max_trials}")
        print("=" * 70)
    
    def _load_model(self):
        if self._model is not None:
            return self._tokenizer, self._model
        
        info = self.config_manager.get_model_info_by_nickname(self.model_nickname)
        if not info:
            raise ValueError(f"Model not found: {self.model_nickname}")
        
        print(f"\nLoading model: {self.model_nickname}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            info["model_identifier"], trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        
        self._model = AutoModelForCausalLM.from_pretrained(
            info["model_identifier"],
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self._model = self._model.to(self.device)
        self._model.eval()
        
        print("Model loaded")
        return self._tokenizer, self._model
    
    def _get_kgw_processor(self, tokenizer) -> WatermarkLogitsProcessor:
        if self._kgw_processor is None:
            self._kgw_processor = WatermarkLogitsProcessor(
                vocab=list(tokenizer.get_vocab().values()),
                gamma=self.kgw_gamma,
                delta=self.kgw_delta,
                seeding_scheme=self.kgw_seeding_scheme,
            )
        return self._kgw_processor
    
    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            print("Loading embedder: hkunlp/instructor-large")
            self._embedder = SentenceTransformer("hkunlp/instructor-large", device=self.device)
            print("Embedder loaded")
        return self._embedder
    
    def _get_pca_model(self):
        if self._pca_model is None:
            if self.use_pca:
                import pickle
                pca_path = os.path.join(os.path.dirname(__file__), "SimMark", "pca_model_16.pkl")
                with open(pca_path, 'rb') as f:
                    self._pca_model = pickle.load(f)
            else:
                self._pca_model = DummyPCA()
        return self._pca_model
    
    def _gen_sent_with_kgw(
        self,
        model,
        tokenizer,
        text_ids: torch.LongTensor,
        gen_config: GenerationConfig,
        stopping_criteria: StoppingCriteriaList,
        kgw_processor: WatermarkLogitsProcessor,
    ) -> Tuple[str, torch.LongTensor]:
        """Generate ONE sentence with KGW watermark."""
        outputs = model.generate(
            text_ids,
            gen_config,
            stopping_criteria=stopping_criteria,
            logits_processor=LogitsProcessorList([kgw_processor]),
            return_dict_in_generate=True,
        )
        outputs = discard_final_token_in_outputs(outputs)
        new_text_ids = outputs.sequences
        new_text = tokenizer.decode(
            new_text_ids[0, text_ids.size(1):], skip_special_tokens=True
        )
        return new_text, new_text_ids
    
    def _check_simmark_constraint(
        self,
        prev_sent: str,
        new_sent: str,
        embedder: SentenceTransformer,
        pca_model,
    ) -> Tuple[bool, float]:
        """Check if new sentence satisfies SimMark's similarity constraint."""
        if prev_sent is None:
            return True, 0.0
        
        instruction = "Represent the sentence for cosine similarity:"
        
        prev_embed = embedder.encode(
            prev_sent, prompt=instruction, 
            convert_to_tensor=True, normalize_embeddings=True
        ).reshape(1, -1)
        
        new_embed = embedder.encode(
            new_sent, prompt=instruction,
            convert_to_tensor=True, normalize_embeddings=True
        ).reshape(1, -1)
        
        sim = cosine_similarity(
            pca_model.transform(prev_embed.cpu().detach().numpy()),
            pca_model.transform(new_embed.cpu().detach().numpy())
        )[0, 0]
        
        a, b = self.simmark_interval
        is_valid = a <= sim <= b
        
        return is_valid, float(sim)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        min_new_tokens: int = 100,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> Tuple[str, GenerationStats, Dict]:
        """Generate text with BOTH KGW and SimMark watermarks."""
        tokenizer, model = self._load_model()
        embedder = self._get_embedder()
        pca_model = self._get_pca_model()
        kgw_processor = self._get_kgw_processor(tokenizer)
        
        bad_words_ids = tokenizer(
            "\n", return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device=self.device).tolist()
        
        gen_config = GenerationConfig(
            max_new_tokens=50,
            min_new_tokens=5,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            repetition_penalty=1.05,
            bad_words_ids=bad_words_ids,
        )
        
        print(f"\nGenerating with Hybrid (KGW + SimMark)...")
        print(f"Prompt: {prompt[:50]}...")
        
        sent_end_criteria = SentenceEndCriteria(tokenizer)
        text = prompt
        text_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        prompt_length = len(text_ids[0])
        sent_end_criteria.update(text)
        
        all_sentences = [prompt]
        total_trials = 0
        maxed_out = 0
        prev_sent = None
        
        while True:
            current_trials = 0
            accepted = False
            
            while not accepted and current_trials < self.simmark_max_trials:
                stopping_criteria = StoppingCriteriaList([sent_end_criteria])
                
                new_text, new_text_ids = self._gen_sent_with_kgw(
                    model=model,
                    tokenizer=tokenizer,
                    text_ids=text_ids,
                    gen_config=gen_config,
                    stopping_criteria=stopping_criteria,
                    kgw_processor=kgw_processor,
                )
                
                current_trials += 1
                total_trials += 1
                
                if new_text == '':
                    print('Empty generation, stopping')
                    break
                
                is_valid, sim_score = self._check_simmark_constraint(
                    prev_sent, new_text, embedder, pca_model
                )
                
                if verbose:
                    status = "OK" if is_valid else "X"
                    print(f"   Trial {current_trials}: sim={sim_score:.4f} [{status}]")
                
                if is_valid or current_trials >= self.simmark_max_trials:
                    if current_trials >= self.simmark_max_trials and not is_valid:
                        maxed_out += 1
                        if verbose:
                            print(f"   Max trials reached, accepting anyway")
                    
                    text += new_text
                    text_ids = new_text_ids
                    all_sentences.append(new_text)
                    prev_sent = new_text
                    sent_end_criteria.update(text)
                    accepted = True
            
            if not accepted:
                break
            
            if (len(text_ids[0]) - prompt_length) >= max_new_tokens - 1:
                break
        
        generated_text = text[len(prompt):].strip()
        
        n_sentences = len(all_sentences) - 1
        stats = GenerationStats(
            total_sentences=n_sentences,
            total_trials=total_trials,
            avg_trials_per_sentence=total_trials / max(n_sentences, 1),
            maxed_out_sentences=maxed_out,
        )
        
        metadata = {
            "type": "hybrid_kgw_simmark",
            "kgw": {
                "gamma": self.kgw_gamma,
                "delta": self.kgw_delta,
                "seeding_scheme": self.kgw_seeding_scheme,
            },
            "simmark": {
                "interval": self.simmark_interval,
                "K": self.simmark_K,
                "max_trials": self.simmark_max_trials,
            },
            "model": self.model_nickname,
        }
        
        print(f"Generated {n_sentences} sentences")
        print(f"Total trials: {total_trials}, Avg: {stats.avg_trials_per_sentence:.2f}")
        print(f"Maxed out: {maxed_out}")
        
        return generated_text, stats, metadata


class HybridWatermarkDetector:
    """Detects BOTH KGW and SimMark watermarks."""
    
    def __init__(
        self,
        tokenizer,
        kgw_gamma: float = 0.25,
        kgw_seeding_scheme: str = "selfhash",
        kgw_z_threshold: float = 3.0,
        simmark_interval: Tuple[float, float] = (0.75, 0.83),
        simmark_K: int = 250,
        simmark_gamma: float = 0.25,  # Adjusted for new interval
        simmark_z_threshold: float = 2.0,
        use_pca: bool = False,
        device: str = "cuda",
    ):
        self.device = device
        
        self.kgw_detector = WatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=kgw_gamma,
            seeding_scheme=kgw_seeding_scheme,
            device=device,
            tokenizer=tokenizer,
            z_threshold=kgw_z_threshold,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )
        self.kgw_z_threshold = kgw_z_threshold
        
        self.simmark_interval = simmark_interval
        self.simmark_K = simmark_K
        self.simmark_gamma = simmark_gamma
        self.simmark_z_threshold = simmark_z_threshold
        self.use_pca = use_pca
        
        self._embedder = None
        self._pca_model = None
    
    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer("hkunlp/instructor-large", device=self.device)
        return self._embedder
    
    def _get_pca_model(self):
        if self._pca_model is None:
            if self.use_pca:
                import pickle
                pca_path = os.path.join(os.path.dirname(__file__), "SimMark", "pca_model_16.pkl")
                with open(pca_path, 'rb') as f:
                    self._pca_model = pickle.load(f)
            else:
                self._pca_model = DummyPCA()
        return self._pca_model
    
    def detect(self, text: str) -> HybridDetectionResult:
        """Detect both KGW and SimMark watermarks."""
        # KGW Detection
        kgw_result = self.kgw_detector.detect(text, return_scores=True, return_prediction=True)
        kgw_z = float(kgw_result.get("z_score", 0.0))
        kgw_pred = kgw_z > self.kgw_z_threshold
        kgw_green_frac = float(kgw_result.get("green_fraction", 0.0))
        kgw_num_tokens = int(kgw_result.get("num_tokens_scored", 0))
        
        # SimMark Detection
        embedder = self._get_embedder()
        pca_model = self._get_pca_model()
        instruction = "Represent the sentence for cosine similarity:"
        
        sentences = sent_tokenize(text)
        n_sentences = len(sentences) - 1
        
        if n_sentences < 1:
            simmark_z = 0.0
            simmark_n_watermark = 0.0
        else:
            embeddings = embedder.encode(
                sentences,
                prompt=instruction,
                batch_size=32,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            
            n_watermark = 0.0
            a, b = self.simmark_interval
            
            for j in range(1, len(sentences)):
                e1 = embeddings[j-1].reshape(1, -1).cpu().detach().numpy()
                e2 = embeddings[j].reshape(1, -1).cpu().detach().numpy()
                
                sim = cosine_similarity(
                    pca_model.transform(e1),
                    pca_model.transform(e2)
                )[0, 0]
                
                if a <= sim <= b:
                    dist_to_interval = 0.0
                else:
                    dist_to_interval = min(abs(sim - a), abs(sim - b))
                
                soft_count = np.exp(-self.simmark_K * dist_to_interval)
                n_watermark += soft_count
            
            simmark_n_watermark = n_watermark
            
            gamma = self.simmark_gamma
            num = n_watermark - gamma * n_sentences
            denom = np.sqrt(n_sentences * gamma * (1 - gamma) + 1e-12)
            simmark_z = float(num / denom)
        
        simmark_pred = simmark_z > self.simmark_z_threshold
        
        return HybridDetectionResult(
            kgw_z_score=kgw_z,
            kgw_prediction=kgw_pred,
            kgw_green_fraction=kgw_green_frac,
            kgw_num_tokens=kgw_num_tokens,
            simmark_z_score=simmark_z,
            simmark_prediction=simmark_pred,
            simmark_n_watermark=simmark_n_watermark,
            simmark_n_sentences=n_sentences,
            both_detected=kgw_pred and simmark_pred,
        )


def run_experiment(
    prompt: str = "Write a short story about a robot learning to paint.",
    model_nickname: str = "llama-3.2-3b",
    max_new_tokens: int = 200,
    verbose: bool = True,
):
    """Run a single hybrid watermark experiment."""
    print("\n" + "=" * 70)
    print("HYBRID WATERMARK EXPERIMENT: KGW + SimMark")
    print("=" * 70)
    
    generator = HybridWatermarkGenerator(
        model_nickname=model_nickname,
        kgw_gamma=0.25,
        kgw_delta=2.0,
        simmark_interval=(0.75, 0.83),
        simmark_K=250,
        simmark_max_trials=50,
    )
    
    text, stats, metadata = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        verbose=verbose,
    )
    
    print(f"\nGenerated Text ({len(text)} chars):")
    print("-" * 50)
    print(text[:500] + ("..." if len(text) > 500 else ""))
    print("-" * 50)
    
    tokenizer, _ = generator._load_model()
    detector = HybridWatermarkDetector(
        tokenizer=tokenizer,
        kgw_gamma=0.25,
        simmark_interval=(0.75, 0.83),
        simmark_K=250,
        simmark_gamma=0.25,  # ~25% of natural distribution in this interval
    )
    
    result = detector.detect(text)
    
    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)
    print(f"\nKGW Detection:")
    print(f"   z-score: {result.kgw_z_score:.4f}")
    print(f"   Green fraction: {result.kgw_green_fraction:.4f}")
    print(f"   Tokens scored: {result.kgw_num_tokens}")
    print(f"   Detected: {'YES' if result.kgw_prediction else 'NO'}")
    
    print(f"\nSimMark Detection:")
    print(f"   z-score: {result.simmark_z_score:.4f}")
    print(f"   Soft watermark count: {result.simmark_n_watermark:.2f}")
    print(f"   Sentence pairs: {result.simmark_n_sentences}")
    print(f"   Detected: {'YES' if result.simmark_prediction else 'NO'}")
    
    print(f"\nHYBRID RESULT: {'BOTH DETECTED' if result.both_detected else 'NOT FULLY DETECTED'}")
    
    results = {
        "prompt": prompt,
        "generated_text": text,
        "stats": asdict(stats),
        "detection": asdict(result),
        "metadata": metadata,
        "timestamp": datetime.now().isoformat(),
    }
    
    filepath = os.path.join(
        generator.results_dir,
        f"hybrid_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {filepath}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Hybrid KGW + SimMark Watermark")
    parser.add_argument("--prompt", type=str, 
                       default="Write a short story about a robot learning to paint.")
    parser.add_argument("--model", type=str, default="llama-3.2-3b")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    run_experiment(
        prompt=args.prompt,
        model_nickname=args.model,
        max_new_tokens=args.max_tokens,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
