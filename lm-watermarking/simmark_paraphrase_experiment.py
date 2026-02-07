"""
SimMark Paraphrase Robustness Experiment

This experiment validates SimMark's claim of paraphrase robustness by:
1. Using llama-3.2-3b to generate text with SimMark watermark (sentence-level similarity-based)
2. Using qwen-3-4b to paraphrase the watermarked text
3. Detecting the watermark before and after paraphrase
4. Comparing against KGW (token-level red/green list) baseline

Based on the SimMark paper:
"SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models"
https://arxiv.org/abs/2502.02787
"""

from __future__ import annotations

import sys
# Fix Windows console encoding for emoji
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SimMark"))

# Import SimMark modules
from SimMark.sampling_utils import (
    cosine_reject_completion,
    euclidean_reject_completion,
    DummyPCA,
)
import SimMark.sampling_utils as sampling_utils

# Import KGW baseline
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from transformers import LogitsProcessorList

# Import config manager
from llama_demos.model_config_manager import ModelConfigManager


@dataclass
class SimMarkDetectionResult:
    """SimMark detection result"""
    z_score: float
    n_watermark: float  # soft count of valid sentences
    n_test_sent: int    # number of sentences tested
    gamma: float        # expected fraction in interval for human text
    mode: str           # 'cosine' or 'euclidean'
    interval: Tuple[float, float]
    K: int              # soft count decay factor
    prediction: bool    # True if z_score > threshold


@dataclass
class KGWDetectionResult:
    """KGW (token-level) detection result"""
    z_score: float
    p_value: float
    prediction: bool
    num_green_tokens: int
    num_tokens_scored: int
    green_fraction: float


@dataclass
class ExperimentResult:
    """Single experiment result"""
    prompt: str
    original_text: str
    paraphrased_text: str
    
    simmark_original: SimMarkDetectionResult
    simmark_paraphrase: SimMarkDetectionResult
    kgw_original: KGWDetectionResult
    kgw_paraphrase: KGWDetectionResult
    
    simmark_survived: bool
    kgw_survived: bool
    
    semantic_similarity: float
    timestamp: str


class SimMarkParaphraseExperiment:
    """
    Experiment comparing SimMark vs KGW under paraphrase attack.
    
    Uses the same setup as multi_llm_chain_experiment:
    - Generator: llama-3.2-3b (local)
    - Paraphraser: qwen-3-4b (local or API)
    """
    
    # Aggressive paraphrase instruction (original - may compress/merge sentences)
    PARAPHRASE_INSTRUCTION_AGGRESSIVE = """/no_think
Paraphrase the following text to preserve its meaning.
CRITICAL RULES:
1. Output ONLY the rewritten text, nothing else.
2. Do NOT include any notes, explanations, labels, or introductory phrases.
3. Do NOT include "Output:", "Here is", or any similar prefixes.
4. Start your response directly with the paraphrased content."""

    # Moderate paraphrase instruction (preserves sentence structure - similar to DIPPER L40)
    PARAPHRASE_INSTRUCTION_MODERATE = """/no_think
Paraphrase the following text using different words and phrasing while preserving BOTH meaning AND structure.
CRITICAL RULES:
1. KEEP THE SAME NUMBER OF SENTENCES - do NOT merge or split sentences.
2. Maintain the same paragraph structure and sentence order.
3. Change vocabulary and sentence phrasing, but preserve the logical flow.
4. Output ONLY the rewritten text, nothing else.
5. Do NOT include any notes, explanations, labels, or prefixes.
6. Start your response directly with the paraphrased content."""

    # Default to moderate (paper-style) paraphrase
    PARAPHRASE_INSTRUCTION = PARAPHRASE_INSTRUCTION_MODERATE
    
    def __init__(
        self,
        generator_model: str = "llama-3.2-3b",
        paraphraser_model: str = "qwen-3-4b",
        device: Optional[str] = None,
        use_api: bool = False,
        simmark_mode: str = "cosine",
        simmark_interval: Tuple[float, float] = (0.75, 0.83),
        simmark_K: int = 250,
        use_pca: bool = False,
        max_trials: int = 50,
        paraphrase_mode: str = "moderate",  # "moderate" or "aggressive"
    ):
        """
        Initialize experiment.
        
        Args:
            generator_model: Model for watermarked generation
            paraphraser_model: Model for paraphrasing
            device: cuda or cpu
            use_api: Use API mode for paraphrasing
            simmark_mode: 'cosine' or 'euclidean'
            simmark_interval: (a, b) interval for valid similarity
            simmark_K: Soft count decay factor
            use_pca: Use PCA for dimensionality reduction (euclidean mode)
            max_trials: Max rejection sampling trials per sentence
            paraphrase_mode: 'moderate' (preserves sentence count) or 'aggressive' (may merge)
        """
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Model config
        config_path = os.path.join(
            os.path.dirname(__file__), "llama_demos", "model_config.json"
        )
        self.config_manager = ModelConfigManager(config_path)
        
        self.generator_model = generator_model
        self.paraphraser_model = paraphraser_model
        self.use_api = use_api
        
        # Paraphrase mode
        self.paraphrase_mode = paraphrase_mode
        if paraphrase_mode == "moderate":
            self._paraphrase_instruction = self.PARAPHRASE_INSTRUCTION_MODERATE
        else:
            self._paraphrase_instruction = self.PARAPHRASE_INSTRUCTION_AGGRESSIVE
        
        # SimMark config
        self.simmark_mode = simmark_mode
        self.simmark_interval = simmark_interval
        self.simmark_K = simmark_K
        self.use_pca = use_pca
        
        # Set max trials in sampling_utils
        sampling_utils.MAX_TRIALS = max_trials
        
        # Model cache
        self._model_cache: Dict = {}
        self._embedder: Optional[SentenceTransformer] = None
        self._pca_model = None
        self._gamma: Optional[float] = None  # Expected fraction for human text
        
        # API client for paraphrasing
        self._api_client = None
        if use_api:
            from hybrid_watermark.model_client import ModelClient
            self._api_client = ModelClient()
            print("🌐 Using API mode for paraphrasing")
        
        # Results directory
        self.results_dir = os.path.join(os.path.dirname(__file__), "simmark_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"📍 SimMark Mode: {simmark_mode}")
        print(f"📍 Interval: [{simmark_interval[0]}, {simmark_interval[1]}]")
        print(f"📍 K: {simmark_K}")
        print(f"📍 Max Trials: {max_trials}")
    
    def _load_model(self, nickname: str):
        """Load model with caching."""
        if nickname in self._model_cache:
            return self._model_cache[nickname]
        
        info = self.config_manager.get_model_info_by_nickname(nickname)
        if not info:
            raise ValueError(f"Model not found: {nickname}")
        
        print(f"📦 Loading model: {nickname}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            info["model_identifier"], trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = AutoModelForCausalLM.from_pretrained(
            info["model_identifier"],
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            model = model.to(self.device)
        model.eval()
        
        self._model_cache[nickname] = {
            "tokenizer": tokenizer,
            "model": model,
            "info": info,
        }
        print(f"�?Model loaded: {nickname}")
        return self._model_cache[nickname]
    
    def _load_tokenizer_only(self, nickname: str):
        """Load only tokenizer."""
        if nickname in self._model_cache:
            return self._model_cache[nickname]["tokenizer"]
        
        cache_key = f"tokenizer_only:{nickname}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        info = self.config_manager.get_model_info_by_nickname(nickname)
        if not info:
            raise ValueError(f"Model not found: {nickname}")
        
        print(f"📦 Loading tokenizer: {nickname}")
        tokenizer = AutoTokenizer.from_pretrained(
            info["model_identifier"], trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        self._model_cache[cache_key] = tokenizer
        print(f"�?Tokenizer loaded: {nickname}")
        return tokenizer
    
    def _get_embedder(self) -> SentenceTransformer:
        """Get or create sentence embedder."""
        if self._embedder is None:
            print("📦 Loading sentence embedder: hkunlp/instructor-large")
            self._embedder = SentenceTransformer("hkunlp/instructor-large", device=self.device)
            print("�?Embedder loaded")
        return self._embedder
    
    def _get_pca_model(self):
        """Get PCA model if needed."""
        if self._pca_model is None:
            if self.use_pca:
                import pickle
                pca_path = os.path.join(os.path.dirname(__file__), "SimMark", "pca_model_16.pkl")
                with open(pca_path, 'rb') as f:
                    self._pca_model = pickle.load(f)
                print(f"�?PCA model loaded from {pca_path}")
            else:
                self._pca_model = DummyPCA()
        return self._pca_model
    
    def _get_instruction(self) -> str:
        """Get embedding instruction based on mode."""
        if self.use_pca:
            return "Represent the sentence for PCA:"
        elif self.simmark_mode == "cosine":
            return "Represent the sentence for cosine similarity:"
        else:
            return "Represent the sentence for euclidean distance:"
    
    def _estimate_gamma(self, sample_texts: Optional[List[str]] = None) -> float:
        """
        Estimate gamma (expected fraction of sentence pairs in interval for human text).
        
        If no sample texts provided, use default values from SimMark paper.
        """
        if self._gamma is not None:
            return self._gamma
        
        # Default gamma values from SimMark paper experiments
        # These are calibrated on human text distributions
        if self.simmark_mode == "cosine":
            # For cosine similarity with interval [0.68, 0.76]
            # Paper shows ~8% of human text pairs fall in this interval
            self._gamma = 0.08
        else:
            # For euclidean distance (varies by dataset)
            self._gamma = 0.08
        
        print(f"📊 Using estimated gamma = {self._gamma:.4f}")
        return self._gamma
    
    # =========================================================================
    # SimMark Generation (Sentence-level watermark via rejection sampling)
    # =========================================================================
    
    def generate_with_simmark(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        min_new_tokens: int = 100,
    ) -> Tuple[str, Dict]:
        """
        Generate text with SimMark watermark using rejection sampling.
        
        Each generated sentence is accepted only if its embedding similarity
        to the previous sentence falls within the target interval [a, b].
        """
        cached = self._load_model(self.generator_model)
        tokenizer = cached["tokenizer"]
        model = cached["model"]
        
        embedder = self._get_embedder()
        pca_model = self._get_pca_model()
        
        # Block newline generation
        bad_words_ids = tokenizer(
            "\n", return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device=self.device).tolist()
        
        gen_config = GenerationConfig.from_pretrained(
            cached["info"]["model_identifier"],
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=0,
            repetition_penalty=1.05,
            bad_words_ids=bad_words_ids,
        )
        
        print(f"\n🌊 Generating with SimMark ({self.simmark_mode} mode)...")
        print(f"   Interval: [{self.simmark_interval[0]}, {self.simmark_interval[1]}]")
        
        if self.simmark_mode == "cosine":
            text = cosine_reject_completion(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                gen_config=gen_config,
                embedder=embedder,
                whole_interval=self.simmark_interval,
                pca_model=pca_model,
                verbose=False,
                device=self.device,
            )
        else:
            text = euclidean_reject_completion(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                gen_config=gen_config,
                embedder=embedder,
                whole_interval=self.simmark_interval,
                pca_model=pca_model,
                verbose=False,
                device=self.device,
            )
        
        # Remove prompt from text (SimMark returns prompt + generated)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        
        metadata = {
            "type": "simmark",
            "mode": self.simmark_mode,
            "interval": self.simmark_interval,
            "K": self.simmark_K,
            "use_pca": self.use_pca,
            "max_trials": sampling_utils.MAX_TRIALS,
            "generator": self.generator_model,
        }
        
        sentences = sent_tokenize(text)
        print(f"   Generated {len(sentences)} sentences, {len(text)} chars")
        
        return text, metadata
    
    def detect_simmark(self, text: str, z_threshold: float = 2.0) -> SimMarkDetectionResult:
        """
        Detect SimMark watermark by computing sentence similarity distribution.
        
        Uses soft counting: sentences close to but outside interval get partial credit.
        """
        embedder = self._get_embedder()
        pca_model = self._get_pca_model()
        instruction = self._get_instruction()
        gamma = self._estimate_gamma()
        
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return SimMarkDetectionResult(
                z_score=0.0,
                n_watermark=0.0,
                n_test_sent=0,
                gamma=gamma,
                mode=self.simmark_mode,
                interval=self.simmark_interval,
                K=self.simmark_K,
                prediction=False,
            )
        
        # Compute embeddings
        embeddings = embedder.encode(
            sentences,
            prompt=instruction,
            batch_size=32,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        
        # Compute similarity/distance between consecutive sentences
        # and accumulate soft count
        n_watermark = 0.0
        n_test_sent = len(sentences) - 1
        
        a, b = self.simmark_interval
        
        for j in range(1, len(sentences)):
            embedding1 = embeddings[j-1].reshape(1, -1).cpu().detach().numpy()
            embedding2 = embeddings[j].reshape(1, -1).cpu().detach().numpy()
            
            if self.simmark_mode == "cosine":
                dist = cosine_similarity(
                    pca_model.transform(embedding1),
                    pca_model.transform(embedding2)
                )[0, 0]
            else:
                dist = euclidean_distances(
                    pca_model.transform(embedding1),
                    pca_model.transform(embedding2)
                )[0, 0]
            
            # Compute distance to interval
            if a <= dist <= b:
                dist_to_interval = 0.0
            else:
                dist_to_interval = min(abs(dist - a), abs(dist - b))
            
            # Soft count: exp(-K * distance)
            soft_count = np.exp(-self.simmark_K * dist_to_interval)
            n_watermark += soft_count
        
        # Z-score computation (same as SimMark paper)
        num = n_watermark - gamma * n_test_sent
        denom = np.sqrt(n_test_sent * gamma * (1 - gamma) + 1e-12)
        z_score = num / denom
        
        return SimMarkDetectionResult(
            z_score=float(z_score),
            n_watermark=float(n_watermark),
            n_test_sent=n_test_sent,
            gamma=gamma,
            mode=self.simmark_mode,
            interval=self.simmark_interval,
            K=self.simmark_K,
            prediction=z_score > z_threshold,
        )
    
    # =========================================================================
    # KGW Baseline (Token-level red/green list watermark)
    # =========================================================================
    
    def generate_with_kgw(
        self,
        prompt: str,
        gamma: float = 0.25,
        delta: float = 2.0,
        max_new_tokens: int = 150,
    ) -> Tuple[str, Dict]:
        """Generate text with KGW (token-level) watermark."""
        cached = self._load_model(self.generator_model)
        tokenizer = cached["tokenizer"]
        model = cached["model"]
        
        processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            seeding_scheme="selfhash",
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        print(f"\n🔴🟢 Generating with KGW (gamma={gamma}, delta={delta})...")
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                logits_processor=LogitsProcessorList([processor]),
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = output[:, inputs["input_ids"].shape[-1]:]
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        
        metadata = {
            "type": "kgw",
            "gamma": gamma,
            "delta": delta,
            "generator": self.generator_model,
        }
        
        print(f"   Generated {len(text)} chars")
        return text, metadata
    
    def detect_kgw(
        self,
        text: str,
        gamma: float = 0.25,
        z_threshold: float = 3.0,
    ) -> KGWDetectionResult:
        """Detect KGW watermark."""
        tokenizer = self._load_tokenizer_only(self.generator_model)
        
        detector = WatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=gamma,
            seeding_scheme="selfhash",
            device=self.device,
            tokenizer=tokenizer,
            z_threshold=z_threshold,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )
        
        result = detector.detect(text, return_scores=True, return_prediction=True)
        
        return KGWDetectionResult(
            z_score=float(result.get("z_score", 0.0)),
            p_value=float(result.get("p_value", 1.0)),
            prediction=bool(result.get("prediction", False)),
            num_green_tokens=int(result.get("num_green_tokens", 0)),
            num_tokens_scored=int(result.get("num_tokens_scored", 0)),
            green_fraction=float(result.get("green_fraction", 0.0)),
        )
    
    # =========================================================================
    # Paraphrase
    # =========================================================================
    
    def paraphrase_text(self, text: str) -> str:
        """Paraphrase text using paraphraser model."""
        prompt = f"""{self._paraphrase_instruction}

Text:
{text}

Output:"""
        
        if self.use_api and self._api_client:
            result = self._api_client.generate(
                model_nickname=self.paraphraser_model,
                prompt=prompt,
                max_new_tokens=int(len(text.split()) * 1.5) + 50,
                temperature=0.7,
                do_sample=True,
                with_watermark=False,
            )
            return self._clean_output(result["generated_text"])
        
        # Local mode
        cached = self._load_model(self.paraphraser_model)
        tokenizer = cached["tokenizer"]
        model = cached["model"]
        
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
        return self._clean_output(
            tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        )
    
    @staticmethod
    def _clean_output(text: str) -> str:
        """Clean paraphrase output."""
        text = text.strip()
        
        # Remove common prefixes
        prefixes = [
            "Output:", "output:", "OUTPUT:",
            "Here is the paraphrased version:",
            "Here's the paraphrased text:",
            "Paraphrased version:",
        ]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        # Remove quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        
        return text
    
    @staticmethod
    def calculate_similarity(text_a: str, text_b: str) -> float:
        """Calculate Jaccard similarity."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        return intersection / union if union > 0 else 0.0
    
    # =========================================================================
    # Main Experiment
    # =========================================================================
    
    def run_single_experiment(
        self,
        prompt: str,
        simmark_z_threshold: float = 2.0,
        kgw_z_threshold: float = 3.0,
        kgw_gamma: float = 0.25,
        kgw_delta: float = 2.0,
    ) -> ExperimentResult:
        """
        Run a single comparison experiment.
        
        1. Generate text with SimMark watermark
        2. Detect SimMark watermark on original
        3. Paraphrase text
        4. Detect SimMark watermark on paraphrased
        5. Repeat steps 1-4 for KGW baseline
        """
        print("\n" + "=" * 70)
        print("SimMark vs KGW Paraphrase Robustness Experiment")
        print("=" * 70)
        print(f"Prompt: {prompt[:100]}...")
        
        # ================= SimMark =================
        simmark_text, simmark_meta = self.generate_with_simmark(prompt)
        simmark_original = self.detect_simmark(simmark_text, simmark_z_threshold)
        print(f"   SimMark Original z-score: {simmark_original.z_score:.4f}")
        
        print("\n📝 Paraphrasing SimMark text...")
        simmark_paraphrased = self.paraphrase_text(simmark_text)
        simmark_para_detection = self.detect_simmark(simmark_paraphrased, simmark_z_threshold)
        print(f"   SimMark Paraphrased z-score: {simmark_para_detection.z_score:.4f}")
        
        # ================= KGW Baseline =================
        kgw_text, kgw_meta = self.generate_with_kgw(
            prompt, gamma=kgw_gamma, delta=kgw_delta
        )
        kgw_original = self.detect_kgw(kgw_text, gamma=kgw_gamma, z_threshold=kgw_z_threshold)
        print(f"   KGW Original z-score: {kgw_original.z_score:.4f}")
        
        print("\n📝 Paraphrasing KGW text...")
        kgw_paraphrased = self.paraphrase_text(kgw_text)
        kgw_para_detection = self.detect_kgw(kgw_paraphrased, gamma=kgw_gamma, z_threshold=kgw_z_threshold)
        print(f"   KGW Paraphrased z-score: {kgw_para_detection.z_score:.4f}")
        
        # Calculate similarity
        simmark_similarity = self.calculate_similarity(simmark_text, simmark_paraphrased)
        kgw_similarity = self.calculate_similarity(kgw_text, kgw_paraphrased)
        
        # Create result
        result = ExperimentResult(
            prompt=prompt,
            original_text=simmark_text,  # Use SimMark text as reference
            paraphrased_text=simmark_paraphrased,
            simmark_original=simmark_original,
            simmark_paraphrase=simmark_para_detection,
            kgw_original=kgw_original,
            kgw_paraphrase=kgw_para_detection,
            simmark_survived=simmark_para_detection.prediction,
            kgw_survived=kgw_para_detection.prediction,
            semantic_similarity=simmark_similarity,
            timestamp=datetime.now().isoformat(),
        )
        
        self._print_summary(result)
        return result
    
    def run_batch_experiment(
        self,
        prompts: List[str],
        simmark_z_threshold: float = 2.0,
        kgw_z_threshold: float = 3.0,
    ) -> Dict:
        """Run batch experiments on multiple prompts."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\n\n{'#' * 70}")
            print(f"Experiment {i+1}/{len(prompts)}")
            print('#' * 70)
            result = self.run_single_experiment(
                prompt,
                simmark_z_threshold=simmark_z_threshold,
                kgw_z_threshold=kgw_z_threshold,
            )
            results.append(asdict(result))
        
        # Aggregate statistics
        simmark_survived = sum(1 for r in results if r["simmark_survived"])
        kgw_survived = sum(1 for r in results if r["kgw_survived"])
        
        summary = {
            "total_experiments": len(prompts),
            "simmark_survival_rate": simmark_survived / len(prompts),
            "kgw_survival_rate": kgw_survived / len(prompts),
            "simmark_avg_original_z": np.mean([r["simmark_original"]["z_score"] for r in results]),
            "simmark_avg_paraphrase_z": np.mean([r["simmark_paraphrase"]["z_score"] for r in results]),
            "kgw_avg_original_z": np.mean([r["kgw_original"]["z_score"] for r in results]),
            "kgw_avg_paraphrase_z": np.mean([r["kgw_paraphrase"]["z_score"] for r in results]),
            "avg_semantic_similarity": np.mean([r["semantic_similarity"] for r in results]),
        }
        
        return {
            "summary": summary,
            "results": results,
            "config": {
                "generator": self.generator_model,
                "paraphraser": self.paraphraser_model,
                "simmark_mode": self.simmark_mode,
                "simmark_interval": self.simmark_interval,
                "simmark_K": self.simmark_K,
                "simmark_z_threshold": simmark_z_threshold,
                "kgw_z_threshold": kgw_z_threshold,
            },
            "timestamp": datetime.now().isoformat(),
        }
    
    def _print_summary(self, result: ExperimentResult):
        """Print comparison summary."""
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        
        print("\n┌─────────────────────┬────────────────┬────────────────�?)
        print("�?      Metric        �?   SimMark     �?     KGW       �?)
        print("├─────────────────────┼────────────────┼────────────────�?)
        
        sim_orig = result.simmark_original.z_score
        kgw_orig = result.kgw_original.z_score
        print(f"�?Original z-score    �?   {sim_orig:>8.4f}    �?   {kgw_orig:>8.4f}    �?)
        
        sim_para = result.simmark_paraphrase.z_score
        kgw_para = result.kgw_paraphrase.z_score
        print(f"�?Paraphrased z-score �?   {sim_para:>8.4f}    �?   {kgw_para:>8.4f}    �?)
        
        sim_decay = sim_orig - sim_para
        kgw_decay = kgw_orig - kgw_para
        print(f"�?z-score decay       �?   {sim_decay:>8.4f}    �?   {kgw_decay:>8.4f}    �?)
        
        sim_surv = "�?YES" if result.simmark_survived else "�?NO"
        kgw_surv = "�?YES" if result.kgw_survived else "�?NO"
        print(f"�?Survived?           �?   {sim_surv:>6}      �?   {kgw_surv:>6}      �?)
        
        print("└─────────────────────┴────────────────┴────────────────�?)
        
        if result.simmark_survived and not result.kgw_survived:
            print("\n🎉 SimMark survived while KGW did not!")
        elif result.simmark_survived and result.kgw_survived:
            print("\n�?Both watermarks survived!")
        elif not result.simmark_survived and not result.kgw_survived:
            print("\n�?Neither watermark survived paraphrasing.")
        else:
            print("\n⚠️ KGW survived but SimMark did not (unexpected!)")
    
    def save_results(self, results: Dict, filename: Optional[str] = None) -> str:
        """Save results to JSON."""
        if filename is None:
            filename = f"simmark_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Results saved: {filepath}")
        return filepath


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SimMark vs KGW Paraphrase Robustness Experiment"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a short story about artificial intelligence discovering emotions.",
        help="Generation prompt",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch experiment with multiple prompts",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cosine", "euclidean"],
        default="cosine",
        help="SimMark mode",
    )
    parser.add_argument(
        "--interval-a",
        type=float,
        default=0.68,
        help="SimMark interval lower bound",
    )
    parser.add_argument(
        "--interval-b",
        type=float,
        default=0.76,
        help="SimMark interval upper bound",
    )
    parser.add_argument(
        "--paraphrase-mode",
        type=str,
        choices=["moderate", "aggressive"],
        default="moderate",
        help="Paraphrase mode: moderate (preserves sentence count) or aggressive (may merge/compress)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=50,
        help="Max rejection sampling trials per sentence",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use API for paraphrasing",
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="Use PCA for dimensionality reduction (euclidean mode)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    experiment = SimMarkParaphraseExperiment(
        simmark_mode=args.mode,
        simmark_interval=(args.interval_a, args.interval_b),
        max_trials=args.max_trials,
        use_api=args.use_api,
        use_pca=args.use_pca,
        paraphrase_mode=args.paraphrase_mode,
    )
    
    print(f"📍 Paraphrase Mode: {args.paraphrase_mode}")
    
    if args.batch:
        # Batch mode with multiple prompts
        prompts = [
            "Write a short story about artificial intelligence discovering emotions.",
            "Explain the concept of quantum computing to a high school student.",
            "Describe a day in the life of an astronaut on the International Space Station.",
            "Write a poem about the changing seasons.",
            "Discuss the ethical implications of genetic engineering.",
        ]
        results = experiment.run_batch_experiment(prompts)
        
        print("\n\n" + "=" * 70)
        print("BATCH EXPERIMENT SUMMARY")
        print("=" * 70)
        summary = results["summary"]
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"SimMark survival rate: {summary['simmark_survival_rate']:.2%}")
        print(f"KGW survival rate: {summary['kgw_survival_rate']:.2%}")
        print(f"SimMark avg original z: {summary['simmark_avg_original_z']:.4f}")
        print(f"SimMark avg paraphrase z: {summary['simmark_avg_paraphrase_z']:.4f}")
        print(f"KGW avg original z: {summary['kgw_avg_original_z']:.4f}")
        print(f"KGW avg paraphrase z: {summary['kgw_avg_paraphrase_z']:.4f}")
        
        experiment.save_results(results)
    else:
        # Single experiment
        result = experiment.run_single_experiment(args.prompt)
        experiment.save_results({"result": asdict(result)})


if __name__ == "__main__":
    main()
