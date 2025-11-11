"""
Multi-LLM watermark chain experiment.

构建一个多模型链路：
- 生成模型负责带水印生成（默认 llama-3.2-3b）。
- 改写模型负责顺次改写（默认 qwen-3-8b）。

提供能力：
1. 按给定提示生成带水印文本。
2. 通过一个或多个改写模型顺次改写文本。
3. 在改写前后检测水印，记录 z-score / p-value 等指标。
4. 计算语义相似度和水印衰减程度。
5. 支持在不同模型组合间对比水印存活率。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from extended_watermark_processor import WatermarkDetector, WatermarkLogitsProcessor
from llama_demos.model_config_manager import ModelConfigManager


@dataclass
class WatermarkDetection:
    z_score: float
    p_value: float
    prediction: bool
    num_green_tokens: int
    num_tokens_scored: int
    green_fraction: float
    z_threshold: float
    confidence: Optional[float] = None


class MultiLLMChainExperiment:
    """多模型 LLM 链路实验器。"""

    def __init__(
        self,
        generator_default: str = "llama-3.2-3b",
        paraphraser_defaults: Optional[List[str]] = None,
        device: Optional[str] = None,
        config_manager: Optional[ModelConfigManager] = None,
    ) -> None:
        if config_manager is None:
            config_manager = ModelConfigManager(
                os.path.join(os.path.dirname(__file__), "..", "llama_demos", "model_config.json")
            )
        self.config_manager = config_manager

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA 不可用，自动回退到 CPU。")
            device = "cpu"

        self.device = device
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.generator_default = generator_default
        self.paraphraser_defaults = paraphraser_defaults or ["qwen-3-8b"]

        self.model_cache: Dict[str, Dict[str, object]] = {}
        self.results_dir = os.path.join(os.path.dirname(__file__), "multi_llm_chain_results")
        os.makedirs(self.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------
    def _load_model(self, nickname: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, Dict]:
        if nickname in self.model_cache:
            cached = self.model_cache[nickname]
            return cached["tokenizer"], cached["model"], cached["info"]  # type: ignore[return-value]

        info = self.config_manager.get_model_info_by_nickname(nickname)
        if not info:
            raise ValueError(f"未找到模型配置: {nickname}")

        tokenizer = AutoTokenizer.from_pretrained(info["model_identifier"], trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
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

        self.model_cache[nickname] = {"tokenizer": tokenizer, "model": model, "info": info}
        return tokenizer, model, info

    # ------------------------------------------------------------------
    # Watermark helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_seeding_scheme(seeding_scheme: str, hash_key: int) -> str:
        if hash_key == 15485863:
            return seeding_scheme
        parts = seeding_scheme.split("-")
        if len(parts) >= 6:
            parts[-1] = str(hash_key)
            return "-".join(parts)
        return f"ff-anchored_minhash_prf-4-True-{hash_key}"

    def _create_watermark_processor(
        self,
        tokenizer: AutoTokenizer,
        watermark_config: Dict,
    ) -> WatermarkLogitsProcessor:
        seeding_scheme = self._normalise_seeding_scheme(
            watermark_config.get("seeding_scheme", "selfhash"),
            watermark_config.get("hash_key", 15485863),
        )

        return WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=watermark_config.get("gamma", 0.25),
            delta=watermark_config.get("delta", 2.0),
            seeding_scheme=seeding_scheme,
        )

    def _create_watermark_detector(
        self,
        tokenizer: AutoTokenizer,
        watermark_config: Dict,
        z_threshold: float,
    ) -> WatermarkDetector:
        seeding_scheme = self._normalise_seeding_scheme(
            watermark_config.get("seeding_scheme", "selfhash"),
            watermark_config.get("hash_key", 15485863),
        )
        return WatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=watermark_config.get("gamma", 0.25),
            seeding_scheme=seeding_scheme,
            device=self.device,
            tokenizer=tokenizer,
            z_threshold=z_threshold,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )

    @staticmethod
    def _ensure_prompt_min_context(prompt: str, tokenizer: AutoTokenizer, processor: WatermarkLogitsProcessor) -> str:
        context_width = getattr(processor, "context_width", 0)
        if not context_width or context_width <= 0:
            return prompt

        candidate = prompt
        prefixes = [
            "Background context: ",
            "Additional detail: ",
            "Warm-up detail: ",
            "Context primer: ",
        ]
        idx = 0
        while True:
            tokenised = tokenizer(candidate, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            if tokenised.shape[-1] >= context_width:
                if candidate != prompt:
                    print(f"[Info] 自动补足提示以满足 {context_width} 个上下文 token。")
                return candidate
            candidate = prefixes[idx % len(prefixes)] + candidate
            idx += 1
            if idx > 8:
                candidate = (" warm-up" * max(1, context_width - tokenised.shape[-1])) + candidate
                return candidate

    # ------------------------------------------------------------------
    # Generation / paraphrase
    # ------------------------------------------------------------------
    def generate_with_watermark(
        self,
        prompt: str,
        generator_nickname: Optional[str] = None,
        watermark_config: Optional[Dict] = None,
        generation_config_name: str = "precise",
    ) -> Tuple[str, Dict, AutoTokenizer]:
        nickname = generator_nickname or self.generator_default
        tokenizer, model, info = self._load_model(nickname)

        watermark_config = watermark_config or self.config_manager.get_watermark_config("default") or {
            "gamma": 0.25,
            "delta": 2.0,
            "seeding_scheme": "selfhash",
            "hash_key": 15485863,
        }
        processor = self._create_watermark_processor(tokenizer, watermark_config)

        prompt_prepared = self._ensure_prompt_min_context(prompt, tokenizer, processor)
        inputs = tokenizer(prompt_prepared, return_tensors="pt").to(self.device)

        generation_config = self.config_manager.get_generation_config(generation_config_name) or {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "do_sample": True,
        }

        generate_kwargs = dict(
            max_new_tokens=generation_config.get("max_new_tokens", 100),
            do_sample=generation_config.get("do_sample", True),
            temperature=generation_config.get("temperature", 0.7),
            logits_processor=LogitsProcessorList([processor]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if "top_p" in generation_config:
            generate_kwargs["top_p"] = generation_config["top_p"]
        if "top_k" in generation_config:
            generate_kwargs["top_k"] = generation_config["top_k"]

        with torch.no_grad():
            output_tokens = model.generate(**inputs, **generate_kwargs)

        generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1] :]
        generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        completion_tokens = int(generated_tokens.shape[-1])

        metadata = {
            "model": info["nickname"],
            "model_identifier": info["model_identifier"],
            "prompt": prompt,
            "prompt_prepared": prompt_prepared,
            "watermark_config": watermark_config,
            "generation_config": generation_config,
            "generated_at": datetime.now().isoformat(),
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return generated_text, metadata, tokenizer

    def paraphrase_text(
        self,
        text: str,
        paraphraser_nickname: str,
        instruction: str = "Paraphrase the following text while preserving its meaning:",
        generation_config_name: str = "precise",
        watermark_config: Optional[Dict] = None,
    ) -> Tuple[str, Dict]:
        tokenizer, model, info = self._load_model(paraphraser_nickname)
        prompt_text = (
            f"{instruction}\n\n"
            f"Original text:\n{text}\n\n"
            f"Paraphrased text:"
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(self.device)
        generation_config = self.config_manager.get_generation_config(generation_config_name) or {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "do_sample": True,
        }
        paraphrase_kwargs = dict(
            max_new_tokens=generation_config.get("max_new_tokens", 200),
            do_sample=generation_config.get("do_sample", True),
            temperature=generation_config.get("temperature", 0.7),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if "top_p" in generation_config:
            paraphrase_kwargs["top_p"] = generation_config["top_p"]
        if "top_k" in generation_config:
            paraphrase_kwargs["top_k"] = generation_config["top_k"]

        # 如果提供了水印配置，在改写时也嵌入水印
        if watermark_config is not None:
            processor = self._create_watermark_processor(tokenizer, watermark_config)
            paraphrase_kwargs["logits_processor"] = LogitsProcessorList([processor])

        with torch.no_grad():
            output_tokens = model.generate(**inputs, **paraphrase_kwargs)

        generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1] :]
        paraphrased_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

        metadata = {
            "model": info["nickname"],
            "model_identifier": info["model_identifier"],
            "instruction": instruction,
            "prompt_text": prompt_text,
            "original_text": text,
            "generation_config": generation_config,
            "paraphrase_watermark_config": watermark_config,
            "paraphrased_at": datetime.now().isoformat(),
            "token_usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(generated_tokens.shape[-1]),
                "total_tokens": int(inputs["input_ids"].shape[-1] + generated_tokens.shape[-1]),
            },
        }
        return paraphrased_text, metadata

    # ------------------------------------------------------------------
    # Detection & analysis helpers
    # ------------------------------------------------------------------
    def detect_watermark(
        self,
        text: str,
        tokenizer: AutoTokenizer,
        watermark_config: Dict,
        z_threshold: float,
    ) -> WatermarkDetection:
        detector = self._create_watermark_detector(tokenizer, watermark_config, z_threshold)
        detection = detector.detect(
            text,
            z_threshold=z_threshold,
            return_scores=True,
            return_prediction=True,
        )
        return WatermarkDetection(
            z_score=float(detection.get("z_score", 0.0)),
            p_value=float(detection.get("p_value", 1.0)),
            prediction=bool(detection.get("prediction", False)),
            num_green_tokens=int(detection.get("num_green_tokens", 0)),
            num_tokens_scored=int(detection.get("num_tokens_scored", 0)),
            green_fraction=float(detection.get("green_fraction", 0.0)),
            z_threshold=z_threshold,
            confidence=float(detection.get("confidence", 0.0)) if "confidence" in detection else None,
        )

    @staticmethod
    def calculate_similarity(text_a: str, text_b: str) -> float:
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        return intersection / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Core experiment routines
    # ------------------------------------------------------------------
    def run_chain(
        self,
        prompt: str,
        generator_model: Optional[str] = None,
        paraphraser_models: Optional[List[str]] = None,
        watermark_config: Optional[Dict] = None,
        z_threshold: float = 3.0,
        paraphrase_instruction: str = "Paraphrase the following text while preserving its meaning:",
        prefetched: Optional[Dict[str, object]] = None,
    ) -> Dict:
        paraphraser_models = paraphraser_models or self.paraphraser_defaults

        if prefetched is not None:
            watermarked_text = cast(str, prefetched["watermarked_text"])
            generation_meta = dict(cast(Dict[str, object], prefetched["generation_metadata"]))
            tokenizer = cast(AutoTokenizer, prefetched["tokenizer"])
            watermark_conf = cast(Dict[str, object], generation_meta.get("watermark_config") or watermark_config or {})
        else:
            watermarked_text, generation_meta, tokenizer = self.generate_with_watermark(
                prompt,
                generator_nickname=generator_model,
                watermark_config=watermark_config,
            )
            watermark_conf = generation_meta["watermark_config"]

        original_detection = self.detect_watermark(
            watermarked_text,
            tokenizer=tokenizer,
            watermark_config=watermark_conf,
            z_threshold=z_threshold,
        )

        paraphraser_results = []
        for paraphraser in paraphraser_models:
            paraphrased_text, paraphrase_meta = self.paraphrase_text(
                watermarked_text,
                paraphraser,
                instruction=paraphrase_instruction,
            )
            paraphrased_detection = self.detect_watermark(
                paraphrased_text,
                tokenizer=tokenizer,
                watermark_config=watermark_conf,
                z_threshold=z_threshold,
            )
            similarity = self.calculate_similarity(watermarked_text, paraphrased_text)
            watermark_decay = original_detection.z_score - paraphrased_detection.z_score
            retention = (
                paraphrased_detection.z_score / original_detection.z_score
                if original_detection.z_score != 0
                else 0.0
            )

            paraphraser_results.append(
                {
                    "paraphraser": paraphraser,
                    "paraphrased_text": paraphrased_text,
                    "metadata": paraphrase_meta,
                    "detection": paraphrased_detection.__dict__,
                    "semantic_similarity": similarity,
                    "watermark_decay": watermark_decay,
                    "z_score_retention": retention,
                }
            )

        survival_count = sum(1 for result in paraphraser_results if result["detection"]["prediction"])
        survival_rate = survival_count / len(paraphraser_results) if paraphraser_results else 0.0

        result = {
            "prompt": prompt,
            "generator_model": generator_model or self.generator_default,
            "paraphraser_models": paraphraser_models,
            "watermark_config": watermark_conf,
            "z_threshold": z_threshold,
            "generated_text": watermarked_text,
            "generation_metadata": generation_meta,
            "original_detection": original_detection.__dict__,
            "paraphraser_results": paraphraser_results,
            "summary": {
                "watermark_survival_rate": survival_rate,
                "watermark_survived_count": survival_count,
                "average_similarity": (
                    sum(item["semantic_similarity"] for item in paraphraser_results) / len(paraphraser_results)
                    if paraphraser_results
                    else 0.0
                ),
                "average_decay": (
                    sum(item["watermark_decay"] for item in paraphraser_results) / len(paraphraser_results)
                    if paraphraser_results
                    else 0.0
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }
        return result

    def run_chain_with_watermarked_paraphrase(
        self,
        prompt: str,
        generator_model: Optional[str] = None,
        paraphraser_models: Optional[List[str]] = None,
        generator_watermark_config: Optional[Dict] = None,
        paraphraser_watermark_configs: Optional[List[Dict]] = None,
        z_threshold: float = 3.0,
        paraphrase_instruction: str = "Paraphrase the following text while preserving its meaning:",
    ) -> Dict:
        """
        运行链路实验，改写阶段也嵌入水印（使用不同的 green/red list）。
        
        Args:
            paraphraser_watermark_configs: 每个改写模型对应的水印配置列表。
                                           如果为 None，则改写时不嵌入水印。
        """
        paraphraser_models = paraphraser_models or self.paraphraser_defaults
        
        # 生成带水印文本
        watermarked_text, generation_meta, tokenizer = self.generate_with_watermark(
            prompt,
            generator_nickname=generator_model,
            watermark_config=generator_watermark_config,
        )
        generator_watermark_conf = generation_meta["watermark_config"]

        # 检测原始生成文本的水印
        original_detection = self.detect_watermark(
            watermarked_text,
            tokenizer=tokenizer,
            watermark_config=generator_watermark_conf,
            z_threshold=z_threshold,
        )

        paraphraser_results = []
        for idx, paraphraser in enumerate(paraphraser_models):
            # 获取改写器的水印配置
            paraphrase_wm_config = None
            if paraphraser_watermark_configs and idx < len(paraphraser_watermark_configs):
                paraphrase_wm_config = paraphraser_watermark_configs[idx]
            
            # 改写文本（可能嵌入新水印）
            paraphrased_text, paraphrase_meta = self.paraphrase_text(
                watermarked_text,
                paraphraser,
                instruction=paraphrase_instruction,
                watermark_config=paraphrase_wm_config,
            )
            
            # 用原始生成器的密钥检测（原水印存活性）
            detection_with_generator_key = self.detect_watermark(
                paraphrased_text,
                tokenizer=tokenizer,
                watermark_config=generator_watermark_conf,
                z_threshold=z_threshold,
            )
            
            # 如果改写时也嵌入了水印，用改写器的密钥检测
            detection_with_paraphraser_key = None
            if paraphrase_wm_config is not None:
                detection_with_paraphraser_key = self.detect_watermark(
                    paraphrased_text,
                    tokenizer=tokenizer,
                    watermark_config=paraphrase_wm_config,
                    z_threshold=z_threshold,
                )
            
            similarity = self.calculate_similarity(watermarked_text, paraphrased_text)
            watermark_decay = original_detection.z_score - detection_with_generator_key.z_score
            retention = (
                detection_with_generator_key.z_score / original_detection.z_score
                if original_detection.z_score != 0
                else 0.0
            )

            paraphraser_results.append(
                {
                    "paraphraser": paraphraser,
                    "paraphrased_text": paraphrased_text,
                    "metadata": paraphrase_meta,
                    "paraphrase_watermark_config": paraphrase_wm_config,
                    "detection_with_generator_key": detection_with_generator_key.__dict__,
                    "detection_with_paraphraser_key": detection_with_paraphraser_key.__dict__ if detection_with_paraphraser_key else None,
                    "semantic_similarity": similarity,
                    "generator_watermark_decay": watermark_decay,
                    "generator_z_score_retention": retention,
                }
            )

        # 统计原水印存活率
        generator_survival_count = sum(
            1 for result in paraphraser_results 
            if result["detection_with_generator_key"]["prediction"]
        )
        generator_survival_rate = generator_survival_count / len(paraphraser_results) if paraphraser_results else 0.0

        # 统计新水印检测率
        paraphraser_detection_count = sum(
            1 for result in paraphraser_results 
            if result["detection_with_paraphraser_key"] and result["detection_with_paraphraser_key"]["prediction"]
        )
        paraphraser_detection_rate = paraphraser_detection_count / len(paraphraser_results) if paraphraser_results else 0.0

        result = {
            "experiment_type": "watermarked_paraphrase_chain",
            "prompt": prompt,
            "generator_model": generator_model or self.generator_default,
            "paraphraser_models": paraphraser_models,
            "generator_watermark_config": generator_watermark_conf,
            "paraphraser_watermark_configs": paraphraser_watermark_configs,
            "z_threshold": z_threshold,
            "generated_text": watermarked_text,
            "generation_metadata": generation_meta,
            "original_detection": original_detection.__dict__,
            "paraphraser_results": paraphraser_results,
            "summary": {
                "generator_watermark_survival_rate": generator_survival_rate,
                "generator_watermark_survived_count": generator_survival_count,
                "paraphraser_watermark_detection_rate": paraphraser_detection_rate,
                "paraphraser_watermark_detected_count": paraphraser_detection_count,
                "average_similarity": (
                    sum(item["semantic_similarity"] for item in paraphraser_results) / len(paraphraser_results)
                    if paraphraser_results
                    else 0.0
                ),
                "average_generator_decay": (
                    sum(item["generator_watermark_decay"] for item in paraphraser_results) / len(paraphraser_results)
                    if paraphraser_results
                    else 0.0
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }
        return result

    def compare_across_models(
        self,
        prompt: str,
        generator_models: Optional[List[str]] = None,
        paraphraser_models: Optional[List[str]] = None,
        watermark_config: Optional[Dict] = None,
    z_threshold: float = 3.0,
    paraphrase_instruction: str = "Paraphrase the following text while preserving its meaning:",
        prefetched_generations: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> Dict:
        generator_models = generator_models or [self.generator_default]
        paraphraser_models = paraphraser_models or self.paraphraser_defaults

        comparison_results = []
        for generator in generator_models:
            result = self.run_chain(
                prompt,
                generator_model=generator,
                paraphraser_models=paraphraser_models,
                watermark_config=watermark_config,
                z_threshold=z_threshold,
                paraphrase_instruction=paraphrase_instruction,
                prefetched=prefetched_generations.get(generator) if prefetched_generations else None,
            )
            comparison_results.append(result)

        overall_survival = [res["summary"]["watermark_survival_rate"] for res in comparison_results]
        average_survival = sum(overall_survival) / len(overall_survival) if overall_survival else 0.0

        summary = {
            "prompt": prompt,
            "generator_models": generator_models,
            "paraphraser_models": paraphraser_models,
            "average_survival_rate": average_survival,
            "highest_survival": max(overall_survival) if overall_survival else 0.0,
            "lowest_survival": min(overall_survival) if overall_survival else 0.0,
        }
        return {"summary": summary, "individual_results": comparison_results}

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def save_results(self, results: Dict, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"multi_llm_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 结果已保存: {filepath}")
        return filepath

    @staticmethod
    def print_summary(result: Dict) -> None:
        summary = result.get("summary", {})
        print("\n" + "=" * 80)
        print("多模型链路水印摘要")
        print("=" * 80)
        print(f"Prompt: {result.get('prompt')}")
        print(f"Generator: {result.get('generator_model')}")
        print(f"Paraphrasers: {', '.join(result.get('paraphraser_models', []))}")
        print(f"Original z-score: {result.get('original_detection', {}).get('z_score', 0.0):.4f}")
        print(f"Watermark survival rate: {summary.get('watermark_survival_rate', 0.0):.2%}")
        print(f"Average similarity: {summary.get('average_similarity', 0.0):.4f}")
        print(f"Average decay: {summary.get('average_decay', 0.0):.4f}")
        print("=" * 80 + "\n")


# ----------------------------------------------------------------------
# CLI helper
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-LLM watermark chain experiment")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a short story about artificial intelligence.",
        help="生成提示语",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="llama-3.2-3b",
        help="生成模型昵称，默认 llama-3.2-3b",
    )
    parser.add_argument(
        "--paraphrasers",
        type=str,
        default="qwen-3-8b",
        help="逗号分隔的改写模型昵称列表",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="是否在多个生成模型之间比较水印存活率（使用 --generator 逗号列表）",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="水印检测 z-score 阈值，默认 3.0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator_models = [name.strip() for name in args.generator.split(",") if name.strip()]
    paraphraser_models = [name.strip() for name in args.paraphrasers.split(",") if name.strip()]

    experiment = MultiLLMChainExperiment(
        generator_default=generator_models[0],
        paraphraser_defaults=paraphraser_models,
    )

    if args.compare and len(generator_models) > 1:
        result = experiment.compare_across_models(
            prompt=args.prompt,
            generator_models=generator_models,
            paraphraser_models=paraphraser_models,
            z_threshold=args.z_threshold,
        )
        experiment.save_results(result)
        for individual in result["individual_results"]:
            experiment.print_summary(individual)
        print("总体存活率对比: ")
        for item in result["individual_results"]:
            print(
                f"  {item['generator_model']}: "
                f"survival={item['summary']['watermark_survival_rate']:.2%}, "
                f"avg decay={item['summary']['average_decay']:.4f}, "
                f"avg sim={item['summary']['average_similarity']:.4f}"
            )
    else:
        result = experiment.run_chain(
            prompt=args.prompt,
            generator_model=generator_models[0],
            paraphraser_models=paraphraser_models,
            z_threshold=args.z_threshold,
        )
        experiment.print_summary(result)
        experiment.save_results(result)


if __name__ == "__main__":
    main()
