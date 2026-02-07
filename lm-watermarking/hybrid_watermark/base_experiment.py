"""
水印实验基础类
提供模型加载、水印处理器/检测器创建、文本生成等公共功能
支持本地模型和 API 模式
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# 确保能导入上级目录的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extended_watermark_processor import WatermarkDetector, WatermarkLogitsProcessor
from llama_demos.model_config_manager import ModelConfigManager


@dataclass
class WatermarkDetection:
    """水印检测结果"""

    z_score: float
    p_value: float
    prediction: bool
    num_green_tokens: int
    num_tokens_scored: int
    green_fraction: float
    z_threshold: float
    confidence: Optional[float] = None


class BaseExperiment:
    """水印实验基础类，提供公共功能"""

    def __init__(
        self,
        default_model: str = "llama-3.2-3b",
        device: Optional[str] = None,
        config_path: Optional[str] = None,
        results_subdir: str = "results",
        use_api: bool = False,
    ):
        """
        初始化实验基础环境

        Args:
            default_model: 默认模型昵称
            device: 运行设备 (cuda/cpu)
            config_path: 模型配置文件路径
            results_subdir: 结果保存子目录
            use_api: 是否使用 API 模式（通过 model_server 访问模型）
        """
        # 配置管理器
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "llama_demos", "model_config.json"
            )
        self.config_manager = ModelConfigManager(config_path)

        # 设备设置
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA 不可用，回退到 CPU")
            device = "cpu"
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32

        # 模型缓存
        self.default_model = default_model
        self.model_cache: Dict[str, Dict] = {}

        # API 模式
        self.use_api = use_api
        self._api_client = None
        if use_api:
            from model_client import ModelClient
            self._api_client = ModelClient()
            print("🌐 使用 API 模式（通过 model_server 访问模型）")

        # 结果目录
        self.results_dir = os.path.join(os.path.dirname(__file__), results_subdir)
        os.makedirs(self.results_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # 模型加载
    # ─────────────────────────────────────────────────────────────
    def load_model(
        self, nickname: str
    ) -> Tuple[AutoTokenizer, AutoModelForCausalLM, Dict]:
        """
        加载模型（带缓存）

        Args:
            nickname: 模型昵称

        Returns:
            (tokenizer, model, info) 元组
        """
        if nickname in self.model_cache:
            c = self.model_cache[nickname]
            return c["tokenizer"], c["model"], c["info"]

        info = self.config_manager.get_model_info_by_nickname(nickname)
        if not info:
            available = self.config_manager.list_model_names()
            raise ValueError(f"未找到模型: {nickname}。可用模型: {', '.join(available)}")

        print(f"📦 加载模型: {nickname} ({info['model_identifier']})")

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

        self.model_cache[nickname] = {
            "tokenizer": tokenizer,
            "model": model,
            "info": info,
        }
        print(f"✅ 模型加载完成: {nickname}")
        return tokenizer, model, info

    def load_tokenizer_only(self, nickname: str) -> Tuple[AutoTokenizer, Dict]:
        """
        只加载分词器（不加载模型权重，用于 API 模式下的水印检测）

        Args:
            nickname: 模型昵称

        Returns:
            (tokenizer, info) 元组
        """
        # 如果完整模型已缓存，直接返回 tokenizer
        if nickname in self.model_cache:
            c = self.model_cache[nickname]
            return c["tokenizer"], c["info"]

        # 检查是否有单独的 tokenizer 缓存
        cache_key = f"tokenizer_only:{nickname}"
        if cache_key in self.model_cache:
            c = self.model_cache[cache_key]
            return c["tokenizer"], c["info"]

        info = self.config_manager.get_model_info_by_nickname(nickname)
        if not info:
            available = self.config_manager.list_model_names()
            raise ValueError(f"未找到模型: {nickname}。可用模型: {', '.join(available)}")

        print(f"📦 加载分词器: {nickname} ({info['model_identifier']})")

        tokenizer = AutoTokenizer.from_pretrained(
            info["model_identifier"], trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 缓存 tokenizer（不缓存 model）
        self.model_cache[cache_key] = {
            "tokenizer": tokenizer,
            "info": info,
        }
        print(f"✅ 分词器加载完成: {nickname}")
        return tokenizer, info

    # ─────────────────────────────────────────────────────────────
    # 水印处理器
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def normalize_seeding_scheme(seeding_scheme: str, hash_key: int) -> str:
        """标准化 seeding_scheme 字符串"""
        if hash_key == 15485863:
            return seeding_scheme
        parts = seeding_scheme.split("-")
        if len(parts) >= 6:
            parts[-1] = str(hash_key)
            return "-".join(parts)
        return f"ff-anchored_minhash_prf-4-True-{hash_key}"

    def create_watermark_processor(
        self, tokenizer: AutoTokenizer, config: Dict
    ) -> WatermarkLogitsProcessor:
        """
        创建水印处理器

        Args:
            tokenizer: 分词器
            config: 水印配置 {gamma, delta, seeding_scheme, hash_key}

        Returns:
            WatermarkLogitsProcessor 实例
        """
        seeding_scheme = self.normalize_seeding_scheme(
            config.get("seeding_scheme", "selfhash"),
            config.get("hash_key", 15485863),
        )
        return WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=config.get("gamma", 0.25),
            delta=config.get("delta", 2.0),
            seeding_scheme=seeding_scheme,
        )

    def create_watermark_detector(
        self, tokenizer: AutoTokenizer, config: Dict, z_threshold: float = 3.0
    ) -> WatermarkDetector:
        """
        创建水印检测器

        Args:
            tokenizer: 分词器
            config: 水印配置
            z_threshold: 检测阈值

        Returns:
            WatermarkDetector 实例
        """
        seeding_scheme = self.normalize_seeding_scheme(
            config.get("seeding_scheme", "selfhash"),
            config.get("hash_key", 15485863),
        )
        return WatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=config.get("gamma", 0.25),
            seeding_scheme=seeding_scheme,
            device=self.device,
            tokenizer=tokenizer,
            z_threshold=z_threshold,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )

    # ─────────────────────────────────────────────────────────────
    # 文本生成
    # ─────────────────────────────────────────────────────────────
    def ensure_min_context(
        self,
        prompt: str,
        tokenizer: AutoTokenizer,
        processor: WatermarkLogitsProcessor,
    ) -> str:
        """
        确保 prompt 满足水印上下文长度要求

        Args:
            prompt: 原始提示词
            tokenizer: 分词器
            processor: 水印处理器

        Returns:
            满足上下文要求的提示词
        """
        context_width = getattr(processor, "context_width", 0)
        if not context_width or context_width <= 0:
            return prompt

        candidate = prompt
        prefixes = ["Background: ", "Context: ", "Note: ", "Info: "]
        idx = 0
        while True:
            tokens = tokenizer(
                candidate, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            if tokens.shape[-1] >= context_width:
                if candidate != prompt:
                    print(f"[Info] 自动补足提示以满足 {context_width} 个上下文 token")
                return candidate
            candidate = prefixes[idx % len(prefixes)] + candidate
            idx += 1
            if idx > 8:
                return (" pad" * context_width) + candidate

    def generate_with_watermark(
        self,
        prompt: str,
        model_nickname: Optional[str] = None,
        watermark_config: Optional[Dict] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[str, Dict, AutoTokenizer]:
        """
        带水印生成文本（自动选择本地或 API 模式）

        Args:
            prompt: 提示词
            model_nickname: 模型昵称（默认使用 default_model）
            watermark_config: 水印配置
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            do_sample: 是否采样
            top_p: nucleus sampling 参数
            top_k: top-k sampling 参数

        Returns:
            (generated_text, metadata, tokenizer) 元组
        """
        nickname = model_nickname or self.default_model
        wm_config = watermark_config or {
            "gamma": 0.25,
            "delta": 2.0,
            "seeding_scheme": "selfhash",
            "hash_key": 15485863,
        }

        # API 模式
        if self.use_api and self._api_client:
            return self._generate_with_watermark_api(
                prompt, nickname, wm_config, max_new_tokens, temperature, do_sample, top_p, top_k
            )

        # 本地模式
        return self._generate_with_watermark_local(
            prompt, nickname, wm_config, max_new_tokens, temperature, do_sample, top_p, top_k
        )

    def _generate_with_watermark_api(
        self,
        prompt: str,
        nickname: str,
        wm_config: Dict,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        top_p: Optional[float],
        top_k: Optional[int],
    ) -> Tuple[str, Dict, AutoTokenizer]:
        """通过 API 生成带水印文本"""
        # 先加载 tokenizer（会被缓存），与 API 请求并行准备
        # 这样后续的水印检测可以直接使用缓存的 tokenizer
        tokenizer, info = self.load_tokenizer_only(nickname)
        
        result = self._api_client.generate(
            model_nickname=nickname,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            gamma=wm_config.get("gamma", 0.25),
            delta=wm_config.get("delta", 2.0),
            seeding_scheme=wm_config.get("seeding_scheme", "selfhash"),
            hash_key=wm_config.get("hash_key", 15485863),
            with_watermark=True,
        )
        
        metadata = {
            "model": nickname,
            "model_identifier": info["model_identifier"],
            "prompt": prompt,
            "watermark_config": wm_config,
            "generation_config": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "top_k": top_k,
            },
            "generated_at": datetime.now().isoformat(),
            "token_usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("prompt_tokens", 0) + result.get("completion_tokens", 0),
            },
            "api_mode": True,
        }
        return result["generated_text"], metadata, tokenizer

    def _generate_with_watermark_local(
        self,
        prompt: str,
        nickname: str,
        wm_config: Dict,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        top_p: Optional[float],
        top_k: Optional[int],
    ) -> Tuple[str, Dict, AutoTokenizer]:
        """本地生成带水印文本"""
        tokenizer, model, info = self.load_model(nickname)
        processor = self.create_watermark_processor(tokenizer, wm_config)

        prompt_prepared = self.ensure_min_context(prompt, tokenizer, processor)
        inputs = tokenizer(prompt_prepared, return_tensors="pt").to(self.device)

        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            logits_processor=LogitsProcessorList([processor]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if top_p is not None:
            generate_kwargs["top_p"] = top_p
        if top_k is not None:
            generate_kwargs["top_k"] = top_k

        with torch.no_grad():
            output = model.generate(**inputs, **generate_kwargs)

        generated = output[:, inputs["input_ids"].shape[-1] :]
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

        metadata = {
            "model": nickname,
            "model_identifier": info["model_identifier"],
            "prompt": prompt,
            "prompt_prepared": prompt_prepared,
            "watermark_config": wm_config,
            "generation_config": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "top_k": top_k,
            },
            "generated_at": datetime.now().isoformat(),
            "token_usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(generated.shape[-1]),
                "total_tokens": int(inputs["input_ids"].shape[-1] + generated.shape[-1]),
            },
        }
        return text, metadata, tokenizer

    # ─────────────────────────────────────────────────────────────
    # 水印检测
    # ─────────────────────────────────────────────────────────────
    def detect_watermark(
        self,
        text: str,
        tokenizer: AutoTokenizer,
        config: Dict,
        z_threshold: float = 3.0,
    ) -> WatermarkDetection:
        """
        检测文本水印

        Args:
            text: 待检测文本
            tokenizer: 分词器
            config: 水印配置
            z_threshold: 检测阈值

        Returns:
            WatermarkDetection 结果
        """
        detector = self.create_watermark_detector(tokenizer, config, z_threshold)
        result = detector.detect(
            text, z_threshold=z_threshold, return_scores=True, return_prediction=True
        )
        return WatermarkDetection(
            z_score=float(result.get("z_score", 0.0)),
            p_value=float(result.get("p_value", 1.0)),
            prediction=bool(result.get("prediction", False)),
            num_green_tokens=int(result.get("num_green_tokens", 0)),
            num_tokens_scored=int(result.get("num_tokens_scored", 0)),
            green_fraction=float(result.get("green_fraction", 0.0)),
            z_threshold=z_threshold,
            confidence=float(result.get("confidence", 0.0))
            if "confidence" in result
            else None,
        )

    # ─────────────────────────────────────────────────────────────
    # 语义相似度（简单版，后续可替换为 sentence-transformers）
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_similarity(text_a: str, text_b: str) -> float:
        """
        计算两段文本的 Jaccard 相似度

        Args:
            text_a: 文本 A
            text_b: 文本 B

        Returns:
            相似度 (0.0 ~ 1.0)
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        return intersection / union if union > 0 else 0.0

    # ─────────────────────────────────────────────────────────────
    # 结果保存与输出
    # ─────────────────────────────────────────────────────────────
    def save_results(self, results: Dict, filename: Optional[str] = None) -> str:
        """
        保存实验结果到 JSON

        Args:
            results: 结果字典
            filename: 文件名（可选）

        Returns:
            保存路径
        """
        if filename is None:
            filename = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 结果已保存: {filepath}")
        return filepath

    @staticmethod
    def print_detection_summary(detection: WatermarkDetection, label: str = "") -> None:
        """打印检测结果摘要"""
        status = "✅ 检测到水印" if detection.prediction else "❌ 未检测到水印"
        prefix = f"[{label}] " if label else ""
        print(
            f"{prefix}{status} | z-score: {detection.z_score:.4f} | "
            f"green: {detection.green_fraction:.2%} ({detection.num_green_tokens}/{detection.num_tokens_scored})"
        )
