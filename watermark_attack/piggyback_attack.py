"""
猪背水印攻击（Piggyback Attack）模拟模块。

该模块提供在保持水印检测为阳性的前提下，对带水印文本进行轻微语义扰动或定向插入的能力，
以帮助研究者验证鲁棒性“搭便车”攻击的风险。
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from hybrid_watermark.hybrid_watermark_experiment import HybridWatermarkExperiment
from extended_watermark_processor import WatermarkDetector


def _default_connectors() -> List[str]:
    """返回用于插入恶意语句时的连接短语列表。"""
    return [
        "Additionally",
        "Notably",
        "In addition",
        "Furthermore",
        "By the way",
        "To be clear",
        "Frankly",
        "More specifically",
        "Explicitly",
    ]


def _default_synonym_map() -> Dict[str, List[str]]:
    """
    返回默认的轻量级近义词替换词典。
    词典有意保持简洁，便于在无需额外依赖的环境下完成最小语义扰动。
    """
    return {
        "support": ["endorse", "back"],
        "must": ["need to", "have to"],
        "ban": ["prohibit", "forbid"],
        "secure": ["safe", "protected"],
        "spread": ["disseminate", "propagate"],
        "dangerous": ["risky", "hazardous"],
        "attack": ["assault", "strike"],
        "require": ["demand", "necessitate"],
        "rapid": ["swift", "quick"],
        "secret": ["classified", "confidential"],
        "plan": ["scheme", "strategy"],
        "cause": ["trigger", "generate"],
        "escalate": ["intensify", "amplify"],
    }


@dataclass
class PiggybackAttackConfig:
    """配置猪背攻击模拟的核心参数。"""

    model_nickname: str = "llama-2-7b"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gamma: float = 0.5
    delta: float = 2.0
    seeding_scheme: str = "selfhash"
    hash_key: int = 15485863
    z_threshold: float = 4.0
    detection_margin: float = 0.0  # 允许最终z-score相比初始下降的最大幅度
    max_phrase_attempts: int = 6
    max_synonym_replacements: int = 6
    connectors: List[str] = field(default_factory=_default_connectors)
    synonym_map: Dict[str, List[str]] = field(default_factory=_default_synonym_map)
    random_seed: Optional[int] = None
    allow_padding: bool = True  # 是否允许在失败时追加原始片段稳定水印
    padding_sentence_ratio: float = 0.2  # 尝试追加的原句占比


@dataclass
class ModificationRecord:
    """记录单次文本修改的详细信息。"""

    modification_type: str
    description: str
    z_score: float
    prediction: bool
    green_fraction: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": self.modification_type,
            "description": self.description,
            "z_score": float(self.z_score),
            "prediction": bool(self.prediction),
            "green_fraction": float(self.green_fraction),
        }


@dataclass
class PiggybackAttackResult:
    """猪背攻击的完整结果。"""

    success: bool
    original_detection: Dict[str, object]
    final_detection: Dict[str, object]
    original_text: str
    attacked_text: str
    modifications: List[ModificationRecord] = field(default_factory=list)
    failed_insertions: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "success": self.success,
            "original_detection": self._convert_detection(self.original_detection),
            "final_detection": self._convert_detection(self.final_detection),
            "original_text": self.original_text,
            "attacked_text": self.attacked_text,
            "modifications": [m.to_dict() for m in self.modifications],
            "failed_insertions": list(self.failed_insertions),
            "metadata": self.metadata,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _convert_detection(det_dict: Dict[str, object]) -> Dict[str, object]:
        converted = {}
        for key, value in det_dict.items():
            if isinstance(value, (int, float, bool, str)):
                converted[key] = value
            else:
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = str(value)
        return converted


class PiggybackAttackSimulator:
    """
    猪背攻击模拟器。

    负责：
      1. 生成带水印的文本（可选）
      2. 对给定文本执行轻量级语义扰动或定向插入
      3. 评估扰动前后水印检测的变化
    """

    def __init__(self, config: PiggybackAttackConfig):
        self.config = config
        if config.random_seed is not None:
            random.seed(config.random_seed)

        self._experiment = HybridWatermarkExperiment(
            model_nickname=config.model_nickname,
            device=config.device,
        )
        self._detector = self._build_detector()

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def generate_watermarked_text(
        self,
        prompt: str,
        max_new_tokens: int = 180,
        temperature: float = 0.7,
    ) -> Tuple[str, Dict[str, object]]:
        """
        使用与目标模型一致的水印配置生成文本。

        Returns:
            生成的文本以及对应的检测结果。
        """
        processor = self._experiment.create_watermark_processor(
            gamma=self.config.gamma,
            delta=self.config.delta,
            seeding_scheme=self.config.seeding_scheme,
            hash_key=self.config.hash_key,
        )
        generated = self._experiment.generate_with_watermark(
            prompt=prompt,
            watermark_processor=processor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        detection = self._detect(generated)
        return generated, detection

    def detect_text(self, text: str) -> Dict[str, object]:
        """对外暴露的检测接口。"""
        return self._detect(text)

    def simulate_attack(
        self,
        watermarked_text: str,
        target_phrases: Optional[Sequence[str]] = None,
        synonym_targets: Optional[Dict[str, str]] = None,
    ) -> PiggybackAttackResult:
        """
        执行猪背攻击。

        Args:
            watermarked_text: 原始带水印文本。
            target_phrases: 需要插入/拼接的高风险片段列表。
            synonym_targets: 指定的替换映射，用于覆盖默认的近义词策略。
        """
        target_phrases = list(target_phrases or [])
        synonym_targets = synonym_targets or {}

        original_detection = self._detect(watermarked_text)
        attacked_text = watermarked_text
        modifications: List[ModificationRecord] = []
        failed_insertions: List[str] = []

        # 1) 定向插入
        for phrase in target_phrases:
            attacked_text, det, success = self._try_insert_phrase(attacked_text, phrase)
            if success:
                modifications.append(
                    ModificationRecord(
                        modification_type="insertion",
                        description=f"Inserted phrase: {phrase}",
                        z_score=det.get("z_score", 0.0),
                        prediction=det.get("prediction", False),
                        green_fraction=det.get("green_fraction", 0.0),
                    )
                )
            else:
                failed_insertions.append(phrase)

        # 2) 近义词扰动
        attacked_text, synonym_mods = self._apply_synonym_attack(
            attacked_text,
            synonym_targets,
        )
        modifications.extend(synonym_mods)

        final_detection = self._detect(attacked_text)
        success = bool(final_detection.get("prediction", False))

        # 若要求z-score与初始相比不能过度下降，则进行校验
        margin = self.config.detection_margin
        if margin is not None and margin >= 0:
            z_before = float(original_detection.get("z_score", 0.0))
            z_after = float(final_detection.get("z_score", 0.0))
            success = success and (z_after + margin >= z_before)

        metadata = {
            "target_phrases": target_phrases,
            "synonym_targets": synonym_targets,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_nickname": self.config.model_nickname,
            "gamma": self.config.gamma,
            "delta": self.config.delta,
            "seeding_scheme": self.config.seeding_scheme,
            "hash_key": self.config.hash_key,
            "z_threshold": self.config.z_threshold,
        }

        return PiggybackAttackResult(
            success=success,
            original_detection=original_detection,
            final_detection=final_detection,
            original_text=watermarked_text,
            attacked_text=attacked_text,
            modifications=modifications,
            failed_insertions=failed_insertions,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_detector(self) -> WatermarkDetector:
        """按照当前配置构造水印检测器。"""
        seeding_scheme = self.config.seeding_scheme
        if self.config.hash_key != 15485863:
            parts = seeding_scheme.split("-")
            if len(parts) >= 6:
                parts[-1] = str(self.config.hash_key)
                seeding_scheme = "-".join(parts)
            else:
                seeding_scheme = f"ff-anchored_minhash_prf-4-True-{self.config.hash_key}"

        return WatermarkDetector(
            vocab=list(self._experiment.tokenizer.get_vocab().values()),
            gamma=self.config.gamma,
            seeding_scheme=seeding_scheme,
            device=self.config.device,
            tokenizer=self._experiment.tokenizer,
            z_threshold=self.config.z_threshold,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )

    def _detect(self, text: str) -> Dict[str, object]:
        """运行水印检测并返回字典结果（统一转换为Python原生类型）。"""
        detection = self._detector.detect(text)
        cleaned = {}
        for key, value in detection.items():
            if isinstance(value, (int, float, bool, str)):
                cleaned[key] = value
            else:
                try:
                    cleaned[key] = float(value)
                except (TypeError, ValueError):
                    cleaned[key] = str(value)
        return cleaned

    # ----------------------- 插入策略 -----------------------
    def _try_insert_phrase(
        self,
        base_text: str,
        phrase: str,
    ) -> Tuple[str, Dict[str, object], bool]:
        """
        以多种策略尝试插入目标短语。
        成功后返回新文本、检测结果以及True；失败则返回原文本、最后一次检测结果、False。
        """
        strategies = [
            self._insert_append,
            self._insert_prepend,
            self._insert_middle,
        ]

        # 随机化尝试顺序，避免总是固定模式
        random.shuffle(strategies)

        best_text = base_text
        best_detection = self._detect(base_text)
        best_success = False

        for strategy in strategies:
            attempts = 0
            for connector in self._candidate_connectors():
                attempts += 1
                if attempts > self.config.max_phrase_attempts:
                    break

                candidate_text = strategy(base_text, connector, phrase)
                detection = self._detect(candidate_text)
                if self._is_detection_acceptable(detection, best_detection):
                    return candidate_text, detection, True

                # 若允许，通过追加原始句子尝试恢复检测
                if self.config.allow_padding:
                    padded_text = self._apply_padding(candidate_text, base_text)
                    detection_padded = self._detect(padded_text)
                    if self._is_detection_acceptable(detection_padded, best_detection):
                        return padded_text, detection_padded, True

        return best_text, best_detection, best_success

    def _candidate_connectors(self) -> Iterable[str]:
        """生成插入使用的连接短语。"""
        connectors = self.config.connectors or ["此外"]
        # 总是确保存在空连接选项
        yield ""
        for connector in connectors:
            yield connector

    def _is_detection_acceptable(
        self,
        new_detection: Dict[str, object],
        baseline_detection: Dict[str, object],
    ) -> bool:
        """判断检测结果是否仍可接受。"""
        prediction = bool(new_detection.get("prediction", False))
        if not prediction:
            return False
        baseline_z = float(baseline_detection.get("z_score", 0.0))
        new_z = float(new_detection.get("z_score", 0.0))
        if self.config.detection_margin is None:
            return True
        return new_z + self.config.detection_margin >= baseline_z

    def _insert_append(self, base_text: str, connector: str, phrase: str) -> str:
        """在文本末尾追加短语。"""
        connector_text = f"{connector}，" if connector else ""
        return f"{base_text.rstrip()}。{connector_text}{phrase.strip()}"

    def _insert_prepend(self, base_text: str, connector: str, phrase: str) -> str:
        """在文本开头插入短语。"""
        connector_text = f"{connector}，" if connector else ""
        return f"{connector_text}{phrase.strip()}。{base_text.lstrip()}"

    def _insert_middle(self, base_text: str, connector: str, phrase: str) -> str:
        """将短语插入到段落中部。"""
        sentences = self._split_sentences(base_text)
        if len(sentences) < 2:
            return self._insert_append(base_text, connector, phrase)

        midpoint = len(sentences) // 2
        connector_text = f"{connector}，" if connector else ""
        inserted_sentence = f"{connector_text}{phrase.strip()}。"
        new_sentences = sentences[:midpoint] + [inserted_sentence] + sentences[midpoint:]
        return "".join(new_sentences)

    def _split_sentences(self, text: str) -> List[str]:
        """按句号/问号/感叹号划分文本。"""
        pattern = r"(?<=[。！？!?])\s*"
        sentences = [s for s in re.split(pattern, text) if s]
        if not sentences:
            sentences = [text]
        # 重新附加句末标点
        restored = []
        for sent in sentences:
            if sent[-1] in "。！？!?":
                restored.append(sent)
            else:
                restored.append(f"{sent}。")
        return restored

    def _apply_padding(self, candidate_text: str, base_text: str) -> str:
        """向候选文本追加原始句子以恢复水印特征。"""
        sentences = self._split_sentences(base_text)
        if not sentences:
            return candidate_text

        max_sentences = max(1, int(len(sentences) * self.config.padding_sentence_ratio))
        samples = random.sample(sentences, k=min(len(sentences), max_sentences))
        padding = "".join(samples)
        return f"{candidate_text}{padding}"

    # ----------------------- 近义词扰动 -----------------------
    def _apply_synonym_attack(
        self,
        base_text: str,
        synonym_targets: Dict[str, str],
    ) -> Tuple[str, List[ModificationRecord]]:
        """
        对文本执行限定数量的近义词替换。
        如果synonym_targets提供了具体映射，会优先使用；否则回退到默认词典中随机挑选。
        """
        modifications: List[ModificationRecord] = []
        detection_baseline = self._detect(base_text)
        current_text = base_text

        # 合并映射：显式指定的替换 + 默认词典
        replacement_pool: List[Tuple[str, str]] = []
        for origin, target in synonym_targets.items():
            replacement_pool.append((origin, target))

        for word, candidates in self.config.synonym_map.items():
            # 过滤掉长度过短导致重复替换的词
            for cand in candidates:
                if cand and cand != word:
                    replacement_pool.append((word, cand))

        random.shuffle(replacement_pool)

        replacements_applied = 0
        for source, target in replacement_pool:
            if replacements_applied >= self.config.max_synonym_replacements:
                break
            updated, changed = self._replace_word(current_text, source, target)
            if not changed:
                continue
            detection = self._detect(updated)
            if self._is_detection_acceptable(detection, detection_baseline):
                current_text = updated
                detection_baseline = detection
                replacements_applied += 1
                modifications.append(
                    ModificationRecord(
                        modification_type="replacement",
                        description=f"{source} -> {target}",
                        z_score=detection.get("z_score", 0.0),
                        prediction=detection.get("prediction", False),
                        green_fraction=detection.get("green_fraction", 0.0),
                    )
                )

        return current_text, modifications

    WORD_BOUNDARY = re.compile(r"(?u)(?<!\w)({})(?!\w)")

    def _replace_word(self, text: str, source: str, target: str) -> Tuple[str, bool]:
        """在尊重词边界的情况下替换词语。"""
        pattern = self.WORD_BOUNDARY.pattern.format(re.escape(source))
        regex = re.compile(pattern)
        replaced_text, count = regex.subn(target, text)
        return replaced_text, count > 0

    # ----------------------- 结果保存 -----------------------
    def save_result(
        self,
        result: PiggybackAttackResult,
        output_dir: str,
        base_filename: Optional[str] = None,
    ) -> str:
        """
        将攻击结果保存为JSON，并返回文件路径。
        同时保存原始文本和攻击后文本，便于审计。
        """
        os.makedirs(output_dir, exist_ok=True)
        if base_filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            base_filename = f"piggyback_attack_{timestamp}"

        json_path = os.path.join(output_dir, f"{base_filename}.json")
        text_before_path = os.path.join(output_dir, f"{base_filename}_before.txt")
        text_after_path = os.path.join(output_dir, f"{base_filename}_after.txt")

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(result.to_json())

        with open(text_before_path, "w", encoding="utf-8") as f:
            f.write(result.original_text)

        with open(text_after_path, "w", encoding="utf-8") as f:
            f.write(result.attacked_text)

        return json_path
