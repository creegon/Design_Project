"""
Multi-LLM watermark chain experiment.

构建一个多模型链路：
- 生成模型负责带水印生成（默认 llama-3.2-3b）。
- 改写模型负责顺次改写（默认 qwen-3-4b）。

提供能力：
1. 按给定提示生成带水印文本。
2. 通过一个或多个改写模型顺次改写文本。
3. 在改写前后检测水印，记录 z-score / p-value 等指标。
4. 计算语义相似度和水印衰减程度。
5. 支持在不同模型组合间对比水印存活率。

重构版本：继承 BaseExperiment，去除重复代码。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict, List, Optional, cast

import torch
from transformers import AutoTokenizer, LogitsProcessorList

from base_experiment import BaseExperiment, WatermarkDetection


class MultiLLMChainExperiment(BaseExperiment):
    """多模型 LLM 链路实验器。"""

    def __init__(
        self,
        generator_default: str = "llama-3.2-3b",
        paraphraser_defaults: Optional[List[str]] = None,
        device: Optional[str] = None,
        use_api: bool = False,
    ) -> None:
        """
        初始化多模型链路实验

        Args:
            generator_default: 默认生成模型昵称
            paraphraser_defaults: 默认改写模型列表
            device: 运行设备
            use_api: 是否使用 API 模式
        """
        super().__init__(
            default_model=generator_default,
            device=device,
            results_subdir="multi_llm_chain_results",
            use_api=use_api,
        )
        self.generator_default = generator_default
        self.paraphraser_defaults = paraphraser_defaults or ["qwen-3-4b"]

    # ------------------------------------------------------------------
    # Paraphrase (链路特有功能)
    # ------------------------------------------------------------------
    
    # 严格的改写指令，防止模型输出额外内容
    # /no_think 用于禁用 Qwen3 的思考模式
    PARAPHRASE_INSTRUCTION = """/no_think
Paraphrase the following text to preserve its meaning.
CRITICAL RULES:
1. Output ONLY the rewritten text, nothing else.
2. Do NOT include any notes, explanations, labels, or introductory phrases.
3. Do NOT include "Output:", "Here is", or any similar prefixes.
4. Start your response directly with the paraphrased content."""

    def paraphrase_text(
        self,
        text: str,
        paraphraser_nickname: str,
        instruction: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        watermark_config: Optional[Dict] = None,
    ) -> tuple[str, Dict]:
        """
        使用指定模型改写文本（自动选择本地或 API 模式）

        Args:
            text: 原始文本
            paraphraser_nickname: 改写模型昵称
            instruction: 改写指令（默认使用严格约束）
            max_new_tokens: 最大生成 token 数（默认为原文 token 数的 1.3 倍）
            temperature: 采样温度
            watermark_config: 可选的水印配置（改写时嵌入新水印）

        Returns:
            (paraphrased_text, metadata) 元组
        """
        instr = instruction or self.PARAPHRASE_INSTRUCTION
        
        # 动态计算 max_new_tokens（如果未指定）
        if max_new_tokens is None:
            # 使用生成模型的 tokenizer 估算原文 token 数
            tokenizer, _ = self.load_tokenizer_only(self.generator_default)
            text_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            max_new_tokens = int(text_tokens * 1.3) + 10  # 1.3 倍 + 少量余量
            max_new_tokens = max(max_new_tokens, 50)  # 最少 50 tokens
        
        # API 模式
        if self.use_api and self._api_client:
            return self._paraphrase_text_api(
                text, paraphraser_nickname, instr, max_new_tokens, temperature
            )
        
        # 本地模式
        return self._paraphrase_text_local(
            text, paraphraser_nickname, instr, max_new_tokens, temperature, watermark_config
        )

    def _paraphrase_text_api(
        self,
        text: str,
        paraphraser_nickname: str,
        instruction: str,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[str, Dict]:
        """通过 API 进行改写"""
        # 构建 paraphrase prompt
        prompt_text = f"""{instruction}

Text:
{text}

Output:"""

        result = self._api_client.generate(
            model_nickname=paraphraser_nickname,
            prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            with_watermark=False,  # paraphrase 不加水印
        )
        
        paraphrased_text = self._clean_paraphrase_output(result["generated_text"])
        
        # 不需要加载 tokenizer，只需获取 model info 用于 metadata
        info = self.config_manager.get_model_info_by_nickname(paraphraser_nickname)
        
        metadata = {
            "model": paraphraser_nickname,
            "model_identifier": info["model_identifier"] if info else paraphraser_nickname,
            "instruction": instruction,
            "original_text": text,
            "paraphrase_watermark_config": None,
            "paraphrased_at": datetime.now().isoformat(),
            "token_usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
            },
            "api_mode": True,
        }
        return paraphrased_text, metadata

    def _paraphrase_text_local(
        self,
        text: str,
        paraphraser_nickname: str,
        instruction: str,
        max_new_tokens: int,
        temperature: float,
        watermark_config: Optional[Dict],
    ) -> tuple[str, Dict]:
        """本地进行改写"""
        tokenizer, model, info = self.load_model(paraphraser_nickname)
        
        prompt_text = f"""{instruction}

Text:
{text}

Output:"""

        inputs = tokenizer(prompt_text, return_tensors="pt").to(self.device)

        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # 如果提供了水印配置，在改写时也嵌入水印
        if watermark_config is not None:
            processor = self.create_watermark_processor(tokenizer, watermark_config)
            generate_kwargs["logits_processor"] = LogitsProcessorList([processor])

        with torch.no_grad():
            output_tokens = model.generate(**inputs, **generate_kwargs)

        generated_tokens = output_tokens[:, inputs["input_ids"].shape[-1] :]
        paraphrased_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        
        # 后处理：移除可能的前缀
        paraphrased_text = self._clean_paraphrase_output(paraphrased_text)

        metadata = {
            "model": info["nickname"],
            "model_identifier": info["model_identifier"],
            "instruction": instruction,
            "original_text": text,
            "paraphrase_watermark_config": watermark_config,
            "paraphrased_at": datetime.now().isoformat(),
            "token_usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(generated_tokens.shape[-1]),
            },
        }
        return paraphrased_text, metadata

    @staticmethod
    def _clean_paraphrase_output(text: str) -> str:
        """清理改写输出中的多余前缀和思考模式内容"""
        cleaned = text.strip()
        
        # 如果存在 "Output:" 或类似标记，只保留其后的内容
        # 这是处理 Qwen3 思考模式输出的关键
        output_markers = ["Output:", "output:", "OUTPUT:", "Final output:", "Paraphrased:"]
        for marker in output_markers:
            if marker in cleaned:
                # 取最后一个 marker 后的内容（避免思考过程中的误匹配）
                parts = cleaned.rsplit(marker, 1)
                if len(parts) > 1:
                    cleaned = parts[1].strip()
                    break
        
        # 常见的无用前缀
        prefixes_to_remove = [
            "Here is the paraphrased version:",
            "Here is the rewritten text:",
            "Here's the paraphrased text:",
            "Here's the rewritten version:",
            "Paraphrased version:",
            "Rewritten text:",
            "Here is:",
            "Here's:",
        ]
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # 移除可能的引号包裹
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1].strip()
        
        # 检测并移除段落级重复
        # 如果文本后半部分与前半部分高度相似，截断到第一次出现的位置
        cleaned = MultiLLMChainExperiment._remove_paragraph_repetition(cleaned)
        
        return cleaned
    
    @staticmethod
    def _remove_paragraph_repetition(text: str) -> str:
        """移除文本中的段落级和行级重复"""
        # 按行分割（保留诗歌格式）
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) < 2:
            return text
        
        # 检测行重复：如果一行在后面再次出现，截断到那里
        seen = {}  # fingerprint -> first occurrence index
        result_lines = []
        for i, line in enumerate(lines):
            # 标准化行用于比较（去掉标点和空格的差异）
            normalized = ''.join(c.lower() for c in line if c.isalnum() or c.isspace()).strip()
            # 取前40个字符作为指纹
            fingerprint = normalized[:40] if len(normalized) > 40 else normalized
            
            # 跳过空指纹
            if not fingerprint:
                result_lines.append(line)
                continue
            
            if fingerprint in seen:
                # 检查是否是连续重复（相邻行重复）
                if seen[fingerprint] == i - 1:
                    # 连续重复，停止
                    break
                # 非连续重复，但只对足够长的行去重
                elif len(fingerprint) > 15:
                    break
            
            seen[fingerprint] = i
            result_lines.append(line)
        
        return '\n'.join(result_lines)

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
        paraphrase_instruction: Optional[str] = None,
        prefetched: Optional[Dict[str, object]] = None,
    ) -> Dict:
        """
        运行单次链路实验：生成 → 改写 → 检测

        Args:
            prompt: 生成提示词
            generator_model: 生成模型（默认使用 generator_default）
            paraphraser_models: 改写模型列表
            watermark_config: 水印配置
            z_threshold: 检测阈值
            paraphrase_instruction: 改写指令（默认使用严格约束）
            prefetched: 预先生成的文本（跳过生成步骤）

        Returns:
            实验结果字典
        """
        paraphraser_models = paraphraser_models or self.paraphraser_defaults
        generator_model = generator_model or self.generator_default

        # 生成或使用预取的文本
        if prefetched is not None:
            watermarked_text = cast(str, prefetched["watermarked_text"])
            generation_meta = dict(cast(Dict[str, object], prefetched["generation_metadata"]))
            tokenizer = cast(AutoTokenizer, prefetched["tokenizer"])
            watermark_conf = cast(Dict[str, object], generation_meta.get("watermark_config") or watermark_config or {})
        else:
            watermarked_text, generation_meta, tokenizer = self.generate_with_watermark(
                prompt,
                model_nickname=generator_model,
                watermark_config=watermark_config,
            )
            watermark_conf = generation_meta["watermark_config"]

        # 检测原始文本水印
        original_detection = self.detect_watermark(
            watermarked_text,
            tokenizer=tokenizer,
            config=watermark_conf,
            z_threshold=z_threshold,
        )

        # 对每个改写模型进行改写和检测
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
                config=watermark_conf,
                z_threshold=z_threshold,
            )
            similarity = self.calculate_similarity(watermarked_text, paraphrased_text)
            watermark_decay = original_detection.z_score - paraphrased_detection.z_score
            retention = (
                paraphrased_detection.z_score / original_detection.z_score
                if original_detection.z_score != 0
                else 0.0
            )

            paraphraser_results.append({
                "paraphraser": paraphraser,
                "paraphrased_text": paraphrased_text,
                "metadata": paraphrase_meta,
                "detection": paraphrased_detection.__dict__,
                "semantic_similarity": similarity,
                "watermark_decay": watermark_decay,
                "z_score_retention": retention,
            })

        # 汇总统计
        survival_count = sum(1 for r in paraphraser_results if r["detection"]["prediction"])
        survival_rate = survival_count / len(paraphraser_results) if paraphraser_results else 0.0

        return {
            "prompt": prompt,
            "generator_model": generator_model,
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
                    sum(r["semantic_similarity"] for r in paraphraser_results) / len(paraphraser_results)
                    if paraphraser_results else 0.0
                ),
                "average_decay": (
                    sum(r["watermark_decay"] for r in paraphraser_results) / len(paraphraser_results)
                    if paraphraser_results else 0.0
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }

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
            paraphraser_watermark_configs: 每个改写模型对应的水印配置列表
        """
        paraphraser_models = paraphraser_models or self.paraphraser_defaults
        generator_model = generator_model or self.generator_default

        # 生成带水印文本
        watermarked_text, generation_meta, tokenizer = self.generate_with_watermark(
            prompt,
            model_nickname=generator_model,
            watermark_config=generator_watermark_config,
        )
        generator_watermark_conf = generation_meta["watermark_config"]

        # 检测原始文本水印
        original_detection = self.detect_watermark(
            watermarked_text,
            tokenizer=tokenizer,
            config=generator_watermark_conf,
            z_threshold=z_threshold,
        )

        paraphraser_results = []
        for idx, paraphraser in enumerate(paraphraser_models):
            # 获取改写器水印配置
            paraphrase_wm_config = None
            if paraphraser_watermark_configs and idx < len(paraphraser_watermark_configs):
                paraphrase_wm_config = paraphraser_watermark_configs[idx]

            # 改写文本
            paraphrased_text, paraphrase_meta = self.paraphrase_text(
                watermarked_text,
                paraphraser,
                instruction=paraphrase_instruction,
                watermark_config=paraphrase_wm_config,
            )

            # 用原始密钥检测
            detection_gen_key = self.detect_watermark(
                paraphrased_text,
                tokenizer=tokenizer,
                config=generator_watermark_conf,
                z_threshold=z_threshold,
            )

            # 用改写器密钥检测（如有）
            detection_para_key = None
            if paraphrase_wm_config is not None:
                detection_para_key = self.detect_watermark(
                    paraphrased_text,
                    tokenizer=tokenizer,
                    config=paraphrase_wm_config,
                    z_threshold=z_threshold,
                )

            similarity = self.calculate_similarity(watermarked_text, paraphrased_text)
            watermark_decay = original_detection.z_score - detection_gen_key.z_score
            retention = (
                detection_gen_key.z_score / original_detection.z_score
                if original_detection.z_score != 0 else 0.0
            )

            paraphraser_results.append({
                "paraphraser": paraphraser,
                "paraphrased_text": paraphrased_text,
                "metadata": paraphrase_meta,
                "paraphrase_watermark_config": paraphrase_wm_config,
                "detection_with_generator_key": detection_gen_key.__dict__,
                "detection_with_paraphraser_key": detection_para_key.__dict__ if detection_para_key else None,
                "semantic_similarity": similarity,
                "generator_watermark_decay": watermark_decay,
                "generator_z_score_retention": retention,
            })

        # 统计
        gen_survival = sum(1 for r in paraphraser_results if r["detection_with_generator_key"]["prediction"])
        para_detection = sum(
            1 for r in paraphraser_results
            if r["detection_with_paraphraser_key"] and r["detection_with_paraphraser_key"]["prediction"]
        )

        return {
            "experiment_type": "watermarked_paraphrase_chain",
            "prompt": prompt,
            "generator_model": generator_model,
            "paraphraser_models": paraphraser_models,
            "generator_watermark_config": generator_watermark_conf,
            "paraphraser_watermark_configs": paraphraser_watermark_configs,
            "z_threshold": z_threshold,
            "generated_text": watermarked_text,
            "generation_metadata": generation_meta,
            "original_detection": original_detection.__dict__,
            "paraphraser_results": paraphraser_results,
            "summary": {
                "generator_watermark_survival_rate": gen_survival / len(paraphraser_results) if paraphraser_results else 0.0,
                "generator_watermark_survived_count": gen_survival,
                "paraphraser_watermark_detection_rate": para_detection / len(paraphraser_results) if paraphraser_results else 0.0,
                "paraphraser_watermark_detected_count": para_detection,
                "average_similarity": (
                    sum(r["semantic_similarity"] for r in paraphraser_results) / len(paraphraser_results)
                    if paraphraser_results else 0.0
                ),
                "average_generator_decay": (
                    sum(r["generator_watermark_decay"] for r in paraphraser_results) / len(paraphraser_results)
                    if paraphraser_results else 0.0
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }

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
        """跨模型比较水印存活率"""
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

        survival_rates = [r["summary"]["watermark_survival_rate"] for r in comparison_results]

        return {
            "summary": {
                "prompt": prompt,
                "generator_models": generator_models,
                "paraphraser_models": paraphraser_models,
                "average_survival_rate": sum(survival_rates) / len(survival_rates) if survival_rates else 0.0,
                "highest_survival": max(survival_rates) if survival_rates else 0.0,
                "lowest_survival": min(survival_rates) if survival_rates else 0.0,
            },
            "individual_results": comparison_results,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def save_results(self, results: Dict, filename: Optional[str] = None) -> str:
        """保存结果（覆盖父类方法以使用特定前缀）"""
        if filename is None:
            filename = f"multi_llm_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return super().save_results(results, filename)

    @staticmethod
    def print_summary(result: Dict) -> None:
        """打印实验摘要"""
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
# CLI
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
        help="生成模型昵称",
    )
    parser.add_argument(
        "--paraphrasers",
        type=str,
        default="qwen-3-4b",
        help="逗号分隔的改写模型列表",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="多生成模型比较模式",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="检测阈值",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="使用 API 模式（通过 model_server 访问模型）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generators = [n.strip() for n in args.generator.split(",") if n.strip()]
    paraphrasers = [n.strip() for n in args.paraphrasers.split(",") if n.strip()]

    experiment = MultiLLMChainExperiment(
        generator_default=generators[0],
        paraphraser_defaults=paraphrasers,
        use_api=args.use_api,
    )

    if args.compare and len(generators) > 1:
        result = experiment.compare_across_models(
            prompt=args.prompt,
            generator_models=generators,
            paraphraser_models=paraphrasers,
            z_threshold=args.z_threshold,
        )
        experiment.save_results(result)
        for individual in result["individual_results"]:
            experiment.print_summary(individual)
        print("总体存活率对比:")
        for item in result["individual_results"]:
            print(
                f"  {item['generator_model']}: "
                f"survival={item['summary']['watermark_survival_rate']:.2%}, "
                f"avg decay={item['summary']['average_decay']:.4f}"
            )
    else:
        result = experiment.run_chain(
            prompt=args.prompt,
            generator_model=generators[0],
            paraphraser_models=paraphrasers,
            z_threshold=args.z_threshold,
        )
        experiment.print_summary(result)
        experiment.save_results(result)


if __name__ == "__main__":
    main()

