"""
Interactive CLI for simulating piggyback watermark attacks.

Provides both manual configuration and preset scenarios so that experiments
can be reproduced quickly (e.g., the Appendix F demo).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from watermark_attack.piggyback_attack import (
    PiggybackAttackConfig,
    PiggybackAttackSimulator,
)
from watermark_attack.piggyback_presets import get_preset, list_presets


DEFAULT_GENERATION_PROMPT = (
    "Provide a vivid historical timeline of computing breakthroughs, cite exact years such as 1945, "
    "1950, 1970, and mention numerical facts like transistor counts, processing speeds, or workforce numbers."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Piggyback watermark attack interactive CLI"
    )
    parser.add_argument("--model", default="llama-2-7b", help="Model nickname from model_config.json")
    parser.add_argument("--device", default=None, help="Execution device, defaults to auto detect")
    parser.add_argument("--gamma", type=float, default=0.5, help="Watermark gamma")
    parser.add_argument("--delta", type=float, default=2.0, help="Watermark delta (generation)")
    parser.add_argument("--hash-key", type=int, default=15485863, help="Watermark hash key")
    parser.add_argument("--z-threshold", type=float, default=4.0, help="Detection threshold")
    parser.add_argument("--detection-margin", type=float, default=0.5, help="Allowed z-score drop after attack")
    parser.add_argument("--max-new-tokens", type=int, default=180, help="Generation max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("watermark_attack", "watermark_attack_results"),
        help="Directory for saving experiment artifacts",
    )
    return parser.parse_args()


class WatermarkAttackCLI:
    def __init__(self, args: argparse.Namespace):
        device = args.device
        if device is None:
            device = "cuda" if PiggybackAttackCLIUtils.has_cuda() else "cpu"

        config = PiggybackAttackConfig(
            model_nickname=args.model,
            device=device,
            gamma=args.gamma,
            delta=args.delta,
            hash_key=args.hash_key,
            z_threshold=args.z_threshold,
            detection_margin=args.detection_margin,
        )
        self.args = args
        self.config = config
        self.simulator = PiggybackAttackSimulator(config)

    def run(self) -> None:
        PiggybackAttackCLIUtils.print_banner("Piggyback Watermark Attack")
        print(f"Model nickname : {self.config.model_nickname}")
        print(f"Device         : {self.config.device}")
        print(f"Watermark      : gamma={self.config.gamma}, delta={self.config.delta}, hash_key={self.config.hash_key}")
        print(f"Detection      : z_threshold={self.config.z_threshold}, margin={self.config.detection_margin}\n")

        base_text = self._acquire_base_text()
        base_detection = self.simulator.detect_text(base_text)
        PiggybackAttackCLIUtils.print_detection("Base detection", base_detection)

        attack_cfg = self._collect_attack_configuration()

        result = self.simulator.simulate_attack(
            watermarked_text=base_text,
            target_phrases=attack_cfg.get("target_phrases"),
            synonym_targets=attack_cfg.get("synonym_map"),
            direct_replacements=attack_cfg.get("direct_replacements"),
            use_default_antonyms=attack_cfg.get("use_default_antonyms", False),
            apply_numeric_heuristic=attack_cfg.get("apply_numeric_heuristic", False),
            numeric_multiplier=attack_cfg.get("numeric_multiplier"),
            numeric_offset=attack_cfg.get("numeric_offset"),
        )

        PiggybackAttackCLIUtils.print_detection("\nFinal detection", result.final_detection)
        print(f"\nAttack success : {'Yes' if result.success else 'No'}")
        print(f"Failed inserts : {result.failed_insertions if result.failed_insertions else 'None'}")
        print(f"Failed replace : {result.failed_replacements if result.failed_replacements else 'None'}\n")

        print("Modification timeline:")
        if result.modifications:
            for mod in result.modifications:
                print(
                    f"  - [{mod.modification_type}] {mod.description} "
                    f"(z={mod.z_score:.3f}, green={mod.green_fraction:.2%})"
                )
        else:
            print("  None")

        self._maybe_save(result)

    # ------------------------------------------------------------------
    def _collect_attack_configuration(self) -> Dict[str, object]:
        print("\nAttack configuration mode:")
        print("  1) Manual input")
        print("  2) Preset scenario")
        choice = input("Pick option (1/2): ").strip()
        if choice == "2":
            return self._collect_preset_attack()
        return self._collect_manual_attack()

    def _collect_manual_attack(self) -> Dict[str, object]:
        phrases = self._prompt_target_phrases()
        direct_replacements = self._prompt_direct_replacements()
        synonym_map = self._prompt_synonym_pairs()
        use_antonyms = input("\nUse default antonym map? (y/n): ").strip().lower() == "y"
        apply_numeric, multiplier, offset = self._prompt_numeric_strategy()

        return {
            "target_phrases": phrases,
            "direct_replacements": direct_replacements,
            "synonym_map": synonym_map,
            "use_default_antonyms": use_antonyms,
            "apply_numeric_heuristic": apply_numeric,
            "numeric_multiplier": multiplier,
            "numeric_offset": offset,
        }

    def _collect_preset_attack(self) -> Dict[str, object]:
        names = list_presets()
        if not names:
            print("No preset scenarios found. Fallback to manual mode.")
            return self._collect_manual_attack()

        print("\nAvailable presets:")
        for idx, name in enumerate(names, start=1):
            preset = get_preset(name) or {}
            desc = preset.get("description", "")
            print(f"  {idx}) {name} - {desc}")

        selection = input("Choose preset number: ").strip()
        try:
            index = int(selection) - 1
            if not (0 <= index < len(names)):
                raise ValueError
        except ValueError:
            print("Invalid selection. Fallback to manual mode.")
            return self._collect_manual_attack()

        preset_name = names[index]
        preset = get_preset(preset_name) or {}
        print(f"\nUsing preset: {preset_name}")

        apply_numeric = bool(preset.get("numeric_multiplier") or preset.get("numeric_offset"))

        return {
            "target_phrases": preset.get("insert_phrases", []),
            "direct_replacements": preset.get("direct_replacements", []),
            "synonym_map": preset.get("synonym_overrides", {}),
            "use_default_antonyms": bool(preset.get("use_antonyms", False)),
            "apply_numeric_heuristic": apply_numeric,
            "numeric_multiplier": preset.get("numeric_multiplier"),
            "numeric_offset": preset.get("numeric_offset"),
        }

    # ------------------------------------------------------------------
    def _acquire_base_text(self) -> str:
        while True:
            print("\nAcquire base text:")
            print("  1) Generate watermarked text")
            print("  2) Load from file")
            print("  3) Paste manually")
            choice = input("Select option (1/2/3): ").strip()

            if choice == "1":
                prompt = input("\nEnter generation prompt: ").strip()
                if not prompt:
                    prompt = DEFAULT_GENERATION_PROMPT
                    print("[Info] Using default numeric-rich prompt.")
                text, detection = self.simulator.generate_watermarked_text(
                    prompt=prompt,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature,
                )
                PiggybackAttackCLIUtils.print_detection("\nGeneration detection", detection)
                print("\nGeneration preview (first 300 chars):")
                PiggybackAttackCLIUtils.print_snippet(text)
                if input("\nUse this text? (y/n): ").strip().lower() == "y":
                    return text
            elif choice == "2":
                path = input("\nEnter path to text file: ").strip()
                if not os.path.exists(path):
                    print("File does not exist.")
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if not text:
                    print("File is empty.")
                    continue
                return text
            elif choice == "3":
                print("\nPaste text, then input a single line 'END' to finish:")
                lines: List[str] = []
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    lines.append(line)
                text = "\n".join(lines).strip()
                if not text:
                    print("Text is empty.")
                    continue
                return text
            else:
                print("Invalid option.")

    def _prompt_target_phrases(self) -> List[str]:
        raw = input("\nPhrases to insert (semicolon separated, leave empty to skip): ").strip()
        if not raw:
            return []
        return [seg.strip() for seg in raw.replace("；", ";").split(";") if seg.strip()]

    def _prompt_direct_replacements(self) -> List[Tuple[str, str]]:
        raw = input("\nDirect replacements (format: src->dst, src2->dst2). Leave empty to skip: ").strip()
        if not raw:
            return []
        replacements: List[Tuple[str, str]] = []
        for part in raw.split(","):
            if "->" not in part:
                continue
            src, dst = part.split("->", 1)
            src = src.strip()
            dst = dst.strip()
            if src and dst:
                replacements.append((src, dst))
        return replacements

    def _prompt_synonym_pairs(self) -> Dict[str, str]:
        raw = input("\nSynonym replacements (format: src->dst, src2->dst2). Leave empty for defaults: ").strip()
        if not raw:
            return {}
        mapping: Dict[str, str] = {}
        for part in raw.split(","):
            if "->" not in part:
                continue
            src, dst = part.split("->", 1)
            src = src.strip()
            dst = dst.strip()
            if src and dst:
                mapping[src] = dst
        return mapping

    def _prompt_numeric_strategy(self) -> Tuple[bool, Optional[float], Optional[int]]:
        raw = input(
            "\nNumeric heuristic (enter multiplier or multiplier,offset; empty to skip): "
        ).strip()
        if not raw:
            return False, None, None
        parts = raw.split(",")
        try:
            multiplier = float(parts[0])
        except ValueError:
            print("Invalid multiplier, numeric heuristic disabled.")
            return False, None, None
        offset: Optional[int] = None
        if len(parts) > 1:
            try:
                offset = int(parts[1])
            except ValueError:
                print("Invalid offset, ignoring offset component.")
                offset = None
        return True, multiplier, offset

    # ------------------------------------------------------------------
    def _maybe_save(self, result) -> None:
        if input("\nSave result? (y/n): ").strip().lower() != "y":
            return
        output_dir = self.args.output_dir
        path = self.simulator.save_result(result, output_dir)
        print(f"Saved to {path}")


class PiggybackAttackCLIUtils:
    @staticmethod
    def has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def print_banner(title: str) -> None:
        line = "=" * 80
        print(f"\n{line}\n{title}\n{line}\n")

    @staticmethod
    def print_detection(label: str, detection: Dict[str, object]) -> None:
        z = float(detection.get("z_score", 0.0))
        pred = detection.get("prediction", False)
        green = float(detection.get("green_fraction", 0.0))
        tokens = detection.get("num_tokens_scored", detection.get("context_window", ""))
        status = "positive" if pred else "negative"
        print(f"{label}: {status} (z={z:.3f}, green={green:.2%}, tokens={tokens})")

    @staticmethod
    def print_snippet(text: str, limit: int = 300) -> None:
        preview = text[:limit]
        if len(text) > limit:
            preview += "..."
        print(preview)


def main() -> None:
    args = parse_args()
    cli = WatermarkAttackCLI(args)
    cli.run()


if __name__ == "__main__":
    main()
