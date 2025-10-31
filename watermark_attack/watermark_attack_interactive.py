"""
猪背水印攻击交互式脚本。

使用方式类似 hybrid_watermark_interactive.py，通过命令行交互完成：
  1. 生成或载入带水印文本
  2. 指定需要插入的“搭便车”敏感片段
  3. 执行轻量语义扰动，评估水印检测是否仍为阳性
  4. 可选择保存实验结果到 watermark_attack_results 目录
"""

import argparse
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from watermark_attack.piggyback_attack import (
    PiggybackAttackConfig,
    PiggybackAttackSimulator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Piggyback watermark attack interactive CLI"
    )
    parser.add_argument("--model", default="llama-2-7b", help="模型昵称（model_config.json 中定义）")
    parser.add_argument("--device", default=None, help="运行设备，默认自动检测")
    parser.add_argument("--gamma", type=float, default=0.5, help="水印 gamma 参数")
    parser.add_argument("--delta", type=float, default=2.0, help="水印 delta 参数（生成时使用）")
    parser.add_argument("--hash-key", type=int, default=15485863, help="水印 hash key")
    parser.add_argument("--z-threshold", type=float, default=4.0, help="检测阈值")
    parser.add_argument("--detection-margin", type=float, default=0.5, help="允许的 z-score 下降幅度")
    parser.add_argument("--max-new-tokens", type=int, default=180, help="生成文本时的最大新增 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("watermark_attack", "watermark_attack_results"),
        help="保存实验结果的目录",
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

    def run(self):
        PiggybackAttackCLIUtils.print_banner("猪背水印攻击交互式实验")
        print(f"模型昵称: {self.config.model_nickname}")
        print(f"设备: {self.config.device}")
        print(f"水印参数: gamma={self.config.gamma}, delta={self.config.delta}, hash_key={self.config.hash_key}")
        print(f"检测阈值: {self.config.z_threshold}, 允许下降: {self.config.detection_margin}\n")

        base_text = self._acquire_base_text()
        base_detection = self.simulator.detect_text(base_text)
        PiggybackAttackCLIUtils.print_detection("原始文本检测", base_detection)

        target_phrases = self._prompt_target_phrases()
        synonym_map = self._prompt_synonym_pairs()

        result = self.simulator.simulate_attack(
            watermarked_text=base_text,
            target_phrases=target_phrases,
            synonym_targets=synonym_map,
        )

        PiggybackAttackCLIUtils.print_detection("\n最终文本检测", result.final_detection)
        print(f"\n攻击成功: {'是' if result.success else '否'}")
        print(f"失败的插入: {result.failed_insertions if result.failed_insertions else '无'}")
        print("\n改动记录:")
        if result.modifications:
            for mod in result.modifications:
                desc = mod.description
                z = mod.z_score
                gf = mod.green_fraction
                print(f"  - [{mod.modification_type}] {desc} (z={z:.3f}, green={gf:.2%})")
        else:
            print("  无 - 未进行近义词替换或插入。")

        self._maybe_save(result)

    # ------------------------------------------------------------------
    def _acquire_base_text(self) -> str:
        """
        获取带水印的原始文本：支持生成、载入或手动粘贴。
        """
        while True:
            print("\n请选择获取原始文本的方式：")
            print("  1) 使用模型生成带水印文本")
            print("  2) 从文件载入已有文本")
            print("  3) 手动粘贴文本")
            choice = input("请输入选项 (1/2/3): ").strip()

            if choice == "1":
                prompt = input("\n请输入生成提示词 (prompt): ").strip()
                if not prompt:
                    print("提示词不能为空。")
                    continue
                text, detection = self.simulator.generate_watermarked_text(
                    prompt=prompt,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature,
                )
                PiggybackAttackCLIUtils.print_detection("\n生成文本检测", detection)
                print("\n生成文本预览（前300字）:")
                PiggybackAttackCLIUtils.print_snippet(text)
                confirm = input("\n是否使用该文本作为基础文本? (y/n): ").strip().lower()
                if confirm == "y":
                    return text

            elif choice == "2":
                path = input("\n请输入文本文件路径: ").strip()
                if not os.path.exists(path):
                    print("文件不存在，请重新选择。")
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if not text:
                    print("文件内容为空。")
                    continue
                return text

            elif choice == "3":
                print("\n请粘贴文本，完成后输入单独一行 END 结束：")
                lines: List[str] = []
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    lines.append(line)
                text = "\n".join(lines).strip()
                if not text:
                    print("文本为空。")
                    continue
                return text
            else:
                print("无效选项，请重新输入。")

    def _prompt_target_phrases(self) -> List[str]:
        """从输入中解析待插入的敏感短语。"""
        raw = input("\n输入需要插入的句子（使用分号分隔，留空表示跳过）: ").strip()
        if not raw:
            return []
        phrases = [seg.strip() for seg in raw.replace("；", ";").split(";") if seg.strip()]
        return phrases

    def _prompt_synonym_pairs(self) -> Dict[str, str]:
        """
        解析显式指定的近义词替换映射，格式：原词->替换词, 原词2->替换词2
        """
        raw = input("\n输入近义词替换（格式: 原词->新词, 原词2->新词2），留空表示使用默认词典: ").strip()
        if not raw:
            return {}
        mapping: Dict[str, str] = {}
        parts = raw.split(",")
        for part in parts:
            if "->" not in part:
                continue
            src, dst = part.split("->", 1)
            src = src.strip()
            dst = dst.strip()
            if src and dst:
                mapping[src] = dst
        return mapping

    def _maybe_save(self, result):
        """是否保存实验记录。"""
        save = input("\n是否保存结果? (y/n): ").strip().lower()
        if save != "y":
            return
        output_dir = self.args.output_dir
        path = self.simulator.save_result(result, output_dir)
        print(f"结果已保存至 {path}")


class PiggybackAttackCLIUtils:
    @staticmethod
    def has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def print_banner(title: str):
        line = "=" * 80
        print(f"\n{line}\n{title}\n{line}\n")

    @staticmethod
    def print_detection(label: str, detection: Dict[str, object]):
        z = detection.get("z_score", 0.0)
        pred = detection.get("prediction", False)
        gf = detection.get("green_fraction", 0.0)
        tokens = detection.get("num_tokens_scored", detection.get("context_window", ""))
        status = "阳性" if pred else "阴性"
        print(
            f"{label}: {status} "
            f"(z={float(z):.3f}, green={float(gf):.2%}, tokens={tokens})"
        )

    @staticmethod
    def print_snippet(text: str, limit: int = 300):
        preview = text[:limit]
        if len(text) > limit:
            preview += "..."
        print(preview)


def main():
    args = parse_args()
    cli = WatermarkAttackCLI(args)
    cli.run()


if __name__ == "__main__":
    main()
