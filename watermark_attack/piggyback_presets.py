"""
预设的猪背攻击场景，便于快速复现论文中的示例或进行批量测试。

每个预设包含：
  - description: 说明
  - direct_replacements: 按顺序执行的替换 (List[Tuple[str, str]])
  - synonym_overrides: 额外的替换映射（与交互输入格式一致）
  - insert_phrases: 需要插入的句子列表
  - use_antonyms: 是否启用默认反义词映射（会替换成 config.antonym_map）
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

PresetConfig = Dict[str, object]


PRESET_LIBRARY: Dict[str, PresetConfig] = {
    "appendix_f_demo": {
        "description": "复现 Appendix F 中的 ENIAC 示例：扩大年份与速度差异并引入语义反转。",
        "direct_replacements": [
            ("70", "700"),
            ("1945", "1445"),
            ("fast", "slow"),
            ("faster", "slower"),
            ("first", "last"),
            ("thousand", "hundred"),
        ],
        "synonym_overrides": {},
        "insert_phrases": [
            "Explicitly, historical archives claim some of these figures were widely disputed."
        ],
        "use_antonyms": False,
    },
    "numbers_only": {
        "description": "仅改变文本中的数字值（乘以 10），适合验证数值敏感度。",
        "direct_replacements": [],
        "synonym_overrides": {},
        "insert_phrases": [],
        "use_antonyms": False,
        "numeric_multiplier": 10.0,
    },
    "antonym_flip": {
        "description": "不改变数字，仅应用默认反义词映射进行语义翻转。",
        "direct_replacements": [],
        "synonym_overrides": {},
        "insert_phrases": [],
        "use_antonyms": True,
    },
}


def list_presets() -> List[str]:
    return list(PRESET_LIBRARY.keys())


def get_preset(name: str) -> PresetConfig | None:
    return PRESET_LIBRARY.get(name)


def iter_presets() -> Iterable[Tuple[str, PresetConfig]]:
    for name, config in PRESET_LIBRARY.items():
        yield name, config
