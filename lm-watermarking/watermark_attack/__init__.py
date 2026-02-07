"""
水印攻击模块初始化。
提供猪背攻击（piggyback attack）模拟相关工具。
"""

from .piggyback_attack import PiggybackAttackSimulator, PiggybackAttackConfig, PiggybackAttackResult

__all__ = [
    "PiggybackAttackSimulator",
    "PiggybackAttackConfig",
    "PiggybackAttackResult",
]
