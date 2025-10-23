"""
Llama 模型配置
定义支持的模型和推荐配置
"""

# 支持的模型配置
SUPPORTED_MODELS = {
    # Llama 2 系列 (推荐，无需特殊权限)
    "llama2-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "description": "Llama 2 7B - 推荐使用，性能良好",
        "vram_required": "~14GB",
        "access": "公开可用",
        "recommended": True
    },
    "llama2-13b": {
        "name": "meta-llama/Llama-2-13b-hf",
        "description": "Llama 2 13B - 更强性能，需要更多显存",
        "vram_required": "~26GB",
        "access": "公开可用",
        "recommended": False
    },
    "llama2-7b-chat": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "description": "Llama 2 7B Chat - 对话优化版本",
        "vram_required": "~14GB",
        "access": "公开可用",
        "recommended": False
    },
    
    # Llama 3.2 系列 (需要访问权限)
    "llama3.2-1b": {
        "name": "meta-llama/Llama-3.2-1B",
        "description": "Llama 3.2 1B - 轻量级模型",
        "vram_required": "~4GB",
        "access": "需要HF权限",
        "recommended": False
    },
    "llama3.2-3b": {
        "name": "meta-llama/Llama-3.2-3B",
        "description": "Llama 3.2 3B - 中等大小模型",
        "vram_required": "~8GB",
        "access": "需要HF权限",
        "recommended": False
    },
    
    # 其他兼容模型
    "opt-6.7b": {
        "name": "facebook/opt-6.7b",
        "description": "OPT 6.7B - Facebook开源模型",
        "vram_required": "~13GB",
        "access": "公开可用",
        "recommended": False
    },
}

# 默认模型
DEFAULT_MODEL = "llama2-7b"

# 推荐的水印参数配置
WATERMARK_CONFIGS = {
    "default": {
        "gamma": 0.25,
        "delta": 2.0,
        "seeding_scheme": "selfhash",
        "description": "默认推荐配置，适用于大多数场景"
    },
    "strong": {
        "gamma": 0.25,
        "delta": 3.0,
        "seeding_scheme": "selfhash",
        "description": "更强的水印，适用于instruction-tuned模型"
    },
    "mild": {
        "gamma": 0.25,
        "delta": 1.0,
        "seeding_scheme": "selfhash",
        "description": "较弱的水印，对文本质量影响更小"
    },
    "robust": {
        "gamma": 0.25,
        "delta": 2.0,
        "seeding_scheme": "minhash",
        "description": "更鲁棒的水印，适合需要抵抗编辑的场景"
    },
}


def get_model_name(model_key: str) -> str:
    """获取模型完整名称"""
    if model_key in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_key]["name"]
    # 如果不在预定义列表中，假设是完整的模型路径
    return model_key


def list_models():
    """列出所有支持的模型"""
    print("\n" + "="*80)
    print("支持的模型列表")
    print("="*80 + "\n")
    
    for key, config in SUPPORTED_MODELS.items():
        status = "✓ 推荐" if config.get("recommended", False) else "  "
        print(f"{status} [{key}]")
        print(f"    名称: {config['name']}")
        print(f"    描述: {config['description']}")
        print(f"    显存需求: {config['vram_required']}")
        print(f"    访问: {config['access']}")
        print()
    
    print(f"默认模型: {DEFAULT_MODEL}")
    print("="*80 + "\n")


def list_watermark_configs():
    """列出水印配置"""
    print("\n" + "="*80)
    print("预设水印配置")
    print("="*80 + "\n")
    
    for key, config in WATERMARK_CONFIGS.items():
        print(f"[{key}]")
        print(f"  描述: {config['description']}")
        print(f"  Gamma: {config['gamma']}")
        print(f"  Delta: {config['delta']}")
        print(f"  Seeding Scheme: {config['seeding_scheme']}")
        print()
    
    print("="*80 + "\n")


def get_watermark_config(config_key: str = "default") -> dict:
    """获取水印配置"""
    if config_key in WATERMARK_CONFIGS:
        config = WATERMARK_CONFIGS[config_key].copy()
        config.pop('description', None)
        return config
    return WATERMARK_CONFIGS["default"]


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--list-models":
        list_models()
    elif len(sys.argv) > 1 and sys.argv[1] == "--list-configs":
        list_watermark_configs()
    else:
        print("用法:")
        print("  python llama_model_config.py --list-models    # 列出所有模型")
        print("  python llama_model_config.py --list-configs   # 列出水印配置")
