"""
模型配置管理器
从 model_config.json 读取和管理API提供商和模型配置
"""

import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path


class ModelConfigManager:
    """模型配置管理器"""
    
    def __init__(self, config_path: str = "model_config.json"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = self._resolve_path(config_path)
        self.config = self._load_config()
        
    def _resolve_path(self, config_path: str) -> Path:
        """解析配置文件路径"""
        # 如果是相对路径，从当前文件所在目录查找
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        return Path(config_path)
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def reload(self):
        """重新加载配置文件"""
        self.config = self._load_config()
    
    def _resolve_api_key(self, api_key: str, provider_name: str) -> str:
        """
        解析API密钥
        
        Args:
            api_key: 配置文件中的API密钥值
            provider_name: API提供商名称
            
        Returns:
            实际的API密钥
        """
        # 如果设置为 "auto"，尝试从系统读取
        if api_key == "auto":
            if provider_name == "HuggingFace":
                # 尝试从 HuggingFace CLI 缓存读取
                hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
                if hf_token_path.exists():
                    with open(hf_token_path, 'r') as f:
                        return f.read().strip()
                else:
                    print(f"⚠️ 警告: HuggingFace token 文件不存在于 {hf_token_path}")
                    print("   请运行: huggingface-cli login")
                    return ""
            else:
                print(f"⚠️ 警告: {provider_name} 不支持自动读取API密钥")
                return ""
        
        # 检查是否是环境变量
        if api_key.startswith("$"):
            env_var = api_key[1:]
            return os.getenv(env_var, "")
        
        return api_key
    
    # ========== API Provider 相关方法 ==========
    
    def get_api_providers(self) -> List[Dict]:
        """获取所有API提供商"""
        return self.config.get("api_providers", [])
    
    def get_api_provider(self, name: str, resolve_key: bool = True) -> Optional[Dict]:
        """
        根据名称获取API提供商配置
        
        Args:
            name: API提供商名称
            resolve_key: 是否解析API密钥（自动读取或从环境变量）
            
        Returns:
            API提供商配置字典，如果不存在则返回None
        """
        for provider in self.get_api_providers():
            if provider.get("name") == name:
                # 复制一份以避免修改原始配置
                provider_copy = provider.copy()
                
                # 解析API密钥
                if resolve_key and "api_key" in provider_copy:
                    provider_copy["api_key"] = self._resolve_api_key(
                        provider_copy["api_key"],
                        name
                    )
                
                return provider_copy
        return None
    
    def list_api_provider_names(self) -> List[str]:
        """列出所有API提供商名称"""
        return [p.get("name") for p in self.get_api_providers()]
    
    # ========== Model 相关方法 ==========
    
    def get_models(self, api_provider: Optional[str] = None) -> List[Dict]:
        """
        获取模型列表
        
        Args:
            api_provider: 如果指定，只返回该提供商的模型
            
        Returns:
            模型配置列表
        """
        models = self.config.get("models", [])
        if api_provider:
            models = [m for m in models if m.get("api_provider") == api_provider]
        return models
    
    def get_model(self, name: str) -> Optional[Dict]:
        """
        根据名称获取模型配置
        
        Args:
            name: 模型名称
            
        Returns:
            模型配置字典，如果不存在则返回None
        """
        for model in self.get_models():
            if model.get("name") == name:
                return model
        return None
    
    def get_model_with_provider(self, name: str, resolve_key: bool = True) -> Optional[Dict]:
        """
        获取模型配置及其API提供商配置
        
        Args:
            name: 模型名称
            resolve_key: 是否解析API密钥（自动读取或从环境变量）
            
        Returns:
            包含model和provider的字典
        """
        model = self.get_model(name)
        if not model:
            return None
        
        provider = self.get_api_provider(model.get("api_provider"), resolve_key=resolve_key)
        return {
            "model": model,
            "provider": provider
        }
    
    def resolve_model_name(self, nickname: str) -> Optional[str]:
        """
        将模型昵称转换为真实的模型标识符
        
        Args:
            nickname: 模型昵称（如 "deepseek-v3"）
            
        Returns:
            真实的模型标识符（如 "deepseek-ai/DeepSeek-V3"），如果不存在则返回None
        """
        model = self.get_model(nickname)
        if model:
            return model.get("model_identifier")
        return None
    
    def get_model_info_by_nickname(self, nickname: str) -> Optional[Dict]:
        """
        通过昵称获取完整的模型信息（包括真实标识符和API配置）
        
        Args:
            nickname: 模型昵称（如 "deepseek-v3"）
            
        Returns:
            包含 model_identifier, provider_config, model_config 的字典
        """
        result = self.get_model_with_provider(nickname, resolve_key=True)
        if not result:
            return None
        
        model = result["model"]
        provider = result["provider"]
        
        return {
            "nickname": nickname,
            "model_identifier": model.get("model_identifier"),
            "model_config": model,
            "provider_config": provider,
            "api_key": provider.get("api_key") if provider else None,
            "base_url": provider.get("base_url") if provider else None,
        }
    
    def list_model_names(self, api_provider: Optional[str] = None) -> List[str]:
        """
        列出所有模型名称
        
        Args:
            api_provider: 如果指定，只返回该提供商的模型名称
        """
        return [m.get("name") for m in self.get_models(api_provider)]
    
    # ========== Watermark Config 相关方法 ==========
    
    def get_watermark_configs(self) -> Dict[str, Dict]:
        """获取所有水印配置"""
        return self.config.get("watermark_configs", {})
    
    def get_watermark_config(self, name: str = "default") -> Optional[Dict]:
        """
        获取指定的水印配置
        
        Args:
            name: 配置名称（default/strong/weak/balanced）
            
        Returns:
            水印配置字典
        """
        return self.get_watermark_configs().get(name)
    
    def list_watermark_config_names(self) -> List[str]:
        """列出所有水印配置名称"""
        return list(self.get_watermark_configs().keys())
    
    # ========== Generation Config 相关方法 ==========
    
    def get_generation_configs(self) -> Dict[str, Dict]:
        """获取所有生成配置"""
        return self.config.get("generation_configs", {})
    
    def get_generation_config(self, name: str = "default") -> Optional[Dict]:
        """
        获取指定的生成配置
        
        Args:
            name: 配置名称（default/creative/precise）
            
        Returns:
            生成配置字典
        """
        return self.get_generation_configs().get(name)
    
    def list_generation_config_names(self) -> List[str]:
        """列出所有生成配置名称"""
        return list(self.get_generation_configs().keys())
    
    # ========== 计算成本 ==========
    
    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        计算API调用成本
        
        Args:
            model_name: 模型名称
            input_tokens: 输入token数量
            output_tokens: 输出token数量
            
        Returns:
            成本（单位：元）
        """
        model = self.get_model(model_name)
        if not model:
            return 0.0
        
        price_in = model.get("price_in", 0.0)
        price_out = model.get("price_out", 0.0)
        
        # 价格通常是每百万token的价格
        cost_in = (input_tokens / 1_000_000) * price_in
        cost_out = (output_tokens / 1_000_000) * price_out
        
        return cost_in + cost_out
    
    # ========== 打印信息 ==========
    
    def print_summary(self):
        """打印配置摘要"""
        print("=" * 70)
        print("模型配置摘要")
        print("=" * 70)
        
        # API提供商
        print(f"\n📡 API提供商 ({len(self.get_api_providers())}个):")
        print("-" * 70)
        for provider in self.get_api_providers():
            print(f"  • {provider['name']}")
            print(f"    Base URL: {provider['base_url']}")
            print(f"    Client: {provider.get('client_type', 'openai')}")
            print(f"    Timeout: {provider.get('timeout', 30)}s")
            if provider.get('description'):
                print(f"    说明: {provider['description']}")
        
        # 模型
        print(f"\n🤖 模型 ({len(self.get_models())}个):")
        print("-" * 70)
        
        # 按API提供商分组
        for provider_name in self.list_api_provider_names():
            models = self.get_models(provider_name)
            if models:
                print(f"\n  [{provider_name}] ({len(models)}个模型)")
                for model in models:
                    print(f"    • {model['name']}")
                    print(f"      ID: {model['model_identifier']}")
                    if model.get('price_in', 0) > 0 or model.get('price_out', 0) > 0:
                        print(f"      价格: ¥{model['price_in']}/M输入, ¥{model['price_out']}/M输出")
                    if model.get('context_length'):
                        print(f"      上下文: {model['context_length']} tokens")
        
        # 水印配置
        print(f"\n💧 水印配置 ({len(self.get_watermark_configs())}个):")
        print("-" * 70)
        for name, config in self.get_watermark_configs().items():
            print(f"  • {name}: gamma={config['gamma']}, delta={config['delta']}")
        
        # 生成配置
        print(f"\n⚙️  生成配置 ({len(self.get_generation_configs())}个):")
        print("-" * 70)
        for name, config in self.get_generation_configs().items():
            print(f"  • {name}: temp={config['temperature']}, max_tokens={config['max_new_tokens']}")
        
        print("\n" + "=" * 70)
    
    def print_model_details(self, model_name: str):
        """打印模型详细信息"""
        result = self.get_model_with_provider(model_name)
        if not result:
            print(f"❌ 模型 '{model_name}' 不存在")
            return
        
        model = result["model"]
        provider = result["provider"]
        
        print("=" * 70)
        print(f"模型详情: {model['name']}")
        print("=" * 70)
        
        print(f"\n📝 基本信息:")
        print(f"  名称: {model['name']}")
        print(f"  标识符: {model['model_identifier']}")
        print(f"  描述: {model.get('description', 'N/A')}")
        
        print(f"\n📡 API提供商:")
        print(f"  名称: {provider['name']}")
        print(f"  Base URL: {provider['base_url']}")
        print(f"  客户端类型: {provider.get('client_type', 'openai')}")
        print(f"  超时: {provider.get('timeout', 30)}秒")
        print(f"  最大重试: {provider.get('max_retry', 2)}次")
        print(f"  重试间隔: {provider.get('retry_interval', 10)}秒")
        
        print(f"\n💰 成本:")
        if model.get('price_in', 0) > 0 or model.get('price_out', 0) > 0:
            print(f"  输入: ¥{model['price_in']}/百万tokens")
            print(f"  输出: ¥{model['price_out']}/百万tokens")
            print(f"\n  示例（10K输入 + 1K输出）:")
            cost = self.calculate_cost(model_name, 10000, 1000)
            print(f"    成本: ¥{cost:.4f}")
        else:
            print(f"  免费或本地运行")
        
        print(f"\n⚙️  参数:")
        print(f"  上下文长度: {model.get('context_length', 'N/A')} tokens")
        if model.get('requires_auth'):
            print(f"  需要认证: 是")
        
        print("\n" + "=" * 70)


def main():
    """命令行工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型配置管理工具")
    parser.add_argument("--config", default="model_config.json", help="配置文件路径")
    parser.add_argument("--list-providers", action="store_true", help="列出所有API提供商")
    parser.add_argument("--list-models", action="store_true", help="列出所有模型")
    parser.add_argument("--provider", help="筛选指定提供商的模型")
    parser.add_argument("--model", help="显示指定模型的详细信息")
    parser.add_argument("--list-watermark", action="store_true", help="列出所有水印配置")
    parser.add_argument("--list-generation", action="store_true", help="列出所有生成配置")
    parser.add_argument("--summary", action="store_true", help="显示配置摘要")
    
    args = parser.parse_args()
    
    try:
        manager = ModelConfigManager(args.config)
        
        if args.summary:
            manager.print_summary()
        elif args.list_providers:
            print("API提供商列表:")
            for name in manager.list_api_provider_names():
                print(f"  • {name}")
        elif args.list_models:
            if args.provider:
                print(f"[{args.provider}] 模型列表:")
                models = manager.get_models(args.provider)
            else:
                print("所有模型列表:")
                models = manager.get_models()
            
            for model in models:
                print(f"  • {model['name']} ({model['api_provider']})")
        elif args.model:
            manager.print_model_details(args.model)
        elif args.list_watermark:
            print("水印配置列表:")
            for name in manager.list_watermark_config_names():
                config = manager.get_watermark_config(name)
                print(f"  • {name}: gamma={config['gamma']}, delta={config['delta']}")
        elif args.list_generation:
            print("生成配置列表:")
            for name in manager.list_generation_config_names():
                config = manager.get_generation_config(name)
                print(f"  • {name}: temp={config['temperature']}, max_tokens={config['max_new_tokens']}")
        else:
            manager.print_summary()
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
