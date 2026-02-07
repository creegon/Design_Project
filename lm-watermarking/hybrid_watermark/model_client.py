"""
Model Client - API client for model server

Automatically resolves model names to ports from config and provides a simple interface
for text generation and watermark detection via API.

Usage:
    from model_client import ModelClient
    
    client = ModelClient()
    text = client.generate("llama-3.2-3b", "Write a poem", with_watermark=True)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import httpx

# Add parent directory to path for config manager
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llama_demos.model_config_manager import ModelConfigManager


def _load_port_map_from_config() -> Dict[str, int]:
    """从 model_config.json 加载端口映射"""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "llama_demos", "model_config.json"
    )
    try:
        config_manager = ModelConfigManager(config_path)
        port_map = {}
        for name in config_manager.list_model_names():
            info = config_manager.get_model_info_by_nickname(name)
            if info:
                model_config = info.get("model_config", {})
                if "server_port" in model_config:
                    port_map[name] = model_config["server_port"]
                elif "server_port" in info:
                    port_map[name] = info["server_port"]
        return port_map
    except Exception:
        return {}


class ModelClient:
    """Client for interacting with model server via API."""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port_map: Optional[Dict[str, int]] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize model client.
        
        Args:
            host: Server host
            port_map: Custom port mapping (model_name -> port), auto-loaded from config if None
            timeout: Request timeout in seconds
        """
        self.host = host
        # 从配置文件加载端口映射
        self.port_map = port_map if port_map is not None else _load_port_map_from_config()
        self.timeout = timeout
        self._server_status_cache: Dict[str, bool] = {}
    
    def get_port(self, model_nickname: str) -> int:
        """Get port for a model by nickname."""
        if model_nickname in self.port_map:
            return self.port_map[model_nickname]
        raise ValueError(
            f"Unknown model or no server_port configured: {model_nickname}. "
            f"Available: {list(self.port_map.keys())}"
        )
    
    def get_base_url(self, model_nickname: str) -> str:
        """Get base URL for a model."""
        port = self.get_port(model_nickname)
        return f"http://{self.host}:{port}"
    
    def is_server_running(self, model_nickname: str) -> bool:
        """Check if server is running for a model."""
        try:
            url = f"{self.get_base_url(model_nickname)}/health"
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                return response.status_code == 200
        except Exception:
            return False
    
    def generate(
        self,
        model_nickname: str,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        gamma: float = 0.25,
        delta: float = 2.0,
        seeding_scheme: str = "selfhash",
        hash_key: int = 15485863,
        with_watermark: bool = True,
    ) -> Dict:
        """
        Generate text using model server API.
        
        Args:
            model_nickname: Model name (e.g., "llama-3.2-3b")
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            gamma: Watermark gamma
            delta: Watermark delta
            seeding_scheme: Watermark seeding scheme
            hash_key: Watermark hash key
            with_watermark: Whether to apply watermark
        
        Returns:
            Dict with generated_text, prompt_tokens, completion_tokens, model
        
        Raises:
            ConnectionError: If server is not running
            httpx.HTTPError: If request fails
        """
        base_url = self.get_base_url(model_nickname)
        endpoint = "/generate" if with_watermark else "/generate_no_watermark"
        
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "gamma": gamma,
            "delta": delta,
            "seeding_scheme": seeding_scheme,
            "hash_key": hash_key,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{base_url}{endpoint}", json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to model server for '{model_nickname}' at {base_url}. "
                f"Start it with: python model_server.py --model {model_nickname}"
            )
    
    def detect(
        self,
        model_nickname: str,
        text: str,
        gamma: float = 0.25,
        seeding_scheme: str = "selfhash",
        hash_key: int = 15485863,
        z_threshold: float = 3.0,
    ) -> Dict:
        """
        Detect watermark using model server API.
        
        Args:
            model_nickname: Model name (for tokenizer)
            text: Text to analyze
            gamma: Watermark gamma
            seeding_scheme: Seeding scheme
            hash_key: Hash key
            z_threshold: Detection threshold
        
        Returns:
            Dict with z_score, p_value, prediction, green_fraction, num_tokens_scored
        """
        base_url = self.get_base_url(model_nickname)
        
        payload = {
            "text": text,
            "gamma": gamma,
            "seeding_scheme": seeding_scheme,
            "hash_key": hash_key,
            "z_threshold": z_threshold,
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{base_url}/detect", json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to model server for '{model_nickname}' at {base_url}. "
                f"Start it with: python model_server.py --model {model_nickname}"
            )
    
    def health(self, model_nickname: str) -> Dict:
        """Get health status of a model server."""
        base_url = self.get_base_url(model_nickname)
        
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/health")
            response.raise_for_status()
            return response.json()
    
    def list_available_models(self) -> list[str]:
        """List all models with known port mappings."""
        return list(self.port_map.keys())
    
    def list_running_servers(self) -> list[str]:
        """List models with running servers."""
        running = []
        for model in self.port_map:
            if self.is_server_running(model):
                running.append(model)
        return running


# Convenience function
def get_model_client() -> ModelClient:
    """Get a configured model client instance."""
    return ModelClient()


if __name__ == "__main__":
    # Quick test
    client = ModelClient()
    print("Available models:", client.list_available_models())
    print("Running servers:", client.list_running_servers())
