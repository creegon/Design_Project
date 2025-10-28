import json
import os
import sys
from pathlib import Path
from datasets import load_dataset

# ensure repo root is on sys.path so we can import hybrid_watermark package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybrid_watermark.hybrid_watermark_experiment import HybridWatermarkExperiment

OUTPUT_DIR = Path("./experiments_output/smoke_llama3_3b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_ag_news_samples(n_samples=5):
    """Load a few samples from AG News dataset (use full text for seeding)"""
    dataset = load_dataset("ag_news", split="test")
    samples = dataset.select(range(n_samples))
    # Use full text as prompt to give seeding context for selfhash
    prompts = [item['text'] for item in samples]
    return prompts

def main():
    try:
        # 打印环境信息
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

        # Use gated Llama model (requires HF access)
        model_nickname = "llama-3.2-3b"
        print(f"\nInitializing HybridWatermarkExperiment with model nickname: {model_nickname}")
        print("Note: this requires your Hugging Face account to have access to the gated model.")
        # Let the experiment helper load the model/tokenizer (it uses model_config.json)
        exp = HybridWatermarkExperiment(model_nickname=model_nickname)
    except Exception as e:
        print("Failed to initialize experiment:", str(e))
        import traceback
        traceback.print_exc()
        return

    # create a simple watermark processor (use default hash_key from config)
    processor = exp.create_watermark_processor(gamma=0.25, delta=2.0, seeding_scheme='selfhash')

    # Get test prompts from AG News
    prompts = load_ag_news_samples(n_samples=5)
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating for prompt {i+1}/{len(prompts)}: {prompt[:100]}...")
        try:
            generated = exp.generate_with_watermark(
                prompt=prompt, 
                watermark_processor=processor, 
                max_new_tokens=80,  # reasonable length for news articles
                temperature=0.7
            )
            results.append({
                "model": model_nickname,
                "prompt": prompt,
                "generated": generated
            })
            print(f"Generated {len(generated.split())} words")
        except Exception as e:
            print(f"Generation failed for prompt {i+1}:", e)
            continue

    result = {
        "model": model_nickname,
        "samples": results
    }

    out_file = OUTPUT_DIR / "smoke_result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved result to: {out_file}")
    if results:
        print("Example generated text (first sample):\n", results[0]["generated"])
    else:
        print("No generations were produced.")

if __name__ == '__main__':
    main()
