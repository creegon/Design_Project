import os
import sys
import json
from pathlib import Path
from datetime import datetime
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
# ensure llama_demos is importable for ModelConfigManager
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'llama_demos')))
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from llama_demos.model_config_manager import ModelConfigManager
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from datasets import load_dataset


def compute_confusion_stats(y_true, scores, thresholds):
    # y_true: list of 0/1
    # scores: list of float (z_scores)
    out = []
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        tp = sum(1 for yt, p in zip(y_true, preds) if yt == 1 and p == 1)
        fn = sum(1 for yt, p in zip(y_true, preds) if yt == 1 and p == 0)
        fp = sum(1 for yt, p in zip(y_true, preds) if yt == 0 and p == 1)
        tn = sum(1 for yt, p in zip(y_true, preds) if yt == 0 and p == 0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        out.append({"threshold": t, "TP": tp, "FN": fn, "FP": fp, "TN": tn, "TPR": tpr, "FPR": fpr, "precision": prec})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='opt-350m')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--decoding', type=str, default='multinomial', choices=['multinomial','beam8'], help='decoding strategy')
    parser.add_argument('--outdir', default='./hybrid_watermark_results')
    args = parser.parse_args()
    
    # Define parameter combinations to test
    gamma_values = [0.25, 0.5]
    delta_values = [1.0, 2.0, 5.0]
    z_thresholds = [4.0, 5.0]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'watermark_run_{timestamp}.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Setup output directories
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    gen_out = Path(f"experiments_output/small_repro_{timestamp}")
    gen_out.mkdir(parents=True, exist_ok=True)

    # Setup model
    logger.info('Resolving model identifier...')
    cfg = ModelConfigManager()
    model_identifier = cfg.resolve_model_name(args.model)
    if model_identifier is None:
        model_identifier = args.model
        logger.warning(f'No model config found for {args.model}, using as direct identifier')

    logger.info(f'Loading tokenizer and model: {model_identifier}')
    
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    # Ensure pad/eos are set for generation (common for LLaMA tokenizers)
    if tokenizer.pad_token_id is None and hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_identifier,
        torch_dtype=torch.float16 if device.type=='cuda' else torch.float32,
        device_map='auto' if device.type=='cuda' else None
    )
    if device.type == 'cpu':
        model = model.to(device)
    # Align model pad token configuration with tokenizer to avoid early EOS/padding issues
    try:
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    model.eval()
    
    # Prepare vocab
    vocab_ids = list(tokenizer.get_vocab().values())
    vocab_ids_tensor = torch.tensor(vocab_ids).to(device)

    def generate_watermarked_text(prompt, gamma, delta, num_beams=1, do_sample=True, max_new_tokens=80, temperature=0.7):
        """Generate text with optional watermark."""
        input_device = next(model.parameters()).device  # 获取模型实际所在的设备
        vocab_ids_tensor = torch.tensor(vocab_ids, device=input_device)
        # Use a simpler seeding scheme ('simple_1') to avoid cross-device issues
        proc = WatermarkLogitsProcessor(
            vocab=vocab_ids_tensor.tolist(),
            gamma=gamma,
            delta=delta,
            seeding_scheme='simple_1'
        )
        
        # 确保输入和词汇表在同一个设备上
        inputs = tokenizer(prompt, return_tensors='pt')
        input_device = next(model.parameters()).device  # 获取模型实际所在的设备
        inputs = {k: v.to(input_device) for k,v in inputs.items()}
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                min_new_tokens=20,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            if num_beams and num_beams > 1:
                gen_kwargs.update(dict(num_beams=num_beams, early_stopping=True))
            
            from transformers import MinLengthLogitsProcessor
            processors = [proc] if proc else []
            processors.append(MinLengthLogitsProcessor(20, eos_token_id=tokenizer.eos_token_id))
            
            output_tokens = model.generate(
                **inputs,
                logits_processor=LogitsProcessorList(processors),
                **gen_kwargs
            )
            
        generated_tokens = output_tokens[:, inputs['input_ids'].shape[-1]:]
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    def generate_without_watermark(prompt, gamma, max_new_tokens=80, temperature=0.7):
        """Generate text without watermark."""
        # Baseline generation: no logits processor at all
        inputs = tokenizer(prompt, return_tensors='pt')
        input_device = next(model.parameters()).device  # 获取模型实际所在的设备
        inputs = {k: v.to(input_device) for k,v in inputs.items()}
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                min_new_tokens=20,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            output_tokens = model.generate(
                **inputs,
                **gen_kwargs
            )
        generated_tokens = output_tokens[:, inputs['input_ids'].shape[-1]:]
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    # Load prompts
    logger.info('Loading prompts from AG News test split...')
    ds = load_dataset('ag_news', split='test')
    prompts = [row['text'] for row in ds.select(range(args.n))]
    logger.info(f'Loaded {len(prompts)} prompts')

    # Dictionary to store all results
    all_results = {}
    
    # Run experiments for each parameter combination
    for gamma in gamma_values:
        for delta in delta_values:
            logger.info(f'\nTesting gamma={gamma}, delta={delta}')
            watermarked_texts = []
            unwatermarked_texts = []

            # Generate texts
            for i, prompt in enumerate(prompts, 1):
                try:
                    if args.decoding == 'beam8':
                        wm = generate_watermarked_text(prompt, gamma, delta, num_beams=8, 
                            do_sample=False, max_new_tokens=args.max_new_tokens)
                    else:
                        wm = generate_watermarked_text(prompt, gamma, delta, num_beams=1, 
                            do_sample=True, max_new_tokens=args.max_new_tokens)
                except Exception as e:
                    logger.error(f'Generation failed (watermarked) at {i}: {str(e)}')
                    wm = ''
                
                try:
                    uwm = generate_without_watermark(prompt, gamma, max_new_tokens=args.max_new_tokens)
                except Exception as e:
                    logger.error(f'Generation failed (unwatermarked) at {i}: {str(e)}')
                    uwm = ''

                watermarked_texts.append({'prompt': prompt, 'text': wm})
                unwatermarked_texts.append({'prompt': prompt, 'text': uwm})

                if i % 10 == 0:
                    logger.info(f'Progress: Generated {i}/{args.n} samples')

            # Save generations for this parameter set
            gen_file = gen_out / f'generations_gamma{gamma}_delta{delta}_{timestamp}.json'
            with open(gen_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'params': {'gamma': gamma, 'delta': delta},
                    'watermarked': watermarked_texts,
                    'unwatermarked': unwatermarked_texts
                }, f, ensure_ascii=False, indent=2)

            # Test different detection thresholds
            for z_threshold in z_thresholds:
                logger.info(f'\nTesting detection with z_threshold={z_threshold}')

                # Detector must use same seeding scheme as generation
                detector = WatermarkDetector(
                    vocab=vocab_ids,
                    gamma=gamma,
                    seeding_scheme='simple_1',
                    device=device,
                    tokenizer=tokenizer,
                    z_threshold=z_threshold
                )
                
                all_scores = []
                y_true = []
                z_scores = []

                # Process watermarked texts
                for item in watermarked_texts:
                    text = item['text']
                    try:
                        if not text.strip():
                            z = float('nan')
                            res = {'z_score': z, 'prediction': False, 'green_fraction': None}
                        else:
                            res = detector.detect(text)
                            z = float(res.get('z_score', float('nan')))
                    except Exception as e:
                        # detection can fail if sequence too short for context width; treat as NaN
                        z = float('nan')
                        res = {'z_score': z, 'prediction': False, 'green_fraction': None, 'error': str(e)}
                    z_scores.append(z)
                    y_true.append(1)
                    all_scores.append({
                        'text': text,
                        'z_score': z,
                        'prediction': res.get('prediction', False),
                        'green_fraction': res.get('green_fraction', None)
                    })

                # Process unwatermarked texts
                for item in unwatermarked_texts:
                    text = item['text']
                    try:
                        if not text.strip():
                            z = float('nan')
                            res = {'z_score': z, 'prediction': False, 'green_fraction': None}
                        else:
                            res = detector.detect(text)
                            z = float(res.get('z_score', float('nan')))
                    except Exception as e:
                        z = float('nan')
                        res = {'z_score': z, 'prediction': False, 'green_fraction': None, 'error': str(e)}
                    z_scores.append(z)
                    y_true.append(0)
                    all_scores.append({
                        'text': text,
                        'z_score': z,
                        'prediction': res.get('prediction', False),
                        'green_fraction': res.get('green_fraction', None)
                    })

                # Store results for this parameter combination
                results_key = f'gamma_{gamma}_delta_{delta}_z_{z_threshold}'
                all_results[results_key] = {
                    'params': {'gamma': gamma, 'delta': delta, 'z_threshold': z_threshold},
                    'scores': all_scores,
                    'z_scores': z_scores,
                    'y_true': y_true,
                }

                # Compute confusion matrix stats
                thresholds = [z_threshold - 0.5, z_threshold, z_threshold + 0.5]
                table = compute_confusion_stats(y_true, z_scores, thresholds)
                all_results[results_key]['confusion_stats'] = table

                # Log summary stats
                valid_scores = [s for s in z_scores if not np.isnan(s)]
                if valid_scores:
                    logger.info(f'Statistics for gamma={gamma}, delta={delta}, z_threshold={z_threshold}:')
                    logger.info(f'  Mean z-score: {np.mean(valid_scores):.3f}')
                    logger.info(f'  Std z-score: {np.std(valid_scores):.3f}')
                    logger.info(f'  Min z-score: {np.min(valid_scores):.3f}')
                    logger.info(f'  Max z-score: {np.max(valid_scores):.3f}')

                # Try to generate ROC curve (drop NaNs first)
                try:
                    from sklearn.metrics import roc_curve, auc
                    zs = np.array(z_scores)
                    yt = np.array(y_true)
                    mask = ~np.isnan(zs)
                    if mask.sum() == 0:
                        raise ValueError("All z-scores are NaN; cannot compute ROC.")
                    fpr, tpr, thr = roc_curve(yt[mask], zs[mask])
                    roc_auc = auc(fpr, tpr)
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(6,6))
                    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
                    plt.plot([0,1],[0,1],'k--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve (gamma={gamma}, delta={delta}, z={z_threshold})')
                    plt.legend()
                    roc_file = outdir / f'roc_gamma{gamma}_delta{delta}_z{z_threshold}_{timestamp}.png'
                    plt.savefig(roc_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f'Saved ROC plot to {roc_file}')
                except Exception as e:
                    logger.error(f'Could not plot ROC: {str(e)}')

    # Save final results
    results_file = outdir / f'all_results_{args.model}_{timestamp}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f'\nSaved all results to {results_file}')
    logger.info('Done.')


if __name__ == '__main__':
    main()