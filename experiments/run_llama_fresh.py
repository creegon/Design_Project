#!/usr/bin/env python3
"""Minimal reproducible harness for watermark generation + detection.

Purpose: quick, easy script to load a model (default: tiny GPT2 for smoke),
generate watermarked and baseline texts, run the detector using the same
configuration as the user's `evaluate_watermark_detection` snippet, and
save a small JSON plus ROC image.

Usage examples (quick smoke):
  PYTHONPATH=. python3 experiments/run_llama_fresh.py --model sshleifer/tiny-gpt2 --n 6

For real Llama runs, pass your Llama HF id and ensure HF token is set.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return logging.getLogger('run_llama_fresh')


def ensure_tokenizer_pad(tokenizer):
    if getattr(tokenizer, 'pad_token_id', None) is None and getattr(tokenizer, 'eos_token_id', None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def generate_one(model, tokenizer, logits_proc, prompt, device, max_new_tokens=64, do_sample=True, num_beams=1):
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # minimal guard to avoid zero-length outputs: use generate's min_new_tokens
    procs = [] if logits_proc is None else [logits_proc]

    gen_kwargs = dict(max_new_tokens=max_new_tokens, min_new_tokens=8, do_sample=do_sample, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if num_beams and num_beams > 1:
        gen_kwargs.update(dict(num_beams=num_beams, do_sample=False))

    # transformers expects a LogitsProcessorList (or None). Build and pass it correctly so
    # the WatermarkLogitsProcessor actually gets applied during generation.
    logits_processor = LogitsProcessorList(procs) if len(procs) > 0 else None
    out = model.generate(**inputs, logits_processor=logits_processor, **gen_kwargs)
    # Extract newly generated tokens
    gen_tokens = out[:, inputs['input_ids'].shape[-1]:]
    return tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]


def detect_texts(detector, texts):
    zs = []
    for t in texts:
        if not t or not t.strip():
            zs.append(float('nan'))
            continue
        try:
            r = detector.detect(t)
            zs.append(float(r.get('z_score', float('nan'))))
        except Exception:
            zs.append(float('nan'))
    return zs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sshleifer/tiny-gpt2')
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--decoding', choices=['multinomial','beam8'], default='multinomial')
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='hybrid_watermark_results_fresh')
    parser.add_argument('--lang', choices=['en','zh'], default='en', help='prompt language: en or zh')
    args = parser.parse_args()

    logger = setup_logger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ensure_tokenizer_pad(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto' if device.type=='cuda' else None)
    if device.type == 'cpu':
        model = model.to(device)
    model.eval()

    vocab_ids = list(tokenizer.get_vocab().values())

    # Detector per user's snippet
    detector = WatermarkDetector(
        vocab=vocab_ids,
        gamma=args.gamma,
        seeding_scheme='selfhash',
        device=model.device,
        tokenizer=tokenizer,
        z_threshold=4.0,
        normalizers=[],
        ignore_repeated_ngrams=True,
    )

    wm_proc = WatermarkLogitsProcessor(vocab=vocab_ids, gamma=args.gamma, delta=args.delta, seeding_scheme='selfhash')
    baseline_proc = WatermarkLogitsProcessor(vocab=vocab_ids, gamma=args.gamma, delta=0.0, seeding_scheme='selfhash')

    # simple prompts (small set) to avoid dataset deps
    if args.lang == 'zh':
        prompts = [
            '请写一个关于机器人学习绘画的短篇故事。',
            '用三句话描述太阳能的优点。',
            '用通俗的语言解释下雨的成因。',
            '给我一个简单的意面晚餐食谱。',
            '用一段话概述《小王子》的主要内容。',
            '列出人工智能系统的主要风险。'
        ][:args.n]
    else:
        prompts = [
            'Write a short story about a robot learning to paint.',
            'Describe the benefits of solar energy in 3 sentences.',
            'Explain what causes rain in simple terms.',
            'Give me a recipe for a quick pasta dinner.',
            'Summarize the book "The Little Prince" in one paragraph.',
            'What are the main risks of AI systems?'
        ][:args.n]

    watermarked_texts = []
    baseline_texts = []

    do_sample = (args.decoding == 'multinomial')
    num_beams = 8 if args.decoding == 'beam8' else 1

    for i, p in enumerate(prompts):
        if not p or not p.strip():
            continue
        try:
            wm = generate_one(model, tokenizer, wm_proc, p, model.device, max_new_tokens=args.max_new_tokens, do_sample=do_sample, num_beams=num_beams)
        except Exception as e:
            logger.exception('watermarked gen failure')
            wm = ''
        try:
            base = generate_one(model, tokenizer, baseline_proc, p, model.device, max_new_tokens=args.max_new_tokens, do_sample=do_sample, num_beams=num_beams)
        except Exception as e:
            logger.exception('baseline gen failure')
            base = ''

        watermarked_texts.append(wm)
        baseline_texts.append(base)

    wz = detect_texts(detector, watermarked_texts)
    bz = detect_texts(detector, baseline_texts)

    all_z = np.array(wz + bz, dtype=float)
    labels = np.array([1]*len(wz) + [0]*len(bz))
    mask = ~np.isnan(all_z)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    result = {
        'model': args.model,
        'gamma': args.gamma,
        'delta': args.delta,
        'prompts': prompts,
        'watermarked_texts': watermarked_texts,
        'baseline_texts': baseline_texts,
        'watermarked_z': wz,
        'baseline_z': bz,
    }

    if mask.sum() > 0:
        fpr, tpr, thr = roc_curve(labels[mask], all_z[mask])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC gamma={args.gamma} delta={args.delta}')
        plt.legend()
        roc_path = out_dir / f'roc_fresh_{args.model.replace("/","_")}_{ts}.png'
        plt.savefig(roc_path, dpi=200, bbox_inches='tight')
        plt.close()
        result['roc_auc'] = float(roc_auc)
        result['roc_path'] = str(roc_path)
    else:
        result['roc_auc'] = None
        result['roc_path'] = None

    out_file = out_dir / f'all_results_fresh_{args.model.replace("/","_")}_{ts}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info('Saved result json to %s', out_file)
    # print a quick z-score summary
    print('watermarked z:', wz)
    print('baseline z:', bz)


if __name__ == '__main__':
    main()
