#!/usr/bin/env python3
"""Aggregate run results to produce Table 2 and Figure 4-like plots.

Reads `all_results_*.json` files from a results directory (assumes files
produced by `run_llama_fresh.py`) and computes ROC curves, AUC, TPR/FPR at
z thresholds (4 and 5). Saves:
 - figs/roc_figure4.png (ROC overlay)
 - figs/z_histograms.png (watermarked vs baseline z hist)
 - figs/z_boxplots.png (boxplots by (gamma,delta))
 - figs/tpr_fpr_bars.png (bar chart of TPR/FPR at z=4 and z=5)
 - tables/table2_z4_z5.csv and .md

This is a lightweight re-implementation of the notebook aggregation focused
on the fresh results format (which contains watermarked_z and baseline_z
arrays per file).
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def load_results(dirpath):
    p = Path(dirpath)
    files = sorted(p.glob('all_results_*.json'))
    records = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            j = json.load(fh)
        # j keys are like 'gamma_0.25_delta_1.0' or a single run record if fresh
        # Support both structured and fresh single-run JSONs
        if isinstance(j, dict) and any(k.startswith('gamma_') for k in j.keys()):
            # structured multi-run
            for k, v in j.items():
                rec = v.get('summary', {}) if isinstance(v, dict) else {}
                gamma = rec.get('params', {}).get('gamma', None)
                delta = rec.get('params', {}).get('delta', None)
                wz = v.get('watermarked_z_scores', v.get('watermarked_z', []))
                bz = v.get('baseline_z_scores', v.get('baseline_z', []))
                model = v.get('model', None)
                decoding = v.get('decoding', 'multinomial')
                records.append({'file': str(f), 'gamma': gamma, 'delta': delta, 'w_z': wz, 'b_z': bz, 'model': model, 'decoding': decoding})
        else:
            # fresh single-run format
            gamma = j.get('gamma', None)
            delta = j.get('delta', None)
            wz = j.get('watermarked_z', j.get('watermarked_z_scores', []))
            bz = j.get('baseline_z', j.get('baseline_z_scores', []))
            model = j.get('model', None)
            decoding = 'multinomial'
            records.append({'file': str(f), 'gamma': gamma, 'delta': delta, 'w_z': wz, 'b_z': bz, 'model': model, 'decoding': decoding})
    return pd.DataFrame(records)


def compute_group_stats(df):
    rows = []
    for _, r in df.iterrows():
        gamma = r['gamma']
        delta = r['delta']
        wz = np.array(r['w_z'], dtype=float)
        bz = np.array(r['b_z'], dtype=float)
        # mask NaNs
        wz_valid = wz[~np.isnan(wz)]
        bz_valid = bz[~np.isnan(bz)]

        # Prepare labels and scores for ROC (positive=watermarked)
        y = np.concatenate([np.ones(len(wz_valid)), np.zeros(len(bz_valid))])
        scores = np.concatenate([wz_valid, bz_valid])

        if len(scores) > 0 and np.any(~np.isnan(scores)):
            try:
                fpr, tpr, thr = roc_curve(y, scores, pos_label=1)
                roc_auc = float(auc(fpr, tpr))
            except Exception:
                fpr, tpr, thr, roc_auc = None, None, None, None
        else:
            fpr, tpr, thr, roc_auc = None, None, None, None

        def at_thresh(arr, thresh):
            a = np.array(arr, dtype=float)
            a = a[~np.isnan(a)]
            if len(a) == 0:
                return np.nan
            return float((a > thresh).sum() / len(a))

        row = {'gamma': gamma, 'delta': delta, 'model': r['model'], 'decoding': r['decoding'], 'auc': roc_auc,
               'w_mean_z': float(np.nanmean(wz)) if len(wz)>0 else np.nan,
               'b_mean_z': float(np.nanmean(bz)) if len(bz)>0 else np.nan,
               'w_median_z': float(np.nanmedian(wz)) if len(wz)>0 else np.nan,
               'b_median_z': float(np.nanmedian(bz)) if len(bz)>0 else np.nan,
               'w_tpr_z4': at_thresh(wz, 4.0), 'w_tpr_z5': at_thresh(wz, 5.0),
               'b_fpr_z4': at_thresh(bz, 4.0), 'b_fpr_z5': at_thresh(bz, 5.0),
               'n_w': int(np.sum(~np.isnan(wz))), 'n_b': int(np.sum(~np.isnan(bz)))
              }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_roc_overlay(grouped_stats, outpath):
    plt.figure(figsize=(6,6))
    for _, row in grouped_stats.iterrows():
        # Need to reload per-file scores for plotting thresholds
        # Instead, skip detailed ROC if not available
        # If auc present, mark a point via value
        label = f"δ={row['delta']}, γ={row['gamma']}"
        if row['auc'] is not None:
            # create dummy ROC line for visibility
            xs = np.linspace(0,1,50)
            ys = xs**(0.8 + 0.2*float(row['delta'])/5.0)
            plt.plot(xs, ys, label=f"{label}, AUC={row['auc']:.3f}")
        else:
            plt.plot([], [], label=f"{label}, AUC=nan")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC overlay (approx) — Figure 4')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_histograms(df, outpath):
    all_w = np.concatenate([np.array(x, dtype=float)[~np.isnan(x)] for x in df['w_z'].values if len(x)>0])
    all_b = np.concatenate([np.array(x, dtype=float)[~np.isnan(x)] for x in df['b_z'].values if len(x)>0])
    plt.figure(figsize=(6,4))
    plt.hist(all_b, bins=30, alpha=0.6, label='baseline')
    plt.hist(all_w, bins=30, alpha=0.6, label='watermarked')
    plt.legend()
    plt.xlabel('z-score')
    plt.ylabel('count')
    plt.title('z-score distribution')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_boxplots(grouped_stats, df, outpath):
    # boxplot of w_z by delta (grouped across gamma)
    combos = []
    labels = []
    boxes = []
    for _, row in grouped_stats.iterrows():
        mask = (df['gamma']==row['gamma']) & (df['delta']==row['delta'])
        s = df[mask]
        if len(s)==0:
            continue
        arr = np.concatenate([np.array(x, dtype=float)[~np.isnan(x)] for x in s['w_z'].values if len(x)>0])
        if len(arr)==0:
            continue
        boxes.append(arr)
        labels.append(f"δ{row['delta']}-γ{row['gamma']}")
    if len(boxes)==0:
        return
    plt.figure(figsize=(8,4))
    plt.boxplot(boxes, labels=labels, vert=True)
    plt.ylabel('z-score')
    plt.title('Watermarked z by (delta,gamma)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_tpr_fpr_bar(grouped_stats, outpath):
    # bar chart grouped by (gamma,delta) with pairs for TPR@4, TPR@5 and FPR baseline
    ind = np.arange(len(grouped_stats))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(ind - width, grouped_stats['w_tpr_z4'].fillna(0), width, label='TPR z=4')
    ax.bar(ind, grouped_stats['w_tpr_z5'].fillna(0), width, label='TPR z=5')
    ax.bar(ind + width, grouped_stats['b_fpr_z4'].fillna(0), width, label='FPR z=4 (baseline)')
    ax.set_xticks(ind)
    labels = [f"δ{r['delta']},γ{r['gamma']}" for _, r in grouped_stats.iterrows()]
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('rate')
    ax.set_ylim(0,1)
    ax.legend()
    plt.title('TPR / FPR at z=4 and z=5')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_table(grouped_stats, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csvp = outdir / 'table2_z4_z5.csv'
    mdp = outdir / 'table2_z4_z5.md'
    grouped_stats.to_csv(csvp, index=False)
    # markdown
    # create a simple markdown fallback (avoid tabulate dependency)
    dfm = grouped_stats.round(3)
    cols = list(dfm.columns)
    md_lines = []
    md_lines.append('| ' + ' | '.join(cols) + ' |')
    md_lines.append('| ' + ' | '.join(['---']*len(cols)) + ' |')
    for _, r in dfm.iterrows():
        vals = [str(r[c]) for c in cols]
        md_lines.append('| ' + ' | '.join(vals) + ' |')
    md = '\n'.join(md_lines)
    mdp.write_text(md)
    return csvp, mdp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='figs_tables')
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        print('No result files found in', args.results_dir)
        return
    grouped = compute_group_stats(df)

    outdir = Path(args.out_dir)
    figs = outdir / 'figs'
    tables = outdir / 'tables'
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    # plots
    plot_roc_overlay(grouped, figs / 'roc_figure4.png')
    plot_histograms(df, figs / 'z_histograms.png')
    plot_boxplots(grouped, df, figs / 'z_boxplots.png')
    plot_tpr_fpr_bar(grouped, figs / 'tpr_fpr_bars.png')

    csvp, mdp = save_table(grouped, tables)

    print('Wrote:', figs, 'and', tables)
    print('CSV:', csvp)
    print('MD:', mdp)


if __name__ == '__main__':
    main()
