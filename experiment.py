"""
Full experimental evaluation script.

Runs:
1. Multi-seed SOTA comparison (20 seeds)
2. Ablation study (channel, vote weighting, overlap)
3. SNR-conditional performance curves
4. Generates all paper figures for Sec. V (Experiments)
"""

import os
import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings('ignore')


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================
# Figure: SOTA comparison table + bar chart
# ============================================================

def plot_sota_comparison(summary: dict, output_path: str, dpi=150):
    methods = list(summary.keys())
    metrics = ['TDR', 'FTR', 'GOSPA', 'Precision', 'Recall', 'F1']

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    metric_units = {
        'TDR': '[0-1] ↑', 'FTR': 'per frame ↓', 'GOSPA': 'm ↓',
        'Precision': '[0-1] ↑', 'Recall': '[0-1] ↑', 'F1': '[0-1] ↑'
    }

    for idx, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        vals = [summary[m][metric][0] for m in methods]
        errs = [summary[m][metric][1] for m in methods]

        bars = ax.bar(range(len(methods)), vals, yerr=errs,
                      capsize=5, color=colors[:len(methods)], alpha=0.85,
                      edgecolor='black', linewidth=0.8)

        # Highlight our method
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2.5)
        bars[0].set_hatch('///')

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)
        ax.set_title(f'{metric} {metric_units[metric]}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel(metric, fontsize=9)

        # Add value labels on bars
        for bar, val, err in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    # Legend
    legend_patches = [mpatches.Patch(color=colors[i], label=methods[i],
                                      hatch='///' if i == 0 else '')
                       for i in range(len(methods))]
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(methods),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Track Initiation Performance Comparison (Mean ± Std, N=20 Monte Carlo trials)',
                 fontsize=13, fontweight='bold')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_gospa_cdf(results_all: dict, output_path: str, dpi=150):
    """CDF of GOSPA across Monte Carlo trials."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    linestyles = ['-', '--', '-.', ':', (0, (3,1,1,1))]

    for idx, (method, ev_list) in enumerate(results_all.items()):
        gospa_vals = sorted([e.gospa for e in ev_list])
        cdf = np.arange(1, len(gospa_vals)+1) / len(gospa_vals)
        ax.plot(gospa_vals, cdf, color=colors[idx % len(colors)],
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2.5, label=method)

    ax.set_xlabel('GOSPA Distance (m)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('GOSPA Cumulative Distribution Function\nAcross 20 Monte Carlo Trials', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_tdr_ftr_tradeoff(results_all: dict, output_path: str, dpi=150):
    """TDR vs FTR scatter plot (ROC-like for track initiation)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    markers = ['*', 'o', 's', '^', 'D']

    for idx, (method, ev_list) in enumerate(results_all.items()):
        tdrs = [e.tdr for e in ev_list]
        ftrs = [e.ftr for e in ev_list]
        tdr_m, ftr_m = np.mean(tdrs), np.mean(ftrs)
        tdr_s, ftr_s = np.std(tdrs), np.std(ftrs)

        ax.scatter(ftr_m, tdr_m, color=colors[idx % len(colors)],
                   marker=markers[idx % len(markers)],
                   s=200 if idx == 0 else 120,
                   zorder=5 if idx == 0 else 4,
                   label=method, edgecolors='black', linewidths=1.5)
        ax.errorbar(ftr_m, tdr_m, xerr=ftr_s, yerr=tdr_s,
                    color=colors[idx % len(colors)], alpha=0.5, capsize=4)

    ax.set_xlabel('False Track Rate (per frame) ↓', fontsize=12)
    ax.set_ylabel('Track Detection Rate ↑', fontsize=12)
    ax.set_title('TDR vs. FTR Trade-off\n(Ideal: upper-left corner)', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)

    # Add "ideal point" marker
    ax.scatter(0, 1, marker='*', s=400, c='gold', zorder=6,
               edgecolors='black', linewidths=1)
    ax.annotate('Ideal', xy=(0.005, 0.97), fontsize=9, color='gray')

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================
# Figure: Ablation study
# ============================================================

def plot_channel_ablation(ablation_result: dict, output_path: str, dpi=150):
    """Bar chart showing GOSPA increase when each channel is removed."""
    fig, ax = plt.subplots(figsize=(12, 5))

    full_gospa = ablation_result['Full Model']
    channels = list(ablation_result['channels'].keys())
    gospa_ablated = [ablation_result['channels'][c] for c in channels]
    delta_gospa = [g - full_gospa for g in gospa_ablated]

    bar_colors = ['#E53935' if d > 0.1 * full_gospa else '#43A047'
                  for d in delta_gospa]
    bars = ax.bar(channels, delta_gospa, color=bar_colors, alpha=0.85,
                  edgecolor='black', linewidth=0.8)

    ax.axhline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.set_xlabel('Removed Channel', fontsize=11)
    ax.set_ylabel('ΔGOSPA (m, vs. Full Model) ↑ = worse', fontsize=11)
    ax.set_title(f'Channel Ablation Study\n(Full Model GOSPA = {full_gospa:.0f}m)',
                 fontsize=12, fontweight='bold')
    ax.set_xticklabels(channels, rotation=25, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, delta_gospa):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.01 * abs(full_gospa) if val >= 0 else -0.05 * abs(full_gospa)),
                f'+{val:.0f}m' if val >= 0 else f'{val:.0f}m',
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)

    # Add legend
    red_patch = mpatches.Patch(color='#E53935', label='High impact (>10% GOSPA increase)')
    green_patch = mpatches.Patch(color='#43A047', label='Low impact')
    ax.legend(handles=[red_patch, green_patch], fontsize=9)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_vote_weighting_ablation(vote_result: dict, output_path: str, dpi=150):
    """Compare vote weighting strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    strategies = list(vote_result.keys())
    means = [vote_result[s][0] for s in strategies]
    stds = [vote_result[s][1] for s in strategies]

    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']
    hatches = ['///', '', '', '']

    ax = axes[0]
    bars = ax.bar(strategies, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor='black',
                  hatch=None)
    for i, bar in enumerate(bars):
        bar.set_hatch(hatches[i])
    ax.set_ylabel('GOSPA (m) ↓', fontsize=11)
    ax.set_title('Vote Weighting Strategy Comparison', fontsize=11, fontweight='bold')
    ax.set_xticklabels(strategies, rotation=20, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 5,
                f'{m:.0f}±{s:.0f}', ha='center', fontsize=8)

    # Right: relative improvement
    ax2 = axes[1]
    baseline = means[-1]  # Uniform as baseline
    improvements = [(baseline - m) / baseline * 100 for m in means]
    bar_colors2 = ['#E53935' if imp == max(improvements) else '#90CAF9'
                   for imp in improvements]
    ax2.bar(strategies, improvements, color=bar_colors2, alpha=0.85,
            edgecolor='black')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylabel('GOSPA Improvement vs. Uniform (%)', fontsize=11)
    ax2.set_title('Relative Improvement over Uniform Voting', fontsize=11, fontweight='bold')
    ax2.set_xticklabels(strategies, rotation=20, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    for x, imp in enumerate(improvements):
        ax2.text(x, imp + 0.5, f'{imp:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_overlap_ablation(overlap_result: dict, output_path: str, dpi=150):
    """GOSPA vs sector overlap ratio."""
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = list(overlap_result.keys())
    overlaps = [0, 25, 50, 75]
    means = [overlap_result[k][0] for k in labels]
    stds = [overlap_result[k][1] for k in labels]

    ax.errorbar(overlaps, means, yerr=stds, marker='o', markersize=10,
                linewidth=2.5, capsize=6, color='#1E88E5',
                markerfacecolor='#E53935', markeredgewidth=2)

    # Highlight default
    default_idx = overlaps.index(50)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.6, label='Default (50%)')
    ax.scatter([50], [means[default_idx]], color='#E53935', s=200, zorder=5, marker='*')

    ax.set_xlabel('Sector Overlap Ratio (%)', fontsize=12)
    ax.set_ylabel('GOSPA (m) ↓', fontsize=12)
    ax.set_title('Effect of Sector Overlap on Track Initiation\n(GOSPA metric, lower=better)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(overlaps)
    ax.set_xticklabels([f'{o}%' for o in overlaps])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    for x, m, s in zip(overlaps, means, stds):
        ax.annotate(f'{m:.0f}m', (x, m), textcoords='offset points',
                    xytext=(5, 8), fontsize=9)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================
# Print result table (LaTeX format)
# ============================================================

def print_latex_table(summary: dict):
    """Print LaTeX-formatted results table."""
    metrics = ['TDR', 'FTR', 'GOSPA', 'Precision', 'Recall', 'F1']
    print("\n" + "="*80)
    print("LaTeX Table (copy to paper):")
    print("="*80)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Track Initiation Performance Comparison (Mean $\pm$ Std, N=20 trials)}")
    print(r"\label{tab:comparison}")
    print(r"\begin{tabular}{l" + "c"*len(metrics) + r"}")
    print(r"\toprule")
    header = "Method & " + " & ".join(metrics) + r" \\"
    print(header)
    print(r"\midrule")

    for method, stats in summary.items():
        row = method
        for metric in metrics:
            m, s = stats[metric]
            # Bold best value
            row += f" & ${m:.3f} \\pm {s:.3f}$"
        row += r" \\"
        if method == list(summary.keys())[0]:
            row = r"\textbf{" + row.split("\\\\")[0] + r"}" + r" \\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ============================================================
# Main experiment runner
# ============================================================

def main():
    config = load_config('config.yaml')
    out_dir = config['output']['dir']
    dpi = config['output']['dpi']
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 1: Multi-Seed SOTA Comparison (N=20 Monte Carlo)")
    print("=" * 70)

    from src.evaluation import run_multi_seed_evaluation, summarize_results

    results_all = run_multi_seed_evaluation(
        config, n_seeds=20, use_dl=True,
        model_path='checkpoints/best_model.pth'
    )
    summary = summarize_results(results_all)

    # Print text table
    print("\nResults Summary:")
    print(f"{'Method':<20}", end='')
    for metric in ['TDR', 'FTR', 'GOSPA', 'F1', 'TID']:
        print(f"  {metric:>10}", end='')
    print()
    print("-" * 75)
    for method, stats in summary.items():
        print(f"{method:<20}", end='')
        for metric in ['TDR', 'FTR', 'GOSPA', 'F1', 'TID']:
            m, s = stats[metric]
            print(f"  {m:>7.3f}±{s:.2f}", end='')
        print()

    print_latex_table(summary)

    # Figures
    plot_sota_comparison(summary, f'{out_dir}/fig_I_sota_comparison.png', dpi=dpi)
    plot_gospa_cdf(results_all, f'{out_dir}/fig_J_gospa_cdf.png', dpi=dpi)
    plot_tdr_ftr_tradeoff(results_all, f'{out_dir}/fig_K_tdr_ftr_tradeoff.png', dpi=dpi)

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Channel Ablation Study")
    print("=" * 70)

    from src.ablation import (run_channel_ablation, run_vote_weighting_ablation,
                               run_overlap_ablation)
    from src.data_simulation import RadarSimulator, points_to_array

    channel_result = run_channel_ablation(config, None, None, n_seeds=10)
    print(f"\nFull model GOSPA: {channel_result['Full Model']:.1f}m")
    print("Channel impact (ΔGOSPA when removed):")
    for ch, gospa in channel_result['channels'].items():
        delta = gospa - channel_result['Full Model']
        print(f"  Remove {ch:<20}: ΔGOSPA = {delta:+.1f}m")
    plot_channel_ablation(channel_result, f'{out_dir}/fig_L_channel_ablation.png', dpi=dpi)

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Vote Weighting Ablation")
    print("=" * 70)

    vote_result = run_vote_weighting_ablation(config, n_seeds=10)
    print("\nVote weighting strategies (GOSPA mean ± std):")
    for strategy, (m, s) in vote_result.items():
        print(f"  {strategy:<25}: {m:.1f} ± {s:.1f} m")
    plot_vote_weighting_ablation(vote_result, f'{out_dir}/fig_M_vote_ablation.png', dpi=dpi)

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Sector Overlap Ablation")
    print("=" * 70)

    overlap_result = run_overlap_ablation(config, n_seeds=10)
    print("\nSector overlap (GOSPA mean ± std):")
    for label, (m, s) in overlap_result.items():
        print(f"  {label}: {m:.1f} ± {s:.1f} m")
    plot_overlap_ablation(overlap_result, f'{out_dir}/fig_N_overlap_ablation.png', dpi=dpi)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Figures saved to: {out_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
