"""
Visualization module for all paper figures.

Generates publication-quality figures for radar track initiation pipeline.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional
import os


def save_fig(fig, path: str, dpi: int = 150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_2d_polar_traces(points: np.ndarray, output_path: str, dpi: int = 150,
                         n_frames: int = 20):
    """
    Fig A: 2D polar plot of multi-frame point traces.
    Left panel: polar plot coloured by frame index (clutter=gray, target=colour gradient).
    Right panel: range–frame waterfall for target tracks.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={'width_ratios': [1.2, 1]})
    ax_polar = fig.add_subplot(121, projection='polar')
    axes[0].remove()   # replace first slot with polar
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)

    # Clutter – subsample
    clutter_mask = points['is_target'] == 0
    target_mask  = points['is_target'] == 1
    rng = np.random.default_rng(0)
    n_show = min(3000, clutter_mask.sum())
    c_idx = rng.choice(np.where(clutter_mask)[0], n_show, replace=False)
    c_pts = points[c_idx]
    ax_polar.scatter(np.deg2rad(c_pts['azimuth']), c_pts['range'] / 1000,
                     s=0.5, c='steelblue', alpha=0.2, label='Clutter')

    # Targets coloured by frame index
    t_pts = points[target_mask]
    sc = ax_polar.scatter(np.deg2rad(t_pts['azimuth']), t_pts['range'] / 1000,
                          c=t_pts['frame_id'], cmap='plasma', s=25, alpha=0.9,
                          marker='*', zorder=5, label='Target')
    cbar = plt.colorbar(sc, ax=ax_polar, shrink=0.6, pad=0.08)
    cbar.set_label('Frame index', fontsize=9)
    ax_polar.set_rmax(500)
    ax_polar.set_rgrids([100, 200, 300, 400, 500])
    ax_polar.set_title('2D Polar Point Traces\n(all frames)', fontsize=11, pad=20)
    ax_polar.legend(loc='lower right', fontsize=8, markerscale=2)

    # Right: range-frame waterfall
    ax2 = axes[1]
    frame_ids = np.arange(n_frames)
    snr_vals_all = []
    for fid in frame_ids:
        mask = (points['frame_id'] == fid) & target_mask
        snr_vals_all.append(points['snr'][mask].mean() if mask.any() else np.nan)

    # Clutter density per frame
    clutter_counts = [(points['frame_id'] == fid).sum() for fid in frame_ids]

    ax2.plot(frame_ids, snr_vals_all, 'r-o', ms=4, linewidth=1.5, label='Mean target SNR')
    ax2_twin = ax2.twinx()
    ax2_twin.bar(frame_ids, clutter_counts, color='steelblue', alpha=0.3, label='Clutter count')
    ax2_twin.set_ylabel('Clutter point count', fontsize=9, color='steelblue')
    ax2.set_xlabel('Frame index', fontsize=10)
    ax2.set_ylabel('Mean target SNR (dB)', fontsize=10, color='red')
    ax2.set_title('Per-frame Statistics', fontsize=11)
    ax2.grid(True, alpha=0.3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.suptitle('Multi-frame 2D Radar Point Trace Data', fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(fig, output_path, dpi)


# Keep backward-compatible alias
def plot_3d_scatter(points: np.ndarray, output_path: str, dpi: int = 150,
                    n_frames: int = 20):
    plot_2d_polar_traces(points, output_path, dpi=dpi, n_frames=n_frames)


def plot_polar_raw(points: np.ndarray, output_path: str,
                   batch_frames: List[int], dpi: int = 150,
                   title: str = 'Raw Point Traces (Polar)'):
    """
    Fig B: Polar plot of raw detections for given frames (like Fig 5-4a).
    """
    fig, axes = plt.subplots(1, len(batch_frames), figsize=(6 * len(batch_frames), 6),
                             subplot_kw={'projection': 'polar'})
    if len(batch_frames) == 1:
        axes = [axes]

    for ax, fid in zip(axes, batch_frames):
        mask = points['frame_id'] == fid
        pts = points[mask]
        az_rad = np.deg2rad(pts['azimuth'])
        ax.scatter(az_rad, pts['range'] / 1000, s=1, c='blue', alpha=0.4)

        # Highlight targets
        t_mask = pts['is_target'] == 1
        if t_mask.any():
            ax.scatter(az_rad[t_mask], pts['range'][t_mask] / 1000,
                       s=30, c='red', marker='*', zorder=5)

        ax.set_title(f'Frame {fid}', fontsize=11)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rmax(500)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    save_fig(fig, output_path, dpi)


def plot_pixelized_image(rgb_image: np.ndarray, output_path: str,
                         frame_r: int, frame_g: int, frame_b: int,
                         dpi: int = 150):
    """
    Fig C: RGB pixelized image of frame triplet (like Fig 5-1b,c).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: show as image (range vs azimuth)
    ax = axes[0]
    ax.imshow(rgb_image, aspect='auto', origin='lower')
    ax.set_xlabel('Azimuth bin', fontsize=10)
    ax.set_ylabel('Range bin', fontsize=10)
    ax.set_title(f'RGB Pixelized: R=frame{frame_r}, G=frame{frame_g}, B=frame{frame_b}',
                 fontsize=10)

    # Add legend patches
    patches = [
        mpatches.Patch(color='red', label=f'Frame {frame_r}'),
        mpatches.Patch(color='green', label=f'Frame {frame_g}'),
        mpatches.Patch(color='blue', label=f'Frame {frame_b}'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=9)

    # Right: per-channel intensity
    ax2 = axes[1]
    ax2.plot(rgb_image[:, :, 0].mean(axis=1), 'r-', linewidth=1, label='R (target)')
    ax2.plot(rgb_image[:, :, 1].mean(axis=1), 'g-', linewidth=1, label='G')
    ax2.plot(rgb_image[:, :, 2].mean(axis=1), 'b-', linewidth=1, label='B')
    ax2.set_xlabel('Range bin', fontsize=10)
    ax2.set_ylabel('Mean pixel value', fontsize=10)
    ax2.set_title('Channel intensity profiles', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, output_path, dpi)


def plot_layer_tensor(tensor: np.ndarray, output_path: str, dpi: int = 150):
    """
    Fig D: 10-channel MTSTE tensor visualization (2×5 grid).

    Ch 0: Normalized range coordinate
    Ch 1: Normalized azimuth coordinate
    Ch 2: Temporal marker
    Ch 3: RGB-R (frame t)
    Ch 4: RGB-G (frame t+1)
    Ch 5: RGB-B (frame t+2)
    Ch 6: SNR map
    Ch 7: Range span map
    Ch 8: Temporal persistence (fraction of frames with detections)
    Ch 9: Temporal SNR centroid (SNR-weighted frame centre-of-mass)
    """
    n_ch = tensor.shape[0]
    channel_names = [
        'Range Coord', 'Azimuth Coord',
        'Time Marker', 'RGB-R (t)',
        'RGB-G (t+1)', 'RGB-B (t+2)',
        'SNR Map', 'Range Span Map',
        'Temporal Persistence', 'SNR Temporal Centroid'
    ][:n_ch]

    colormaps = ['viridis', 'plasma', 'coolwarm', 'Reds',
                 'Greens', 'Blues', 'hot', 'YlOrRd',
                 'magma', 'cividis'][:n_ch]

    ncols = 5
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes = np.array(axes).ravel()

    for i in range(n_ch):
        ax = axes[i]
        ch = tensor[i]
        vmax = ch.max() if ch.max() > 0 else 1.0
        im = ax.imshow(ch, aspect='auto', origin='lower', cmap=colormaps[i],
                       vmin=0, vmax=vmax)
        ax.set_title(f'Ch{i}: {channel_names[i]}', fontsize=8, fontweight='bold')
        ax.set_xlabel('Azimuth bin', fontsize=7)
        ax.set_ylabel('Range bin', fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for j in range(n_ch, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('10-Channel MTSTE: Multi-Layer Temporal-Spatial Tensor Encoding',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(fig, output_path, dpi)


def plot_sector_windows(sectors, points: np.ndarray, output_path: str,
                        max_range_km: float = 500.0, dpi: int = 150):
    """
    Fig E: Sliding sector windows on polar plot (like Fig 5-3).
    Shows 3 consecutive sector windows.
    """
    n = min(len(sectors), 3)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6),
                             subplot_kw={'projection': 'polar'})
    if n == 1:
        axes = [axes]

    colors = ['#2196F3', '#4CAF50', '#F44336']
    labels = ['a) Sector Window 1', 'b) Sector Window 2', 'c) Sector Window 3']

    for k, (ax, sec) in enumerate(zip(axes, sectors[:n])):
        # Plot all points (background)
        f_mask = ((points['frame_id'] >= sec.frame_start) &
                  (points['frame_id'] <= sec.frame_end))
        pts_all = points[f_mask]

        # Subsample
        n_show = min(500, len(pts_all))
        idx = np.random.choice(len(pts_all), n_show, replace=False)
        pts_show = pts_all[idx]

        az_rad = np.deg2rad(pts_show['azimuth'])
        ax.scatter(az_rad, pts_show['range'] / 1000, s=1, c='gray', alpha=0.3)

        # Draw sector boundary
        az_min = np.deg2rad(sec.az_min)
        az_max = np.deg2rad(sec.az_max % 360)
        r_min_km = sec.range_min / 1000
        r_max_km = sec.range_max / 1000

        theta_fill = np.linspace(az_min, az_max, 100)
        ax.fill_between(theta_fill, r_min_km, r_max_km,
                        alpha=0.3, color=colors[k])
        ax.plot([az_min, az_min], [r_min_km, r_max_km], color=colors[k], lw=2)
        ax.plot([az_max, az_max], [r_min_km, r_max_km], color=colors[k], lw=2)
        theta_arc = np.linspace(az_min, az_max, 100)
        ax.plot(theta_arc, [r_min_km] * 100, color=colors[k], lw=2)
        ax.plot(theta_arc, [r_max_km] * 100, color=colors[k], lw=2)

        ax.set_title(labels[k], fontsize=11, pad=15)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rmax(max_range_km)
        ax.set_rgrids([100, 200, 300, 400, 500])

    plt.tight_layout()
    save_fig(fig, output_path, dpi)


def plot_clutter_histograms(batch_results: List[Dict], output_path: str,
                             dpi: int = 150):
    """
    Fig F: 3D clutter distribution histograms per batch (like Fig 5-4b).
    """
    n_batches = min(len(batch_results), 4)
    fig = plt.figure(figsize=(7 * n_batches, 7))

    for k, result in enumerate(batch_results[:n_batches]):
        ax = fig.add_subplot(1, n_batches, k + 1, projection='3d')
        dm = result['density_map']
        R, A = dm.shape

        # Downsample for plotting
        step = max(1, R // 40)
        dm_ds = dm[::step, ::step]
        r_c = result['range_centers'][::step] / 1000
        a_c = result['az_centers'][::step]

        xpos, ypos = np.meshgrid(np.arange(len(r_c)), np.arange(len(a_c)), indexing='ij')
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)
        dz = dm_ds.ravel()
        dx = dy = 0.8

        colors = cm.viridis(dz / max(dz.max(), 1.0))
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8, shade=True)

        ax.set_xlabel('Range bin', fontsize=8, labelpad=5)
        ax.set_ylabel('Azimuth bin', fontsize=8, labelpad=5)
        ax.set_zlabel('Count', fontsize=8)
        f0, f1 = result['frame_start'], result['frame_end']
        ax.set_title(f'Batch {k + 1}\n(frames {f0}-{f1})\nn={result["n_total"]}',
                     fontsize=9)

    plt.suptitle('Clutter Distribution Statistics per Batch', fontsize=13)
    plt.tight_layout()
    save_fig(fig, output_path, dpi)


def plot_track_initiation_results(points: np.ndarray,
                                  true_tracks: Dict,
                                  initiated_tracks: List,
                                  confidence_map: np.ndarray,
                                  output_path: str,
                                  dpi: int = 150):
    """
    Fig G: Track initiation results on polar plot.
    Shows: all points (gray), high-confidence points (colored by grade),
    initiated tracks (solid lines), true tracks (dashed lines).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                             subplot_kw={'projection': 'polar'})

    for ax_idx, (ax, title) in enumerate(zip(axes, ['QASH Track Initiation', 'True Tracks'])):
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rmax(500)
        ax.set_title(title, fontsize=12, pad=15)
        ax.set_rgrids([100, 200, 300, 400, 500], ['100', '200', '300', '400', '500km'])

    # Left: QASH results
    ax = axes[0]
    # Background: all points
    n_show = min(2000, len(points))
    idx = np.random.choice(len(points), n_show, replace=False)
    pts_bg = points[idx]
    az_bg = np.deg2rad(pts_bg['azimuth'])
    ax.scatter(az_bg, pts_bg['range'] / 1000, s=0.5, c='lightgray', alpha=0.5)

    # High-confidence points colored by quality
    high_mask = confidence_map >= 0.5
    pts_hc = points[high_mask]
    conf_hc = confidence_map[high_mask]
    if len(pts_hc) > 0:
        az_hc = np.deg2rad(pts_hc['azimuth'])
        sc = ax.scatter(az_hc, pts_hc['range'] / 1000,
                        c=conf_hc, cmap='hot', s=8, alpha=0.8,
                        vmin=0.5, vmax=1.0)
        plt.colorbar(sc, ax=ax, shrink=0.6, label='DL Confidence')

    # Initiated tracks
    track_colors = plt.cm.Set1(np.linspace(0, 1, max(len(initiated_tracks), 1)))
    for t_idx, track in enumerate(initiated_tracks):
        if not track.supporting_points:
            continue
        sp_sorted = sorted(track.supporting_points, key=lambda p: p['frame_id'])
        azes = np.deg2rad([p['azimuth'] for p in sp_sorted])
        rngs = [p['range'] / 1000 for p in sp_sorted]
        ax.plot(azes, rngs, '-', color=track_colors[t_idx % len(track_colors)],
                linewidth=2, alpha=0.9, label=f'Track {track.track_id}')
        ax.scatter(azes, rngs, color=track_colors[t_idx % len(track_colors)],
                   s=20, zorder=5)

    if initiated_tracks:
        ax.legend(loc='upper right', fontsize=7, ncol=2)

    # Right: true tracks
    ax2 = axes[1]
    # Background
    ax2.scatter(az_bg, pts_bg['range'] / 1000, s=0.5, c='lightgray', alpha=0.5)

    true_colors = plt.cm.tab10(np.linspace(0, 1, max(len(true_tracks), 1)))
    for t_idx, (tid, traj) in enumerate(true_tracks.items()):
        # traj is [n_frames, 2]: (range_m, azimuth_deg); r=0 means invalid
        r_km = traj[:, 0] / 1000
        az_rad = np.deg2rad(traj[:, 1])
        valid = r_km > 0
        ax2.plot(az_rad[valid], r_km[valid], '--',
                 color=true_colors[t_idx % len(true_colors)],
                 linewidth=2, alpha=0.9, label=f'Target {tid}')
        ax2.scatter(az_rad[valid], r_km[valid],
                    color=true_colors[t_idx % len(true_colors)], s=15, zorder=5)

    ax2.legend(loc='upper right', fontsize=8)

    plt.suptitle('Track Initiation Results', fontsize=14)
    plt.tight_layout()
    save_fig(fig, output_path, dpi)


def plot_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_score: np.ndarray, output_path: str,
                                 dpi: int = 150):
    """
    Fig H: Classification metrics — confusion matrix + ROC + PR curves.
    """
    from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                                  precision_recall_curve, average_precision_score,
                                  classification_report)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion matrix
    ax = axes[0]
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Clutter', 'Target'])
    ax.set_yticklabels(['Clutter', 'Target'])
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    # ROC curve
    ax2 = axes[1]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax2.set_xlabel('False Positive Rate', fontsize=10)
    ax2.set_ylabel('True Positive Rate', fontsize=10)
    ax2.set_title('ROC Curve', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Precision-Recall curve
    ax3 = axes[2]
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    ax3.plot(rec, prec, 'r-', linewidth=2, label=f'PR curve (AP={ap:.3f})')
    ax3.set_xlabel('Recall', fontsize=10)
    ax3.set_ylabel('Precision', fontsize=10)
    ax3.set_title('Precision-Recall Curve', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Print report
    report = classification_report(y_true, y_pred,
                                    target_names=['Clutter', 'Target'])
    print("\n=== Classification Report ===")
    print(report)

    plt.tight_layout()
    save_fig(fig, output_path, dpi)


def plot_enhanced_pixelization(rgb_image: np.ndarray,
                                persistence_map: np.ndarray,
                                temporal_centroid: np.ndarray,
                                output_path: str,
                                frame_r: int, frame_g: int, frame_b: int,
                                dpi: int = 150):
    """
    Fig C2: Enhanced pixelization showing RGB, persistence, and temporal centroid.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    ax = axes[0]
    ax.imshow(rgb_image, aspect='auto', origin='lower')
    ax.set_title(f'RGB Triplet\nR=f{frame_r}, G=f{frame_g}, B=f{frame_b}', fontsize=10)
    ax.set_xlabel('Azimuth bin'); ax.set_ylabel('Range bin')
    patches = [mpatches.Patch(color='red', label=f'Frame {frame_r}'),
               mpatches.Patch(color='green', label=f'Frame {frame_g}'),
               mpatches.Patch(color='blue', label=f'Frame {frame_b}')]
    ax.legend(handles=patches, fontsize=8, loc='upper right')

    ax2 = axes[1]
    im2 = ax2.imshow(persistence_map, aspect='auto', origin='lower',
                     cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_title('Temporal Persistence\n(fraction of frames with detection)', fontsize=10)
    ax2.set_xlabel('Azimuth bin'); ax2.set_ylabel('Range bin')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = axes[2]
    im3 = ax3.imshow(temporal_centroid, aspect='auto', origin='lower',
                     cmap='coolwarm', vmin=0, vmax=1)
    ax3.set_title('Temporal SNR Centroid\n(0=early, 1=late frame)', fontsize=10)
    ax3.set_xlabel('Azimuth bin'); ax3.set_ylabel('Range bin')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Channel profiles
    ax4 = axes[3]
    ax4.plot(rgb_image[:, :, 0].mean(axis=1), 'r-', lw=1.5, label='Red (t)')
    ax4.plot(rgb_image[:, :, 1].mean(axis=1), 'g-', lw=1.5, label='Green (t+1)')
    ax4.plot(rgb_image[:, :, 2].mean(axis=1), 'b-', lw=1.5, label='Blue (t+2)')
    ax4.plot(persistence_map.mean(axis=1), 'k--', lw=1.5, label='Persistence')
    ax4.set_xlabel('Range bin', fontsize=10)
    ax4.set_ylabel('Mean value', fontsize=10)
    ax4.set_title('Channel Intensity Profiles', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Enhanced Multi-Resolution Pixelization', fontsize=13)
    plt.tight_layout()
    save_fig(fig, output_path, dpi)


def plot_snr_distributions(batch_results: List[Dict], output_path: str,
                            dpi: int = 150):
    """
    Additional: SNR distribution comparison per batch (target vs clutter).
    """
    n = min(len(batch_results), 4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for k, (ax, result) in enumerate(zip(axes, batch_results[:n])):
        if len(result['clutter_snr']) > 0:
            ax.hist(result['clutter_snr'], bins=50, density=True,
                    alpha=0.6, color='blue', label='Clutter')
        if len(result['target_snr']) > 0:
            ax.hist(result['target_snr'], bins=20, density=True,
                    alpha=0.6, color='red', label='Target')
        ax.set_xlabel('SNR (dB)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Batch {k + 1}: frames {result["frame_start"]}-{result["frame_end"]}',
                     fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add text stats
        if len(result['clutter_snr']) > 0:
            ax.axvline(result['clutter_snr'].mean(), color='blue',
                       linestyle='--', alpha=0.8)

    plt.suptitle('SNR Distribution: Target vs Clutter per Batch', fontsize=13)
    plt.tight_layout()
    save_fig(fig, output_path, dpi)
