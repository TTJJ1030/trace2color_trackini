"""
Main pipeline orchestrator for radar track initiation system.

Runs the full pipeline:
1. Data simulation
2. Pixelization + Layering
3. Block sliding coverage
4. Clutter distribution analysis
5. DL model inference (or threshold-based baseline)
6. QASH track initiation
7. Generate all paper figures
"""

import os
import sys
import yaml
import numpy as np
import torch

from src.data_simulation import RadarSimulator, save_scenario, load_scenario, points_to_array
from src.pixelization import Pixelizer
from src.layering import Layerer
from src.block_sliding import BlockSlidingCoverage
from src.clutter_analysis import ClutterAnalyzer
from src.model import CBAMUNet
from src.track_initiation import QASHTrackInitiator, compute_gospa
from src.visualization import (
    plot_2d_polar_traces, plot_3d_scatter, plot_polar_raw, plot_pixelized_image,
    plot_enhanced_pixelization, plot_layer_tensor, plot_sector_windows,
    plot_clutter_histograms, plot_track_initiation_results,
    plot_classification_metrics, plot_snr_distributions,
)


def load_config(path: str = 'config.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict, use_dl: bool = False,
                 model_path: str = 'checkpoints/best_model.pth'):
    """
    Run the full radar track initiation pipeline.

    Args:
        config: configuration dictionary
        use_dl: use trained DL model for confidence scoring
        model_path: path to trained model checkpoint
    """
    out_dir = config['output']['dir']
    dpi = config['output']['dpi']
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # =========================================================
    # Step 1: Data Simulation
    # =========================================================
    print("\n" + "=" * 60)
    print("Step 1: Simulating radar scenario...")
    data_path = 'data/simulated_scenario.npz'

    sim = RadarSimulator(config)
    points_list = sim.simulate()
    true_tracks = sim.get_true_tracks()
    save_scenario(points_list, true_tracks, data_path)

    points, _ = load_scenario(data_path)
    n_frames = int(points['frame_id'].max()) + 1
    n_targets = int((points['is_target'] == 1).sum())
    n_clutter = int((points['is_target'] == 0).sum())
    print(f"  Frames: {n_frames}, Targets: {n_targets}, Clutter: {n_clutter}")

    # =========================================================
    # Fig A: 2D polar multi-frame traces
    # =========================================================
    print("\nGenerating Fig A: 2D polar point trace plot...")
    plot_2d_polar_traces(points, f'{out_dir}/fig_A_2d_polar_traces.png',
                         dpi=dpi, n_frames=n_frames)

    # =========================================================
    # Fig B: Raw polar plots per batch
    # =========================================================
    print("Generating Fig B: Polar raw data plots...")
    batch_frames = list(range(min(4, n_frames)))
    plot_polar_raw(points, f'{out_dir}/fig_B_polar_raw.png',
                   batch_frames=batch_frames[:2], dpi=dpi,
                   title='Raw Point Traces - First Two Frames')

    # =========================================================
    # Step 2: Pixelization
    # =========================================================
    print("\n" + "=" * 60)
    print("Step 2: Pixelization...")
    pixelizer = Pixelizer(config)
    rgb_triplet_0 = pixelizer.pixelize_batch(points, 0)
    rgb_triplet_1 = pixelizer.pixelize_batch(points, 1)
    print(f"  Pixelized image shape: {rgb_triplet_0.shape}")

    # =========================================================
    # Fig C: Pixelized RGB images (standard + enhanced)
    # =========================================================
    print("Generating Fig C: Pixelized RGB images...")
    plot_pixelized_image(rgb_triplet_0, f'{out_dir}/fig_C_pixelized_triplet0.png',
                         0, 1, 2, dpi=dpi)
    plot_pixelized_image(rgb_triplet_1, f'{out_dir}/fig_C_pixelized_triplet1.png',
                         1, 2, 3, dpi=dpi)

    # Enhanced pixelization: persistence + temporal centroid
    print("Generating Fig C2: Enhanced pixelization...")
    enh0 = pixelizer.pixelize_triplet_enhanced(points, 0, 1, 2)
    plot_enhanced_pixelization(
        enh0['rgb'], enh0['persistence'], enh0['temporal_centroid'],
        f'{out_dir}/fig_C2_enhanced_pixelization.png', 0, 1, 2, dpi=dpi)

    # =========================================================
    # Step 3: Layering (build 10-channel MTSTE tensor)
    # =========================================================
    print("\n" + "=" * 60)
    print("Step 3: Building 10-channel MTSTE tensors...")
    layerer = Layerer(config)
    tensor_0 = layerer.build_tensor(points, 0, 1, 2)
    label_map_0 = layerer.build_label_map(points, 0, 1, 2)
    print(f"  Tensor shape: {tensor_0.shape}")
    print(f"  Label map: {label_map_0.sum()} target cells / {label_map_0.size} total")

    # =========================================================
    # Fig D: Layer tensor visualization
    # =========================================================
    print("Generating Fig D: Multi-layer tensor visualization...")
    plot_layer_tensor(tensor_0, f'{out_dir}/fig_D_layer_tensor.png', dpi=dpi)

    # =========================================================
    # Step 4: Block sliding coverage
    # =========================================================
    print("\n" + "=" * 60)
    print("Step 4: Block sliding coverage...")
    bsc = BlockSlidingCoverage(config)
    stats = bsc.summarize(n_frames)
    print(f"  {stats['n_sectors']} sectors | "
          f"size: {stats['sector_range_width_km']:.0f}km × "
          f"{stats['sector_az_width_deg']:.1f}°")

    viz_sectors = bsc.get_visualization_sectors(n_frames, n_show=3)

    # =========================================================
    # Fig E: Sliding sector windows
    # =========================================================
    print("Generating Fig E: Sliding sector windows...")
    plot_sector_windows(viz_sectors, points,
                        f'{out_dir}/fig_E_sector_windows.png',
                        max_range_km=500.0, dpi=dpi)

    # =========================================================
    # Step 5: Clutter distribution analysis
    # =========================================================
    print("\n" + "=" * 60)
    print("Step 5: Clutter distribution analysis...")
    analyzer = ClutterAnalyzer(config)
    batch_results = analyzer.analyze_all_batches(points)
    print(f"  {len(batch_results)} batches analyzed")
    for k, br in enumerate(batch_results[:2]):
        thr = analyzer.compute_adaptive_threshold(br, pfa=1e-4)
        print(f"  Batch {k}: n={br['n_total']}, "
              f"adaptive threshold={thr:.1f} dB, "
              f"K-dist nu={br['kdist_nu']:.2f}, b={br['kdist_b']:.2f}")

    # =========================================================
    # Fig F: Clutter histograms
    # =========================================================
    print("Generating Fig F: Clutter distribution histograms...")
    plot_clutter_histograms(batch_results, f'{out_dir}/fig_F_clutter_histograms.png', dpi=dpi)
    plot_snr_distributions(batch_results, f'{out_dir}/fig_F2_snr_distributions.png', dpi=dpi)

    # =========================================================
    # Step 6: Model inference (DL or threshold baseline)
    # =========================================================
    print("\n" + "=" * 60)
    conf_threshold = config['model']['conf_threshold']

    if use_dl and os.path.exists(model_path):
        print("Step 6: DL model inference (CBAM-UNet)...")
        confidence_per_point = run_dl_inference(config, points, model_path)
    else:
        print("Step 6: Using SNR-threshold baseline confidence scoring...")
        confidence_per_point = compute_snr_baseline_confidence(points, config)

    pos_rate = (confidence_per_point >= conf_threshold).mean()
    print(f"  Points above threshold ({conf_threshold}): {pos_rate:.1%}")

    # =========================================================
    # Step 7: QASH track initiation
    # =========================================================
    print("\n" + "=" * 60)
    print("Step 7: QASH track initiation...")
    initiator = QASHTrackInitiator(config)

    # Run QASH on the full dataset
    frame_start = 0
    frame_end = min(n_frames - 1, config['track_initiation']['n_frames_window'] - 1)
    tracks = initiator.initiate_tracks(points, confidence_per_point,
                                        frame_start, frame_end)

    # Run baseline for comparison
    baseline_tracks = initiator.baseline_hough_initiation(points, frame_start, frame_end)

    print(f"  QASH initiated {len(tracks)} tracks")
    print(f"  Baseline initiated {len(baseline_tracks)} tracks")

    # Evaluate GOSPA
    gospa_qash = compute_gospa(true_tracks, tracks)
    gospa_baseline = compute_gospa(true_tracks, baseline_tracks)
    print(f"\n  GOSPA (QASH):     {gospa_qash:.1f} m")
    print(f"  GOSPA (Baseline): {gospa_baseline:.1f} m")
    if gospa_baseline > 0:
        improvement = (gospa_baseline - gospa_qash) / gospa_baseline * 100
        print(f"  Improvement: {improvement:+.1f}%")

    # =========================================================
    # Fig G: Track initiation results
    # =========================================================
    print("\nGenerating Fig G: Track initiation results...")
    plot_track_initiation_results(
        points, true_tracks, tracks, confidence_per_point,
        f'{out_dir}/fig_G_track_initiation.png', dpi=dpi)

    # Also plot baseline results
    plot_track_initiation_results(
        points, true_tracks, baseline_tracks, confidence_per_point,
        f'{out_dir}/fig_G_baseline_tracks.png', dpi=dpi)

    # =========================================================
    # Summary metrics
    # =========================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"All figures saved to: {out_dir}/")
    print(f"\nKey Results:")
    print(f"  True targets:           {config['simulation']['n_targets']}")
    print(f"  QASH tracks initiated:  {len(tracks)}")
    print(f"  GOSPA (QASH):          {gospa_qash:.1f} m")
    print(f"  GOSPA (Baseline):      {gospa_baseline:.1f} m")
    print(f"  Improvement:           {improvement if gospa_baseline > 0 else 'N/A'}%")

    return {
        'tracks': tracks,
        'gospa_qash': gospa_qash,
        'gospa_baseline': gospa_baseline,
        'batch_results': batch_results,
    }


def run_dl_inference(config: dict, points: np.ndarray,
                      model_path: str) -> np.ndarray:
    """Run trained CBAM-UNet model and map per-pixel confidences back to points."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfg = config['model']

    model = CBAMUNet(in_channels=model_cfg['n_channels'], n_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    layerer = Layerer(config)
    pixelizer = layerer.pixelizer
    range_edges, az_edges = pixelizer.get_grid_edges(points)
    nr = len(range_edges) - 1
    naz = len(az_edges) - 1

    n_frames = int(points['frame_id'].max()) + 1
    confidence_per_point = np.zeros(len(points), dtype=np.float32)
    count_per_point = np.zeros(len(points), dtype=np.int32)

    # Process each triplet
    for start in range(0, n_frames - 2):
        fr, fg, fb = start, start + 1, start + 2
        tensor = layerer.build_tensor(points, fr, fg, fb)
        tensor_t = torch.from_numpy(tensor[None]).to(device)  # [1, 10, H, W]

        with torch.no_grad():
            conf_map = model.predict_confidence(tensor_t)[0].cpu().numpy()  # [H, W]

        # Map confidences back to points
        for fid in [fr, fg, fb]:
            f_mask = points['frame_id'] == fid
            pts_f = points[f_mask]
            ri = np.searchsorted(range_edges[1:], pts_f['range'], side='right')
            azi = np.searchsorted(az_edges[1:], pts_f['azimuth'], side='right')
            ri = np.clip(ri, 0, nr - 1)
            azi = np.clip(azi, 0, naz - 1)
            idx_f = np.where(f_mask)[0]
            for k, (i, j) in enumerate(zip(ri, azi)):
                confidence_per_point[idx_f[k]] += conf_map[i, j]
                count_per_point[idx_f[k]] += 1

    # Average across triplets
    valid = count_per_point > 0
    confidence_per_point[valid] /= count_per_point[valid]
    return confidence_per_point


def compute_snr_baseline_confidence(points: np.ndarray, config: dict) -> np.ndarray:
    """
    Compute per-point confidence using SNR-threshold baseline.
    Normalizes SNR to [0, 1] with sigmoid-like scaling.
    """
    snr = points['snr'].astype(np.float32)
    threshold = config['layering']['snr_threshold']
    # Sigmoid-like: sigmoid((SNR - threshold) / scale)
    scale = 5.0
    confidence = 1.0 / (1.0 + np.exp(-(snr - threshold) / scale))
    return confidence.astype(np.float32)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Radar Track Initiation Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--use-dl', action='store_true',
                        help='Use trained DL model (requires checkpoints/best_model.pth)')
    parser.add_argument('--model-path', default='checkpoints/best_model.pth',
                        help='Path to trained model checkpoint')
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_pipeline(config, use_dl=args.use_dl, model_path=args.model_path)
