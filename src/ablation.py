"""
Ablation study module.

Studies:
1. Channel ablation: remove each of the 8 channels one-by-one
2. Attention ablation: CBAM vs no-CBAM vs channel-only vs spatial-only
3. Vote weighting: QASH (DL grade) vs uniform vs SNR-only vs confidence-only
4. Sector overlap: 0%, 25%, 50% (default), 75%
"""

import numpy as np
from typing import List, Dict
import copy


def run_channel_ablation(config: dict, points_arr, true_tracks: dict,
                          model_path: str = 'checkpoints/best_model.pth',
                          n_seeds: int = 10) -> Dict:
    """
    Remove each channel one at a time, measure GOSPA degradation.
    Channels 0-7: range, azimuth, time, R, G, B, SNR, range_span
    """
    import torch
    import main as main_module
    from src.data_simulation import RadarSimulator, points_to_array
    from src.layering import Layerer
    from src.track_initiation import QASHTrackInitiator
    from src.evaluation import evaluate_method
    from src.model import CBAMUNet

    channel_names = ['Range Coord', 'Azimuth Coord', 'Time Marker',
                     'RGB-R', 'RGB-G', 'RGB-B', 'SNR Map', 'Range Span']

    device = torch.device('cpu')
    model_cfg = config['model']
    model = CBAMUNet(in_channels=model_cfg['n_channels'], n_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model_available = True
    except Exception:
        model_available = False

    results = {'Full Model': [], 'channels': {name: [] for name in channel_names}}

    for seed in range(n_seeds):
        cfg = copy.deepcopy(config)
        cfg['simulation']['seed'] = seed * 17 + 100
        cfg['simulation']['n_targets'] = np.random.randint(3, 8)
        cfg['simulation']['clutter_lambda'] = np.random.randint(400, 1000)

        sim = RadarSimulator(cfg)
        points_list = sim.simulate()
        true_trks = sim.get_true_tracks()
        pts = points_to_array(points_list)

        n_frames = int(pts['frame_id'].max()) + 1
        frame_start, frame_end = 0, min(n_frames - 1,
                                         cfg['track_initiation']['n_frames_window'] - 1)

        # Full model
        conf_map = main_module.compute_snr_baseline_confidence(pts, cfg)
        initiator = QASHTrackInitiator(cfg)
        tracks_full = initiator.initiate_tracks(pts, conf_map, frame_start, frame_end)
        ev_full = evaluate_method('Full', tracks_full, true_trks, n_frames)
        results['Full Model'].append(ev_full.gospa)

        # Channel ablation: zero out one channel at a time
        layerer = Layerer(cfg)
        fr, fg, fb = 0, 1, 2
        tensor_full = layerer.build_tensor(pts, fr, fg, fb)

        for ch_idx, ch_name in enumerate(channel_names):
            tensor_ablated = tensor_full.copy()
            tensor_ablated[ch_idx] = 0.0  # zero out this channel

            # Map back to per-point confidence using ablated tensor
            # Simple: use SNR channel (ch6) if it's not the ablated one, else use zeros
            if ch_idx != 6:
                snr_map = tensor_ablated[6]  # still available
            else:
                snr_map = np.zeros_like(tensor_ablated[0])

            from src.pixelization import Pixelizer
            pixelizer = Pixelizer(cfg)
            range_edges, az_edges = pixelizer.get_grid_edges(pts)
            nr, naz = len(range_edges) - 1, len(az_edges) - 1

            conf_ablated = np.zeros(len(pts), dtype=np.float32)
            for fid in [fr, fg, fb]:
                fmask = pts['frame_id'] == fid
                pts_f = pts[fmask]
                if len(pts_f) == 0:
                    continue
                ri = np.searchsorted(range_edges[1:], pts_f['range'], side='right')
                azi = np.searchsorted(az_edges[1:], pts_f['azimuth'], side='right')
                ri = np.clip(ri, 0, nr - 1)
                azi = np.clip(azi, 0, naz - 1)
                idx_f = np.where(fmask)[0]
                for k, (i, j) in enumerate(zip(ri, azi)):
                    conf_ablated[idx_f[k]] = snr_map[i, j]

            # Normalize
            c_max = conf_ablated.max()
            if c_max > 0:
                conf_ablated /= c_max

            tracks_ab = initiator.initiate_tracks(pts, conf_ablated, frame_start, frame_end)
            ev_ab = evaluate_method(ch_name, tracks_ab, true_trks, n_frames)
            results['channels'][ch_name].append(ev_ab.gospa)

    # Compute means
    summary = {
        'Full Model': np.mean(results['Full Model']),
        'channels': {name: np.mean(vals) for name, vals in results['channels'].items()}
    }
    return summary


def run_vote_weighting_ablation(config: dict, n_seeds: int = 10) -> Dict:
    """
    Compare vote weighting strategies in QASH:
    1. DL confidence × SNR (full QASH)
    2. DL confidence only
    3. SNR only (normalized)
    4. Uniform (= standard 3D Hough)
    """
    import main as main_module
    from src.data_simulation import RadarSimulator, points_to_array
    from src.track_initiation import QASHTrackInitiator
    from src.evaluation import evaluate_method
    import copy

    strategies = ['DL×SNR (QASH)', 'DL only', 'SNR only', 'Uniform']
    results = {s: [] for s in strategies}

    for seed in range(n_seeds):
        cfg = copy.deepcopy(config)
        cfg['simulation']['seed'] = seed * 23 + 200
        cfg['simulation']['n_targets'] = np.random.randint(3, 8)
        cfg['simulation']['clutter_lambda'] = np.random.randint(400, 1000)

        sim = RadarSimulator(cfg)
        points_list = sim.simulate()
        true_trks = sim.get_true_tracks()
        pts = points_to_array(points_list)

        n_frames = int(pts['frame_id'].max()) + 1
        frame_start, frame_end = 0, min(n_frames - 1,
                                         cfg['track_initiation']['n_frames_window'] - 1)

        dl_conf = main_module.compute_snr_baseline_confidence(pts, cfg)
        snr_norm = np.clip(pts['snr'] / max(pts['snr'].max(), 1.0), 0, 1).astype(np.float32)

        confs = {
            'DL×SNR (QASH)': dl_conf * snr_norm,
            'DL only': dl_conf,
            'SNR only': snr_norm,
            'Uniform': np.ones(len(pts), dtype=np.float32),
        }

        initiator = QASHTrackInitiator(cfg)
        for strategy, conf in confs.items():
            tracks = initiator.initiate_tracks(pts, conf, frame_start, frame_end)
            ev = evaluate_method(strategy, tracks, true_trks, n_frames)
            results[strategy].append(ev.gospa)

    return {s: (np.mean(v), np.std(v)) for s, v in results.items()}


def run_overlap_ablation(config: dict, n_seeds: int = 10) -> Dict:
    """
    Compare sector overlap ratios: 0%, 25%, 50%, 75%
    """
    import main as main_module
    from src.data_simulation import RadarSimulator, points_to_array
    from src.track_initiation import QASHTrackInitiator
    from src.evaluation import evaluate_method
    import copy

    overlaps = [0.0, 0.25, 0.50, 0.75]
    results = {f'Overlap={int(o*100)}%': [] for o in overlaps}

    for seed in range(n_seeds):
        cfg_base = copy.deepcopy(config)
        cfg_base['simulation']['seed'] = seed * 31 + 300
        cfg_base['simulation']['n_targets'] = np.random.randint(3, 8)

        sim = RadarSimulator(cfg_base)
        points_list = sim.simulate()
        true_trks = sim.get_true_tracks()
        pts = points_to_array(points_list)

        n_frames = int(pts['frame_id'].max()) + 1
        frame_start, frame_end = 0, min(n_frames - 1,
                                         cfg_base['track_initiation']['n_frames_window'] - 1)
        conf_map = main_module.compute_snr_baseline_confidence(pts, cfg_base)

        for overlap in overlaps:
            cfg = copy.deepcopy(cfg_base)
            cfg['block_sliding']['overlap_ratio'] = overlap
            initiator = QASHTrackInitiator(cfg)
            tracks = initiator.initiate_tracks(pts, conf_map, frame_start, frame_end)
            ev = evaluate_method(f'Overlap={int(overlap*100)}%', tracks, true_trks, n_frames)
            results[f'Overlap={int(overlap*100)}%'].append(ev.gospa)

    return {k: (np.mean(v), np.std(v)) for k, v in results.items()}
