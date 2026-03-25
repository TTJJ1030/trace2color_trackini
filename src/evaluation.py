"""
Comprehensive evaluation metrics for radar track initiation.

Metrics:
- Track Detection Rate (TDR): fraction of true targets with initiated track
- False Track Rate (FTR): false tracks per unit time per unit area
- GOSPA (Generalized Optimal Sub-Pattern Assignment)
- OSPA (Optimal Sub-Pattern Assignment)
- Track Initiation Delay (TID): frames until track confirmed
- Precision / Recall at track level
- SNR-conditional performance curves
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    method_name: str
    tdr: float           # Track Detection Rate [0,1]
    ftr: float           # False Track Rate (per frame)
    gospa: float         # GOSPA distance (m)
    ospa: float          # OSPA distance (m)
    precision: float     # track-level precision
    recall: float        # track-level recall
    f1: float            # F1 score
    tid_mean: float      # mean track initiation delay (frames)
    tid_std: float       # std of initiation delay
    n_initiated: int     # number of initiated tracks
    n_true: int          # number of true targets
    n_false: int         # number of false tracks
    n_missed: int        # number of missed targets


def polar_to_xy(r: float, az: float) -> Tuple[float, float]:
    az_rad = np.deg2rad(az)
    return r * np.sin(az_rad), r * np.cos(az_rad)


def compute_gospa(true_tracks: Dict, initiated_tracks: List,
                  c: float = 5000.0, p: int = 2, alpha: float = 2.0) -> Tuple[float, Dict]:
    """
    GOSPA metric with decomposition.
    Returns (total_gospa, components_dict)
    """
    if not true_tracks and not initiated_tracks:
        return 0.0, {'localization': 0.0, 'missed': 0.0, 'false': 0.0}

    true_positions = []
    for tid, traj in true_tracks.items():
        valid = traj[:, 2] > 0
        if valid.any():
            last_valid = traj[valid][-1]
            x, y = polar_to_xy(last_valid[2], last_valid[3])
            true_positions.append([x, y])

    est_positions = []
    for t in initiated_tracks:
        if t.supporting_points:
            last_pt = sorted(t.supporting_points, key=lambda p: p['frame_id'])[-1]
            x, y = polar_to_xy(last_pt['range'], last_pt['azimuth'])
            est_positions.append([x, y])

    n_t = len(true_positions)
    n_e = len(est_positions)

    if n_t == 0:
        false_cost = (c**p / alpha) * n_e
        return float(false_cost**(1/p)), {'localization': 0, 'missed': 0, 'false': float(false_cost)}
    if n_e == 0:
        missed_cost = (c**p / alpha) * n_t
        return float(missed_cost**(1/p)), {'localization': 0, 'missed': float(missed_cost), 'false': 0}

    tp = np.array(true_positions)
    ep = np.array(est_positions)

    cost = np.zeros((n_t, n_e))
    for i in range(n_t):
        for j in range(n_e):
            d = np.linalg.norm(tp[i] - ep[j])
            cost[i, j] = min(d, c)

    row_ind, col_ind = linear_sum_assignment(cost**p)

    loc_cost = sum(cost[r, c]**p for r, c in zip(row_ind, col_ind)
                   if cost[r, c] < c)
    matched_t = set(row_ind[cost[row_ind, col_ind] < c])
    matched_e = set(col_ind[cost[row_ind, col_ind] < c])

    missed = n_t - len(matched_t)
    false = n_e - len(matched_e)
    missed_cost = (c**p / alpha) * missed
    false_cost = (c**p / alpha) * false

    total = (loc_cost + missed_cost + false_cost) / max(n_t, n_e)
    return float(total**(1/p)), {
        'localization': float(loc_cost / max(n_t, n_e)),
        'missed': float(missed_cost / max(n_t, n_e)),
        'false': float(false_cost / max(n_t, n_e)),
    }


def compute_ospa(true_tracks: Dict, initiated_tracks: List,
                 c: float = 5000.0, p: int = 2) -> float:
    """OSPA metric (no cardinality penalty decomposition)."""
    true_positions = []
    for tid, traj in true_tracks.items():
        valid = traj[:, 2] > 0
        if valid.any():
            last_valid = traj[valid][-1]
            x, y = polar_to_xy(last_valid[2], last_valid[3])
            true_positions.append([x, y])

    est_positions = []
    for t in initiated_tracks:
        if t.supporting_points:
            last_pt = sorted(t.supporting_points, key=lambda p: p['frame_id'])[-1]
            x, y = polar_to_xy(last_pt['range'], last_pt['azimuth'])
            est_positions.append([x, y])

    n_t, n_e = len(true_positions), len(est_positions)
    if n_t == 0 and n_e == 0:
        return 0.0
    if n_t == 0 or n_e == 0:
        return c

    n = max(n_t, n_e)
    tp = np.array(true_positions)
    ep = np.array(est_positions)

    # Pad smaller set with zeros (will have cost c)
    if n_t < n:
        tp = np.vstack([tp, np.zeros((n - n_t, 2))])
    if n_e < n:
        ep = np.vstack([ep, np.zeros((n - n_e, 2))])

    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = np.linalg.norm(tp[i] - ep[j])
            cost[i, j] = min(d, c)**p

    row_ind, col_ind = linear_sum_assignment(cost)
    ospa_val = (cost[row_ind, col_ind].sum() / n) ** (1/p)
    return float(ospa_val)


def compute_tdr_ftr(true_tracks: Dict, initiated_tracks: List,
                    assoc_threshold: float = 10000.0,
                    n_frames: int = 20) -> Tuple[float, float, int, int]:
    """
    Compute Track Detection Rate and False Track Rate.

    TDR = (# true targets with associated initiated track) / (# true targets)
    FTR = (# false tracks) / n_frames

    Returns: (TDR, FTR, n_detected, n_false)
    """
    true_positions = {}
    for tid, traj in true_tracks.items():
        valid = traj[:, 2] > 0
        if valid.any():
            last_valid = traj[valid][-1]
            x, y = polar_to_xy(last_valid[2], last_valid[3])
            true_positions[tid] = np.array([x, y])

    est_positions = []
    for t in initiated_tracks:
        if t.supporting_points:
            last_pt = sorted(t.supporting_points, key=lambda p: p['frame_id'])[-1]
            x, y = polar_to_xy(last_pt['range'], last_pt['azimuth'])
            est_positions.append(np.array([x, y]))

    n_true = len(true_positions)
    n_est = len(est_positions)

    if n_true == 0:
        return 1.0, n_est / max(n_frames, 1), 0, n_est

    if n_est == 0:
        return 0.0, 0.0, 0, 0

    tp = np.array(list(true_positions.values()))
    ep = np.array(est_positions)

    cost = np.zeros((n_true, n_est))
    for i in range(n_true):
        for j in range(n_est):
            cost[i, j] = np.linalg.norm(tp[i] - ep[j])

    row_ind, col_ind = linear_sum_assignment(cost)
    associated = sum(1 for r, c in zip(row_ind, col_ind)
                     if cost[r, c] <= assoc_threshold)

    tdr = associated / n_true
    ftr = (n_est - associated) / max(n_frames, 1)
    return float(tdr), float(ftr), int(associated), int(n_est - associated)


def compute_track_initiation_delay(true_tracks: Dict, initiated_tracks: List,
                                    assoc_threshold: float = 10000.0) -> Tuple[float, float]:
    """
    Track Initiation Delay: for each matched true-initiated track pair,
    compute the frame at which the initiated track first appears vs.
    the first true detection frame.

    Returns: (mean_delay_frames, std_delay_frames)
    """
    true_first_frame = {}
    true_positions = {}
    for tid, traj in true_tracks.items():
        valid = traj[:, 2] > 0
        if valid.any():
            first_valid_idx = np.where(valid)[0][0]
            true_first_frame[tid] = first_valid_idx
            last_valid = traj[valid][-1]
            x, y = polar_to_xy(last_valid[2], last_valid[3])
            true_positions[tid] = np.array([x, y])

    if not true_positions:
        return 0.0, 0.0

    delays = []
    for t in initiated_tracks:
        if not t.supporting_points:
            continue
        last_pt = sorted(t.supporting_points, key=lambda p: p['frame_id'])[-1]
        x_est, y_est = polar_to_xy(last_pt['range'], last_pt['azimuth'])

        # Find closest true target
        min_dist = float('inf')
        matched_tid = None
        for tid, pos in true_positions.items():
            d = np.linalg.norm(pos - np.array([x_est, y_est]))
            if d < min_dist:
                min_dist = d
                matched_tid = tid

        if min_dist <= assoc_threshold and matched_tid is not None:
            first_pt = sorted(t.supporting_points, key=lambda p: p['frame_id'])[0]
            initiated_frame = first_pt['frame_id']
            true_frame = true_first_frame.get(matched_tid, 0)
            delay = max(0, initiated_frame - true_frame)
            delays.append(delay)

    if not delays:
        return float('nan'), float('nan')
    return float(np.mean(delays)), float(np.std(delays))


def evaluate_method(method_name: str, initiated_tracks: List,
                    true_tracks: Dict, n_frames: int = 20,
                    gospa_c: float = 5000.0) -> EvaluationResult:
    """Compute all metrics for one method."""
    gospa_val, _ = compute_gospa(true_tracks, initiated_tracks, c=gospa_c)
    ospa_val = compute_ospa(true_tracks, initiated_tracks, c=gospa_c)
    # Association threshold: 10% of max range (generous for GOSPA scale)
    assoc_thr = gospa_c * 4.0
    tdr, ftr, n_detected, n_false = compute_tdr_ftr(
        true_tracks, initiated_tracks, assoc_threshold=assoc_thr, n_frames=n_frames)
    tid_mean, tid_std = compute_track_initiation_delay(
        true_tracks, initiated_tracks, assoc_threshold=assoc_thr)

    n_true = len(true_tracks)
    n_initiated = len(initiated_tracks)
    n_missed = n_true - n_detected

    precision = n_detected / max(n_initiated, 1)
    recall = tdr
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return EvaluationResult(
        method_name=method_name,
        tdr=tdr, ftr=ftr, gospa=gospa_val, ospa=ospa_val,
        precision=precision, recall=recall, f1=f1,
        tid_mean=tid_mean if not np.isnan(tid_mean) else 999.0,
        tid_std=tid_std if not np.isnan(tid_std) else 0.0,
        n_initiated=n_initiated, n_true=n_true,
        n_false=n_false, n_missed=n_missed,
    )


def run_multi_seed_evaluation(config: dict, n_seeds: int = 20,
                               use_dl: bool = True,
                               model_path: str = 'checkpoints/best_model.pth'):
    """
    Run evaluation across multiple random seeds for statistical robustness.
    Returns dict of method_name -> list of EvaluationResult
    """
    import yaml
    from src.data_simulation import RadarSimulator, points_to_array
    from src.track_initiation import QASHTrackInitiator
    from src.sota_comparison import (Hough3DTrackInitiator,
                                      CFARNNTrackInitiator,
                                      SlidingWindowCorrelation)
    import main as main_module

    results_all = {
        'QASH (Ours)': [],
        '3D-Hough': [],
        'CFAR-NN': [],
        'SWC': [],
        'SNR-Threshold': [],
    }

    for seed in range(n_seeds):
        cfg = dict(config)
        cfg['simulation'] = dict(config['simulation'])
        cfg['simulation']['seed'] = seed * 13 + 42
        cfg['simulation']['n_targets'] = np.random.randint(3, 10)
        cfg['simulation']['clutter_lambda'] = np.random.randint(400, 1500)

        sim = RadarSimulator(cfg)
        points_list = sim.simulate()
        true_tracks = sim.get_true_tracks()
        points = points_to_array(points_list)

        n_frames = int(points['frame_id'].max()) + 1
        frame_start = 0
        frame_end = min(n_frames - 1, cfg['track_initiation']['n_frames_window'] - 1)

        initiator = QASHTrackInitiator(cfg)
        hough3d = Hough3DTrackInitiator(cfg)
        cfar_nn = CFARNNTrackInitiator(cfg)
        swc = SlidingWindowCorrelation(cfg)

        # QASH with DL
        if use_dl:
            try:
                conf_map = main_module.run_dl_inference(cfg, points, model_path)
            except Exception:
                conf_map = main_module.compute_snr_baseline_confidence(points, cfg)
        else:
            conf_map = main_module.compute_snr_baseline_confidence(points, cfg)

        tracks_qash = initiator.initiate_tracks(points, conf_map, frame_start, frame_end)
        tracks_3dh = hough3d.initiate_tracks(points, frame_start, frame_end)
        tracks_cfar = cfar_nn.initiate_tracks(points, frame_start, frame_end)
        tracks_swc = swc.initiate_tracks(points, frame_start, frame_end)

        # SNR-threshold baseline (uniform confidence)
        uniform_conf = (points['snr'] >= cfg['layering']['snr_threshold']).astype(np.float32)
        tracks_snr = initiator.initiate_tracks(points, uniform_conf, frame_start, frame_end)

        for method, tracks in [('QASH (Ours)', tracks_qash),
                                 ('3D-Hough', tracks_3dh),
                                 ('CFAR-NN', tracks_cfar),
                                 ('SWC', tracks_swc),
                                 ('SNR-Threshold', tracks_snr)]:
            ev = evaluate_method(method, tracks, true_tracks, n_frames=n_frames)
            results_all[method].append(ev)

    return results_all


def summarize_results(results_all: Dict) -> Dict:
    """Compute mean ± std for each metric across seeds."""
    summary = {}
    for method, ev_list in results_all.items():
        summary[method] = {
            'TDR':       (np.mean([e.tdr for e in ev_list]),
                          np.std([e.tdr for e in ev_list])),
            'FTR':       (np.mean([e.ftr for e in ev_list]),
                          np.std([e.ftr for e in ev_list])),
            'GOSPA':     (np.mean([e.gospa for e in ev_list]),
                          np.std([e.gospa for e in ev_list])),
            'OSPA':      (np.mean([e.ospa for e in ev_list]),
                          np.std([e.ospa for e in ev_list])),
            'Precision': (np.mean([e.precision for e in ev_list]),
                          np.std([e.precision for e in ev_list])),
            'Recall':    (np.mean([e.recall for e in ev_list]),
                          np.std([e.recall for e in ev_list])),
            'F1':        (np.mean([e.f1 for e in ev_list]),
                          np.std([e.f1 for e in ev_list])),
            'TID':       (np.mean([e.tid_mean for e in ev_list if e.tid_mean < 900]),
                          np.std([e.tid_mean for e in ev_list if e.tid_mean < 900])),
        }
    return summary
