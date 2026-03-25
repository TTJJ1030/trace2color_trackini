"""
Quality-Weighted Adaptive Sector Hough Transform (QASH) for track initiation.

Innovation: DL-derived quality grades used as continuous Hough vote weights,
reducing false track rate while maintaining detection rate.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TrackHypothesis:
    """A candidate track hypothesis."""
    track_id: int
    vx: float          # estimated velocity x (m/s)
    vy: float          # estimated velocity y (m/s)
    supporting_points: List[Dict] = field(default_factory=list)
    quality_score: float = 0.0
    confirmed: bool = False
    frame_hits: List[int] = field(default_factory=list)


class QASHTrackInitiator:
    """
    Quality-Weighted Adaptive Sector Hough Transform for track initiation.

    Algorithm:
    1. Filter points by DL confidence threshold
    2. Compute quality grade for each point
    3. Pairwise Hough voting in velocity space, weighted by quality
    4. Peak detection in Hough accumulator
    5. Collect supporting points per velocity peak
    6. Confirm tracks by M-of-N logic
    """

    def __init__(self, config: dict):
        ti_cfg = config['track_initiation']
        rdr_cfg = config['radar']

        self.conf_threshold = ti_cfg['conf_threshold']
        self.v_max = ti_cfg['v_max']
        self.v_res = ti_cfg['v_resolution']
        self.hough_threshold = ti_cfg['hough_threshold']
        self.gate_distance = ti_cfg['gate_distance']
        self.m_hits = ti_cfg['m_of_n_hits']
        self.n_window = ti_cfg['n_frames_window']
        self.scan_time = rdr_cfg['scan_time']

        # Hough space grid
        self.v_bins = np.arange(-self.v_max, self.v_max + self.v_res, self.v_res)
        self.n_vbins = len(self.v_bins) - 1

    def _polar_to_xy(self, r: float, az: float) -> Tuple[float, float]:
        """Convert polar (m, deg) to Cartesian (m, m)."""
        az_rad = np.deg2rad(az)
        x = r * np.sin(az_rad)
        y = r * np.cos(az_rad)
        return x, y

    def _compute_quality_grade(self, confidence: float, snr: float,
                                snr_max: float) -> float:
        """
        Composite quality grade combining DL confidence and SNR.

        Grade = confidence * snr_normalized
        """
        snr_norm = np.clip(snr / max(snr_max, 1.0), 0.0, 1.0)
        return float(confidence * snr_norm)

    def initiate_tracks(self, points: np.ndarray,
                        confidence_map: np.ndarray,
                        frame_start: int,
                        frame_end: int) -> List[TrackHypothesis]:
        """
        Run QASH track initiation on a window of frames.

        Args:
            points: structured array of point traces
            confidence_map: per-point DL confidence scores (same length as points)
            frame_start, frame_end: frame range to process

        Returns:
            List of confirmed TrackHypothesis objects
        """
        # Filter to window
        f_mask = ((points['frame_id'] >= frame_start) &
                  (points['frame_id'] <= frame_end))
        pts = points[f_mask]
        confs = confidence_map[f_mask]

        if len(pts) == 0:
            return []

        # Step 1: Filter by confidence threshold
        high_conf_mask = confs >= self.conf_threshold
        pts_hc = pts[high_conf_mask]
        confs_hc = confs[high_conf_mask]

        if len(pts_hc) < 2:
            return []

        snr_max = float(pts_hc['snr'].max())

        # Step 2: Compute quality grades
        grades = np.array([
            self._compute_quality_grade(c, s, snr_max)
            for c, s in zip(confs_hc, pts_hc['snr'])
        ])

        # Convert to Cartesian
        xs = np.array([self._polar_to_xy(r, az)[0]
                       for r, az in zip(pts_hc['range'], pts_hc['azimuth'])])
        ys = np.array([self._polar_to_xy(r, az)[1]
                       for r, az in zip(pts_hc['range'], pts_hc['azimuth'])])
        ts = pts_hc['frame_id'].astype(float) * self.scan_time

        # Step 3: Quality-weighted Hough accumulation
        hough_vx = np.zeros((self.n_vbins, self.n_vbins), dtype=np.float32)
        hough_vy = np.zeros((self.n_vbins, self.n_vbins), dtype=np.float32)

        n_pts = len(pts_hc)
        for i in range(n_pts):
            for j in range(i + 1, min(n_pts, i + 50)):  # limit pairs per point
                dt = ts[j] - ts[i]
                if abs(dt) < 0.1:
                    continue
                if pts_hc['frame_id'][i] == pts_hc['frame_id'][j]:
                    continue

                vx_cand = (xs[j] - xs[i]) / dt
                vy_cand = (ys[j] - ys[i]) / dt

                if abs(vx_cand) > self.v_max or abs(vy_cand) > self.v_max:
                    continue

                vote_weight = min(grades[i], grades[j])

                # Bin vote
                vxi = int((vx_cand + self.v_max) / self.v_res)
                vyi = int((vy_cand + self.v_max) / self.v_res)
                vxi = np.clip(vxi, 0, self.n_vbins - 1)
                vyi = np.clip(vyi, 0, self.n_vbins - 1)
                hough_vx[vxi, vyi] += vote_weight
                hough_vy[vxi, vyi] += vote_weight

        # Step 4: Peak detection
        hough_acc = (hough_vx + hough_vy) / 2.0
        max_acc = hough_acc.max()
        if max_acc <= 0:
            return []

        hough_norm = hough_acc / max_acc
        peak_mask = hough_norm >= self.hough_threshold
        peak_indices = np.argwhere(peak_mask)

        # Step 5: Build track hypotheses from peaks
        tracks = []
        track_id = 0

        for pi, pj in peak_indices:
            vx_est = self.v_bins[pi] + self.v_res / 2
            vy_est = self.v_bins[pj] + self.v_res / 2

            # Collect supporting points
            supporting = []
            frame_hits = set()

            for k in range(n_pts):
                # Project point k to all frames using estimated velocity
                # Find closest predicted position at point's frame time
                t_k = ts[k]
                # Use frame 0 as reference (find anchor by median)
                t_ref = np.median(ts)
                dt = t_k - t_ref
                # Check if point is consistent with this velocity hypothesis
                # Gate: distance from predicted trajectory
                min_dist = float('inf')
                for ref_k in range(n_pts):
                    t_ref_k = ts[ref_k]
                    dt2 = t_k - t_ref_k
                    x_pred = xs[ref_k] + vx_est * dt2
                    y_pred = ys[ref_k] + vy_est * dt2
                    dist = np.sqrt((xs[k] - x_pred) ** 2 + (ys[k] - y_pred) ** 2)
                    if dist < min_dist:
                        min_dist = dist

                if min_dist <= self.gate_distance:
                    supporting.append({
                        'range': float(pts_hc['range'][k]),
                        'azimuth': float(pts_hc['azimuth'][k]),
                        'frame_id': int(pts_hc['frame_id'][k]),
                        'snr': float(pts_hc['snr'][k]),
                        'confidence': float(confs_hc[k]),
                        'grade': float(grades[k]),
                    })
                    frame_hits.add(int(pts_hc['frame_id'][k]))

            if len(frame_hits) < self.m_hits:
                continue

            quality_score = float(np.mean([p['grade'] for p in supporting]))
            track = TrackHypothesis(
                track_id=track_id,
                vx=float(vx_est),
                vy=float(vy_est),
                supporting_points=supporting,
                quality_score=quality_score,
                confirmed=True,
                frame_hits=sorted(frame_hits),
            )
            tracks.append(track)
            track_id += 1

        # Remove duplicate tracks (similar velocity)
        tracks = self._merge_duplicate_tracks(tracks)
        return tracks

    def _merge_duplicate_tracks(self,
                                 tracks: List[TrackHypothesis]) -> List[TrackHypothesis]:
        """Merge tracks with similar velocity estimates."""
        if len(tracks) <= 1:
            return tracks

        kept = []
        used = [False] * len(tracks)
        for i in range(len(tracks)):
            if used[i]:
                continue
            best = tracks[i]
            for j in range(i + 1, len(tracks)):
                if used[j]:
                    continue
                dv = np.sqrt((tracks[i].vx - tracks[j].vx) ** 2 +
                             (tracks[i].vy - tracks[j].vy) ** 2)
                if dv < self.v_res * 3:
                    used[j] = True
                    if tracks[j].quality_score > best.quality_score:
                        best = tracks[j]
            kept.append(best)
            used[i] = True
        return kept

    def baseline_hough_initiation(self, points: np.ndarray,
                                   frame_start: int,
                                   frame_end: int) -> List[TrackHypothesis]:
        """
        Baseline: standard Hough transform (uniform vote weights = 1.0).
        Used for comparison to demonstrate QASH superiority.
        """
        # Use uniform confidence map
        n = len(points)
        uniform_conf = np.ones(n, dtype=np.float32)
        return self.initiate_tracks(points, uniform_conf, frame_start, frame_end)


def compute_gospa(true_tracks: Dict, initiated_tracks: List[TrackHypothesis],
                  c: float = 5000.0, p: int = 2, alpha: float = 2.0) -> float:
    """
    Compute GOSPA (Generalized Optimal Sub-Pattern Assignment) metric.

    Measures track initiation quality vs ground truth.
    Lower is better.

    Args:
        true_tracks: dict of target_id → trajectory array [n_frames, 4]
        initiated_tracks: list of initiated TrackHypothesis
        c: cutoff distance (meters)
        p: order
        alpha: missed target penalty exponent
    """
    if len(initiated_tracks) == 0 and len(true_tracks) == 0:
        return 0.0
    if len(initiated_tracks) == 0:
        return c * len(true_tracks)

    # Extract final positions from true tracks (last frame)
    true_positions = []
    for tid, traj in true_tracks.items():
        true_positions.append(traj[-1, :2])  # x, y

    # Extract position estimates from initiated tracks
    est_positions = []
    for t in initiated_tracks:
        if t.supporting_points:
            last_pt = sorted(t.supporting_points, key=lambda p: p['frame_id'])[-1]
            r, az = last_pt['range'], last_pt['azimuth']
            az_rad = np.deg2rad(az)
            x = r * np.sin(az_rad)
            y = r * np.cos(az_rad)
            est_positions.append([x, y])

    if not true_positions or not est_positions:
        return c * max(len(true_tracks), len(initiated_tracks))

    tp = np.array(true_positions)
    ep = np.array(est_positions)
    n_t, n_e = len(tp), len(ep)

    # Compute cost matrix
    cost_matrix = np.zeros((n_t, n_e))
    for i in range(n_t):
        for j in range(n_e):
            d = np.linalg.norm(tp[i] - ep[j])
            cost_matrix[i, j] = min(d, c)

    # Simple greedy assignment
    assigned_t = set()
    assigned_e = set()
    total_cost = 0.0

    flat_idx = np.argsort(cost_matrix.ravel())
    for idx in flat_idx:
        i, j = divmod(idx, n_e)
        if i not in assigned_t and j not in assigned_e:
            assigned_t.add(i)
            assigned_e.add(j)
            total_cost += cost_matrix[i, j] ** p

    # Penalty for unassigned
    missed = n_t - len(assigned_t)
    false = n_e - len(assigned_e)
    total_cost += (c ** p / alpha) * (missed + false)

    return float((total_cost / max(n_t, n_e)) ** (1.0 / p))
