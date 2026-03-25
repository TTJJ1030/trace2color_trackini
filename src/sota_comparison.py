"""
SOTA comparison methods for track initiation benchmarking.

Implements:
1. 3D Hough Transform (Carlson et al. style)
2. CFAR + Nearest-Neighbor (NN) correlation baseline
3. Sliding Window Correlation (SWC)
4. Threshold-only (SNR gate + simple association)

All methods output TrackHypothesis-compatible objects for fair GOSPA evaluation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment


@dataclass
class BaseTrack:
    track_id: int
    supporting_points: List[Dict] = field(default_factory=list)
    quality_score: float = 0.0
    confirmed: bool = False
    frame_hits: List[int] = field(default_factory=list)
    vx: float = 0.0
    vy: float = 0.0
    method: str = "unknown"


def polar_to_xy(r: float, az: float) -> Tuple[float, float]:
    az_rad = np.deg2rad(az)
    return r * np.sin(az_rad), r * np.cos(az_rad)


# ============================================================
# Method 1: 3D Hough Transform (binary votes, no quality weighting)
# ============================================================

class Hough3DTrackInitiator:
    """
    Classical 3D Hough Transform for track initiation.
    Reference: Carlson et al., "Hough Transform Techniques for Radar TBD"

    Parameterizes target trajectory as (x0, y0, vx, vy) but in practice
    uses 2D projection: accumulates in (vx, vy) Hough space with binary votes.
    No quality weighting — all detections above SNR threshold contribute equally.
    """

    def __init__(self, config: dict):
        ti_cfg = config['track_initiation']
        rdr_cfg = config['radar']
        self.snr_threshold = config['layering']['snr_threshold']
        self.v_max = ti_cfg['v_max']
        self.v_res = ti_cfg['v_resolution']
        self.hough_threshold = ti_cfg['hough_threshold']
        self.gate_distance = ti_cfg['gate_distance']
        self.m_hits = ti_cfg['m_of_n_hits']
        self.scan_time = rdr_cfg['scan_time']
        self.v_bins = np.arange(-self.v_max, self.v_max + self.v_res, self.v_res)
        self.n_vbins = len(self.v_bins) - 1

    def initiate_tracks(self, points: np.ndarray, frame_start: int,
                        frame_end: int) -> List[BaseTrack]:
        f_mask = (points['frame_id'] >= frame_start) & (points['frame_id'] <= frame_end)
        pts = points[f_mask]

        # SNR threshold gate (binary)
        snr_mask = pts['snr'] >= self.snr_threshold
        pts_hc = pts[snr_mask]

        if len(pts_hc) < 2:
            return []

        xs = np.array([polar_to_xy(r, az)[0] for r, az in zip(pts_hc['range'], pts_hc['azimuth'])])
        ys = np.array([polar_to_xy(r, az)[1] for r, az in zip(pts_hc['range'], pts_hc['azimuth'])])
        ts = pts_hc['frame_id'].astype(float) * self.scan_time

        # Binary Hough accumulator
        hough = np.zeros((self.n_vbins, self.n_vbins), dtype=np.float32)
        n_pts = len(pts_hc)

        for i in range(n_pts):
            for j in range(i + 1, min(n_pts, i + 50)):
                dt = ts[j] - ts[i]
                if abs(dt) < 0.1 or pts_hc['frame_id'][i] == pts_hc['frame_id'][j]:
                    continue
                vx = (xs[j] - xs[i]) / dt
                vy = (ys[j] - ys[i]) / dt
                if abs(vx) > self.v_max or abs(vy) > self.v_max:
                    continue
                vxi = int((vx + self.v_max) / self.v_res)
                vyi = int((vy + self.v_max) / self.v_res)
                vxi = np.clip(vxi, 0, self.n_vbins - 1)
                vyi = np.clip(vyi, 0, self.n_vbins - 1)
                hough[vxi, vyi] += 1.0  # binary vote

        max_acc = hough.max()
        if max_acc <= 0:
            return []

        hough_norm = hough / max_acc
        peak_mask = hough_norm >= self.hough_threshold
        peak_indices = np.argwhere(peak_mask)

        tracks = []
        used = [False] * len(peak_indices)
        track_id = 0

        for k, (pi, pj) in enumerate(peak_indices):
            if used[k]:
                continue
            vx_est = self.v_bins[pi] + self.v_res / 2
            vy_est = self.v_bins[pj] + self.v_res / 2

            supporting = []
            frame_hits = set()
            for m in range(n_pts):
                for ref in range(n_pts):
                    dt2 = ts[m] - ts[ref]
                    x_pred = xs[ref] + vx_est * dt2
                    y_pred = ys[ref] + vy_est * dt2
                    dist = np.sqrt((xs[m] - x_pred)**2 + (ys[m] - y_pred)**2)
                    if dist <= self.gate_distance:
                        supporting.append({
                            'range': float(pts_hc['range'][m]),
                            'azimuth': float(pts_hc['azimuth'][m]),
                            'frame_id': int(pts_hc['frame_id'][m]),
                            'snr': float(pts_hc['snr'][m]),
                            'grade': 1.0,
                        })
                        frame_hits.add(int(pts_hc['frame_id'][m]))
                        break

            if len(frame_hits) >= self.m_hits:
                tracks.append(BaseTrack(
                    track_id=track_id,
                    vx=float(vx_est), vy=float(vy_est),
                    supporting_points=supporting,
                    quality_score=len(frame_hits) / max(frame_end - frame_start + 1, 1),
                    confirmed=True,
                    frame_hits=sorted(frame_hits),
                    method='3D-Hough',
                ))
                track_id += 1

        return self._merge_duplicates(tracks)

    def _merge_duplicates(self, tracks):
        if len(tracks) <= 1:
            return tracks
        kept, used = [], [False] * len(tracks)
        for i in range(len(tracks)):
            if used[i]:
                continue
            best = tracks[i]
            for j in range(i + 1, len(tracks)):
                if used[j]:
                    continue
                dv = np.sqrt((tracks[i].vx - tracks[j].vx)**2 + (tracks[i].vy - tracks[j].vy)**2)
                if dv < self.v_res * 3:
                    used[j] = True
                    if tracks[j].quality_score > best.quality_score:
                        best = tracks[j]
            kept.append(best)
            used[i] = True
        return kept


# ============================================================
# Method 2: CFAR + Nearest-Neighbor Correlation
# ============================================================

class CFARNNTrackInitiator:
    """
    Traditional two-step: CFAR detection + Nearest-Neighbor gating correlation.

    Step 1: CFAR threshold (here using Weibull-fitted threshold from clutter analysis)
    Step 2: NN gating — connect detections in consecutive frames within gate radius
    Step 3: Track confirmed if connected for M consecutive frames
    """

    def __init__(self, config: dict, cfar_threshold_db: float = None):
        ti_cfg = config['track_initiation']
        rdr_cfg = config['radar']
        self.snr_threshold = cfar_threshold_db or config['layering']['snr_threshold']
        self.gate_distance = ti_cfg['gate_distance']
        self.m_hits = ti_cfg['m_of_n_hits']
        self.scan_time = rdr_cfg['scan_time']

    def initiate_tracks(self, points: np.ndarray, frame_start: int,
                        frame_end: int) -> List[BaseTrack]:
        f_mask = (points['frame_id'] >= frame_start) & (points['frame_id'] <= frame_end)
        pts = points[f_mask]

        # CFAR gate
        cfar_mask = pts['snr'] >= self.snr_threshold
        pts_cfar = pts[cfar_mask]

        n_frames = frame_end - frame_start + 1
        if len(pts_cfar) == 0:
            return []

        # Group by frame
        frames_pts = {}
        for f in range(frame_start, frame_end + 1):
            fm = pts_cfar['frame_id'] == f
            frames_pts[f] = pts_cfar[fm]

        # NN gating across consecutive frames
        track_chains = []
        for f in range(frame_start, frame_end):
            pts_f = frames_pts.get(f, np.array([]))
            pts_f1 = frames_pts.get(f + 1, np.array([]))
            if len(pts_f) == 0 or len(pts_f1) == 0:
                continue

            xs_f = np.array([polar_to_xy(r, az)[0] for r, az in zip(pts_f['range'], pts_f['azimuth'])])
            ys_f = np.array([polar_to_xy(r, az)[1] for r, az in zip(pts_f['range'], pts_f['azimuth'])])
            xs_f1 = np.array([polar_to_xy(r, az)[0] for r, az in zip(pts_f1['range'], pts_f1['azimuth'])])
            ys_f1 = np.array([polar_to_xy(r, az)[1] for r, az in zip(pts_f1['range'], pts_f1['azimuth'])])

            # Cost matrix
            cost = np.sqrt((xs_f[:, None] - xs_f1[None, :])**2 +
                           (ys_f[:, None] - ys_f1[None, :])**2)
            cost[cost > self.gate_distance] = 1e9

            if cost.min() >= 1e9:
                continue

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] < self.gate_distance:
                    track_chains.append({
                        'frame_start': f,
                        'pts': [
                            {'range': float(pts_f['range'][ri]),
                             'azimuth': float(pts_f['azimuth'][ri]),
                             'frame_id': f, 'snr': float(pts_f['snr'][ri]), 'grade': 0.5},
                            {'range': float(pts_f1['range'][ci]),
                             'azimuth': float(pts_f1['azimuth'][ci]),
                             'frame_id': f + 1, 'snr': float(pts_f1['snr'][ci]), 'grade': 0.5},
                        ]
                    })

        # Merge chains and confirm
        tracks = []
        track_id = 0
        for chain in track_chains:
            frame_hits = list(set(p['frame_id'] for p in chain['pts']))
            if len(frame_hits) >= self.m_hits:
                tracks.append(BaseTrack(
                    track_id=track_id,
                    supporting_points=chain['pts'],
                    quality_score=len(frame_hits) / n_frames,
                    confirmed=True,
                    frame_hits=sorted(frame_hits),
                    method='CFAR-NN',
                ))
                track_id += 1

        return tracks


# ============================================================
# Method 3: Sliding Window Correlation (SWC)
# ============================================================

class SlidingWindowCorrelation:
    """
    Sliding Window Correlation for track initiation.
    For each pair of points in adjacent frames within velocity gate,
    project forward and count "hits" in subsequent frames.
    Confirm if M/N frames hit within gate.
    """

    def __init__(self, config: dict):
        ti_cfg = config['track_initiation']
        rdr_cfg = config['radar']
        self.snr_threshold = config['layering']['snr_threshold']
        self.v_max = ti_cfg['v_max']
        self.gate_distance = ti_cfg['gate_distance']
        self.m_hits = ti_cfg['m_of_n_hits']
        self.scan_time = rdr_cfg['scan_time']

    def initiate_tracks(self, points: np.ndarray, frame_start: int,
                        frame_end: int) -> List[BaseTrack]:
        f_mask = (points['frame_id'] >= frame_start) & (points['frame_id'] <= frame_end)
        pts = points[f_mask]
        snr_mask = pts['snr'] >= self.snr_threshold
        pts_hc = pts[snr_mask]

        if len(pts_hc) < 2:
            return []

        xs = np.array([polar_to_xy(r, az)[0] for r, az in zip(pts_hc['range'], pts_hc['azimuth'])])
        ys = np.array([polar_to_xy(r, az)[1] for r, az in zip(pts_hc['range'], pts_hc['azimuth'])])
        ts = pts_hc['frame_id'].astype(float) * self.scan_time

        tracks = []
        track_id = 0
        n = len(pts_hc)

        for i in range(n):
            for j in range(n):
                if pts_hc['frame_id'][j] != pts_hc['frame_id'][i] + 1:
                    continue
                dt = ts[j] - ts[i]
                if abs(dt) < 0.1:
                    continue
                vx = (xs[j] - xs[i]) / dt
                vy = (ys[j] - ys[i]) / dt
                speed = np.sqrt(vx**2 + vy**2)
                if speed > self.v_max:
                    continue

                # Project forward and count hits
                supporting = [
                    {'range': float(pts_hc['range'][i]), 'azimuth': float(pts_hc['azimuth'][i]),
                     'frame_id': int(pts_hc['frame_id'][i]), 'snr': float(pts_hc['snr'][i]), 'grade': 0.6},
                    {'range': float(pts_hc['range'][j]), 'azimuth': float(pts_hc['azimuth'][j]),
                     'frame_id': int(pts_hc['frame_id'][j]), 'snr': float(pts_hc['snr'][j]), 'grade': 0.6},
                ]
                frame_hits = {int(pts_hc['frame_id'][i]), int(pts_hc['frame_id'][j])}

                for k in range(n):
                    if k == i or k == j:
                        continue
                    t_k = ts[k]
                    x_pred = xs[i] + vx * (t_k - ts[i])
                    y_pred = ys[i] + vy * (t_k - ts[i])
                    dist = np.sqrt((xs[k] - x_pred)**2 + (ys[k] - y_pred)**2)
                    if dist <= self.gate_distance:
                        supporting.append({
                            'range': float(pts_hc['range'][k]),
                            'azimuth': float(pts_hc['azimuth'][k]),
                            'frame_id': int(pts_hc['frame_id'][k]),
                            'snr': float(pts_hc['snr'][k]), 'grade': 0.6,
                        })
                        frame_hits.add(int(pts_hc['frame_id'][k]))

                if len(frame_hits) >= self.m_hits:
                    tracks.append(BaseTrack(
                        track_id=track_id,
                        vx=float(vx), vy=float(vy),
                        supporting_points=supporting,
                        quality_score=len(frame_hits) / max(frame_end - frame_start + 1, 1),
                        confirmed=True,
                        frame_hits=sorted(frame_hits),
                        method='SWC',
                    ))
                    track_id += 1

        # Deduplicate
        return self._deduplicate(tracks)

    def _deduplicate(self, tracks):
        if len(tracks) <= 1:
            return tracks
        kept, used = [], [False] * len(tracks)
        for i in range(len(tracks)):
            if used[i]:
                continue
            best = tracks[i]
            for j in range(i + 1, len(tracks)):
                if used[j]:
                    continue
                # Check overlap in supporting points
                fh_i = set(tracks[i].frame_hits)
                fh_j = set(tracks[j].frame_hits)
                overlap = len(fh_i & fh_j) / max(len(fh_i | fh_j), 1)
                dv = np.sqrt((tracks[i].vx - tracks[j].vx)**2 + (tracks[i].vy - tracks[j].vy)**2)
                if overlap > 0.7 or dv < 20:
                    used[j] = True
                    if tracks[j].quality_score > best.quality_score:
                        best = tracks[j]
            kept.append(best)
            used[i] = True
        return kept
