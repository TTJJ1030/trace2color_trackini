"""
Radar point trace data simulator.

Simulates a multi-target radar scenario with:
- Constant-velocity targets with Swerling-I RCS fluctuation
- K-distributed clutter
- Realistic measurement noise
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os


@dataclass
class PointTrace:
    """Single radar point trace measurement."""
    range: float       # meters
    azimuth: float     # degrees [0, 360)
    snr: float         # dB
    amplitude: float   # linear
    range_span: float  # meters (range extent of detection)
    az_span: float     # degrees (azimuth extent)
    frame_id: int      # scan frame index
    is_target: bool    # ground truth label
    target_id: int     # -1 for clutter


@dataclass
class TargetState:
    """True target state."""
    target_id: int
    x: float    # meters (east)
    y: float    # meters (north)
    vx: float   # m/s
    vy: float   # m/s
    rcs_mean: float  # mean RCS (m^2)


class RadarSimulator:
    """
    Simulates radar point trace data for track initiation research.

    Generates multi-frame point trace data with realistic target and clutter
    characteristics suitable for image-based processing.
    """

    def __init__(self, config: dict):
        self.cfg = config
        rng_cfg = config['radar']
        sim_cfg = config['simulation']

        self.max_range = rng_cfg['max_range']
        self.min_range = rng_cfg['min_range']
        self.scan_time = rng_cfg['scan_time']
        self.n_frames = sim_cfg['n_frames']
        self.n_targets = sim_cfg['n_targets']
        self.pd = sim_cfg['pd']
        self.sigma_r = sim_cfg['sigma_r']
        self.sigma_az = sim_cfg['sigma_az']
        self.clutter_lambda = sim_cfg['clutter_lambda']
        self.k_shape = sim_cfg['clutter_snr_shape']
        self.k_scale = sim_cfg['clutter_snr_scale']
        self.target_snr_mean = sim_cfg['target_snr_mean']
        self.target_snr_std = sim_cfg['target_snr_std']
        self.v_min = sim_cfg['target_speed_min']
        self.v_max = sim_cfg['target_speed_max']
        self.seed = sim_cfg.get('seed', 42)

        self.rng = np.random.default_rng(self.seed)

    def _init_targets(self) -> List[TargetState]:
        """Initialize random constant-velocity targets."""
        targets = []
        for i in range(self.n_targets):
            r = self.rng.uniform(self.min_range * 2, self.max_range * 0.8)
            az = self.rng.uniform(0, 360)
            az_rad = np.deg2rad(az)
            x = r * np.sin(az_rad)
            y = r * np.cos(az_rad)

            speed = self.rng.uniform(self.v_min, self.v_max)
            heading = self.rng.uniform(0, 2 * np.pi)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)

            rcs_mean = self.rng.uniform(1.0, 10.0)
            targets.append(TargetState(i, x, y, vx, vy, rcs_mean))
        return targets

    def _swerling1_snr(self, rcs_mean: float, r: float) -> float:
        """Compute Swerling-I SNR: RCS exponentially distributed, constant scan-to-scan."""
        rcs = self.rng.exponential(rcs_mean)
        # Range-dependent SNR: SNR ~ RCS / r^4
        snr_linear = rcs / (r / 100000) ** 4 * 100
        snr_db = 10 * np.log10(max(snr_linear, 1e-6))
        # Add noise
        snr_db += self.rng.normal(0, 2.0)
        return float(np.clip(snr_db, -5, 50))

    def _kdist_snr(self) -> float:
        """Sample SNR from K-distribution (clutter)."""
        # K-distribution: gamma-mixture; approximate via gamma-gamma compound
        u = self.rng.gamma(self.k_shape, self.k_scale)
        snr_linear = self.rng.gamma(1.0, u)
        snr_db = 10 * np.log10(max(snr_linear, 1e-6))
        return float(np.clip(snr_db, -10, 30))

    def _xy_to_polar(self, x: float, y: float):
        r = np.sqrt(x ** 2 + y ** 2)
        az = np.rad2deg(np.arctan2(x, y)) % 360
        return r, az

    def simulate(self) -> List[PointTrace]:
        """Run full simulation, return list of all point traces."""
        targets = self._init_targets()
        all_points: List[PointTrace] = []

        for frame in range(self.n_frames):
            t = frame * self.scan_time

            # Target detections
            for tgt in targets:
                # Update position
                x = tgt.x + tgt.vx * t
                y = tgt.y + tgt.vy * t

                r_true, az_true = self._xy_to_polar(x, y)
                if r_true < self.min_range or r_true > self.max_range:
                    continue

                # Detection probability gate
                if self.rng.random() > self.pd:
                    continue

                # Measurement noise
                r_meas = r_true + self.rng.normal(0, self.sigma_r)
                az_meas = (az_true + self.rng.normal(0, self.sigma_az)) % 360
                r_meas = max(r_meas, self.min_range)

                snr = self._swerling1_snr(tgt.rcs_mean, r_true)
                amp = 10 ** (snr / 20)

                # Realistic extents
                range_span = self.rng.uniform(200, 800)
                az_span = self.rng.uniform(0.1, 0.5)

                all_points.append(PointTrace(
                    range=float(r_meas),
                    azimuth=float(az_meas),
                    snr=float(snr),
                    amplitude=float(amp),
                    range_span=float(range_span),
                    az_span=float(az_span),
                    frame_id=frame,
                    is_target=True,
                    target_id=tgt.target_id,
                ))

            # Clutter points
            n_clutter = self.rng.poisson(self.clutter_lambda)
            for _ in range(n_clutter):
                # Clutter concentrated at low-mid ranges (realistic)
                r_c = self.rng.choice([
                    self.rng.uniform(self.min_range, self.max_range * 0.4),
                    self.rng.uniform(self.min_range, self.max_range * 0.7),
                ], p=[0.7, 0.3])
                az_c = self.rng.uniform(0, 360)
                snr_c = self._kdist_snr()
                amp_c = 10 ** (snr_c / 20)
                range_span_c = self.rng.uniform(100, 2000)
                az_span_c = self.rng.uniform(0.05, 2.0)

                all_points.append(PointTrace(
                    range=float(r_c),
                    azimuth=float(az_c),
                    snr=float(snr_c),
                    amplitude=float(amp_c),
                    range_span=float(range_span_c),
                    az_span=float(az_span_c),
                    frame_id=frame,
                    is_target=False,
                    target_id=-1,
                ))

        return all_points

    def get_true_tracks(self) -> Dict[int, np.ndarray]:
        """Return true target trajectories as dict: target_id → [n_frames, 2] array (range, azimuth).

        Uses pure 2D polar coordinates. r=0 indicates target is out of surveillance area.
        """
        targets = self._init_targets()
        tracks = {}
        for tgt in targets:
            traj = []
            for frame in range(self.n_frames):
                t = frame * self.scan_time
                x = tgt.x + tgt.vx * t
                y = tgt.y + tgt.vy * t
                r, az = self._xy_to_polar(x, y)
                if r < self.min_range or r > self.max_range:
                    traj.append([0.0, 0.0])   # out of range: sentinel
                else:
                    traj.append([r, az])
            tracks[tgt.target_id] = np.array(traj)
        return tracks


def points_to_array(points: List[PointTrace]) -> np.ndarray:
    """Convert list of PointTrace to numpy structured array."""
    n = len(points)
    dtype = np.dtype([
        ('range', np.float32),
        ('azimuth', np.float32),
        ('snr', np.float32),
        ('amplitude', np.float32),
        ('range_span', np.float32),
        ('az_span', np.float32),
        ('frame_id', np.int32),
        ('is_target', np.int32),
        ('target_id', np.int32),
    ])
    arr = np.zeros(n, dtype=dtype)
    for i, p in enumerate(points):
        arr[i] = (p.range, p.azimuth, p.snr, p.amplitude,
                  p.range_span, p.az_span, p.frame_id,
                  int(p.is_target), p.target_id)
    return arr


def save_scenario(points: List[PointTrace], tracks: Dict[int, np.ndarray],
                  path: str):
    """Save simulated scenario to .npz file."""
    arr = points_to_array(points)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    track_arrays = {f'track_{k}': v for k, v in tracks.items()}
    np.savez(path,
             points_range=arr['range'],
             points_azimuth=arr['azimuth'],
             points_snr=arr['snr'],
             points_amplitude=arr['amplitude'],
             points_range_span=arr['range_span'],
             points_az_span=arr['az_span'],
             points_frame_id=arr['frame_id'],
             points_is_target=arr['is_target'],
             points_target_id=arr['target_id'],
             **track_arrays)
    print(f"Saved {len(points)} points to {path}")


def load_scenario(path: str):
    """Load scenario from .npz file, return (points_array, tracks_dict)."""
    data = np.load(path)
    n = len(data['points_range'])
    dtype = np.dtype([
        ('range', np.float32), ('azimuth', np.float32),
        ('snr', np.float32), ('amplitude', np.float32),
        ('range_span', np.float32), ('az_span', np.float32),
        ('frame_id', np.int32), ('is_target', np.int32),
        ('target_id', np.int32),
    ])
    arr = np.zeros(n, dtype=dtype)
    for field in ['range', 'azimuth', 'snr', 'amplitude', 'range_span',
                  'az_span', 'frame_id', 'is_target', 'target_id']:
        arr[field] = data[f'points_{field}']

    tracks = {}
    for key in data.files:
        if key.startswith('track_'):
            tid = int(key.split('_')[1])
            tracks[tid] = data[key]
    return arr, tracks
