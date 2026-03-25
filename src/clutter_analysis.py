"""
Clutter distribution statistical analysis module.

Analyzes clutter and target distributions per batch of frames.
Estimates K-distribution/Weibull parameters and produces density maps.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def fit_weibull(data: np.ndarray) -> Tuple[float, float]:
    """Fit Weibull distribution to data, return (shape c, scale lambda)."""
    data = data[data > 0]
    if len(data) < 5:
        return 1.0, 1.0
    try:
        c, loc, scale = stats.weibull_min.fit(data, floc=0)
        return float(c), float(scale)
    except Exception:
        return 1.0, float(np.mean(data))


def fit_kdist(data: np.ndarray) -> Tuple[float, float]:
    """
    Fit K-distribution to data using method of moments.

    K-distribution: shape (nu), scale (b).
    Mean = b * Gamma(nu + 1/2) / Gamma(nu)
    Variance = b^2 * nu (approximation for large nu)
    """
    data = data[data > 0]
    if len(data) < 5:
        return 1.0, 1.0
    m1 = np.mean(data)
    m2 = np.mean(data ** 2)
    if m1 <= 0 or m2 <= 0:
        return 1.0, 1.0
    # Method of moments: nu ≈ m1^2 / (m2 - m1^2)
    nu = m1 ** 2 / max(m2 - m1 ** 2, 1e-6)
    b = m1 / max(nu, 1e-6)
    return float(max(nu, 0.1)), float(max(b, 0.1))


class ClutterAnalyzer:
    """
    Statistical analysis of clutter and target distributions per frame batch.

    Produces:
    - Per-cell density histograms (3D bar plots)
    - Distribution parameter estimates
    - SNR histograms per batch
    """

    def __init__(self, config: dict):
        ca_cfg = config['clutter_analysis']
        rdr_cfg = config['radar']

        self.batch_size = ca_cfg['batch_size']
        self.n_range_bins = ca_cfg['n_range_bins_hist']
        self.n_az_bins = ca_cfg['n_az_bins_hist']
        self.max_range = rdr_cfg['max_range']
        self.min_range = rdr_cfg['min_range']

    def analyze_batch(self, points: np.ndarray,
                      frame_start: int) -> Dict:
        """
        Analyze one batch of batch_size frames.

        Args:
            points: full point trace array
            frame_start: first frame in this batch

        Returns:
            dict with density_map, snr_dist_params, count stats
        """
        frame_end = frame_start + self.batch_size - 1
        mask = (points['frame_id'] >= frame_start) & (points['frame_id'] <= frame_end)
        batch_pts = points[mask]

        range_edges = np.linspace(self.min_range, self.max_range, self.n_range_bins + 1)
        az_edges = np.linspace(0, 360, self.n_az_bins + 1)

        # 2D density histogram
        density_map, _, _ = np.histogram2d(
            batch_pts['range'], batch_pts['azimuth'],
            bins=[range_edges, az_edges]
        )

        # SNR distribution (all detections)
        snr_all = batch_pts['snr']
        weibull_c, weibull_scale = fit_weibull(snr_all - snr_all.min() + 0.1)

        # Separate analysis if labels available
        target_mask = batch_pts['is_target'] == 1
        clutter_mask = ~target_mask

        target_snr = batch_pts['snr'][target_mask]
        clutter_snr = batch_pts['snr'][clutter_mask]

        kdist_nu, kdist_b = fit_kdist(clutter_snr - clutter_snr.min() + 0.1
                                       if len(clutter_snr) > 0 else np.array([1.0]))

        r_centers = (range_edges[:-1] + range_edges[1:]) / 2
        az_centers = (az_edges[:-1] + az_edges[1:]) / 2

        return {
            'frame_start': frame_start,
            'frame_end': frame_end,
            'density_map': density_map,
            'range_centers': r_centers,
            'az_centers': az_centers,
            'n_total': len(batch_pts),
            'n_target': int(target_mask.sum()),
            'n_clutter': int(clutter_mask.sum()),
            'snr_all': snr_all,
            'target_snr': target_snr,
            'clutter_snr': clutter_snr,
            'weibull_shape': weibull_c,
            'weibull_scale': weibull_scale,
            'kdist_nu': kdist_nu,
            'kdist_b': kdist_b,
        }

    def analyze_all_batches(self, points: np.ndarray) -> List[Dict]:
        """Analyze all batches across the full dataset."""
        n_frames = int(points['frame_id'].max()) + 1
        results = []
        for f0 in range(0, n_frames - self.batch_size + 1, self.batch_size):
            result = self.analyze_batch(points, f0)
            results.append(result)
        return results

    def compute_adaptive_threshold(self, batch_result: Dict,
                                   pfa: float = 1e-4) -> float:
        """
        Compute adaptive SNR detection threshold for given false alarm rate.

        Uses fitted Weibull distribution of clutter SNR.
        """
        clutter_snr = batch_result['clutter_snr']
        if len(clutter_snr) < 5:
            return batch_result.get('weibull_scale', 10.0)
        c, scale = batch_result['weibull_shape'], batch_result['weibull_scale']
        # Threshold = F_weibull_inv(1 - pfa)
        threshold = scale * (-np.log(pfa)) ** (1.0 / c)
        return float(threshold)

    def clutter_density_map(self, points: np.ndarray) -> np.ndarray:
        """
        Compute average clutter density map across all frames.

        Returns [n_range_bins, n_az_bins] array of mean clutter counts per cell.
        """
        n_frames = int(points['frame_id'].max()) + 1
        range_edges = np.linspace(self.min_range, self.max_range, self.n_range_bins + 1)
        az_edges = np.linspace(0, 360, self.n_az_bins + 1)

        total_map = np.zeros((self.n_range_bins, self.n_az_bins))
        clutter_pts = points[points['is_target'] == 0]
        density, _, _ = np.histogram2d(
            clutter_pts['range'], clutter_pts['azimuth'],
            bins=[range_edges, az_edges]
        )
        total_map += density / max(n_frames, 1)
        return total_map
