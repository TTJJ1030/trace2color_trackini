"""
Pixelization module.

Converts point trace data to 2D polar grid RGB images.
Frame triplets are mapped to R/G/B channels for temporal encoding.

Enhanced version adds:
  - Temporal persistence map (fraction of frames with any detection per cell)
  - Temporal SNR centroid (SNR-weighted temporal centre of mass per cell)
  - Multi-resolution option: fine grid + coarse grid for scale diversity
"""

import numpy as np
from typing import Optional, Tuple, Dict


class Pixelizer:
    """
    Converts radar point traces to RGB polar images.

    The polar grid is parameterized by range and azimuth bins.
    Consecutive frame triplets are color-coded: (R, G, B) = (frame_t, frame_t+1, frame_t+2).
    Each pixel stores the maximum SNR of all points mapping to that cell.
    """

    def __init__(self, config: dict):
        pix_cfg = config['pixelization']
        rdr_cfg = config['radar']

        self.mode = pix_cfg.get('mode', 'fixed')
        self.n_range_bins = pix_cfg['n_range_bins']
        self.n_az_bins = pix_cfg['n_azimuth_bins']
        self.frames_per_triplet = pix_cfg.get('frames_per_triplet', 3)

        self.max_range = rdr_cfg['max_range']
        self.min_range = rdr_cfg['min_range']

    def _build_grid(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute range and azimuth bin edges."""
        if self.mode == 'adaptive':
            ranges = np.sort(np.unique(points['range']))
            azs = np.sort(np.unique(points['azimuth']))
            if len(ranges) > 1:
                dr = np.min(np.diff(ranges))
            else:
                dr = (self.max_range - self.min_range) / self.n_range_bins
            if len(azs) > 1:
                daz = np.min(np.diff(azs))
            else:
                daz = 360.0 / self.n_az_bins
            range_edges = np.arange(self.min_range, self.max_range + dr, dr)
            az_edges = np.arange(0, 360 + daz, daz)
            # Cap to reasonable size
            if len(range_edges) > 2000:
                range_edges = np.linspace(self.min_range, self.max_range, self.n_range_bins + 1)
            if len(az_edges) > 2000:
                az_edges = np.linspace(0, 360, self.n_az_bins + 1)
        else:
            range_edges = np.linspace(self.min_range, self.max_range, self.n_range_bins + 1)
            az_edges = np.linspace(0, 360, self.n_az_bins + 1)
        return range_edges, az_edges

    def pixelize_triplet(self, points: np.ndarray, frame_r: int,
                         frame_g: int, frame_b: int) -> np.ndarray:
        """
        Create RGB image for one triplet of frames.

        Args:
            points: structured array with fields range, azimuth, snr, frame_id
            frame_r: frame index → Red channel
            frame_g: frame index → Green channel
            frame_b: frame index → Blue channel

        Returns:
            RGB image array [n_range_bins, n_az_bins, 3], values in [0, 1]
        """
        range_edges, az_edges = self._build_grid(points)
        nr = len(range_edges) - 1
        naz = len(az_edges) - 1
        image = np.zeros((nr, naz, 3), dtype=np.float32)

        for ch_idx, fid in enumerate([frame_r, frame_g, frame_b]):
            mask = points['frame_id'] == fid
            pts = points[mask]
            if len(pts) == 0:
                continue
            # Bin points into grid
            ri = np.searchsorted(range_edges[1:], pts['range'], side='right')
            azi = np.searchsorted(az_edges[1:], pts['azimuth'], side='right')
            # Clip to valid range
            ri = np.clip(ri, 0, nr - 1)
            azi = np.clip(azi, 0, naz - 1)

            # Accumulate max SNR per cell
            snr_vals = pts['snr']
            for i, j, s in zip(ri, azi, snr_vals):
                if s > image[i, j, ch_idx]:
                    image[i, j, ch_idx] = s

        # Normalize each channel to [0, 1]
        for ch in range(3):
            ch_max = image[:, :, ch].max()
            if ch_max > 0:
                image[:, :, ch] /= ch_max

        return image

    def pixelize_all_triplets(self, points: np.ndarray) -> list:
        """
        Generate all overlapping triplet images from the full dataset.

        Returns list of dicts: {image, frame_r, frame_g, frame_b}
        """
        n_frames = int(points['frame_id'].max()) + 1
        results = []
        step = 1  # overlapping triplets
        for start in range(0, n_frames - self.frames_per_triplet + 1, step):
            fr, fg, fb = start, start + 1, start + 2
            img = self.pixelize_triplet(points, fr, fg, fb)
            results.append({
                'image': img,
                'frame_r': fr,
                'frame_g': fg,
                'frame_b': fb,
            })
        return results

    def _bin_points(self, pts: np.ndarray,
                    range_edges: np.ndarray, az_edges: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (range_indices, azimuth_indices) for a set of points."""
        nr = len(range_edges) - 1
        naz = len(az_edges) - 1
        ri = np.searchsorted(range_edges[1:], pts['range'], side='right')
        azi = np.searchsorted(az_edges[1:], pts['azimuth'], side='right')
        ri = np.clip(ri, 0, nr - 1)
        azi = np.clip(azi, 0, naz - 1)
        return ri, azi

    def pixelize_triplet_enhanced(self, points: np.ndarray,
                                   frame_r: int, frame_g: int,
                                   frame_b: int) -> Dict[str, np.ndarray]:
        """
        Enhanced pixelization returning RGB, persistence, and temporal centroid maps.

        Returns dict with keys:
          'rgb'               : [H, W, 3]  colour-coded SNR image
          'persistence'       : [H, W]     fraction of 3 frames with ≥1 detection
          'temporal_centroid' : [H, W]     SNR-weighted temporal CoM in [0, 1]
                                           (0 = all energy in frame_r,
                                            1 = all energy in frame_b)
        """
        range_edges, az_edges = self._build_grid(points)
        nr = len(range_edges) - 1
        naz = len(az_edges) - 1

        rgb = np.zeros((nr, naz, 3), dtype=np.float32)
        hit_count = np.zeros((nr, naz), dtype=np.float32)   # 0–3
        snr_weighted_t = np.zeros((nr, naz), dtype=np.float32)
        snr_total = np.zeros((nr, naz), dtype=np.float32)

        for ch_idx, (fid, t_norm) in enumerate(
                zip([frame_r, frame_g, frame_b], [0.0, 0.5, 1.0])):
            mask = points['frame_id'] == fid
            pts = points[mask]
            if len(pts) == 0:
                continue
            ri, azi = self._bin_points(pts, range_edges, az_edges)
            snr_vals = np.clip(pts['snr'], 0, None)   # linear-safe

            # RGB channel: max SNR per cell
            for i, j, s in zip(ri, azi, snr_vals):
                if s > rgb[i, j, ch_idx]:
                    rgb[i, j, ch_idx] = float(s)

            # Persistence: mark cells that received ≥1 point this frame
            present = np.zeros((nr, naz), dtype=np.float32)
            np.add.at(present, (ri, azi), 1.0)
            hit_count += (present > 0).astype(np.float32)

            # Temporal centroid accumulation
            for i, j, s in zip(ri, azi, snr_vals):
                snr_weighted_t[i, j] += s * t_norm
                snr_total[i, j] += s

        # Normalize RGB channels to [0, 1]
        for ch in range(3):
            ch_max = rgb[:, :, ch].max()
            if ch_max > 0:
                rgb[:, :, ch] /= ch_max

        persistence = hit_count / 3.0   # already in [0, 1]

        # Temporal centroid: avoid division by zero
        safe_total = np.where(snr_total > 0, snr_total, 1.0)
        temporal_centroid = np.where(snr_total > 0,
                                     snr_weighted_t / safe_total,
                                     0.0).astype(np.float32)

        return {
            'rgb': rgb,
            'persistence': persistence,
            'temporal_centroid': temporal_centroid,
        }

    def pixelize_batch(self, points: np.ndarray, frame_start: int) -> np.ndarray:
        """Pixelize exactly frames [frame_start, frame_start+1, frame_start+2]."""
        return self.pixelize_triplet(points, frame_start, frame_start + 1, frame_start + 2)

    def get_grid_shape(self, points: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """Return (n_range_bins, n_az_bins) for the grid."""
        if self.mode == 'adaptive' and points is not None:
            re, ae = self._build_grid(points)
            return len(re) - 1, len(ae) - 1
        return self.n_range_bins, self.n_az_bins

    def get_grid_edges(self, points: Optional[np.ndarray] = None):
        """Return (range_edges, az_edges)."""
        if self.mode == 'adaptive' and points is not None:
            return self._build_grid(points)
        range_edges = np.linspace(self.min_range, self.max_range, self.n_range_bins + 1)
        az_edges = np.linspace(0, 360, self.n_az_bins + 1)
        return range_edges, az_edges

    def get_cell_centers(self, points: Optional[np.ndarray] = None):
        """Return (range_centers, az_centers) for the grid cells."""
        re, ae = self.get_grid_edges(points)
        r_centers = (re[:-1] + re[1:]) / 2
        az_centers = (ae[:-1] + ae[1:]) / 2
        return r_centers, az_centers
