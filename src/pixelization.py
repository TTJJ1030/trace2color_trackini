"""
Pixelization module.

Converts point trace data to 2D polar grid RGB images.
Frame triplets are mapped to R/G/B channels for temporal encoding.
"""

import numpy as np
from typing import Optional, Tuple


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
