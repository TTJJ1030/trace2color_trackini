"""
Layering module.

Constructs 8-channel multi-layer tensors from point trace data and pixelized images.

Channels:
  0: Normalized range coordinate
  1: Normalized azimuth coordinate
  2: Temporal marker (normalized frame index)
  3: RGB-R (frame t)
  4: RGB-G (frame t+1)
  5: RGB-B (frame t+2)
  6: SNR map (max SNR per cell, normalized)
  7: Range span map (range extent per cell, normalized)

The quality grade (initially threshold-based, later DL-updated) is stored separately.
"""

import numpy as np
from typing import Optional, Tuple
from .pixelization import Pixelizer


class Layerer:
    """
    Builds multi-channel tensor (MTSTE: Multi-layer Temporal-Spatial Tensor Encoding)
    from point trace data.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.pixelizer = Pixelizer(config)
        self.snr_threshold = config['layering']['snr_threshold']
        self.max_range = config['radar']['max_range']
        self.min_range = config['radar']['min_range']

    def build_tensor(self, points: np.ndarray, frame_r: int,
                     frame_g: int, frame_b: int,
                     quality_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Build 8-channel tensor for a triplet of frames.

        Args:
            points: full point trace structured array
            frame_r, frame_g, frame_b: frame indices for R/G/B channels
            quality_map: optional pre-computed quality map [H, W]; if None,
                         uses SNR threshold

        Returns:
            tensor: [8, H, W] float32
        """
        range_edges, az_edges = self.pixelizer.get_grid_edges(points)
        nr = len(range_edges) - 1
        naz = len(az_edges) - 1

        r_centers = (range_edges[:-1] + range_edges[1:]) / 2
        az_centers = (az_edges[:-1] + az_edges[1:]) / 2

        tensor = np.zeros((8, nr, naz), dtype=np.float32)

        # Channel 0: normalized range coordinate
        r_norm = (r_centers - self.min_range) / (self.max_range - self.min_range)
        tensor[0] = r_norm[:, np.newaxis] * np.ones((nr, naz))

        # Channel 1: normalized azimuth coordinate
        az_norm = az_centers / 360.0
        tensor[1] = np.ones((nr, naz)) * az_norm[np.newaxis, :]

        # Channel 2: temporal marker (mean frame index normalized)
        mean_frame = (frame_r + frame_g + frame_b) / 3.0
        n_frames = int(points['frame_id'].max()) + 1
        tensor[2] = mean_frame / max(n_frames - 1, 1)

        # Channels 3-5: RGB from pixelization
        rgb_image = self.pixelizer.pixelize_triplet(points, frame_r, frame_g, frame_b)
        tensor[3] = rgb_image[:, :, 0]  # R
        tensor[4] = rgb_image[:, :, 1]  # G
        tensor[5] = rgb_image[:, :, 2]  # B

        # Channels 6, 7: SNR map and range span map (all 3 frames combined)
        snr_map = np.zeros((nr, naz), dtype=np.float32)
        rspan_map = np.zeros((nr, naz), dtype=np.float32)
        snr_max_global = max(points['snr'].max(), 1.0)
        rspan_max = max(points['range_span'].max(), 1.0)

        for fid in [frame_r, frame_g, frame_b]:
            mask = points['frame_id'] == fid
            pts = points[mask]
            if len(pts) == 0:
                continue
            ri = np.searchsorted(range_edges[1:], pts['range'], side='right')
            azi = np.searchsorted(az_edges[1:], pts['azimuth'], side='right')
            ri = np.clip(ri, 0, nr - 1)
            azi = np.clip(azi, 0, naz - 1)
            for i, j, s, rs in zip(ri, azi, pts['snr'], pts['range_span']):
                if s > snr_map[i, j]:
                    snr_map[i, j] = s
                if rs > rspan_map[i, j]:
                    rspan_map[i, j] = rs

        tensor[6] = snr_map / snr_max_global
        tensor[7] = rspan_map / rspan_max

        return tensor

    def build_label_map(self, points: np.ndarray, frame_r: int,
                        frame_g: int, frame_b: int) -> np.ndarray:
        """
        Build ground truth label map [H, W] for training.

        A cell is labeled as target (1) if any target point falls in it
        across the triplet of frames.
        """
        range_edges, az_edges = self.pixelizer.get_grid_edges(points)
        nr = len(range_edges) - 1
        naz = len(az_edges) - 1
        label_map = np.zeros((nr, naz), dtype=np.int64)

        for fid in [frame_r, frame_g, frame_b]:
            mask = (points['frame_id'] == fid) & (points['is_target'] == 1)
            pts = points[mask]
            if len(pts) == 0:
                continue
            ri = np.searchsorted(range_edges[1:], pts['range'], side='right')
            azi = np.searchsorted(az_edges[1:], pts['azimuth'], side='right')
            ri = np.clip(ri, 0, nr - 1)
            azi = np.clip(azi, 0, naz - 1)
            label_map[ri, azi] = 1

        return label_map

    def build_initial_quality_map(self, points: np.ndarray,
                                  frame_r: int, frame_g: int,
                                  frame_b: int) -> np.ndarray:
        """
        Build initial quality grade map using SNR threshold (before DL).

        Returns [H, W] float32 in [0, 1].
        """
        range_edges, az_edges = self.pixelizer.get_grid_edges(points)
        nr = len(range_edges) - 1
        naz = len(az_edges) - 1
        snr_map = np.zeros((nr, naz), dtype=np.float32)

        for fid in [frame_r, frame_g, frame_b]:
            mask = points['frame_id'] == fid
            pts = points[mask]
            if len(pts) == 0:
                continue
            ri = np.searchsorted(range_edges[1:], pts['range'], side='right')
            azi = np.searchsorted(az_edges[1:], pts['azimuth'], side='right')
            ri = np.clip(ri, 0, nr - 1)
            azi = np.clip(azi, 0, naz - 1)
            for i, j, s in zip(ri, azi, pts['snr']):
                if s > snr_map[i, j]:
                    snr_map[i, j] = s

        quality = (snr_map >= self.snr_threshold).astype(np.float32)
        return quality

    def extract_patches(self, tensor: np.ndarray, label_map: np.ndarray,
                        patch_size: int = 64,
                        stride: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract overlapping patches from tensor and label map for training.

        Returns:
            patches: [N, 8, patch_size, patch_size]
            labels: [N, patch_size, patch_size]
        """
        C, H, W = tensor.shape
        patches, labels = [], []
        for r in range(0, H - patch_size + 1, stride):
            for c in range(0, W - patch_size + 1, stride):
                patch = tensor[:, r:r + patch_size, c:c + patch_size]
                label = label_map[r:r + patch_size, c:c + patch_size]
                patches.append(patch)
                labels.append(label)
        if not patches:
            return np.zeros((0, C, patch_size, patch_size)), np.zeros((0, patch_size, patch_size))
        return np.stack(patches), np.stack(labels)

    def build_all_tensors(self, points: np.ndarray) -> list:
        """
        Build tensors for all overlapping triplets in the dataset.

        Returns list of dicts with keys: tensor, label_map, frame_r, frame_g, frame_b
        """
        n_frames = int(points['frame_id'].max()) + 1
        results = []
        for start in range(0, n_frames - 2):
            fr, fg, fb = start, start + 1, start + 2
            tensor = self.build_tensor(points, fr, fg, fb)
            label_map = self.build_label_map(points, fr, fg, fb)
            results.append({
                'tensor': tensor,
                'label_map': label_map,
                'frame_r': fr,
                'frame_g': fg,
                'frame_b': fb,
            })
        return results
