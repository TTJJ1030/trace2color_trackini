"""
Block sliding coverage module.

Divides the full radar coverage area into overlapping sector windows
based on target speed constraints, and processes each sector independently.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class SectorWindow:
    """Describes one sector processing window."""
    sector_id: int
    range_min: float
    range_max: float
    az_min: float
    az_max: float
    frame_start: int
    frame_end: int

    def contains_point(self, r: float, az: float, frame: int) -> bool:
        az_norm = az % 360
        az_min_norm = self.az_min % 360
        az_max_norm = self.az_max % 360
        if az_min_norm <= az_max_norm:
            az_ok = az_min_norm <= az_norm <= az_max_norm
        else:
            az_ok = az_norm >= az_min_norm or az_norm <= az_max_norm
        r_ok = self.range_min <= r <= self.range_max
        f_ok = self.frame_start <= frame <= self.frame_end
        return r_ok and az_ok and f_ok


class BlockSlidingCoverage:
    """
    Implements adaptive block sliding coverage for radar track initiation.

    Given prior knowledge of target speed and radar parameters, computes
    sector sizes that guarantee a moving target stays within at least one
    sector for P consecutive frames.
    """

    def __init__(self, config: dict):
        self.cfg = config
        bs_cfg = config['block_sliding']
        rdr_cfg = config['radar']

        self.p_frames = bs_cfg['p_frames']
        self.v_max = bs_cfg['v_max']
        self.scan_time = rdr_cfg['scan_time']
        self.overlap_ratio = bs_cfg['overlap_ratio']
        self.n_az_sectors = bs_cfg['n_az_sectors']
        self.max_range = rdr_cfg['max_range']
        self.min_range = rdr_cfg['min_range']

    def compute_sector_sizes(self) -> Tuple[float, float]:
        """
        Compute sector width in range and azimuth based on target kinematics.

        A target moving at v_max for p_frames scans moves at most:
            delta_r = v_max * p_frames * scan_time

        Sector width = 2 * delta_r (to contain full motion).
        Azimuth width: angular equivalent at mid-range.
        """
        delta_r = self.v_max * self.p_frames * self.scan_time
        sector_range_width = 2.0 * delta_r  # meters

        # Angular width: conservative — based on max tangential motion at min range
        mid_range = (self.max_range + self.min_range) / 2
        delta_az_rad = np.arctan2(delta_r, mid_range)
        sector_az_width = np.rad2deg(delta_az_rad) * 2  # degrees
        sector_az_width = max(sector_az_width, 30.0)  # minimum 30 deg

        return sector_range_width, sector_az_width

    def generate_sectors(self, n_frames: int) -> List[SectorWindow]:
        """
        Generate all sliding sector windows covering the full surveillance area.

        Returns:
            List of SectorWindow objects (may overlap significantly).
        """
        sector_range_width, sector_az_width = self.compute_sector_sizes()
        range_step = sector_range_width * (1 - self.overlap_ratio)
        az_step = sector_az_width * (1 - self.overlap_ratio)
        frame_step = max(1, self.p_frames // 2)

        sectors = []
        sector_id = 0

        r_starts = np.arange(self.min_range, self.max_range, range_step)
        az_starts = np.arange(0, 360, az_step)
        f_starts = range(0, n_frames - self.p_frames + 1, frame_step)

        for r0 in r_starts:
            r1 = min(r0 + sector_range_width, self.max_range)
            for a0 in az_starts:
                a1 = a0 + sector_az_width
                for f0 in f_starts:
                    f1 = f0 + self.p_frames - 1
                    sectors.append(SectorWindow(
                        sector_id=sector_id,
                        range_min=float(r0),
                        range_max=float(r1),
                        az_min=float(a0),
                        az_max=float(a1),
                        frame_start=int(f0),
                        frame_end=int(f1),
                    ))
                    sector_id += 1

        return sectors

    def get_sector_points(self, points: np.ndarray,
                          sector: SectorWindow) -> np.ndarray:
        """Extract points belonging to a given sector."""
        r = points['range']
        az = points['azimuth'] % 360
        fid = points['frame_id']

        r_mask = (r >= sector.range_min) & (r <= sector.range_max)
        f_mask = (fid >= sector.frame_start) & (fid <= sector.frame_end)

        az_min = sector.az_min % 360
        az_max = sector.az_max % 360
        if az_min <= az_max:
            az_mask = (az >= az_min) & (az <= az_max)
        else:
            az_mask = (az >= az_min) | (az <= az_max)

        return points[r_mask & az_mask & f_mask]

    def get_visualization_sectors(self, n_frames: int,
                                  n_show: int = 3) -> List[SectorWindow]:
        """
        Return a few representative sector windows for visualization (like Fig 5-3).
        Shows consecutive azimuth sectors at a fixed range band.
        """
        sector_range_width, sector_az_width = self.compute_sector_sizes()
        az_step = sector_az_width * (1 - self.overlap_ratio)
        r0 = self.min_range + sector_range_width
        r1 = r0 + sector_range_width
        result = []
        for k in range(n_show):
            a0 = k * az_step
            a1 = a0 + sector_az_width
            result.append(SectorWindow(
                sector_id=k,
                range_min=float(r0),
                range_max=float(r1),
                az_min=float(a0),
                az_max=float(a1),
                frame_start=k * max(1, self.p_frames // 2),
                frame_end=k * max(1, self.p_frames // 2) + self.p_frames - 1,
            ))
        return result

    def summarize(self, n_frames: int) -> Dict:
        """Return coverage statistics."""
        sectors = self.generate_sectors(n_frames)
        rw, aw = self.compute_sector_sizes()
        return {
            'n_sectors': len(sectors),
            'sector_range_width_km': rw / 1000,
            'sector_az_width_deg': aw,
            'overlap_ratio': self.overlap_ratio,
            'p_frames': self.p_frames,
        }
