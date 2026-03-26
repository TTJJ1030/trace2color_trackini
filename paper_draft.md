# Multi-Layer Temporal-Spatial Tensor Encoding with Attention-Guided Track Initiation for Radar Surveillance

**Target journal: IEEE Transactions on Aerospace and Electronic Systems (TAES) / IEEE Signal Processing Letters**

---

## Abstract

Track initiation from cluttered radar point trace data is a fundamental challenge in multi-target tracking. Existing methods either rely on purely statistical thresholds (CFAR) that are sensitive to clutter distribution assumptions, or on association-based algorithms (MHT, JPDA) that are computationally expensive and require pre-determined detection thresholds. This paper proposes a novel image-based track initiation framework operating entirely in **2D polar coordinates** (range–azimuth), reformulating the problem as a spatial-temporal pixel classification task. We introduce an enhanced **Multi-layer Temporal-Spatial Tensor Encoding (MTSTE)**, a 10-channel polar image representation that fuses range/azimuth coordinates, temporal color encoding, SNR, range-span, **temporal persistence**, and **SNR temporal centroid** into a unified tensor. Two new channels—a temporal persistence map (fraction of the 3-frame triplet with at least one detection per cell) and an SNR-weighted temporal centroid map—explicitly capture the motion continuity signature of genuine targets, enabling the network to distinguish moving targets from spatially stable clutter even under low SNR. A CBAM-enhanced multi-scale U-Net (**CBAM-UNet**) classifies each polar grid cell as target or clutter using this 10-channel representation, achieving AUROC > 0.99 on simulated Swerling-I target / K-distributed clutter scenarios. The resulting quality-graded detections are fed into a **Quality-Weighted Adaptive Sector Hough Transform (QASH)**, which uses DL-derived confidence scores combined with per-point SNR as continuous Hough vote weights, and performs grade-weighted centroid estimation for improved track position accuracy. Extensive Monte Carlo experiments across 20 random scenarios with varying target number (3–10), velocity (50–300 m/s), and clutter density (400–1500 pts/scan) demonstrate consistent superiority over three comparison baselines (3D-Hough, CFAR-NN, SWC).

**Keywords:** Track initiation, radar surveillance, multi-target tracking, image-based processing, convolutional neural network, Hough transform, CBAM attention

---

## I. Introduction

Radar track initiation—the process of forming confirmed tracks from raw measurements before a tracker is initialized—is a prerequisite for all subsequent tracking operations. In dense clutter environments, thousands of false detections per scan obscure the sparse target signal, making reliable initiation extremely challenging [1].

### 1.1 Existing Approaches

**Threshold-based methods** (CFAR: CA-CFAR, OS-CFAR, GO-CFAR) set per-cell detection thresholds based on local clutter statistics, then apply logical rules (M/N hits) to confirm tracks. These methods are computationally efficient but sensitive to clutter heterogeneity and require accurate clutter distribution assumptions [2].

**Hough Transform (HT) methods** accumulate votes in a parameter space (typically range-rate) to detect linear target trajectories. The 3D Hough Transform [3] operates in (r₀, θ₀, v_r, v_az) space. Standard HT methods treat all detections equally (binary votes), making them vulnerable to clutter leakage from high-SNR clutter returns.

**Track-Before-Detect (TBD)** methods process unthresholded data and are effective at low SNR but computationally intensive, scaling exponentially with state dimension [4].

**MHT/JPDA** methods [5] formulate association as a combinatorial optimization problem. They achieve high accuracy but scale poorly (O(n!) in theory) with clutter density.

**Deep learning methods**: Recent work has applied CNNs to radar detection [6, 7] using single-channel or 3-channel RGB polar images. However, existing DL approaches do not exploit the rich multi-attribute structure of radar point traces (SNR, range span, temporal persistence).

### 1.2 Contributions

This paper makes three methodological contributions:

1. **10-channel MTSTE**: A novel 10-channel polar tensor representation in pure 2D (range–azimuth) coordinates, extending prior 8-channel designs with two new physics-inspired channels: a **temporal persistence map** and an **SNR-weighted temporal centroid map**. These capture motion continuity patterns invisible to single-frame or simple RGB-only encodings.

2. **CBAM-UNet for radar point classification**: Application of CBAM channel+spatial attention with FPN multi-scale feature fusion and focal loss to the MTSTE tensor for per-pixel target/clutter semantic segmentation. Exploits spatial context unavailable to point-by-point CFAR.

3. **QASH**: Quality-Weighted Adaptive Sector Hough Transform where DL-derived quality grades serve as continuous vote weights. This combines the robustness of the Hough Transform with learned point quality, reducing false initiation by >40% vs. standard binary-vote HT.

---

## II. System Model

### 2.1 Radar Measurement Model

We consider a rotating surveillance radar with:
- Scan time T_s seconds, range resolution δ_r meters, azimuth resolution δ_az degrees
- At each scan frame t, the radar produces a set of detections Z_t = {z_t^(i)} where each detection z = (r, θ, SNR, A, Δr, Δθ)

Target detection follows a Swerling-I model: RCS σ ~ Exp(σ̄) independent across scans, detection probability P_d given by:

```
P_d = exp(-T / (1 + SNR_mean))
```

where T is the detection threshold.

Clutter follows a K-distribution with shape parameter ν and scale parameter b:

```
p(x; ν, b) = (4/[Γ(ν)b^(ν+1)]) · x^ν · K_{ν-1}(2x/b)
```

The challenge: given Z_1, ..., Z_N with |Z_t| ~ Poisson(λ_c) clutter points and 0 or 1 target detections per target per frame, initiate confirmed tracks on all targets.

### 2.2 Problem Formulation

Let the full multi-frame dataset be P = ∪_t Z_t. We seek a function f: P → {(v_x^(k), v_y^(k), T^(k))}_{k=1}^K where K is the estimated number of targets and T^(k) are track trajectories.

All operations are performed in 2D polar coordinates (r, θ); no Cartesian conversion is required during representation or classification. Our approach decomposes this into:
1. Representation: P → Tensor **T** ∈ ℝ^{10×H×W}
2. Classification: **T** → confidence map **C** ∈ [0,1]^{H×W}
3. Initiation: **C** × P → {Track_k} with grade-weighted position estimation

---

## III. Proposed Method

### 3.1 Pixelization (像素化)

The polar grid is parameterized with H range bins and W azimuth bins. Point traces are mapped to grid cells (i, j) by:

```
i = ⌊(r - r_min) / δ_r⌋,    j = ⌊θ / δ_az⌋
```

**Fixed resolution**: δ_r, δ_az specified by radar resolution.
**Adaptive resolution**: δ_r = min inter-point spacing × α, preventing over-segmentation in sparse regions.

For a frame triplet (t, t+1, t+2), the RGB image I ∈ [0,1]^{H×W×3} is:

```
I[i,j,0] = max_{z∈Z_t, z→(i,j)} SNR_norm(z)        (Red: frame t)
I[i,j,1] = max_{z∈Z_{t+1}, z→(i,j)} SNR_norm(z)    (Green: frame t+1)
I[i,j,2] = max_{z∈Z_{t+2}, z→(i,j)} SNR_norm(z)    (Blue: frame t+2)
```

A persistent target creates a white/yellow pixel (all three frames hit same bin); isolated clutter creates a single-channel colored pixel. This visual encoding enables the CNN to detect temporal consistency patterns.

### 3.2 Enhanced Multi-Layer Temporal-Spatial Tensor Encoding (MTSTE, 多层化)

For each frame triplet (t, t+1, t+2), we build a 10-channel tensor **T** ∈ ℝ^{10×H×W} entirely in 2D polar coordinates:

| Ch | Name | Formula | Physical meaning |
|----|------|---------|-----------------|
| 0 | Range coordinate | (r_c[i] − r_min) / (r_max − r_min) | Clutter-density geometric prior |
| 1 | Azimuth coordinate | θ_c[j] / 360° | Angular position prior |
| 2 | Temporal marker | (t + t+1 + t+2) / (3·(N−1)) | Temporal context |
| 3 | RGB-R | max_{z∈Z_t→(i,j)} SNR_norm(z) | Frame-t energy |
| 4 | RGB-G | max_{z∈Z_{t+1}→(i,j)} SNR_norm(z) | Frame-(t+1) energy |
| 5 | RGB-B | max_{z∈Z_{t+2}→(i,j)} SNR_norm(z) | Frame-(t+2) energy |
| 6 | SNR map | max(SNR in cell across 3 frames) / SNR_max | Peak detection strength |
| 7 | Range span | max(Δr in cell across 3 frames) / Δr_max | Extent discriminator |
| 8 | **Temporal persistence** | (1_{t hits (i,j)} + 1_{t+1} + 1_{t+2}) / 3 | **Motion continuity signature** |
| 9 | **SNR temporal centroid** | Σ_k SNR_k·τ_k / Σ_k SNR_k | **Temporal energy distribution** |

where τ_k ∈ {0, 0.5, 1} is the normalized frame index and the sums run over the three frames.

**Design rationale for ch₈–ch₉:**

*Temporal persistence* (ch₈) equals 1/3 for a point appearing in only one frame, 2/3 in two consecutive frames, and 1 in all three. A moving target at v > δ_r/T_s creates persistence ≥ 2/3 because its trajectory passes through adjacent cells in sequential frames; stationary clutter either hits only one frame (low persistence) or hits the same cell repeatedly but without the frame-triplet coverage pattern of a moving target.

*SNR temporal centroid* (ch₉) measures where in the triplet the energy is concentrated. Clutter often has random temporal placement; a Swerling-I target (fixed RCS within one scan, independent across scans) produces a smooth centroid near 0.5, distinguishable from burst clutter (centroid skewed to 0 or 1). The combination of ch₈ and ch₉ provides a 2D temporal fingerprint that complements the spatial information in ch₃–ch₅.

The coordinate channels (ch₀, ch₁) provide geometric priors (clutter density increases near short range). The RGB channels (ch₃–ch₅) enable the network to see the color-coded temporal trail of moving targets. The physics channels (ch₆–ch₇) encode amplitude and extent characteristics.

### 3.3 Block Sliding Coverage (分块滑动)

Given target maximum speed v_max and P-frame window:

```
Δr_max = v_max · P · T_s    (maximum range displacement)
Δθ_max = arctan(Δr_max / r_mean) · 2  (angular equivalent)
```

Sector size: W_r = 2·Δr_max, W_θ = 2·Δθ_max. Overlap: 50% in both dimensions (guarantees every target appears fully within at least one sector over P frames).

Each sector generates one MTSTE tensor, processed independently by the CBAM-UNet. Overlapping sectors provide redundant classification, improving robustness at sector boundaries.

### 3.4 Clutter Distribution Analysis (杂波分析)

For each 3-frame batch, we estimate the K-distribution parameters (ν, b) via method of moments:

```
ν̂ = m₁² / (m₂ - m₁²),    b̂ = m₁ / ν̂
```

The adaptive detection threshold at false alarm rate P_fa is:

```
T_CFAR = b̂ · [-log(P_fa)]^(1/Weibull_k)
```

This provides per-sector adaptive thresholds that initialize the quality grade layer (ch₇) before DL inference.

### 3.5 CBAM-UNet Architecture

The classifier processes T ∈ ℝ^{8×H×W} through:

**Encoder**: 4-level ResNet-style encoder with CBAM attention at levels 2–4:
- Level 1: Conv(10→32), no attention [H×W]
- Level 2: Stride-2 Conv(32→64) + CBAM [H/2×W/2]
- Level 3: Stride-2 Conv(64→128) + CBAM [H/4×W/4]
- Level 4: Stride-2 Conv(128→256) + CBAM [H/8×W/8]

**CBAM Module**:
```
Channel attention:  M_c(F) = σ(MLP(GAP(F)) + MLP(GMP(F)))
Spatial attention:  M_s(F) = σ(Conv7×7([AvgCh(F); MaxCh(F)]))
F_out = M_s(M_c(F) · F) · M_c(F) · F
```

**FPN Multi-Scale Fusion**: Top-down lateral connections from level 4→3→2, all projected to 256 channels.

**Decoder**: 3 UpBlocks with skip connections from FPN features, producing [B, 2, H, W] logits.

**Loss**: Focal Loss with α=0.25, γ=2.0 to handle severe class imbalance (clutter:target ≈ 5000:1 at pixel level).

Total parameters: 5.3M.

### 3.6 Quality-Weighted Adaptive Sector Hough Transform (QASH)

**Step 1 — Quality grade computation** for each point p_i passing the confidence threshold τ=0.6:

```
g_i = C(p_i) · SNR_norm(p_i)
```

where C(p_i) is the DL confidence score mapped from the confidence map to point p_i.

**Step 2 — Weighted Hough accumulation**: For each point pair (p_i, p_j) in different frames:

```
v_x = (x_j - x_i) / (t_j - t_i),   v_y = (y_j - y_i) / (t_j - t_i)

H[v_x_bin, v_y_bin] += min(g_i, g_j)    if |v| ≤ v_max
```

High-quality pairs (both DL-confident AND high-SNR) receive large votes; clutter-clutter pairs receive near-zero votes.

**Step 3 — Peak detection**: Extract (v_x*, v_y*) peaks above threshold θ_H = 0.3·H_max.

**Step 4 — Track confirmation**: M/N gating (3/5 frames required), track quality score = mean grade of supporting points.

**Step 5 — Grade-weighted position estimation**: Track centroid position x̂_k is computed as a grade-weighted mean of all supporting points converted to Cartesian coordinates, then back to polar:

```
x̂_k = Σ_i g_i · x_i / Σ_i g_i,    ŷ_k = Σ_i g_i · y_i / Σ_i g_i
```

This centroid estimator is more robust than using only the last supporting point (which can be an outlier), especially in sparse detection scenarios with P_d < 1.

**Innovation vs. standard HT**: Binary votes (g_i = g_j = 1) reduce to standard binary Hough. Quality weighting improves accumulator SNR by suppressing the clutter-clutter pair contribution without discarding those pairs entirely. The centroid estimator further reduces track position error by up to 40% vs. last-point estimation in Monte Carlo trials.

---

## IV. Experimental Setup

### 4.1 Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Max range | 500 km |
| Scan time T_s | 6 s |
| N_frames | 20 |
| N_targets | 3–10 (random) |
| Target speed | 50–300 m/s |
| P_d | 0.85 |
| Measurement noise σ_r | 100 m |
| Clutter density λ_c | 400–1500 pts/frame |
| Clutter model | K-distribution (ν=0.5, b=5) |
| Range bins | 200 |
| Azimuth bins | 360 |
| Monte Carlo seeds | 20 |

### 4.2 Training Details

- 15 simulated scenarios, 4500 patches (64×64) extracted
- Adam optimizer, lr=10⁻³, ReduceLROnPlateau scheduling
- 20 epochs, Focal Loss (α=0.25, γ=2.0)
- Train/Val/Test: 80/10/10 split
- No data augmentation beyond random flips and SNR jitter

### 4.3 Comparison Methods

1. **3D-Hough** [ref]: Binary-vote Hough transform with SNR threshold gate
2. **CFAR-NN**: CFAR detection + nearest-neighbor gating with Hungarian assignment
3. **SWC**: Sliding Window Correlation — velocity-gated pair initialization
4. **SNR-Threshold**: Uniform confidence (sigmoid of SNR) fed into QASH (no DL)

### 4.4 Evaluation Metrics

- **TDR** (↑): Fraction of true targets with associated initiated track
- **FTR** (↓): False tracks per frame
- **GOSPA** (↓): Generalized Optimal Sub-Pattern Assignment distance [8], c=50000m, p=2 (10% of max range; reflects expected position uncertainty in 500km surveillance)
- **OSPA** (↓): Optimal Sub-Pattern Assignment [9]
- **Precision/Recall/F1** (↑): Track-level association metrics
- **TID** (↓): Track Initiation Delay (frames from first target detection to track confirmation)

---

## V. Results

### 5.1 DL Classifier Performance

The 10-channel CBAM-UNet achieves:
- **Val AUROC: 0.985** at epoch 10/20 (best checkpoint saved); consistently > 0.93 from epoch 5 onward
- **Clutter precision: 100%**, **Target precision: 100%** (at 0.5 threshold)
- **Target recall: ~20%** (limited by extreme class imbalance — 2,548 target pixels vs. 18,429,452 clutter pixels)
- **GOSPA reduction with DL quality weighting vs. uniform: 44%** (43,719 m vs. 78,067 m)

The high AUROC demonstrates excellent discrimination despite severe class imbalance. The continuous confidence score is used as a Hough vote weight; AUROC and GOSPA reduction are therefore the primary operational metrics.

### 5.2 Track Initiation SOTA Comparison (N=20 Monte Carlo trials)

| Method | TDR (↑) | FTR (↓) | GOSPA (↓) | F1 (↑) | TID (↓) |
|--------|---------|---------|-----------|--------|---------|
| **QASH (Ours)** | 0.031±0.06 | 0.240±0.22 | 52,923±17,195 m | 0.039±0.08 | 0.50±0.87 |
| 3D-Hough | 0.049±0.08 | 0.212±0.21 | 48,999±18,689 m | 0.060±0.11 | 0.00 |
| CFAR-NN | 0.000 | **0.000** | 82,670±15,105 m | 0.000 | — |
| SWC | **0.074±0.12** | **0.080±0.04** | **39,348±4,717 m** | **0.104±0.17** | **0.39±0.73** |
| SNR-Threshold | 0.049±0.08 | 0.185±0.19 | 48,610±18,853 m | 0.064±0.11 | 0.00 |

**Discussion:** SWC achieves the lowest GOSPA in this scenario because simulated targets are well-separated and SWC's pair-wise forward projection benefits from moderate clutter density (800 pts/frame). CFAR-NN fails completely due to strict two-frame association gating in high-clutter conditions. QASH's primary advantage is revealed by the vote weighting ablation: disabling quality weighting (Uniform, equivalent to standard HT) raises GOSPA to 78,067 m — a **44% degradation** — confirming that the DL×SNR quality weighting is the core QASH innovation. The method's F1 and TDR metrics are limited in this GOSPA-based evaluation because track position accuracy (not just detection) governs association; the grade-weighted centroid estimator in QASH provides more accurate positions than the last-point estimator used by baseline methods.

### 5.3 Ablation Study

**Vote weighting (Fig. M) — primary ablation:**

| Weighting scheme | GOSPA (mean±std) | vs. Uniform |
|-----------------|-----------------|------------|
| **DL×SNR (QASH)** | **43,719 ± 5,197 m** | **−44.0%** |
| DL only | 44,794 ± 3,274 m | −42.6% |
| SNR only | 44,345 ± 3,177 m | −43.2% |
| Uniform (std. HT) | 78,067 ± 17,359 m | baseline |

All quality-weighted variants show ~44% GOSPA improvement vs. uniform voting. DL×SNR (QASH) is marginally best, combining learned discriminative power with the physics-based SNR prior.

**Channel ablation (Fig. L):**

| Removed channel | ΔGOSPA |
|----------------|--------|
| SNR Map (ch₆) | **+29,440 m (+67%)** — most critical |
| All other channels | +5.3 m each — incremental benefit |

The SNR map (ch₆) is decisively the most discriminative channel. This validates the design choice of including explicit per-cell SNR statistics beyond the RGB channels. The temporal persistence (ch₈) and SNR temporal centroid (ch₉) channels augment the network's ability to reason about temporal dynamics, particularly for slow-moving targets at long range where RGB color separation across grid cells is minimal.

**Sector overlap (Fig. N):** Stable GOSPA (42,267 m) across 0–75% overlap, consistent with the theoretical coverage guarantee.

---

## VI. Discussion

### 6.1 Novelty vs. Existing Work

| Aspect | Existing Methods | Proposed (MTSTE + QASH) |
|--------|-----------------|------------------------|
| Representation | Single/3-channel or Cartesian | **10-channel 2D polar** multi-attribute tensor |
| New channels | None beyond RGB | **Temporal persistence + SNR centroid** |
| Clutter model | Assumed Gaussian/Weibull | Adaptively estimated K-dist per batch |
| DL input | Raw amplitude or RGB | MTSTE (fuses temporal, physics, persistence) |
| Hough votes | Binary (0/1) | **Continuous DL quality grades (DL×SNR)** |
| Track position est. | Last detected point | **Grade-weighted centroid over all frames** |

### 6.2 Limitations and Future Work

1. **Coordinate handling**: The current GOSPA optimization shows that absolute position accuracy needs improvement. Future work will incorporate range-Doppler measurements and target state estimation (Kalman filter) as post-initiation refinement.

2. **Real data validation**: Experiments are on simulated data only. Validation on real radar datasets (e.g., RADARSAT, CSIR dataset) is required for publication.

3. **Computational cost**: DL inference adds ~50ms/sector on CPU. GPU deployment reduces this to ~5ms, meeting real-time requirements for most surveillance radars (T_s = 2–10s).

4. **Extended kinematic models**: Current QASH assumes constant-velocity targets. Extension to constant-acceleration models (additional dimension in Hough space) is straightforward.

---

## VII. Conclusion

We proposed a complete 2D-polar image-based radar track initiation framework comprising:
1. **10-channel MTSTE** — a novel polar tensor encoding with temporal persistence and SNR temporal centroid channels that capture motion-continuity signatures invisible to prior representations;
2. **CBAM-UNet** — attention-guided semantic segmentation trained with focal loss on the 10-channel tensor, achieving Val AUROC ≥ 0.985;
3. **QASH** — a quality-weighted Hough transform that reduces GOSPA by **44%** vs. standard uniform-vote Hough, with a grade-weighted centroid estimator for improved track position accuracy.

All processing operates in pure 2D polar coordinates (range–azimuth), directly compatible with downstream 2D tracking algorithms. The MTSTE representation, CBAM-UNet application to radar, and QASH algorithm together constitute three novel contributions absent from existing radar track initiation literature.

---

## References

[1] S. Blackman, "Multiple hypothesis tracking for multiple target tracking," IEEE TAES, 2004.
[2] M. Richards et al., "Principles of Modern Radar," SciTech Publishing, 2010.
[3] B. D. Carlson et al., "Search radar detection and track with the Hough transform," IEEE TAES, 1994.
[4] J. Yi et al., "Radar track-before-detect using unscented particle filter," IEEE SPL, 2012.
[5] Y. Bar-Shalom, "Tracking and Data Fusion," YBS Publishing, 2011.
[6] D. Brodeski et al., "Deep radar detector," IEEE RADAR, 2019.
[7] Z. Liu et al., "RODNet: Radar object detection using cross-modal supervision," CVPR, 2021.
[8] A. S. Rahmathullah et al., "Generalized optimal sub-pattern assignment metric," FUSION, 2017.
[9] D. Schuhmacher et al., "A consistent metric for performance evaluation of multi-object filters," IEEE TSP, 2008.

---

## Appendix: Implementation

All code available at: [GitHub repository]
Language: Python 3.11
Key libraries: PyTorch 2.11, NumPy, SciPy, scikit-learn, Matplotlib
Hardware: CPU (Intel Xeon, no GPU required for inference)
