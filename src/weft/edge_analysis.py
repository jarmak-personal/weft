"""Edge detection and contour-based candidate generation for WEFT.

Replaces template-driven candidates with primitives placed along actual
image features — edges, contours, gradient flow, and region boundaries.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .primitives import Primitive


def _sobel_edges(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel edge magnitude and orientation for a tile.

    Returns (magnitude, angle, luminance) where angle is in [0, pi).
    """
    lum = 0.2126 * tile[..., 0] + 0.7152 * tile[..., 1] + 0.0722 * tile[..., 2]

    # 3x3 Sobel kernels applied via convolution.
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    h, w = lum.shape
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    for dy in range(3):
        for dx in range(3):
            patch = lum[max(0, dy - 1):h - 1 + dy, max(0, dx - 1):w - 1 + dx]
            target = gx[max(1, dy):min(h, h - 1 + dy), max(1, dx):min(w, w - 1 + dx)]
            sz = min(patch.shape[0], target.shape[0]), min(patch.shape[1], target.shape[1])
            gx[max(1, dy):max(1, dy) + sz[0], max(1, dx):max(1, dx) + sz[1]] += kx[dy, dx] * patch[:sz[0], :sz[1]]
            gy[max(1, dy):max(1, dy) + sz[0], max(1, dx):max(1, dx) + sz[1]] += ky[dy, dx] * patch[:sz[0], :sz[1]]

    mag = np.sqrt(gx * gx + gy * gy)
    angle = np.arctan2(gy, gx) % np.pi  # [0, pi)
    return mag, angle, lum


def _dominant_orientations(angle: np.ndarray, mag: np.ndarray, n_bins: int = 12) -> list[float]:
    """Extract dominant edge orientations from gradient histogram.

    Returns angles (radians) of the top histogram peaks.
    """
    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    hist = np.zeros(n_bins, dtype=np.float32)
    flat_angle = angle.ravel()
    flat_mag = mag.ravel()

    for i in range(n_bins):
        mask = (flat_angle >= bin_edges[i]) & (flat_angle < bin_edges[i + 1])
        hist[i] = float(np.sum(flat_mag[mask]))

    if hist.max() < 1e-6:
        return [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Return orientations of bins with weight > 20% of max.
    threshold = hist.max() * 0.2
    peaks = []
    for i in range(n_bins):
        if hist[i] >= threshold:
            peaks.append(float((bin_edges[i] + bin_edges[i + 1]) / 2.0))
    return peaks if peaks else [float(bin_edges[np.argmax(hist)] + np.pi / (2 * n_bins))]


def _extract_edge_positions(
    mag: np.ndarray, threshold_pct: float = 60.0
) -> list[tuple[float, float]]:
    """Extract pixel positions where edge magnitude exceeds a percentile threshold."""
    threshold = float(np.percentile(mag, threshold_pct))
    if threshold < 1e-6:
        return []
    ys, xs = np.where(mag >= threshold)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def _fit_line_to_points(
    points: list[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    """Fit a line segment to a set of 2D points via PCA."""
    if len(points) < 2:
        return None
    pts = np.array(points, dtype=np.float32)
    cx, cy = pts.mean(axis=0)
    centered = pts - np.array([cx, cy])
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Principal direction = largest eigenvector
    direction = eigvecs[:, 1]
    projections = centered @ direction
    t_min, t_max = float(projections.min()), float(projections.max())
    x0 = cx + direction[0] * t_min
    y0 = cy + direction[1] * t_min
    x1 = cx + direction[0] * t_max
    y1 = cy + direction[1] * t_max
    return (x0, y0, x1, y1)


def _cluster_edge_points(
    points: list[tuple[float, float]], n_clusters: int = 4
) -> list[list[tuple[float, float]]]:
    """Simple k-means clustering of edge points."""
    if len(points) <= n_clusters:
        return [points] if points else []

    pts = np.array(points, dtype=np.float32)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pts), n_clusters, replace=False)
    centers = pts[idx].copy()

    for _ in range(15):
        dists = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for j in range(n_clusters):
            mask = labels == j
            if mask.any():
                new_centers[j] = pts[mask].mean(axis=0)
        if np.allclose(new_centers, centers, atol=0.1):
            break
        centers = new_centers

    clusters: list[list[tuple[float, float]]] = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(points[i])
    return [c for c in clusters if len(c) >= 2]


def _ct(a) -> tuple[float, float, float]:
    return (max(0.0, min(1.0, float(a[0]))),
            max(0.0, min(1.0, float(a[1]))),
            max(0.0, min(1.0, float(a[2]))))


def generate_edge_driven_candidates(
    tile: np.ndarray,
    quality: int = 75,
) -> list[Primitive]:
    """Generate primitives placed along actual image features.

    This replaces the template-driven _generate_candidates with content-aware
    primitive placement guided by Sobel edge detection.
    """
    h, w = tile.shape[:2]
    hf, wf = float(h - 1), float(w - 1)
    mag, angle, lum = _sobel_edges(tile)
    pixels = tile.reshape(-1, 3).astype(np.float32)
    avg = _ct(tile.mean(axis=(0, 1)))

    cands: list[Primitive] = []

    # ── CONST PATCHES (palette-based, same as before — these work well) ──
    from .encoder import _kmeans_colors
    palette: list[tuple[float, float, float]] = [avg]
    for k in (2, 3, 4):
        for c in _kmeans_colors(pixels, k):
            ct = _ct(c)
            if not any(sum((a - b) ** 2 for a, b in zip(ct, p)) < 0.002 for p in palette):
                palette.append(ct)

    for color in palette:
        for a in (1.0, 0.7, 0.4, 0.2):
            cands.append(Primitive(kind=0, geom=(), color0=color, alpha=a))

    # Per-quadrant averages
    qh, qw = h // 2, w // 2
    for qr in range(2):
        for qc in range(2):
            qcolor = _ct(tile[qr * qh:(qr + 1) * qh, qc * qw:(qc + 1) * qw].mean(axis=(0, 1)))
            for a in (0.5, 0.3):
                cands.append(Primitive(kind=0, geom=(), color0=qcolor, alpha=a))

    # ── LINEAR PATCHES along dominant edge orientations ──
    orientations = _dominant_orientations(angle, mag)
    for theta in orientations:
        vx, vy = math.cos(theta), math.sin(theta)
        # Perpendicular to edge = gradient direction for color split
        nx, ny = -vy, vx
        cx, cy = wf * 0.5, hf * 0.5

        x0 = cx - nx * wf * 0.6
        y0 = cy - ny * hf * 0.6
        x1 = cx + nx * wf * 0.6
        y1 = cy + ny * hf * 0.6

        # Split colors by which side of the line each pixel is on
        ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        side = (xs - x0) * (y1 - y0) - (ys - y0) * (x1 - x0)
        a_mask = side >= 0
        b_mask = ~a_mask
        c0 = _ct(tile[a_mask].mean(axis=0)) if a_mask.any() else avg
        c1 = _ct(tile[b_mask].mean(axis=0)) if b_mask.any() else avg

        for a in (1.0, 0.7, 0.4):
            cands.append(Primitive(kind=1, geom=(x0, y0, x1, y1), color0=c0, color1=c1, alpha=a))

        # Also generate with offset centers (25%, 75% of tile)
        for offset_frac in (0.25, 0.75):
            ox = cx + nx * wf * (offset_frac - 0.5)
            oy = cy + ny * hf * (offset_frac - 0.5)
            lx0 = ox - nx * wf * 0.4
            ly0 = oy - ny * hf * 0.4
            lx1 = ox + nx * wf * 0.4
            ly1 = oy + ny * hf * 0.4
            cands.append(Primitive(kind=1, geom=(lx0, ly0, lx1, ly1), color0=c0, color1=c1, alpha=0.5))

    # ── LINES at detected edge positions ──
    edge_points = _extract_edge_positions(mag, threshold_pct=60.0)
    if edge_points:
        clusters = _cluster_edge_points(edge_points, n_clusters=min(6, max(2, len(edge_points) // 4)))
        avg_mag = float(np.mean(mag))
        base_alpha = min(1.0, 0.25 + avg_mag * 3.0)

        # Edge color from high-gradient pixels
        grad_thresh = float(np.percentile(mag, 75))
        emask = mag >= grad_thresh
        edge_color = _ct(tile[emask].mean(axis=0)) if emask.any() else avg

        for cluster in clusters:
            line = _fit_line_to_points(cluster)
            if line is None:
                continue
            lx0, ly0, lx1, ly1 = line
            # Clamp to tile bounds
            lx0 = max(0.0, min(wf, lx0))
            ly0 = max(0.0, min(hf, ly0))
            lx1 = max(0.0, min(wf, lx1))
            ly1 = max(0.0, min(hf, ly1))

            # Sample color at the line position
            mid_x = int(max(0, min(w - 1, round((lx0 + lx1) / 2))))
            mid_y = int(max(0, min(h - 1, round((ly0 + ly1) / 2))))
            line_color = _ct(tile[mid_y, mid_x])

            for thickness in (0.4, 0.8, 1.5, 2.5):
                for lc in (line_color, edge_color):
                    cands.append(Primitive(
                        kind=2,
                        geom=(lx0, ly0, lx1, ly1, thickness),
                        color0=lc, alpha=base_alpha,
                    ))

    # ── CURVES fitted to edge contours ──
    if edge_points and len(edge_points) >= 6:
        # Fit quadratic curves through edge clusters
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            pts = np.array(cluster, dtype=np.float32)
            # Sort by position along principal axis
            cx_c, cy_c = pts.mean(axis=0)
            centered = pts - np.array([cx_c, cy_c])
            cov = centered.T @ centered
            _, eigvecs = np.linalg.eigh(cov)
            proj = centered @ eigvecs[:, 1]
            order = np.argsort(proj)
            sorted_pts = pts[order]

            # Start, control (midpoint with offset), end
            p0 = sorted_pts[0]
            p1 = sorted_pts[-1]
            mid = sorted_pts[len(sorted_pts) // 2]
            # Control point = midpoint pushed away from the chord
            chord_mid = (p0 + p1) / 2
            ctrl = 2 * mid - chord_mid  # reflect midpoint across chord

            x0, y0 = max(0.0, min(wf, float(p0[0]))), max(0.0, min(hf, float(p0[1])))
            cx_v, cy_v = max(0.0, min(wf, float(ctrl[0]))), max(0.0, min(hf, float(ctrl[1])))
            x1, y1 = max(0.0, min(wf, float(p1[0]))), max(0.0, min(hf, float(p1[1])))

            mid_xi = int(max(0, min(w - 1, round(float(mid[0])))))
            mid_yi = int(max(0, min(h - 1, round(float(mid[1])))))
            curve_color = _ct(tile[mid_yi, mid_xi])

            for thickness in (0.5, 1.0, 2.0):
                cands.append(Primitive(
                    kind=3,
                    geom=(x0, y0, cx_v, cy_v, x1, y1, thickness),
                    color0=curve_color, alpha=base_alpha if edge_points else 0.5,
                ))

    # ── TRIANGLES from k-means region boundaries ──
    if quality >= 60:
        from .encoder import _kmeans_colors as _km
        km3 = _km(pixels, 3)
        km_labels = np.argmin(
            np.sum((pixels[:, None, :] - km3[None, :, :]) ** 2, axis=2), axis=1,
        ).reshape(h, w)
        for lab in range(3):
            m = km_labels == lab
            ys_m, xs_m = np.where(m)
            if len(ys_m) >= 3:
                ix_min = np.argmin(xs_m)
                ix_max = np.argmax(xs_m)
                cxm = float(np.mean(xs_m))
                cym = float(np.mean(ys_m))
                geom = (
                    max(0.0, min(wf, float(xs_m[ix_min]))),
                    max(0.0, min(hf, float(ys_m[ix_min]))),
                    max(0.0, min(wf, float(xs_m[ix_max]))),
                    max(0.0, min(hf, float(ys_m[ix_max]))),
                    max(0.0, min(wf, cxm)),
                    max(0.0, min(hf, cym)),
                )
                tc = _ct(km3[lab])
                for a in (0.6, 0.3):
                    cands.append(Primitive(kind=4, geom=geom, color0=tc, alpha=a))

    # Deduplicate
    from .encoder import _dedup_candidates
    return _dedup_candidates(cands)
