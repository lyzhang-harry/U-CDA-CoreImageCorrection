import numpy as np
from scipy.ndimage import uniform_filter1d, binary_erosion
from utils.feature_extraction import to_float01, rgb_to_lab, lab_to_rgb


def harmonic_fill(fixed, values, W, iters=8000, tol=1e-8):
    # smoothly fill the transition gaps between target and source anchors
    fixed = fixed.astype(np.bool_)
    c = np.zeros(W, np.float32)
    c[fixed] = values[fixed]
    for _ in range(iters):
        old = c.copy()
        for i in range(1, W - 1):
            if not fixed[i]:
                c[i] = 0.5 * (c[i - 1] + c[i + 1])
        if not fixed[0] and W > 1: c[0] = c[1]
        if not fixed[W - 1] and W > 1: c[W - 1] = c[W - 2]
        if np.max(np.abs(c - old)) < tol: break
    return c


def _scan_left_right_spans(edge_cols: np.ndarray, ref_cols: np.ndarray, min_w: int):
    # identify the exact bounds of the degraded edges
    W = edge_cols.size
    c_idx = np.where(ref_cols)[0]
    l_edge, r_edge = c_idx[0], c_idx[-1] + 1

    # scan left side
    i = l_edge - 1
    while i >= 0 and not edge_cols[i]: i -= 1
    if i >= 0:
        j = i
        while j >= 0 and edge_cols[j]: j -= 1
        L_start, L_end = j + 1, l_edge - 1
    else:
        L_end = l_edge - 1
        L_start = max(0, L_end - min_w + 1)
    if L_end - L_start + 1 < min_w: L_start = max(0, L_end - min_w + 1)

    # scan right side
    i = r_edge
    while i < W and not edge_cols[i]: i += 1
    if i < W:
        j = i
        while j < W and edge_cols[j]: j += 1
        R_start, R_end = r_edge, j - 1
    else:
        R_start = r_edge
        R_end = min(W - 1, R_start + min_w - 1)
    if R_end - R_start + 1 < min_w: R_end = min(W - 1, R_start + min_w - 1)

    return (L_start, L_end), (R_start, R_end)


def _spans_to_mask(left_span, right_span, W):
    mask = np.zeros(W, dtype=bool)
    Ls, Le = left_span
    Rs, Re = right_span
    if Le >= Ls: mask[Ls:Le + 1] = True
    if Re >= Rs: mask[Rs:Re + 1] = True
    return mask


def robust_col_stat(arr2d, mask2d):
    # compute median stats per col, ignoring missing data and nans
    H, W = arr2d.shape
    med = np.zeros(W, dtype=np.float32)
    for x in range(W):
        v = arr2d[:, x][mask2d[:, x]]
        med[x] = float(np.median(v)) if v.size > 0 else np.nan
    idx = np.arange(W)
    good = ~np.isnan(med)
    if good.any():
        med[~good] = np.interp(idx[~good], idx[good], med[good])
    else:
        med[:] = 0.0
    return med


def ab_constrained_detrend(rgb_in, valid, ref_cols, rgb_ref=None):
    # remove the lingering slow-drifting color cast in lab space
    img01 = to_float01(rgb_in)
    L, a, b = rgb_to_lab(img01)
    H, W = L.shape

    median_L = np.median(L[valid])
    L_threshold = 30.0 if median_L >= 40.0 else 15.0
    mask_stats = valid & (a < 85.0) & (b < 85.0) & (L > L_threshold)
    mask_stats = binary_erosion(mask_stats, iterations=2)
    cols_data = [robust_col_stat(ch, mask_stats) for ch in (a, b)]

    cidx = np.where(ref_cols)[0]
    refs = [0.0, 0.0]
    mid_idx = int((cidx[0] + cidx[-1]) / 2) if cidx.size > 0 else W // 2
    for i, ch in enumerate([a, b]):
        if cidx.size > 0:
            roi = ch[:, cidx[0]:cidx[-1] + 1]
            mask_roi = mask_stats[:, cidx[0]:cidx[-1] + 1]
            if mask_roi.sum() < 100: mask_roi = valid[:, cidx[0]:cidx[-1] + 1]
            refs[i] = float(np.median(roi[mask_roi])) if mask_roi.any() else float(np.median(roi))
        else:
            refs[i] = float(np.median(ch[:, max(0, mid_idx - 50):min(W, mid_idx + 50)]))

    center_a_ref, center_b_ref = refs
    raw_diffs = [cols_data[i] - refs[i] for i in range(2)]

    base_win = max(9, W // 8)
    win_slow = int(base_win * 1.5) if median_L < 40.0 else base_win
    win_fast = max(5, win_slow // 8)
    mask_float = (~ref_cols).astype(np.float32)
    blur_r = max(5, W // 20)
    alpha_map = uniform_filter1d(mask_float, size=blur_r * 2 + 1, mode='nearest')
    alpha_map = np.clip(alpha_map * 1.2, 0.0, 1.0)

    trends = []
    for i in range(2):
        trend_slow = uniform_filter1d(raw_diffs[i], size=win_slow, mode='nearest')
        trend_fast = uniform_filter1d(raw_diffs[i], size=win_fast, mode='nearest')
        smooth_trend = alpha_map * trend_fast + (1.0 - alpha_map) * trend_slow
        safe_trend = np.minimum(np.maximum(0, smooth_trend), np.maximum(0, raw_diffs[i]))
        trends.append(safe_trend)

    col_L = robust_col_stat(L, valid)
    max_L = col_L.max() if col_L.max() > 1e-3 else 1.0
    base_shadow_factor = np.clip(col_L / max_L, 0.5, 1.0)

    final_trend_a = (trends[0] * 1.0 * np.clip(base_shadow_factor + 0.4, 0.9, 1.0))[None, :]
    final_trend_b = (trends[1] * 0.80 * np.clip(base_shadow_factor + 0.35, 0.85, 1.0))[None, :]

    a2 = np.maximum(a - final_trend_a, center_a_ref - 2.0)
    target_b = b - final_trend_b
    real_floor_b = max(-4.0, center_b_ref - 1.0)

    mask_pos_b = b > 0
    b2 = b.copy()
    b2[mask_pos_b] = np.maximum(target_b[mask_pos_b], real_floor_b)

    decay_w = 100
    decay_map = np.ones(W, dtype=np.float32)
    if W > 2 * decay_w:
        t = np.linspace(0, 1, decay_w)
        curve = t * t * (3 - 2 * t)
        decay_map[:decay_w] = curve
        decay_map[-decay_w:] = curve[::-1]

    a2 = center_a_ref + (a2 - center_a_ref) * decay_map[None, :]
    b2 = center_b_ref + (b2 - center_b_ref) * decay_map[None, :]

    pad_w = 20
    if W > 2 * pad_w:
        a2[:, :pad_w] = a2[:, pad_w:pad_w + 1]
        b2[:, :pad_w] = b2[:, pad_w:pad_w + 1]
        a2[:, -pad_w:] = a2[:, -pad_w - 1:-pad_w]
        b2[:, -pad_w:] = b2[:, -pad_w - 1:-pad_w]

    ref_chans = [a, b] if rgb_ref is None else rgb_to_lab(to_float01(rgb_ref))[1:3]
    final_chans = [a2, b2]

    for i in range(2):
        c_idxs = np.where(ref_cols)[0]
        peak_curr = float(
            np.percentile(final_chans[i][:, c_idxs[0]:c_idxs[-1] + 1].mean(axis=0), 95)) if c_idxs.size > 0 else 0.0
        peak_orig = float(
            np.percentile(ref_chans[i][:, c_idxs[0]:c_idxs[-1] + 1].mean(axis=0), 95)) if c_idxs.size > 0 else 0.0
        final_chans[i] = np.clip(final_chans[i] - (peak_curr - peak_orig), -128, 127)

    return lab_to_rgb(L, final_chans[0], final_chans[1]), trends[0], trends[1]