import os
import cv2
import numpy as np
from PIL import Image
from skimage import color as skcolor


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def imread(path):
    # read rgb properly and ignore alpha channels if they sneak in
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None: raise ValueError
        if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            a = img[..., 3:4].astype(np.float32) / 255.0
            rgb = img[..., :3].astype(np.float32)
            rgb = rgb * a + (1 - a) * 0.0
            img = rgb.astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))


def imwrite(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def to_float01(rgb):
    return rgb.astype(np.float32) / 255.0


def rgb_to_hsv01(rgb01):
    hsv = cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    return hsv[..., 0] / 180.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0


def hsv01_to_rgb(h, s, v):
    hsv = np.stack([np.clip(h, 0, 1) * 180.0, np.clip(s, 0, 1) * 255.0, np.clip(v, 0, 1) * 255.0], axis=-1).astype(
        np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def rgb_to_lab(rgb01):
    lab = skcolor.rgb2lab(np.clip(rgb01, 0, 1))
    return lab[..., 0], lab[..., 1], lab[..., 2]


def lab_to_rgb(L, a, b):
    lab = np.stack([L, a, b], axis=-1)
    rgb = skcolor.lab2rgb(np.clip(lab, [0, -128, -128], [100, 127, 127]))
    return (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_in_orange(h, s):
    # locate the orange/red diffuse reflection from the core box
    hc = 19 / 360.0
    k = 40.0
    dh = np.minimum(np.abs(h - hc), 1.0 - np.abs(h - hc))
    mem_h = np.exp(- (dh * 360.0 / k) ** 2)
    gate = np.clip((s - 0.2) / 0.3, 0.0, 1.0)
    return mem_h * gate


def _dilate1d_bool(mask, r):
    if r <= 0: return mask
    kern = np.ones(2 * r + 1, np.int32)
    vals = np.convolve(mask.astype(np.int32), kern, mode="same")
    return vals > 0


def build_masks_and_ref(rgb, ref_width_frac, lum_weight, edge_outer_frac=0.3, edge_thres_pct=70, edge_min_frac=0.12,
                        edge_expand_frac=0.05):
    # establish source (center) and target (edges) domain masks based on stats
    H, W, _ = rgb.shape
    rgb01 = to_float01(rgb)
    h, s, _ = rgb_to_hsv01(rgb01)
    Y = 0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]

    valid = Y > 0.03
    Y_top = np.zeros(W, np.float32)
    for x in range(W):
        col = Y[:, x][valid[:, x]]
        if col.size < 10: col = Y[:, x]
        k = max(1, int(0.2 * col.size))
        Y_top[x] = float(np.partition(col, -k)[-k:].mean())

    xs = np.arange(W)
    dist = np.minimum(xs, W - 1 - xs).astype(np.float32)
    outer = dist <= edge_outer_frac * W
    Yn = (Y_top - Y_top.min()) / (Y_top.max() - Y_top.min() + 1e-6)

    mem = soft_in_orange(h, s)
    mem_col = mem.mean(axis=0)
    score = np.zeros(W, np.float32)

    score[outer] = lum_weight * (1.0 - Yn[outer]) + (1.0 - lum_weight) * mem_col[outer]
    thr = np.percentile(score[outer], edge_thres_pct)
    edge_cols = score >= thr

    min_w = int(edge_min_frac * W)
    if edge_cols[:W // 2].sum() < min_w: edge_cols[:min_w] = True
    if edge_cols[W // 2:].sum() < min_w: edge_cols[W - min_w:] = True
    edge_cols = _dilate1d_bool(edge_cols, int(edge_expand_frac * W))

    peak_idx = np.argmax(Y_top)
    half_ref_w = int(W * ref_width_frac)
    l = max(0, peak_idx - half_ref_w)
    r = min(W, peak_idx + half_ref_w)

    ref_cols = np.zeros(W, np.bool_)
    ref_cols[l:r] = True

    allow = np.zeros(W, np.bool_)
    allow[:l] = True
    allow[r:] = True
    edge_cols = edge_cols & allow

    return valid, Y_top, ref_cols, edge_cols


def build_stage1_features(rgb, valid, Y_top):
    # flatten the 2d core image into 1d column sequences
    H, W, _ = rgb.shape
    xs = np.linspace(0, 1, W, dtype=np.float32)
    rgb01 = to_float01(rgb)
    h, s, v = rgb_to_hsv01(rgb01)

    mem = soft_in_orange(h, s)
    mem_mu = mem.mean(axis=0)
    s_mu = s.mean(axis=0)

    gx = cv2.Sobel((v * 255.0).astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel((v * 255.0).astype(np.uint8), cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy).mean(axis=0)
    if gm.max() > 0: gm = (gm - gm.min()) / (gm.max() - gm.min() + 1e-6)

    Yt = (Y_top - Y_top.min()) / (Y_top.max() - Y_top.min() + 1e-6)
    feats = np.stack([xs, Yt, mem_mu, s_mu, gm], axis=1).astype(np.float32)
    return feats, {"Y_top": Y_top}