import numpy as np
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000


def calc_std_y(img_rgb):
    # measure the std of the luminance curve to verify illumination flatness
    if img_rgb is None: return np.nan
    img_y = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    H, W = img_y.shape
    k = max(1, int(H * 0.2))
    col_means = []

    for x in range(W):
        col = img_y[:, x]
        valid_pixels = col[col > 5.0]
        if len(valid_pixels) < k: valid_pixels = col
        top_vals = np.sort(valid_pixels)[::-1][:k]
        col_means.append(np.mean(top_vals))

    return float(np.std(np.array(col_means) / 255.0))


def calc_side_color_diff(img_rgb):
    # compute physical delta E between degraded edge zones and center
    if img_rgb is None: return np.nan
    H, W, _ = img_rgb.shape
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)

    w_side = int(W * 0.12)
    w_center_half = int(W * 0.10)
    center_x = W // 2

    center_crop = img_lab[:, center_x - w_center_half: center_x + w_center_half, :]
    if center_crop.shape[1] == 0: return 0.0

    mean_c = np.mean(center_crop, axis=(0, 1))
    left_crop = img_lab[:, :w_side, :]
    right_crop = img_lab[:, W - w_side:, :]

    mean_l = np.mean(left_crop, axis=(0, 1)) if left_crop.shape[1] > 0 else mean_c
    mean_r = np.mean(right_crop, axis=(0, 1)) if right_crop.shape[1] > 0 else mean_c

    dist_l = np.sqrt((mean_l[1] - mean_c[1]) ** 2 + (mean_l[2] - mean_c[2]) ** 2)
    dist_r = np.sqrt((mean_r[1] - mean_c[1]) ** 2 + (mean_r[2] - mean_c[2]) ** 2)

    return float((dist_l + dist_r) / 2.0)


def calc_supervised_metrics(gt, res):
    # calculate psnr, ssim, and full ciede2000 against ground truth
    if gt is None or res is None: return np.nan, np.nan, np.nan
    h, w = gt.shape[:2]
    if res.shape[:2] != (h, w): res = cv2.resize(res, (w, h))

    p = psnr(gt, res, data_range=255)
    try:
        s = ssim(gt, res, win_size=11, channel_axis=2, data_range=255)
    except:
        s = ssim(gt, res, win_size=11, multichannel=True, data_range=255)

    g_lab = rgb2lab(gt)
    r_lab = rgb2lab(res)
    c = np.mean(deltaE_ciede2000(g_lab, r_lab))
    return float(p), float(s), float(c)


def evaluate_niqe(img_rgb, device):
    # wrapper for naturalness image quality evaluator
    try:
        import pyiqa
        niqe_metric = pyiqa.create_metric('niqe', device=device, as_loss=False)
        t_img = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            return float(niqe_metric(t_img).item())
    except Exception as e:
        return np.nan