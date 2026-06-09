import os
import glob
import math
import gc
import time
import random
import pandas as pd
import numpy as np
import torch
import argparse
from tqdm import tqdm

from models.network import ColumnBackbone, Heads
from models.mk_mmd_loss import mmd_mk_1d_unbiased, mmd_mk_nd_unbiased
from models.mk_mmd_loss import torch_rgb_to_hsv, torch_hsv_to_rgb, torch_rgb_to_lab, torch_smoothstep
from utils.feature_extraction import ensure_dir, imread, imwrite, to_float01, rgb_to_hsv01, hsv01_to_rgb
from utils.feature_extraction import build_masks_and_ref, build_stage1_features
from utils.post_processing import ab_constrained_detrend, harmonic_fill, _scan_left_right_spans, _spans_to_mask
from utils.metrics import calc_std_y, calc_side_color_diff, calc_supervised_metrics, evaluate_niqe

# ---------------------------------------------------------
# Hyperparameters and Configurations
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNIFIED_MMD_SAMPLES = 2000
EDGE_MIN_FRAC = 0.12
EDGE_ALPHA = 0.80

S1_REF_BOOST = 1.02
HL1_Y_LOW, HL1_Y_HIGH, HL1_STRENGTH = 0.5, 0.9, 0.80

W_Y_ALIGN, W_Y_FLAT, W_Y_TV  = 1.6, 0.4, 0.1
W_Y_SMOOTH, W_HSV_SMOOTH = 0.03, 0.03
W_MMD_Y, W_MMD_HS, W_MMD_AB = 1.5, 1.5, 1.5
W_Y_ID, W_ID_HS, W_ID_AB = 0.4, 0.4, 0.4


MMD_KERNEL_BW_Y = [0.01, 0.02, 0.05, 0.10]
MMD_KERNEL_BW_HS = [0.02, 0.05, 0.10]
MMD_KERNEL_BW_AB = [2.0, 4.0, 8.0]

LR_BB, LR_Y, LR_HS, LR_AB = 4e-4, 3e-4, 5e-4, 5e-4
WARMUP_STEPS = 100
COSINE_MIN_RATIO = 0.2

# List of synthetic images used for test set evaluation
TEST_SET = [
    "HC145_5_1", "HC145_5_2", "HC145_5_3",
    "HC146_4_1", "HC146_4_2", "HC146_4_3",
    "HC149_1_1", "HC149_1_2", "HC149_1_3",
    "HC155_4_1", "HC155_4_2", "HC155_4_3",
    "HC240_1_1", "HC240_1_2", "HC240_1_3"
]


def set_seed(seed=2024):
    """Freeze random number generators for bit-exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cosine_warmup_lr(step, base_lr, total_steps):
    """Apply linear warmup followed by cosine annealing."""
    if step <= WARMUP_STEPS:
        return base_lr * (0.1 + 0.9 * step / max(1, WARMUP_STEPS))
    t = min(1.0, max(0.0, (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)))
    return base_lr * (COSINE_MIN_RATIO + (1 - COSINE_MIN_RATIO) * 0.5 * (1 + math.cos(math.pi * t)))


def apply_stages(rgb0, g, b, ds, dh, da, db, edge_cols, ref_cols):
    """Cascaded correction: Brightness -> HSV -> Lab."""
    img01 = to_float01(rgb0)
    Y = 0.299 * img01[..., 0] + 0.587 * img01[..., 1] + 0.114 * img01[..., 2]
    Y2 = np.clip(Y * g[None, :] + b[None, :], 0.0, 1.0)
    scale = (Y2 / np.maximum(1e-6, Y))[:, :, None]

    w_hl = torch_smoothstep(torch.tensor(HL1_Y_LOW), torch.tensor(HL1_Y_HIGH), torch.tensor(Y)).numpy()[:, :, None]
    scale = 1.0 + (scale - 1.0) * (1.0 - HL1_STRENGTH * w_hl)
    rgb1 = np.clip(img01 * scale, 0, 1)
    # HSV correction with spatial consistency
    H, W, _ = rgb1.shape
    anchors = (edge_cols | ref_cols)
    dh_anchor = np.zeros(W, np.float32)
    dh_anchor[edge_cols] = EDGE_ALPHA * dh[edge_cols]
    dh_field = harmonic_fill(anchors, dh_anchor, W)

    ds_anchor = np.ones(W, np.float32)
    ds_anchor[edge_cols] = EDGE_ALPHA * (ds[edge_cols] - 1.0) + 1.0
    ds_field = harmonic_fill(anchors, ds_anchor, W)

    h, s, v = rgb_to_hsv01(rgb1)
    h2 = (h + dh_field[None, :]) % 1.0
    s2 = np.clip(s * ds_field[None, :], 0, 1.0)
    rgb2 = to_float01(hsv01_to_rgb(h2, s2, v))

    # Lab correction with spatial consistency
    da_anchor = np.zeros(W, np.float32)
    db_anchor = np.zeros(W, np.float32)
    da_anchor[edge_cols] = EDGE_ALPHA * da[edge_cols]
    db_anchor[edge_cols] = EDGE_ALPHA * db[edge_cols]
    da_field = harmonic_fill(anchors, da_anchor, W)
    db_field = harmonic_fill(anchors, db_anchor, W)

    from utils.feature_extraction import rgb_to_lab, lab_to_rgb
    L, a, b_ch = rgb_to_lab(rgb2)
    rgb3 = lab_to_rgb(L, a + da_field[None, :], b_ch + db_field[None, :])

    return rgb3


def train_single_image(rgb, feats, stats_y, ref_cols, edge_cols, valid, iters=500):
    """Unsupervised co-training via MK-MMD distribution alignment."""
    H, W, _ = rgb.shape
    X = torch.from_numpy(feats.T).unsqueeze(0).to(DEVICE)
    bb = ColumnBackbone(F=feats.shape[1], C=48).to(DEVICE)
    hd = Heads(C=48, out_per_head=2).to(DEVICE)

    opt = torch.optim.AdamW([
        {"params": bb.parameters(), "lr": LR_BB},
        {"params": hd.y_head.parameters(), "lr": LR_Y},
        {"params": hd.hs_head.parameters(), "lr": LR_HS},
        {"params": hd.ab_head.parameters(), "lr": LR_AB},
    ], weight_decay=1e-4)
    for pg in opt.param_groups: pg["base_lr"] = pg["lr"]

    Ytop = torch.from_numpy(stats_y["Y_top"]).to(DEVICE)
    ref_w = torch.from_numpy(ref_cols.astype(np.float32)).to(DEVICE)
    tgt_w = 1.0 - ref_w
    edge_w = torch.from_numpy(edge_cols.astype(np.float32)).to(DEVICE)
    mu_ref = float(stats_y["Y_top"][ref_cols].mean()) * S1_REF_BOOST

    rgb01 = to_float01(rgb)
    validCPU = torch.from_numpy(valid.astype(np.bool_))
    ref_colsCPU = torch.from_numpy(ref_cols.astype(np.bool_))
    edge_colsCPU = torch.from_numpy(edge_cols.astype(np.bool_))
    idx_center = torch.nonzero((validCPU & ref_colsCPU[None, :]).reshape(-1), as_tuple=False).squeeze(1)
    idx_target = torch.nonzero((validCPU & edge_colsCPU[None, :]).reshape(-1), as_tuple=False).squeeze(1)
    RGB_flat = torch.from_numpy(rgb01.reshape(-1, 3)).float()

    def clamp_heads(y, hs, ab):
        g = (torch.tanh(y[0]) + 1) / 2 * 2.7 + 0.8
        b = torch.tanh(y[1]) * 0.2 + 0.05
        ds = (torch.tanh(hs[0]) + 1) / 2 * 2.3 + 0.9
        dh = torch.tanh(hs[1]) * (20.0 / 360.0)
        da = torch.tanh(ab[0]) * 40.0
        db = torch.tanh(ab[1]) * 40.0
        return g, b, ds, dh, da, db

    def lap2(x):
        return (x[:-2] - 2 * x[1:-1] + x[2:]).square().mean()

    def hs_embed(h, s):
        ang = 2 * math.pi * h
        return torch.stack([torch.cos(ang), torch.sin(ang), s], dim=1)

    # Optimization loop
    for t in range(1, iters + 1):
        for pg in opt.param_groups: pg["lr"] = cosine_warmup_lr(t, pg["base_lr"], iters)

        hfeat = bb(X)
        y_o, hs_o, ab_o = hd(hfeat)
        gain, bias, ds, dh, da, db = clamp_heads(y_o[0], hs_o[0], ab_o[0])

        ns, nt = min(UNIFIED_MMD_SAMPLES, idx_center.numel()), min(UNIFIED_MMD_SAMPLES, idx_target.numel())
        sel_c = idx_center[torch.randperm(idx_center.numel())[:ns]]
        sel_t = idx_target[torch.randperm(idx_target.numel())[:nt]]
        jt_t = (sel_t % W).to(DEVICE)

        rgb_c_ref = RGB_flat[sel_c].to(DEVICE)
        rgb_t_src = RGB_flat[sel_t].to(DEVICE)

        g_t = gain[jt_t].unsqueeze(1)
        b_t = bias[jt_t].unsqueeze(1)
        Y_t_raw = 0.299 * rgb_t_src[:, 0:1] + 0.587 * rgb_t_src[:, 1:2] + 0.114 * rgb_t_src[:, 2:3]
        Y_t_target = torch.clamp(Y_t_raw * g_t + b_t, 0.0, 1.0)
        scale_t = Y_t_target / (Y_t_raw + 1e-6)

        w_hl = torch_smoothstep(HL1_Y_LOW, HL1_Y_HIGH, Y_t_raw)
        scale_t = 1.0 + (scale_t - 1.0) * (1.0 - HL1_STRENGTH * w_hl)
        rgb_t_s1 = torch.clamp(rgb_t_src * scale_t, 0.0, 1.0)

        hsv_t = torch_rgb_to_hsv(rgb_t_s1)
        ds_t, dh_t = ds[jt_t], dh[jt_t]
        h_t_new = (hsv_t[:, 0] + dh_t) % 1.0
        s_t_new = torch.clamp(hsv_t[:, 1] * ds_t, 0.0, 1.0)
        v_t_new = hsv_t[:, 2]
        rgb_t_s2 = torch_hsv_to_rgb(torch.stack([h_t_new, s_t_new, v_t_new], dim=1))

        lab_t = torch_rgb_to_lab(rgb_t_s2)
        a_t_final = lab_t[:, 1] + da[jt_t]
        b_t_final = lab_t[:, 2] + db[jt_t]

        Y_pred_col = Ytop * gain + bias
        L_y = (tgt_w * (Y_pred_col - mu_ref).abs()).mean() + 0.7 * (edge_w * (Y_pred_col - mu_ref).abs()).mean()
        L_id = (ref_w * ((gain - 1.0).abs() + 0.5 * bias.abs())).mean()
        L_sm = lap2(gain) + lap2(bias)
        L_flatY = torch.var(Y_pred_col, unbiased=False)
        L_tvY = torch.mean(torch.abs(Y_pred_col[1:] - Y_pred_col[:-1]))

        Y_c_ref = 0.299 * rgb_c_ref[:, 0] + 0.587 * rgb_c_ref[:, 1] + 0.114 * rgb_c_ref[:, 2]
        Y_t_res = 0.299 * rgb_t_s1[:, 0] + 0.587 * rgb_t_s1[:, 1] + 0.114 * rgb_t_s1[:, 2]
        mmdY = mmd_mk_1d_unbiased(Y_c_ref, Y_t_res, MMD_KERNEL_BW_Y)

        hsv_c = torch_rgb_to_hsv(rgb_c_ref)
        Xhs = hs_embed(hsv_c[:, 0], hsv_c[:, 1])
        Yhs = hs_embed(h_t_new, s_t_new)
        mmdHS = mmd_mk_nd_unbiased(Xhs, Yhs, MMD_KERNEL_BW_HS)
        L_idHS = (ref_w * (ds - 1.0).pow(2)).mean() + (ref_w * dh.pow(2)).mean()
        L_hsv_sm = lap2(ds) + lap2(dh)

        lab_c = torch_rgb_to_lab(rgb_c_ref)
        Xab = torch.stack([lab_c[:, 1], lab_c[:, 2]], dim=1)
        Yab = torch.stack([a_t_final, b_t_final], dim=1)
        mmdAB = mmd_mk_nd_unbiased(Xab, Yab, MMD_KERNEL_BW_AB)
        L_idAB = (ref_w * da.pow(2)).mean() + (ref_w * db.pow(2)).mean()

        L_total = (W_Y_ALIGN * L_y + W_Y_ID * L_id + W_Y_SMOOTH * L_sm + W_Y_FLAT * L_flatY + W_Y_TV * L_tvY +
                   W_MMD_Y * mmdY + W_MMD_HS * mmdHS + W_ID_HS * L_idHS + W_HSV_SMOOTH * L_hsv_sm +
                   W_MMD_AB * mmdAB + W_ID_AB * L_idAB)

        opt.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(list(bb.parameters()) + list(hd.parameters()), 1.0)
        opt.step()

    with torch.no_grad():
        hfeat = bb(X)
        y_o, hs_o, ab_o = hd(hfeat)
        gain, bias, ds, dh, da, db = clamp_heads(y_o[0], hs_o[0], ab_o[0])

    return gain.cpu().numpy(), bias.cpu().numpy(), ds.cpu().numpy(), dh.cpu().numpy(), da.cpu().numpy(), db.cpu().numpy()


def process_and_evaluate(args):
    set_seed(2024)
    ensure_dir(args.output_dir)
    images = [f for f in glob.glob(os.path.join(args.input_dir, "*.*")) if
              f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    results = []

    if args.mode == 'synthetic':
        base_metrics = ["PSNR", "SSIM", "CIEDE", "Std_Y", "dE_Side", "NIQE"]
    else:
        base_metrics = ["Std_Y", "dE_Side", "NIQE"]

    # Building columns with Change and Ratio metrics
    columns_order = ["Image Name"]
    for m in base_metrics:
        columns_order.extend([f"{m}_Before", f"{m}_After", f"{m}_Change", f"{m}_Ratio"])
    columns_order.append("Time (s)")

    for img_path in tqdm(images, desc=f"Processing [{args.mode.upper()}] Mode Images"):
        name = os.path.splitext(os.path.basename(img_path))[0]
        rgb0 = imread(img_path)
        if rgb0 is None: continue

        row = {col: np.nan for col in columns_order}
        row["Image Name"] = name

        start_t = time.time()

        valid, Y_top0, ref_cols, edge_cols = build_masks_and_ref(rgb0, args.center_ratio, lum_weight=0.7)
        feats1, stats1 = build_stage1_features(rgb0, valid, Y_top0)

        min_w = int(EDGE_MIN_FRAC * rgb0.shape[1])
        l_span, r_span = _scan_left_right_spans(edge_cols, ref_cols, min_w)
        edge_cols = _spans_to_mask(l_span, r_span, rgb0.shape[1])

        g, b, ds, dh, da, db = train_single_image(rgb0, feats1, stats1, ref_cols, edge_cols, valid, iters=500)

        torch.cuda.empty_cache()
        gc.collect()

        rgb_cascade = apply_stages(rgb0, g, b, ds, dh, da, db, edge_cols, ref_cols)
        rgb_final, _, _ = ab_constrained_detrend(rgb_cascade, valid, ref_cols, rgb_ref=rgb0)
        imwrite(os.path.join(args.output_dir, f"{name}_corrected.png"), rgb_final)

        H, W, C = rgb0.shape
        separator = np.full((H, 15, C), 255, dtype=np.uint8)
        comparison_img = np.concatenate((rgb0, separator, rgb_final), axis=1)
        imwrite(os.path.join(args.output_dir, f"{name}_comparison.png"), comparison_img)

        row["Time (s)"] = float(time.time() - start_t)

        if args.mode == 'synthetic':
            gt_path = os.path.join(args.gt_dir, name + ".jpg")
            if not os.path.exists(gt_path): gt_path = os.path.join(args.gt_dir, name + ".png")
            img_gt = imread(gt_path) if os.path.exists(gt_path) else None

            if img_gt is not None:
                row["PSNR_After"], row["SSIM_After"], row["CIEDE_After"] = calc_supervised_metrics(img_gt, rgb_final)
                row["PSNR_Before"], row["SSIM_Before"], row["CIEDE_Before"] = calc_supervised_metrics(img_gt, rgb0)

        row["Std_Y_Before"], row["Std_Y_After"] = calc_std_y(rgb0), calc_std_y(rgb_final)
        row["dE_Side_Before"], row["dE_Side_After"] = calc_side_color_diff(rgb0), calc_side_color_diff(rgb_final)
        row["NIQE_Before"], row["NIQE_After"] = evaluate_niqe(rgb0, DEVICE), evaluate_niqe(rgb_final, DEVICE)

        for m in base_metrics:
            b_val = row.get(f"{m}_Before", np.nan)
            a_val = row.get(f"{m}_After", np.nan)
            if pd.notna(b_val) and pd.notna(a_val):
                row[f"{m}_Change"] = a_val - b_val
                row[f"{m}_Ratio"] = (a_val - b_val) / b_val if b_val != 0 else np.nan

        results.append(row)

    if results:
        df = pd.DataFrame(results)
        df = df.reindex(columns=columns_order)

        def compute_avg_row(df_sub, row_title):
            avg_dict = {col: np.nan for col in columns_order}
            avg_dict["Image Name"] = row_title
            if df_sub.empty: return avg_dict

            for m in base_metrics:
                mean_b = df_sub[f"{m}_Before"].mean()
                mean_a = df_sub[f"{m}_After"].mean()
                avg_dict[f"{m}_Before"] = mean_b
                avg_dict[f"{m}_After"] = mean_a

                change = mean_a - mean_b
                avg_dict[f"{m}_Change"] = change

                avg_dict[f"{m}_Ratio"] = change / mean_b if mean_b != 0 else np.nan

            avg_dict["Time (s)"] = df_sub["Time (s)"].mean()
            return avg_dict

        all_avg_row = compute_avg_row(df, "ALL_AVERAGE")

        df_test = df[df["Image Name"].isin(TEST_SET)]
        test_avg_row = compute_avg_row(df_test, "TEST_AVERAGE")

        df = pd.concat([df, pd.DataFrame([all_avg_row, test_avg_row])], ignore_index=True)

        report_path = os.path.join(args.output_dir, f"Evaluation_Report_{args.mode}.xlsx")
        df.to_excel(report_path, index=False)
        print(f"\nProcessing completed. Metrics logged to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-CDA Training and Evaluation Pipeline")
    parser.add_argument("--mode", type=str, choices=['synthetic', 'real'], default='synthetic', help="Execution mode")
    parser.add_argument("--input_dir", type=str, default="./data/synthetic", help="Raw input images directory")
    parser.add_argument("--gt_dir", type=str, default="./data/synthetic/label",
                        help="Ground truth directory (for synthetic mode)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory for corrected images and reports")
    parser.add_argument("--center_ratio", type=float, default=0.10, help="Distortion-free center width ratio")

    args = parser.parse_args()
    process_and_evaluate(args)