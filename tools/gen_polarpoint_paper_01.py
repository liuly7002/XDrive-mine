import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) 你需要修改的路径
# =========================
JSON_PATH = "/home/liulei/ll/PPBEV_ll/0145.json"
OUT_PATH  = "/home/liulei/ll/PPBEV_ll/overlay_from_json_0145.png"

# =========================
# 2) 极点结构参数（必须和你生成json时一致）
# =========================
ROWS, COLS = 25, 36
EGO_POS = (96, 152)

# 下面这些要与你 generate_multi_fov_poles_angle_focus 的默认参数保持一致
FOVS = [100, 70, 50]
LAYERS_PER_FOV = [18, 5, 2]
R_START = 10
R_TOTAL = 140
BASE_PTS = 36
OFFSET = 10
ANGLE_FOCUS_RATIO = 1.0
ANGLE_FOCUS_FACTOR = 2.0
FOCUS_MODE = "absolute"
LAYER_SPACING_FACTOR = 3.0

# =========================
# 3) 你的极点生成函数（保持一致）
# =========================
def generate_multi_fov_poles_angle_focus(center,
                                         fovs=FOVS,
                                         layers_per_fov=LAYERS_PER_FOV,
                                         r_start=R_START,
                                         r_total=R_TOTAL,
                                         base_pts=BASE_PTS,
                                         offset=OFFSET,
                                         angle_focus_ratio=ANGLE_FOCUS_RATIO,
                                         angle_focus_factor=ANGLE_FOCUS_FACTOR,
                                         focus_mode=FOCUS_MODE,
                                         layer_spacing_factor=LAYER_SPACING_FACTOR):
    pole_list = []

    total_layers = sum(layers_per_fov)

    r_ratios = np.linspace(1, layer_spacing_factor, total_layers)
    r_ratios = r_ratios / r_ratios.sum()
    dr_layers = (r_total - r_start) * r_ratios
    radii_global = r_start + np.cumsum(dr_layers)

    if focus_mode == "absolute":
        min_fov = min(fovs)
        focus_angle_abs = (min_fov * angle_focus_ratio) / 2
    else:
        focus_angle_abs = None

    layer_idx = 0
    for fov, num_layers in zip(fovs, layers_per_fov):
        points_per_layer = base_pts

        if focus_mode == "relative":
            u = np.linspace(-1, 1, points_per_layer)
            u = np.sign(u) * (np.abs(u) ** angle_focus_factor)
            angles = np.deg2rad(u * (fov / 2.0))
        elif focus_mode == "absolute":
            focus_angle = min(focus_angle_abs, fov / 2)
            u = np.linspace(-1, 1, points_per_layer)
            u = np.sign(u) * (np.abs(u) ** angle_focus_factor)
            angles = np.deg2rad(u * focus_angle)

            max_side_angle = fov / 2
            side_factor = max_side_angle / focus_angle
            angles = np.sign(angles) * (np.abs(angles) * side_factor)
        else:
            raise ValueError("focus_mode must be 'relative' or 'absolute'")

        radii = radii_global[layer_idx: layer_idx + num_layers]
        for r in radii:
            for a in angles:
                dx = r * np.cos(a + np.pi/2)
                dy = r * np.sin(a + np.pi/2)
                pole_list.append([center[0] + dx, center[1] - dy - offset])

        layer_idx += num_layers

    return np.array(pole_list, dtype=np.float32)

# =========================
# 4) 读取 JSON -> 恢复到与 poles 顺序一致的一维 risks
#    你的保存逻辑是：
#      risks_2d = risks.reshape(rows, cols)  (近->远)
#      risks_2d = flipud(risks_2d)           (远->近)  然后写入json
#    所以现在读出来要 flipud 回来，才能与 poles 的生成顺序一致
# =========================
def load_risks_from_json(json_path, rows=ROWS, cols=COLS):
    with open(json_path, "r") as f:
        risks_2d_far_to_near = np.array(json.load(f), dtype=np.float32)

    assert risks_2d_far_to_near.shape == (rows, cols), \
        f"JSON shape {risks_2d_far_to_near.shape} != ({rows},{cols})"

    # flip 回去：恢复成 近->远 的行顺序，才能对齐 poles 的 append 顺序
    risks_2d_near_to_far = np.flipud(risks_2d_far_to_near)

    # 展平回一维，与 poles 一一对应
    risks_1d = risks_2d_near_to_far.reshape(-1)
    return risks_1d

# =========================
# 5) 绘制（这里给你两种：透明overlay / 或者纯白底）
# =========================
# def plot_poles_overlay(poles, risks, out_path, canvas_size=(192, 192), transparent=True,
#                        point_size=10, cmap_name="RdYlGn_r", vmin=0.0, vmax=1.0):
#     W, H = canvas_size
#
#     # fig = plt.figure(figsize=(W/100.0, H/100.0), dpi=100)
#     UPSCALE = 10  # 10x 超采样（你可以用 8~20）
#     fig = plt.figure(figsize=(W * UPSCALE / 100.0,
#                               H * UPSCALE / 100.0),
#                      dpi=100)
#
#     ax = fig.add_axes([0, 0, 1, 1])
#
#     if transparent:
#         fig.patch.set_alpha(0.0)
#         ax.set_facecolor((0, 0, 0, 0))
#     else:
#         ax.set_facecolor((1, 1, 1, 1))
#
#     ax.set_xlim(0, W)
#     ax.set_ylim(H, 0)
#     ax.axis("off")
#
#     ax.scatter(
#         poles[:, 0] * UPSCALE,
#         poles[:, 1] * UPSCALE,
#         c=risks,
#         cmap=plt.get_cmap(cmap_name),
#         s=point_size * UPSCALE,
#         vmin=vmin, vmax=vmax,
#         linewidths=0
#     )
#
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     fig.savefig(out_path, transparent=transparent, dpi=300, bbox_inches=None, pad_inches=0)
#     plt.close(fig)

def plot_poles_overlay(
    poles, risks, out_path,
    canvas_size=(192, 192),
    transparent=True,
    point_size=10,
    cmap_name="RdYlGn_r",
    vmin=0.0, vmax=1.0,
    out_px=4096  # 最终输出边长像素（PPT足够清晰：2048/3072/4096）
):
    W, H = canvas_size

    # 直接指定最终像素大小：out_px x out_px
    dpi = 200
    fig_w_in = out_px / dpi
    fig_h_in = out_px / dpi

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    if transparent:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((0, 0, 0, 0))
    else:
        ax.set_facecolor((1, 1, 1, 1))

    # 坐标系仍然是 192x192（与你的 poles 一致）
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    ax.scatter(
        poles[:, 0],
        poles[:, 1],
        c=risks,
        cmap=plt.get_cmap(cmap_name),
        s=point_size,          # 点大小不需要乘UPSCALE了
        vmin=vmin, vmax=vmax,
        linewidths=0
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, transparent=transparent, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main():
    # 1) 读 risks
    risks = load_risks_from_json(JSON_PATH)

    # 2) 生成 poles（确保数量匹配）
    poles = generate_multi_fov_poles_angle_focus(center=np.array(EGO_POS), base_pts=COLS)

    if len(poles) != len(risks):
        raise ValueError(f"poles num {len(poles)} != risks num {len(risks)}. "
                         f"请检查 ROWS/COLS 与 layers_per_fov/base_pts 是否一致。")

    # 3) 输出透明叠加层（用于PPT叠加）
    # plot_poles_overlay(
    #     poles, risks, OUT_PATH,
    #     canvas_size=(1920, 1920),
    #     transparent=True,
    #     point_size=2
    # )

    plot_poles_overlay(
        poles, risks, OUT_PATH,
        canvas_size=(192, 192),
        transparent=True,
        point_size=300,  # PPT里更醒目（你可以试 8~20）
        out_px=4096  # 2048/3072/4096 都可
    )

    print(f"[OK] Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
