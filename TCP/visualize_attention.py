
# 全局开关
VISUALIZE_ATTENTION = False

def overlay_attention_on_img(img, attn_map):
    import numpy as np
    import torch

    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().permute(1,2,0).numpy()

    # 归一化
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # resize到img大小（假设用cv2或skimage）
    from cv2 import resize, COLORMAP_JET, applyColorMap, addWeighted
    attn_resized = resize(attn_map, (img.shape[1], img.shape[0]))
    heatmap = applyColorMap((attn_resized*255).astype(np.uint8), COLORMAP_JET)

    overlay = addWeighted(img.astype(np.uint8), 0.6, heatmap, 0.4, 0)
    return overlay

