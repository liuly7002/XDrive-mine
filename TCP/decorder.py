import torch
import torch.nn as nn
import torch.nn.functional as F


# 轻量: 深度可分离卷积
class DWSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)  # 深度卷积
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)               # 逐点卷积
        self.bn = nn.BatchNorm2d(out_ch)   # 对每个通道做归一化(通道之间互不干扰)
        self.act = nn.ReLU(inplace=False)  # 对每个值做ReLu(值之间互不干扰)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)
# 轻量: 轴向注意
class AxialAttention1D(nn.Module):
    def __init__(self, channels, axis='row', reduction=4):
        super().__init__()
        self.axis = axis
        hid = max(8, channels // reduction)
        self.q = nn.Conv1d(channels, hid, 1, bias=False)
        self.k = nn.Conv1d(channels, hid, 1, bias=False)
        self.v = nn.Conv1d(channels, channels, 1, bias=False)

    def forward(self, x):  # x: [B,C,H,W]
        if self.axis == 'row':
            B, C, H, W = x.shape
            xv = x.permute(0, 3, 1, 2).contiguous().view(B*W, C, H)  # W 个序列，每个长度 H  xv.shape = [2*29, 128, 8] = [58, 128, 8]
        else:
            B, C, H, W = x.shape
            xv = x.permute(0, 2, 1, 3).contiguous().view(B*H, C, W)  # H 个序列，每个长度 W

        q = self.q(xv)                 # [B*, hid, L]  if self.axis == 'row': L = H else: L = W
        k = self.k(xv)                 # [B*, hid, L]
        v = self.v(xv)                 # [B*, C,   L]
        attn = torch.softmax(torch.bmm(q.transpose(1,2), k) / (q.shape[1]**0.5), dim=-1)  # [B*, L, L]  注意力矩阵
        out = torch.bmm(v, attn.transpose(1,2))  # [B*, C, L]  v: [58,128,8], attn.T: [58,8,8] => out: [58,128,8]

        if self.axis == 'row':
            out = out.view(B, W, C, H).permute(0, 2, 3, 1).contiguous()  # [B,C,H,W]
        else:
            out = out.view(B, H, C, W).permute(0, 2, 1, 3).contiguous()

        # === 生成可视化用的 2D map (注意,这只是为了验证网络训练结果的,对网络本身训练没帮助) ===
        attn_intensity = attn.mean(dim=1)  # [B*, L] 每个位置的平均注意力强度
        if self.axis == 'row':
            attn_map = attn_intensity.view(B, W, H).permute(0, 2, 1)  # [B,H,W]
        else:
            attn_map = attn_intensity.view(B, H, W)  # [B,H,W]
        # 🔥 归一化到 [0,1]
        attn_map = (attn_map - attn_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / \
                   (attn_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
        self.last_attn_map2d = attn_map.detach()

        return out

# 预测头: P输入 512×8×29 → 输出 1×25×36
class PolarPointHead(nn.Module):
    def __init__(self, in_ch=512, mid_ch=128, out_h=25, out_w=36,
                 use_block3=True, use_block4=True,
                 use_res3=False, use_res4=False,
                 use_axial=True, use_coord=True):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w

        # switches (store flags first)
        self.use_block3 = use_block3
        self.use_block4 = use_block4
        self.use_res3   = use_res3
        self.use_res4   = use_res4
        self.use_axial  = use_axial
        self.use_coord  = use_coord

        # always-created blocks (baseline)
        self.block1 = DWSeparableConv(in_ch, mid_ch, k=3, s=1, p=1)
        self.block2 = DWSeparableConv(mid_ch, mid_ch, k=3, s=1, p=1)

        if self.use_block3:
            self.block3 = DWSeparableConv(mid_ch, mid_ch, k=3, s=1, p=1)
        else:
            self.block3 = None

        if self.use_block4:
            self.block4 = DWSeparableConv(mid_ch, mid_ch, k=3, s=1, p=1)
        else:
            self.block4 = None

        # axial attention only created if requested
        if self.use_axial:
            self.ax_row = AxialAttention1D(mid_ch, axis='row', reduction=4)
            self.ax_col = AxialAttention1D(mid_ch, axis='col', reduction=4)
        else:
            self.ax_row = None
            self.ax_col = None

        self.attn_fuse_conv = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1, bias=True)

        # fuse and head: fuse requires coord channels; if you disabled coord, you
        # can still keep fuse but ensure its expected input channels match.
        if self.use_coord:
            self.fuse = DWSeparableConv(mid_ch + 2, mid_ch, k=3, s=1, p=1)
        else:
            self.fuse = None

        self.head = nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True)

    @torch.no_grad()
    def _coord_grid(self, B, H, W, device):
        # r ∈ [0,1], 0 = 近、1 = 远；theta ∈ [-1,1] 左 -> 右
        r = torch.linspace(1, 0, H, device=device).view(1, H, 1).expand(B, H, W)
        theta = torch.linspace(-1, 1, W, device=device).view(1, 1, W).expand(B, H, W)
        grid = torch.stack([r, theta], dim=1)  # [B,2,H,W]
        return grid

    def forward(self, cnn_feature):
        x = self.block1(cnn_feature)
        x = self.block2(x)

        # # block3
        # if self.block3 is not None:
        #     if self.use_res3:
        #         x = self.block3(x) + x
        #     else:
        #         x = self.block3(x)
        #
        # # block4
        # if self.block4 is not None:
        #     if self.use_res4:
        #         x_res = x
        #         x = self.block4(x) + x_res
        #     else:
        #         x = self.block4(x)

        # attention
        if self.ax_row is not None and self.ax_col is not None:
            out_row = self.ax_row(x)
            out_col = self.ax_col(x)
            attn_row_map = getattr(self.ax_row, 'last_attn_map2d', None)
            attn_col_map = getattr(self.ax_col, 'last_attn_map2d', None)
        else:
            out_row = out_col = x
            attn_row_map = attn_col_map = None
        self.last_attn_maps = {'row': attn_row_map, 'col': attn_col_map}
        # x = 0.5 * (out_row + out_col)
        fused_feats = torch.cat([out_row, out_col], dim=1)  # [B, 2C, H, W]
        x = self.attn_fuse_conv(fused_feats)

        x = F.interpolate(x, size=(self.out_h, self.out_w), mode='bilinear', align_corners=False) # 直接线性插值到25行x36列

        B = x.size(0)
        if self.use_coord:
            coord = self._coord_grid(B, self.out_h, self.out_w, x.device)
            x = torch.cat([x, coord], dim=1)
            x = self.fuse(x)
        else:
            # do nothing; ensure x has mid_ch channels (it does)
            pass

        logits = self.head(x)
        return logits
