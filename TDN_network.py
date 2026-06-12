import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange
from torch_wavelets import DWT_2D, IDWT_2D
from torch.autograd import Function
import pywt

class DecomNet(nn.Module):
    def __init__(self, dim=24):
        super(DecomNet, self).__init__()
        self.net1_conv = nn.Sequential(nn.Conv2d(dim * 2**3, dim * 2**4, 3, 2, 1,groups=dim * 2**3, bias=False),
                                        nn.LeakyReLU())
        self.net2_conv = nn.Sequential(nn.Conv2d(dim * 2**4, dim * 2**5, 3, 2, 1,groups=dim * 2**4, bias=False),
                                        nn.BatchNorm2d(dim * 2**5),
                                        nn.LeakyReLU())

        self.net4_pool = nn.AdaptiveAvgPool2d(1)
        self.net5_linear = nn.Linear((dim * 2**5), 1)

        self.down1 = Downsample(dim*2)

        self.TDN_R = TDN(dim=24)
        # self.TDN_L = TDN_L(dim=24)

    def forward(self, input_im):
        R, fea_L1, fea_L2, fea_L3 = self.TDN_R(input_im)
        # R = input_im
        # fea_L1 = torch.randn([1,48, 128, 128]).to("cuda")
        # fea_L3 = torch.randn([1,96, 64, 64]).to("cuda")
        # L_down = F.interpolate(
        #     input_im,
        #     size=(128, 128),
        #     mode='bilinear',
        #     align_corners=False
        # )
        fea_down1 = self.down1(fea_L1)
        # fea_down2 = self.down2(torch.cat([fea_L2, fea_down1], 1))
        fea_L = torch.cat([fea_L3, fea_down1], 1)

        feats1 = self.net1_conv(fea_L)
        feats2 = self.net2_conv(feats1)
        #feats3 = self.net3_conv(feats2)
        feats3 = feats2
        feats4 = self.net4_pool(feats3)
        b,_,_,_ = feats4.shape
        feats5 = feats4.reshape(b, -1)
        L_dot = self.net5_linear(feats5).reshape(b,1,1,1)
        R = torch.sigmoid(R)
        L_dot = torch.sigmoid(L_dot)
        L = L_dot.expand_as(input_im[:, -1:, ...])
        return R, L

##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()
#
#         hidden_features = int(dim * ffn_expansion_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
#
#         self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)
#
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x = self.dwconv(x)
#         x = F.gelu(x)
#         x = self.project_out(x)
#         return x

# DWT Feed-Forward Network (DWT-FFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, wave='haar'):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # 图中左侧 1x1
        # [B, C, H, W] -> [B, hidden_features, H, W]
        self.project_in = nn.Conv2d(
            dim,
            hidden_features,
            kernel_size=1,
            bias=bias
        )

        self.dwt = DWT_2D(wave)
        self.idwt = IDWT_2D(wave)

        # 只处理 LL 子带
        # [B, hidden_features, H/2, W/2] -> [B, hidden_features, H/2, W/2]
        self.ll_conv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias
        )

        self.act = nn.GELU()

        # 图中右侧 1x1
        # [B, hidden_features, H, W] -> [B, C, H, W]
        self.project_out = nn.Conv2d(
            hidden_features,
            dim,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h0, w0 = x.shape

        x = self.project_in(x)  # [B, hidden_features, H, W]

        # -------------------------------------------------
        # 2) DWT 前补齐到偶数尺寸
        # -------------------------------------------------
        pad_h = h0 % 2
        pad_w = w0 % 2

        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            # [B, hidden_features, H', W']

        # -------------------------------------------------
        # 3) DWT 分解
        # 假设 DWT_2D 输出:
        # [B, 4*hidden_features, H'/2, W'/2]
        # -------------------------------------------------
        x_dwt = self.dwt(x)

        x_ll, x_lh, x_hl, x_hh = torch.chunk(x_dwt, 4, dim=1)
        # 每个子带:
        # [B, hidden_features, H'/2, W'/2]

        # -------------------------------------------------
        # 4) 只处理 LL 子带
        # 对应流程图中的 3x3 + GELU
        # -------------------------------------------------
        x_ll = self.ll_conv(x_ll)
        x_ll = self.act(x_ll)
        # [B, hidden_features, H'/2, W'/2]

        # -------------------------------------------------
        # 5) 高频子带直接旁路
        # -------------------------------------------------
        x_dwt = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # [B, 4*hidden_features, H'/2, W'/2]

        # -------------------------------------------------
        # 6) IDWT 重建
        # -------------------------------------------------
        x = self.idwt(x_dwt)
        # [B, hidden_features, H', W']

        # 去掉 padding，恢复原始尺寸
        x = x[:, :, :h0, :w0]
        # [B, hidden_features, H, W]

        # -------------------------------------------------
        # 7) 右侧 1x1
        # -------------------------------------------------
        x = self.project_out(x)
        # [B, C, H, W]

        return x


##########################################################################
## MDLA improved by designing Multi-scale Convolution
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#
#         self.qkv_dwconv_3 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
#         self.qkv_dwconv_5 = nn.Conv2d(dim * 3, dim * 3, kernel_size=5, stride=1, padding=2, groups=dim * 3, bias=bias)
#         self.qkv_dwconv_7 = nn.Conv2d(dim * 3, dim * 3, kernel_size=7, stride=1, padding=3, groups=dim * 3, bias=bias)
#
#         self.q_proj = nn.Conv2d(dim * 3, dim, kernel_size=1 ,stride=1, padding=0, bias=bias)
#         self.k_proj = nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1, padding=0, bias=bias)
#         self.v_proj = nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1, padding=0, bias=bias)
#
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         x = self.qkv(x)
#         qkv_1 = self.qkv_dwconv_3(x)
#         q_1, k_1, v_1 = qkv_1.chunk(3, dim=1)
#
#         qkv_2 = self.qkv_dwconv_5(x)
#         q_2, k_2, v_2 = qkv_2.chunk(3, dim=1)
#
#         qkv_3 = self.qkv_dwconv_7(x)
#         q_3, k_3, v_3 = qkv_3.chunk(3, dim=1)
#
#         q = self.q_proj(torch.cat([q_1, q_2, q_3], dim=1))
#         k = self.k_proj(torch.cat([k_1, k_2, k_3], dim=1))
#         v = self.v_proj(torch.cat([v_1, v_2, v_3], dim=1))
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         out = self.project_out(out)
#         return out


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         qkv = self.qkv_dwconv(self.qkv(x))
#         q, k, v = qkv.chunk(3, dim=1)
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         out = self.project_out(out)
#         return out
class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters  # .to(dtype=torch.float16)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        # self.w_ll = self.w_ll.to(dtype=torch.float16)
        # self.w_lh = self.w_lh.to(dtype=torch.float16)
        # self.w_hl = self.w_hl.to(dtype=torch.float16)
        # self.w_hh = self.w_hh.to(dtype=torch.float16)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


# ===================== 2. 多头自注意力（基础模块） =====================
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = (dim // num_heads) ** -0.5
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#     def forward(self, q, k, v):
#         b, c, h, w = q.shape
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         return out
#
#
# # ===================== 3. 核心模块：DWT-FSA 注意力块 =====================
# # 完全对应你提供的 Transformer 结构图
# class DWT_FSA_Block(nn.Module):
#     def __init__(self, dim, num_heads=8, LayerNorm_type='WithBias'):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.norm = LayerNorm(dim, LayerNorm_type)  # 归一化
#
#         # Q/K/V 特征提取：1x1卷积 + 3x3卷积（对应图中结构）
#         self.qkv = nn.Sequential(
#             nn.Conv2d(dim, dim, 1),
#             nn.Conv2d(dim, dim * 3, 3, padding=1)
#         )
#
#         # 小波变换模块
#         self.dwt = DWT_2D(wave='haar')
#         self.idwt = IDWT_2D(wave='haar')
#
#         # 双分支注意力
#         self.attn_spatial = Attention(dim, num_heads)  # 空间自注意力
#         self.attn_freq = Attention(dim, num_heads)  # 频域自注意力
#
#         # 输出卷积 + 残差
#         self.proj_dwt = nn.Conv2d(dim, dim, 1)
#         self.proj = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         residual = x  # 残差连接
#
#         # 1. 归一化 + 生成 Q/K/V
#         x = self.norm(x)
#         qkv = self.qkv(x).chunk(3, dim=1)  # 拆分为 Q, K, V
#         q, k, v = qkv
#
#         # ===================== 分支1：空间自注意力 =====================
#         attn_s = self.attn_spatial(q, k, v)
#
#         # ===================== 分支2：DWT频域自注意力 =====================
#         # 小波分解：[B, C, H, W] → [B, 4C, H/2, W/2]
#         # q_dwt = self.dwt(q)
#         # k_dwt = self.dwt(k)
#         # v_dwt = self.dwt(v)
#         q_ll, q_lh, q_hl, q_hh = torch.chunk(self.dwt(q), chunks=4, dim=1)
#         k_ll, k_lh, k_hl, k_hh = torch.chunk(self.dwt(k), chunks=4, dim=1)
#         v_ll, v_lh, v_hl, v_hh = torch.chunk(self.dwt(v), chunks=4, dim=1)
#         # 频域子带内注意力计算
#         # attn_f = self.attn_freq(q_dwt, k_dwt, v_dwt)
#         attn_ll = self.attn_freq(q_ll, k_ll, v_ll)
#         attn_lh = self.attn_freq(q_lh, k_lh, v_lh)
#         attn_hl = self.attn_freq(q_hl, k_hl, v_hl)
#         attn_hh = self.attn_freq(q_hh, k_hh, v_hh)
#         # 逆小波重构：[B, 4C, H/2, W/2] → [B, C, H, W]
#         attn_f = torch.cat([attn_ll, attn_lh, attn_hl, attn_hh], dim=1)
#         attn_f = self.idwt(attn_f)
#
#         # ===================== 双分支融合 + 输出 =====================
#         # 空间注意力 × 频域注意力（对应图中逐元素相乘）
#         attn_f = self.proj_dwt(attn_f)
#         fuse = attn_s * attn_f
#         out = self.proj(fuse) + residual  # 残差连接
#         return out
#DWT-6

# #DWT-10
class Attention(nn.Module):
    # ⚠️ 加上了 num_heads 参数，记得外部调用时一定要传进来，避免除零错误！
    def __init__(self, dim, num_heads, bias=False, wave='haar'):
        super().__init__()

        # 断言：确保总通道数能被头数整除
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads

        # [B, C, H, W] -> [B, C, H, W]
        self.qkv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.qkv2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.qkv3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)

        self.dwt = DWT_2D(wave)
        self.idwt = IDWT_2D(wave)

        # 频域注意力温度参数改为多头形态 [num_heads, 1, 1]
        self.temperature_freq = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 流程图右侧唯一一个 1×1
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h0, w0 = x.shape

        # 1) Q, K, V
        q = self.qkv1(x)  # [B, C, H, W]
        k = self.qkv2(x)  # [B, C, H, W]
        v = self.qkv3(x)  # [B, C, H, W]

        # 2) padding，保证 H/W 为偶数
        pad_h = h0 % 2
        pad_w = w0 % 2

        if pad_h != 0 or pad_w != 0:
            q = F.pad(q, (0, pad_w, 0, pad_h), mode='reflect')  # [B, C, H', W']
            k = F.pad(k, (0, pad_w, 0, pad_h), mode='reflect')  # [B, C, H', W']
            v = F.pad(v, (0, pad_w, 0, pad_h), mode='reflect')  # [B, C, H', W']

        # 3) DWT
        # DWT 输出: [B, 4C, H'/2, W'/2]

        q_ll, q_lh, q_hl, q_hh = torch.chunk(self.dwt(q), 4, dim=1)
        k_ll, k_lh, k_hl, k_hh = torch.chunk(self.dwt(k), 4, dim=1)
        v_ll, v_lh, v_hl, v_hh = torch.chunk(self.dwt(v), 4, dim=1)

        # 每个子带: [B, C, H'/2, W'/2]
        _, _, h1, w1 = q_ll.shape

        # -------------------------------------------------
        # 4) LL 子带 C×C 频域注意力 (🔥 核心修改：升级为多头)
        # -------------------------------------------------
        # 拆出 head 维度：[B, C, H'/2, W'/2] -> [B, head, C_head, HW/4]
        q_f = rearrange(q_ll, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_ll, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_ll_r = rearrange(v_ll, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f = F.normalize(q_f, dim=-1)  # [B, head, C_head, HW/4]
        k_f = F.normalize(k_f, dim=-1)  # [B, head, C_head, HW/4]

        # 计算多头注意力矩阵：[B, head, C_head, HW/4] @ [B, head, HW/4, C_head] -> [B, head, C_head, C_head]
        freq_attn = (q_f @ k_f.transpose(-2, -1)) * self.temperature_freq
        freq_attn = freq_attn.softmax(dim=-1)  # [B, head, C_head, C_head]

        # 将多头注意力作用于 V：[B, head, C_head, C_head] @ [B, head, C_head, HW/4] -> [B, head, C_head, HW/4]
        out_ll = freq_attn @ v_ll_r

        # 拼回完整的 C 并恢复 2D：[B, head, C_head, HW/4] -> [B, C, H'/2, W'/2]
        out_ll = rearrange(out_ll, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h1, w=w1)

        # -------------------------------------------------
        # 5) 高频三支路逐元素门控 (天然支持多头独立性，保持不变)
        # -------------------------------------------------
        gate_lh = torch.sigmoid(q_lh * k_lh)  # [B, C, H'/2, W'/2]
        gate_hl = torch.sigmoid(q_hl * k_hl)  # [B, C, H'/2, W'/2]
        gate_hh = torch.sigmoid(q_hh * k_hh)  # [B, C, H'/2, W'/2]

        out_lh = v_lh * gate_lh  # [B, C, H'/2, W'/2]
        out_hl = v_hl * gate_hl  # [B, C, H'/2, W'/2]
        out_hh = v_hh * gate_hh  # [B, C, H'/2, W'/2]

        # 6) concat
        freq_out = torch.cat([out_ll, out_lh, out_hl, out_hh], dim=1)  # [B, 4C, H'/2, W'/2]

        # 7) IDWT
        freq_out = self.idwt(freq_out)  # [B, C, H', W']

        # 8) 去掉 padding
        freq_out = freq_out[:, :, :h0, :w0]  # [B, C, H, W]

        # 9) 1×1 输出投影
        out = self.out_proj(freq_out)  # [B, C, H, W]

        return out

##########################################################################

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        #self.attn = Attention(dim,  bias)
        # self.attn = DWT_FSA_Block(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Downsample4(nn.Module):
    def __init__(self, n_feat):
        super(Downsample4, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(4))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
# improved the information Multi-scale conv and lightweights design
class TDN(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[1, 2, 2, 2], #2，3，3，4
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'
                 ):

        super(TDN, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2 ** 1), int(dim), kernel_size=1, bias=bias)
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_dec_level3 = out_enc_level3

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1, inp_enc_level2, out_enc_level2, out_enc_level3


