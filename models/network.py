"""
models/network.py

Retinex 分解网络主模型。

RetinexPointRaw 接收低光图像，通过 TDN（Transformer-based Decomposition Network）
提取多尺度小波特征，输出反射分量 R 和光照标量 L。
内部包含完整的 Transformer Block、DWT-FFN、多头频域注意力等组件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange
from .wavelets import DWT_2D, IDWT_2D
import pywt


# ============================== RetinexPointRaw ==================================

class RetinexPointRaw(nn.Module):
    def __init__(self, dim=24):
        super(RetinexPointRaw, self).__init__()
        self.net1_conv = nn.Sequential(nn.Conv2d(dim * 2**3, dim * 2**4, 3, 2, 1,groups=dim * 2**3, bias=False),
                                        nn.LeakyReLU())
        self.net2_conv = nn.Sequential(nn.Conv2d(dim * 2**4, dim * 2**5, 3, 2, 1,groups=dim * 2**4, bias=False),
                                        nn.BatchNorm2d(dim * 2**5),
                                        nn.LeakyReLU())

        self.net4_pool = nn.AdaptiveAvgPool2d(1)
        self.net5_linear = nn.Linear((dim * 2**5), 1)

        self.down1 = Downsample(dim*2)

        self.TDN_R = TDN(dim=24)

    def forward(self, input_im):
        R, fea_L1, fea_L2, fea_L3 = self.TDN_R(input_im)
        fea_down1 = self.down1(fea_L1)
        fea_L = torch.cat([fea_L3, fea_down1], 1)

        feats1 = self.net1_conv(fea_L)
        feats2 = self.net2_conv(feats1)
        feats3 = feats2
        feats4 = self.net4_pool(feats3)
        b,_,_,_ = feats4.shape
        feats5 = feats4.reshape(b, -1)
        L_dot = self.net5_linear(feats5).reshape(b,1,1,1)
        R = torch.sigmoid(R)
        L_dot = torch.sigmoid(L_dot)
        L = L_dot.expand_as(input_im[:, -1:, ...])
        return R, L


# ============================== LayerNorm ==================================

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


# ======================== DWT Feed-Forward Network =========================

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, wave='haar'):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwt = DWT_2D(wave)
        self.idwt = IDWT_2D(wave)

        self.ll_conv = nn.Conv2d(
            hidden_features, hidden_features,
            kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias
        )

        self.act = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h0, w0 = x.shape

        x = self.project_in(x)

        pad_h = h0 % 2
        pad_w = w0 % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x_dwt = self.dwt(x)
        x_ll, x_lh, x_hl, x_hh = torch.chunk(x_dwt, 4, dim=1)

        x_ll = self.ll_conv(x_ll)
        x_ll = self.act(x_ll)

        x_dwt = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

        x = self.idwt(x_dwt)
        x = x[:, :, :h0, :w0]

        x = self.project_out(x)
        return x


# ========================= DWT-FSA Attention ==============================

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=False, wave='haar'):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads

        self.qkv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.qkv2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.qkv3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)

        self.dwt = DWT_2D(wave)
        self.idwt = IDWT_2D(wave)

        self.temperature_freq = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.out_proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h0, w0 = x.shape

        q = self.qkv1(x)
        k = self.qkv2(x)
        v = self.qkv3(x)

        pad_h = h0 % 2
        pad_w = w0 % 2
        if pad_h != 0 or pad_w != 0:
            q = F.pad(q, (0, pad_w, 0, pad_h), mode='reflect')
            k = F.pad(k, (0, pad_w, 0, pad_h), mode='reflect')
            v = F.pad(v, (0, pad_w, 0, pad_h), mode='reflect')

        q_ll, q_lh, q_hl, q_hh = torch.chunk(self.dwt(q), 4, dim=1)
        k_ll, k_lh, k_hl, k_hh = torch.chunk(self.dwt(k), 4, dim=1)
        v_ll, v_lh, v_hl, v_hh = torch.chunk(self.dwt(v), 4, dim=1)

        _, _, h1, w1 = q_ll.shape

        # LL 子带多头频域注意力
        q_f = rearrange(q_ll, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_ll, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_ll_r = rearrange(v_ll, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f = F.normalize(q_f, dim=-1)
        k_f = F.normalize(k_f, dim=-1)

        freq_attn = (q_f @ k_f.transpose(-2, -1)) * self.temperature_freq
        freq_attn = freq_attn.softmax(dim=-1)

        out_ll = freq_attn @ v_ll_r
        out_ll = rearrange(out_ll, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h1, w=w1)

        # 高频三支路门控
        gate_lh = torch.sigmoid(q_lh * k_lh)
        gate_hl = torch.sigmoid(q_hl * k_hl)
        gate_hh = torch.sigmoid(q_hh * k_hh)

        out_lh = v_lh * gate_lh
        out_hl = v_hl * gate_hl
        out_hh = v_hh * gate_hh

        freq_out = torch.cat([out_ll, out_lh, out_hl, out_hh], dim=1)

        freq_out = self.idwt(freq_out)
        freq_out = freq_out[:, :, :h0, :w0]

        out = self.out_proj(freq_out)
        return out


# ======================== Transformer Block ================================

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ======================== Building Blocks ==================================

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Downsample4(nn.Module):
    def __init__(self, n_feat):
        super(Downsample4, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(4))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# ============================ TDN Backbone =================================

class TDN(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[1, 2, 2, 2],
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

        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1, inp_enc_level2, out_enc_level2, out_enc_level3


# ========================== DWTTransformer ===============================

class DWTTransformer(nn.Module):
    """DWT-based Transformer for L branch refinement."""
    
    def __init__(self, dim, num_heads=4, num_blocks=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(DWTTransformer, self).__init__()
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        return self.blocks(x)


# ========================== RetinexPixelClassic ===========================

class RetinexPixelClassic(nn.Module):
    """RetinexPixelClassic — 逐像素光照版本。

    R 分支与 RetinexPointRaw 完全相同（TDN Transformer U-Net）。
    L 分支替换为轻量 CNN，输出 [B, 1, H, W] 逐像素光照图，
    与 Diff-TDN 原版一致。
    """

    def __init__(self, dim=24, l_channel=32):
        super(RetinexPixelClassic, self).__init__()
        self.TDN_R = TDN(dim=dim)
        # 逐像素光照 CNN — 与 Diff-TDN 原版一致
        self.conv0 = nn.Conv2d(3, l_channel, 3, padding=1, padding_mode='replicate')
        self.convs = nn.Sequential(
            nn.Conv2d(l_channel, l_channel, 5, padding=2, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(l_channel, l_channel, 3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(l_channel, l_channel, 3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
        )
        self.recon = nn.Conv2d(l_channel, 1, 1)

    def forward(self, input_im):
        R, *_ = self.TDN_R(input_im)
        R = torch.sigmoid(R)
        feats = self.conv0(input_im)
        feats = self.convs(feats)
        L = torch.sigmoid(self.recon(feats))   # [B, 1, H, W]
        return R, L


# ========================== RetinexPixelTrans =============================

class RetinexPixelTrans(nn.Module):
    """RetinexPixelTrans — 基于 Transformer 的逐像素光照版本。

    R 分支与 RetinexPointRaw 完全相同（TDN Transformer U-Net）。
    L 分支从 cat(fea_L3, fea_down1) 开始，经过降维、两次上采样、
    DWTTransformer 处理，输出逐像素光照图。
    DWTTransformer 使用 1 个 block，与 encoder_level1 对称。
    """

    def __init__(self, dim=24, l_heads=1, l_ffn_expansion=2.66, bias=False, LayerNorm_type='WithBias'):
        super(RetinexPixelTrans, self).__init__()
        self.TDN_R = TDN(dim=dim)
        self.down1 = Downsample(dim * 2)

        # L 分支：从 8C 降到 4C
        self.l_reduce = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        
        # 两次上采样：4C -> 2C -> C
        self.l_up1 = Upsample(dim * 4)  # 4C -> 2C, H/4 -> H/2
        self.l_up2 = Upsample(dim * 2)  # 2C -> C, H/2 -> H
        
        # DWTTransformer 处理全分辨率特征 (1 block, 对称 encoder_level1)
        self.l_transformer = DWTTransformer(
            dim=dim,
            num_heads=l_heads,
            num_blocks=1,
            ffn_expansion_factor=l_ffn_expansion,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )
        
        # 最终输出：C -> 1
        self.l_output = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input_im):
        R, fea_L1, fea_L2, fea_L3 = self.TDN_R(input_im)
        
        # L 分支
        fea_down1 = self.down1(fea_L1)
        fea_L = torch.cat([fea_L3, fea_down1], 1)  # B, 8C, H/4, W/4
        
        l_feats = self.l_reduce(fea_L)        # B, 4C, H/4, W/4
        l_feats = self.l_up1(l_feats)          # B, 2C, H/2, W/2
        l_feats = self.l_up2(l_feats)          # B, C, H, W
        l_feats = self.l_transformer(l_feats)  # B, C, H, W
        L = torch.sigmoid(self.l_output(l_feats))  # B, 1, H, W
        
        R = torch.sigmoid(R)
        return R, L

# ========================== RetinexPixelTransMinus ========================
class RetinexPixelTransMinus(nn.Module):
    """RetinexPixelTransMinus — L 分支使用 fea_down1 - fea_L3 做差版本。

    R 分支与 RetinexPixelTrans 完全相同（TDN Transformer U-Net）。
    L 分支从 fea_down1 - fea_L3 开始（逐像素做差，dim*4 通道），
    经过两次上采样、DWTTransformer 处理，输出逐像素光照图。
    去掉了 RetinexPixelTrans 中的 1x1 通道降维卷积。
    """

    def __init__(self, dim=24, l_heads=1, l_ffn_expansion=2.66, bias=False, LayerNorm_type='WithBias'):
        super(RetinexPixelTransMinus, self).__init__()
        self.TDN_R = TDN(dim=dim)
        self.down1 = Downsample(dim * 2)

        # 两次上采样：4C -> 2C -> C
        self.l_up1 = Upsample(dim * 4)  # 4C -> 2C, H/4 -> H/2
        self.l_up2 = Upsample(dim * 2)  # 2C -> C, H/2 -> H

        # DWTTransformer 处理全分辨率特征 (1 block, 对称 encoder_level1)
        self.l_transformer = DWTTransformer(
            dim=dim,
            num_heads=l_heads,
            num_blocks=1,
            ffn_expansion_factor=l_ffn_expansion,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )

        # 最终输出：C -> 1
        self.l_output = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input_im):
        R, fea_L1, fea_L2, fea_L3 = self.TDN_R(input_im)

        # L 分支：做差
        fea_down1 = self.down1(fea_L1)
        l_feats = fea_down1 - fea_L3  # B, 4C, H/4, W/4

        l_feats = self.l_up1(l_feats)          # B, 2C, H/2, W/2
        l_feats = self.l_up2(l_feats)          # B, C, H, W
        l_feats = self.l_transformer(l_feats)  # B, C, H, W
        L = torch.sigmoid(self.l_output(l_feats))  # B, 1, H, W

        R = torch.sigmoid(R)
        return R, L




