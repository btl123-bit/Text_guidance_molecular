
import torch as th
import torch.nn as nn
from torch.nn import SiLU
#from timm import trunc_normal_
import einops
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import time
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        #print(inner_dim)
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        w=context   #atte1:None         atte2:context
        h = self.heads      #8

        q = self.to_q(x)    # 1，256 ，128
        if context==None:
            text = x
        else:
            text = context["states"]       # 1,216,768
        k = self.to_k(text)  # atte1:1，256，128   atte2:1,216,128
        v = self.to_v(text)  # atte1:1，256，128   atte2:1,216,128

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))     # 8,216,128

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale        # atte1:8，256，256，

        if context!=None:
            mask = context["mask"]

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            mask = mask.to(torch.bool)
            sim.masked_fill_(~mask, max_neg_value)
            # atte2:8, 4096，77   self.scale=8

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)        # 2: 8,256,216

        out = einsum('b i j, b j d -> b i d', attn, v)       #  1： 8，256，16
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)  # 1: 1,256，16
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=768, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    # def forward(self, x, context=None):
    #     return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x            # 1，256，128
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads=8,
                 depth=1, dropout=0., context_dim=768):
        super().__init__()
        self.in_channels = in_channels
        d_head = self.in_channels//8
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=768)
                for d in range(depth)]
        )
        #self.transformer_blocks = BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)

        """self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))"""

        self.proj_out = nn.Conv2d(inner_dim,in_channels, kernel_size=1,stride=1,padding=0)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape    # 1，128，64，64
        x_in = x            #提前保存了初始的X  (即来自上一个resblock的x）
        x = self.norm(x)
        x = self.proj_in(x)      # 1，128，16，16
                                # nn.Conv2d
        x = rearrange(x, 'b c h w -> b (h w) c')        #1，256 ，128

        for block in self.transformer_blocks:
            x = block(x, context=context)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  #1，128，64，64
        x = self.proj_out(x)        #1，128，64，64  值全部是0
        return x + x_in             #相当于还是输出的 X



class ResBlock(nn.Module):
    def __init__(self,channels,emb_channels,out_channels=None,dropout=0.0):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),
            SiLU(),
            nn.Conv2d(channels, self.out_channels, kernel_size=3, stride=1, padding=1), #Conv2d
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
            )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.out_channels),   ##GroupNorm32(32, channels)
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1,padding=1)),
            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels,self.out_channels, kernel_size=1)

    def forward(self, x, emb):

        h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):        # len(emb_out.shape)=2    len(h.shape)=4
            emb_out = emb_out[..., None]                # 补None

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = th.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)
        x = self.skip_connection(x)

        return x + h
def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class UNetModel(nn.Module):

    def __init__(
            self,
            dropout=0.0,
            mlp_ratio=4.
    ):
        super().__init__()
        self.dropout = dropout,
        self.mlp_ratio=mlp_ratio,

        time_embed_dim = 128     # 512

        self.time_embed = nn.Sequential(
            nn.Linear(128, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        #encoder
        self.conv2d_1 = nn.Conv2d(3,128 , kernel_size=3, stride=1, padding=1)

        self.resblock1_1 = ResBlock(128, time_embed_dim, out_channels=128)
        self.SpatialTransformer1_1 = SpatialTransformer(in_channels=128)
        self.resblock1_2 = ResBlock(128, time_embed_dim, out_channels=128)
        self.SpatialTransformer1_2 = SpatialTransformer(in_channels=128)

        self.conv1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True) #下采样

        self.resblock2_1 = ResBlock(128, time_embed_dim, out_channels=256)
        self.SpatialTransformer2_1 = SpatialTransformer(in_channels=256)
        self.resblock2_2 = ResBlock(256, time_embed_dim, out_channels=256)
        self.SpatialTransformer2_2 = SpatialTransformer(in_channels=256)

        self.conv2 = nn.Conv2d(256, out_channels= 256, kernel_size=3, stride=2, padding=1, bias=True)

        self.resblock3_1 = ResBlock(256, time_embed_dim, out_channels=384)
        self.SpatialTransformer3_1 = SpatialTransformer(in_channels=384)
        self.resblock3_2 = ResBlock(384, time_embed_dim, out_channels=384)
        self.SpatialTransformer3_2 = SpatialTransformer(in_channels=384)

        self.conv3 = nn.Conv2d(384, out_channels=384, kernel_size=3, stride=2, padding=1, bias=True)

        self.resblock4_1 = ResBlock(384, time_embed_dim, out_channels=512)
        self.resblock4_2 = ResBlock(512, time_embed_dim, out_channels=512)


        #decoder
        self.resblock5_1 = ResBlock(512, time_embed_dim, out_channels=512)  # ch*4+ch*3
        self.SpatialTransformer4_1 = SpatialTransformer(in_channels=512)
        self.resblock5_2 = ResBlock(512 + 512, time_embed_dim, out_channels=512)
        self.resblock5_3 = ResBlock(512 + 384, time_embed_dim, out_channels=512)

        self.upsample1 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.Conv2d(512, out_channels=512, kernel_size=3, stride=1, padding=1))

        self.resblock6_1 = ResBlock(512 + 384, time_embed_dim, out_channels=384)
        self.SpatialTransformer5_1 = SpatialTransformer(in_channels=384)
        self.resblock6_2 = ResBlock(384 + 384, time_embed_dim, out_channels=384)
        self.SpatialTransformer5_2 = SpatialTransformer(in_channels=384)
        self.resblock6_3 = ResBlock(384 + 256, time_embed_dim, out_channels=384)
        self.SpatialTransformer5_3 = SpatialTransformer(in_channels=384)
        self.upsample2 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.Conv2d(384, out_channels=384, kernel_size=3, stride=1, padding=1))


        self.resblock7_1 = ResBlock(384 + 256, time_embed_dim, out_channels=256)
        self.SpatialTransformer6_1 = SpatialTransformer(in_channels=256)
        self.resblock7_2 = ResBlock(256 + 256, time_embed_dim, out_channels=256)
        self.SpatialTransformer6_2 = SpatialTransformer(in_channels=256)
        self.resblock7_3 = ResBlock(256 + 128, time_embed_dim, out_channels=256)
        self.SpatialTransformer6_3 = SpatialTransformer(in_channels=256)
        self.upsample3 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1,padding=1))

        self.resblock8_1 = ResBlock(256 + 128, time_embed_dim, out_channels=128)
        self.SpatialTransformer7_1 = SpatialTransformer(in_channels=128)
        self.resblock8_2 = ResBlock(128 + 128, time_embed_dim, out_channels=128)
        self.SpatialTransformer7_2 = SpatialTransformer(in_channels=128)
        self.resblock8_3 = ResBlock(128 + 128, time_embed_dim, out_channels=128)
        self.SpatialTransformer7_3 = SpatialTransformer(in_channels=128)

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=128),
            SiLU(),
            zero_module(nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1, bias=True)),
        )

    def forward(self, x, timesteps, context):

        #print(context)
        hs = []
        x = x.type(th.float32)     #(1,3,16,16)
        temb = self.time_embed(timestep_embedding(timesteps, 128))

        #encoder
        x = self.conv2d_1(x)      #(1,128,16,16)
        hs.append(x)

        x = self.resblock1_1(x, temb)    #(1,128,16,16)
        x = self.SpatialTransformer1_1(x , context)
        hs.append(x)
        x = self.resblock1_2(x, temb)     #(1,128,16,16)
        x = self.SpatialTransformer1_2(x, context)
        hs.append(x)
        x = self.conv1(x)                #(1,128,8,8)
        hs.append(x)

        x = self.resblock2_1(x, temb)    #(1,256,8,8)
        x = self.SpatialTransformer2_1(x, context)
        hs.append(x)
        x = self.resblock2_2(x, temb)     #(1,256,8,8)
        x = self.SpatialTransformer2_2(x, context)
        hs.append(x)
        x = self.conv2(x)                 #(1,256,4,4)
        hs.append(x)

        x = self.resblock3_1(x, temb)     #(1,384,4,4)
        x = self.SpatialTransformer3_1(x, context)
        hs.append(x)
        x = self.resblock3_2(x, temb)     #(1,384,4,4)
        x = self.SpatialTransformer3_2(x, context)
        hs.append(x)
        x = self.conv3(x)                 #(1,384,2,2)
        hs.append(x)

        x = self.resblock4_1(x, temb)     #(1,512,2,2)
        #x = self.SpatialTransformer4_1(x, context)
        hs.append(x)
        x = self.resblock4_2(x, temb)     #(1,512,2,2)
        hs.append(x)


        #decoder
        x = self.resblock5_1(hs.pop(), temb)                               #(1,512,2,2)
        x = self.SpatialTransformer4_1(x, context)
        x = self.resblock5_2(torch.cat([x, hs.pop()], dim=1), temb)        #(1,512,2,2)

        x = self.resblock5_3(torch.cat([x, hs.pop()], dim=1), temb)        #(1,512,2,2)

        x = self.upsample1(x)                                               #(1,512,4,4)

        x = self.resblock6_1(torch.cat([x, hs.pop()], dim=1), temb)        #(1,384,4,4)
        x = self.SpatialTransformer5_1(x, context)
        x = self.resblock6_2(torch.cat([x, hs.pop()], dim=1), temb)         #(1,384,4,4)
        x = self.SpatialTransformer5_2(x, context)
        x = self.resblock6_3(torch.cat([x, hs.pop()], dim=1), temb)         #(1,384,4,4)
        x = self.SpatialTransformer5_3(x, context)
        x = self.upsample2(x)                                               #(1,384,8,8)

        x = self.resblock7_1(torch.cat([x,hs.pop()], dim=1), temb)      #(1,256,8,8)
        x = self.SpatialTransformer6_1(x, context)
        x = self.resblock7_2(torch.cat([x, hs.pop()], dim=1), temb)     #(1,256,8,8)
        x = self.SpatialTransformer6_2(x, context)
        x = self.resblock7_3(torch.cat([x, hs.pop()], dim=1), temb)      #(1,256,8,8)
        x = self.SpatialTransformer6_3(x, context)
        x = self.upsample3(x)                                           #(1,256,16,16)

        x = self.resblock8_1(torch.cat([x,hs.pop()], dim=1), temb)      #(1,128,16,16)
        x = self.SpatialTransformer7_1(x, context)
        x = self.resblock8_2(torch.cat([x, hs.pop()], dim=1), temb)     #(1,128,16,16)
        x = self.SpatialTransformer7_2(x, context)
        x = self.resblock8_3(torch.cat([x, hs.pop()], dim=1), temb)      #(1,128,16,16)
        x = self.SpatialTransformer7_3(x, context)
        x = self.out(x)                                                 #（1，3，16，16）

        x = x.type(x.dtype)
        return x