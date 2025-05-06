import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class STViT(nn.Module):
    """
    Spatial-Temporal ViT (used in ablation study, section 4.2)
    """

    def __init__(
        self,
        img_res,
        patch_size,
        num_classes,
        num_channels,
        timeseries_len,
        dim,
        depth,
        heads,
        pool,
        dim_head,
        dropout=0.0,
        emb_dropout=0.0,
        scale_dim=4,
    ):
        super().__init__()
        self.image_size = img_res
        self.patch_size = patch_size
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = num_classes
        self.num_frames = timeseries_len
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.pool = pool
        self.scale_dim = scale_dim
        assert self.pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        assert (
            self.image_size % self.patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = self.num_patches_1d**2
        patch_dim = num_channels * self.patch_size**2
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            ),
            nn.Linear(patch_dim, self.dim),
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_frames, num_patches, self.dim)
        )
        print("pos embedding: ", self.pos_embedding.shape)
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        print("space token: ", self.space_token.shape)
        self.space_transformer = Transformer(
            self.dim,
            self.depth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            self.dropout,
        )
        self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        print("temporal token: ", self.temporal_token.shape)
        self.temporal_transformer = Transformer(
            self.dim,
            self.depth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            self.dropout,
        )
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes * self.patch_size**2),
        )

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape
        x += self.pos_embedding  # [:, :, :(n + 1)]
        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.space_transformer(x)
        x = rearrange(
            x, "(b t) ... -> b t ...", b=b
        )  # use only space token, location 0
        cls_temporal_tokens = repeat(
            self.temporal_token,
            "() () d -> b t k d",
            b=b,
            t=1,
            k=self.num_patches_1d**2,
        )
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b * self.num_patches_1d**2, self.num_frames + 1, self.dim)
        x = self.temporal_transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.mlp_head(x)
        x = x.reshape(B, self.num_patches_1d**2, self.patch_size**2, self.num_classes)
        x = x.reshape(B, H * W, self.num_classes)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormLocal(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        # print('before fn: ', x.shape)
        x = self.fn(x, **kwargs)
        # print('after fn: ', x.shape)
        return x


class Conv1x1Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        # print(x.shape)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        # print(q.shape, k.shape, v.shape)
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class ReAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange("b h i j -> b i j h"),
            nn.LayerNorm(heads),
            Rearrange("b i j h -> b h i j"),
        )

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        # attention

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum("b h i j, h g -> b g i j", attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class LeFF(nn.Module):

    def __init__(self, dim=192, scale=4, depth_kernel=3):
        super().__init__()

        scale_dim = dim * scale
        self.up_proj = nn.Sequential(
            nn.Linear(dim, scale_dim),
            Rearrange("b n c -> b c n"),
            nn.BatchNorm1d(scale_dim),
            nn.GELU(),
            Rearrange("b c (h w) -> b c h w", h=14, w=14),
        )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(
                scale_dim,
                scale_dim,
                kernel_size=depth_kernel,
                padding=1,
                groups=scale_dim,
                bias=False,
            ),
            nn.BatchNorm2d(scale_dim),
            nn.GELU(),
            Rearrange("b c h w -> b (h w) c", h=14, w=14),
        )

        self.down_proj = nn.Sequential(
            nn.Linear(scale_dim, dim),
            Rearrange("b n c -> b c n"),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            Rearrange("b c n -> b n c"),
        )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x


class LCAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        q = q[:, :, -1, :].unsqueeze(2)  # Only Lth element use as query

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out
