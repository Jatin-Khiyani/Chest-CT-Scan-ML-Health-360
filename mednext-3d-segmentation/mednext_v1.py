"""
MedNeXt-3D: Transformer-driven Scaling of ConvNets for Medical Image Segmentation
Reference: Roy et al., MICCAI 2023 (https://arxiv.org/abs/2303.09975)

Standalone PyTorch implementation decoupled from nnU-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class LayerNorm(nn.Module):
    """
    LayerNorm supporting channels_first (B, C, D, H, W) or channels_last.
    """
    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if x.ndim == 5:
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            elif x.ndim == 4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MedNeXtBlock(nn.Module):
    """
    MedNeXt 3D Block:
    Depthwise 3D Conv -> GroupNorm -> 1x1x1 Conv (Expansion) -> GELU -> 1x1x1 Conv (Compression) + Residual
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = 'group',
        n_groups: int = None,
        dim: str = '3d',
        grn: bool = False
    ):
        super().__init__()
        self.do_res = do_res
        self.dim = dim
        conv = nn.Conv3d if dim == '3d' else nn.Conv2d

        # 1. Depthwise Convolution
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # 2. Normalization
        if norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        elif norm_type == 'layer':
            self.norm = LayerNorm(normalized_shape=in_channels, data_format='channels_first')
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        # 3. Expansion Layer (Conv 1x1x1)
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.act = nn.GELU()

        # 4. Compression Layer (Conv 1x1x1)
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.grn = grn
        if grn:
            shape = (1, exp_r * in_channels, 1, 1, 1) if dim == '3d' else (1, exp_r * in_channels, 1, 1)
            self.grn_beta = nn.Parameter(torch.zeros(*shape), requires_grad=True)
            self.grn_gamma = nn.Parameter(torch.zeros(*shape), requires_grad=True)

    def forward(self, x, dummy_tensor=None):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))

        if self.grn:
            dims = (-3, -2, -1) if self.dim == '3d' else (-2, -1)
            gx = torch.norm(x1, p=2, dim=dims, keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1

        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):
    """
    MedNeXt Downsampling Block (Stride 2)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = False,
        norm_type: str = 'group',
        dim: str = '3d',
        grn: bool = False
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        conv = nn.Conv3d if dim == '3d' else nn.Conv2d
        self.resample_do_res = do_res

        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res
        return x1


class MedNeXtUpBlock(MedNeXtBlock):
    """
    MedNeXt Upsampling Block (ConvTranspose Stride 2)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = False,
        norm_type: str = 'group',
        dim: str = '3d',
        grn: bool = False
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        self.resample_do_res = do_res
        self.dim = dim
        conv_t = nn.ConvTranspose3d if dim == '3d' else nn.ConvTranspose2d

        if do_res:
            self.res_conv = conv_t(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = conv_t(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)
        if self.dim == '3d':
            x1 = F.pad(x1, (1, 0, 1, 0, 1, 0))
        else:
            x1 = F.pad(x1, (1, 0, 1, 0))

        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == '3d':
                res = F.pad(res, (1, 0, 1, 0, 1, 0))
            else:
                res = F.pad(res, (1, 0, 1, 0))
            x1 = x1 + res
        return x1


class OutBlock(nn.Module):
    """
    Final output projection block to num_classes
    """
    def __init__(self, in_channels: int, n_classes: int, dim: str = '3d'):
        super().__init__()
        conv = nn.Conv3d if dim == '3d' else nn.Conv2d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        return self.conv_out(x)


class MedNeXt(nn.Module):
    """
    MedNeXt Full U-Net Architecture (Supports 2D and 3D)
    """
    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        n_classes: int,
        exp_r: int = 4,
        kernel_size: int = 3,
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,
        do_res: bool = True,
        do_res_up_down: bool = True,
        checkpoint_style: str = None,
        block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],
        norm_type: str = 'group',
        dim: str = '3d',
        grn: bool = True
    ):
        super().__init__()
        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.outside_block_checkpointing = (checkpoint_style == 'outside_block')
        assert dim in ['2d', '3d']
        self.dim = dim

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        conv = nn.Conv3d if dim == '3d' else nn.Conv2d
        self.stem = conv(in_channels, n_channels, kernel_size=1)

        if isinstance(exp_r, int):
            exp_r = [exp_r for _ in range(len(block_counts))]

        # Encoder stages
        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(n_channels, n_channels, exp_r[0], enc_kernel_size, do_res, norm_type, dim=dim, grn=grn)
            for _ in range(block_counts[0])
        ])
        self.down_0 = MedNeXtDownBlock(n_channels, 2 * n_channels, exp_r[1], enc_kernel_size, do_res_up_down, norm_type, dim=dim, grn=grn)

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(n_channels * 2, n_channels * 2, exp_r[1], enc_kernel_size, do_res, norm_type, dim=dim, grn=grn)
            for _ in range(block_counts[1])
        ])
        self.down_1 = MedNeXtDownBlock(2 * n_channels, 4 * n_channels, exp_r[2], enc_kernel_size, do_res_up_down, norm_type, dim=dim, grn=grn)

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(n_channels * 4, n_channels * 4, exp_r[2], enc_kernel_size, do_res, norm_type, dim=dim, grn=grn)
            for _ in range(block_counts[2])
        ])
        self.down_2 = MedNeXtDownBlock(4 * n_channels, 8 * n_channels, exp_r[3], enc_kernel_size, do_res_up_down, norm_type, dim=dim, grn=grn)

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(n_channels * 8, n_channels * 8, exp_r[3], enc_kernel_size, do_res, norm_type, dim=dim, grn=grn)
            for _ in range(block_counts[3])
        ])
        self.down_3 = MedNeXtDownBlock(8 * n_channels, 16 * n_channels, exp_r[4], enc_kernel_size, do_res_up_down, norm_type, dim=dim, grn=grn)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(n_channels * 16, n_channels * 16, exp_r[4], dec_kernel_size, do_res, norm_type, dim=dim, grn=grn)
            for _ in range(block_counts[4])
        ])

        # Decoder stages
        self.up_3 = MedNeXtUpBlock(16 * n_channels, 8 * n_channels, exp_r[5], dec_kernel_size, do_res_up_down, norm_type, dim=dim, grn=grn)
        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(n_channels * 8, n_channels * 8, exp_r[5], dec_kernel_size, do_res, norm_type, dim=dim, grn=grn)
            for _ in range(block_counts[5])
        ])

        self.up_2 = MedNeXtUpBlock(8 * n_channels, 4 * n_channels, exp_r[6], dec_kernel_size, do_res_up_down, norm_type, dim=dim, grn=grn)
        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(n_channels * 4, n_channels * 4, exp_r[6], dec_kernel_size, do_res, norm_type, dim=dim, grn=grn)
            for _ in range(block_counts[6])
        ])

        self.up_1 = MedNeXtUpBlock(4 * n_channels, 2 * n_channels, exp_r[7], dec_kernel_size, do_res_up_down, norm_type, dim=dim, grn=grn)
        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(n_channels * 2, n_channels * 2, exp_r[7], dec_kernel_size, do_res, norm_type, dim=dim, grn=grn)
            for _ in range(block_counts[7])
        ])

        self.up_0 = MedNeXtUpBlock(2 * n_channels, n_channels, exp_r[8], dec_kernel_size, do_res_up_down, norm_type, dim=dim, grn=grn)
        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(n_channels, n_channels, exp_r[8], dec_kernel_size, do_res, norm_type, dim=dim, grn=grn)
            for _ in range(block_counts[8])
        ])

        # Output heads
        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=n_channels * 16, n_classes=n_classes, dim=dim)

    def _checkpoint_sequential(self, sequential_block, x):
        for layer in sequential_block:
            x = checkpoint.checkpoint(layer, x, self.dummy_tensor, use_reentrant=False)
        return x

    def forward(self, x):
        x = self.stem(x)

        if self.outside_block_checkpointing and self.training:
            x_res_0 = self._checkpoint_sequential(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor, use_reentrant=False)

            x_res_1 = self._checkpoint_sequential(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor, use_reentrant=False)

            x_res_2 = self._checkpoint_sequential(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor, use_reentrant=False)

            x_res_3 = self._checkpoint_sequential(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor, use_reentrant=False)

            x = self._checkpoint_sequential(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor, use_reentrant=False)
            x = self._checkpoint_sequential(self.dec_block_3, x_res_3 + x_up_3)
            if self.do_ds:
                x_ds_3 = self.out_3(x)

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor, use_reentrant=False)
            x = self._checkpoint_sequential(self.dec_block_2, x_res_2 + x_up_2)
            if self.do_ds:
                x_ds_2 = self.out_2(x)

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor, use_reentrant=False)
            x = self._checkpoint_sequential(self.dec_block_1, x_res_1 + x_up_1)
            if self.do_ds:
                x_ds_1 = self.out_1(x)

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor, use_reentrant=False)
            x = self._checkpoint_sequential(self.dec_block_0, x_res_0 + x_up_0)
            x = self.out_0(x)
        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)

            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)

            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)

            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            x = self.dec_block_3(x_res_3 + x_up_3)
            if self.do_ds:
                x_ds_3 = self.out_3(x)

            x_up_2 = self.up_2(x)
            x = self.dec_block_2(x_res_2 + x_up_2)
            if self.do_ds:
                x_ds_2 = self.out_2(x)

            x_up_1 = self.up_1(x)
            x = self.dec_block_1(x_res_1 + x_up_1)
            if self.do_ds:
                x_ds_1 = self.out_1(x)

            x_up_0 = self.up_0(x)
            x = self.dec_block_0(x_res_0 + x_up_0)
            x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        return x


# Model Factory Functions

def create_mednextv1_small(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=2,
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
    )


def create_mednextv1_base(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
    )


def create_mednextv1_medium(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
        checkpoint_style='outside_block'
    )


def create_mednextv1_large(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        checkpoint_style='outside_block'
    )


def create_mednext_v1(num_input_channels=1, num_classes=1, model_id='S', kernel_size=3, deep_supervision=False):
    model_dict = {
        'S': create_mednextv1_small,
        'B': create_mednextv1_base,
        'M': create_mednextv1_medium,
        'L': create_mednextv1_large,
    }
    if model_id.upper() not in model_dict:
        raise ValueError(f"Invalid model_id '{model_id}'. Choose from: S, B, M, L")
    return model_dict[model_id.upper()](num_input_channels, num_classes, kernel_size, deep_supervision)

