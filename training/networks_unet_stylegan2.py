from .networks_stylegan2 import DiscriminatorEpilogue, MappingNetwork, Conv2dLayer
import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
# from linear_attention_transformer import ImageLinearAttention

def leaky_relu(p=0.2):
    return torch.nn.LeakyReLU(p)

class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class Rezero(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.fn(x.to(torch.float32))* self.g

def convert_bias_to_float16(model):
    for module in model.modules():
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data = module.bias.data.to(torch.float16)

# def attn_and_ff(chan):
#     return torch.nn.Sequential(
#             Residual(Rezero(ImageLinearAttention(chan, norm_queries=True)),),
#             Residual(Rezero(torch.nn.Sequential(torch.nn.Conv2d(chan, chan * 2, 1), leaky_relu(),
#                                                 torch.nn.Conv2d(chan * 2, chan, 1))))
#     )

# attn_and_ff = lambda chan: torch.nn.Sequential(*[
#     Residual(Rezero(ImageLinearAttention(chan, norm_queries = True))),
#     Residual(Rezero(torch.nn.Sequential(torch.nn.Conv2d(chan, chan * 2, 1), leaky_relu(), torch.nn.Conv2d(chan * 2, chan, 1))))
# ])

@persistence.persistent_class
class UnetDiscriminatorUpBlock(torch.nn.Module):
    def __init__(self,
         in_channels,  # Number of input channels, 0 = first block.
         out_channels,  # Number of output channels.
         first_layer_idx,  # Index of the first layer.
         architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
         activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
         resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
         conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
         use_fp16=False,  # Use FP16 for this block?
         fp16_channels_last=False,  # Use channels-last memory format with FP16?
         freeze_layers=0,  # Freeze-D: Number of layers to freeze.
         is_last_layer=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))


        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv0 = Conv2dLayer(in_channels, in_channels, kernel_size=3, activation=activation,
                                 trainable=next(trainable_iter), conv_clamp=conv_clamp,
                                 channels_last=self.channels_last)
        self.conv1 = Conv2dLayer(in_channels, out_channels, kernel_size=3, activation=activation,
                                 trainable=next(trainable_iter), conv_clamp=conv_clamp,
                                 channels_last=self.channels_last)

        if architecture == 'resnet':
            # Divide by 2 because in_channels includes concat x with residual layer
            self.skip = Conv2dLayer(in_channels // 2, out_channels, kernel_size=1, bias=False, up=2,
                                    trainable=next(trainable_iter), resample_filter=resample_filter,
                                    channels_last=self.channels_last)


    def forward(self, x, res):
        dtype = torch.float16 if self.use_fp16 else torch.float32
        memory_format = torch.channels_last if self.channels_last else torch.contiguous_format

        # Input.
        if x is not None:
            # TODO: FIX THIS
            # misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.up(x)
            x = torch.cat([x, res], dim=1)
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.up(x)
            x = torch.cat([x, res], dim=1)
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
        return x


@persistence.persistent_class
class UnetDiscriminatorDownBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        # # Residual layer
        # self.conv_res = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
        #     trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        if (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            # res = self.conv_res(x, gain=np.sqrt(0.5))
            res = x.clone()
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            res = self.conv_res(x, gain=np.sqrt(0.5))
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img, res

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

@persistence.persistent_class
class UnetDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 16384,    # Overall multiplier for the number of channels. original: 32768
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        # log2(resolution) converts resolution to power of 2
        # e.g. log2(1024) = 10, log2(512) = 9
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        # Channel base default = 32768 = 2^15
        # Dividing channel base by resolution gives the number of channels for each resolution
        # e.g. 32768 / 1024 = 2^15 / 2^10 = 2^5 = 32
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = UnetDiscriminatorDownBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                               first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'down_block{res}', block)
            cur_layer_idx += block.num_layers
            # attn_fn = attn_and_ff(out_channels)
            # setattr(self, f'attn_block{res}', attn_fn)

        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)

        in_out_channels = []
        for res in self.block_resolutions[:-2][::-1]:
            chan = channels_dict[res]
            next_chan = chan if res == self.block_resolutions[0] else channels_dict[res * 2]
            in_out_channels.append((chan * 2, next_chan, res))
        # in_out_channels.append((channels_dict[self.block_resolutions[0]] * 2, 3))

        up_blocks = []
        for in_channels, out_channels, res in in_out_channels:
            use_fp16 = (res >= 64)
            up_block = UnetDiscriminatorUpBlock(in_channels, out_channels, first_layer_idx=cur_layer_idx, use_fp16=use_fp16, is_last_layer=out_channels == 3)
            cur_layer_idx += up_block.num_layers
            up_blocks.append(up_block)
        self.up_blocks = torch.nn.ModuleList(up_blocks)
        self.dec_conv_out = torch.nn.Conv2d(in_out_channels[-1][-1], 1, 1)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, update_emas=False, **block_kwargs):
        _ = update_emas # unused
        enc_x = None
        dec_x = None
        residuals = []
        for res in self.block_resolutions:
            block = getattr(self, f'down_block{res}')
            # attn_block = getattr(self, f'attn_block{res}')
            enc_x, img, unet_res = block(enc_x, img, **block_kwargs)
            residuals.append(unet_res)
            # if attn_block is not None:
            #     enc_x = attn_block(enc_x)
            if res == self.block_resolutions[-3]:
                dec_x = torch.empty_like(enc_x)
                dec_x.copy_(enc_x)

        for (up_block, residual) in zip(self.up_blocks, residuals[:-2][::-1]):
            dec_x = up_block(dec_x, residual)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        enc_x = self.b4(enc_x, img, cmap)
        dec_x = self.dec_conv_out(dec_x.to(dtype=torch.float32))
        return enc_x, dec_x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'