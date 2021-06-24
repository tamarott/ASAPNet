import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ASAPNetsBlock as ASAPNetsBlock
from models.networks.architecture import MySeparableBilinearDownsample as BilinearDownsample
import torch.nn.utils.spectral_norm as spectral_norm
import torch as th
from math import pi
from math import log2
import time


class ASAPNetsGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(lr_instance=True)
        parser.set_defaults(no_instance_dist=True)
        parser.set_defaults(hr_coor="cosine")
        return parser

    def __init__(self, opt, hr_stream=None, lr_stream=None, fast=False):
        super(ASAPNetsGenerator, self).__init__()
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if (opt.no_instance_edge & opt.no_instance_dist) else 1)
        self.lr_instance = opt.lr_instance
        self.learned_ds_factor = opt.learned_ds_factor #(S2 in sec. 3.2)
        self.gpu_ids = opt.gpu_ids

        # calculates the total downsampling factor in order to get the final low-res grid of parameters (S=S1xS2 in sec. 3.2)
        self.downsampling = opt.crop_size // (16 * opt.aspect_ratio)


        self.highres_stream = ASAPNetsHRStream(self.downsampling, num_inputs=self.num_inputs,
                                               num_outputs=opt.output_nc, width=opt.hr_width,
                                               depth=opt.hr_depth, coordinates=opt.hr_coor,
                                               no_one_hot=opt.no_one_hot, lr_instance=opt.lr_instance,
                                               **hr_stream)

        num_params = self.highres_stream.num_params
        num_inputs_lr = self.highres_stream.num_inputs + (1 if opt.lr_instance else 0)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        self.lowres_stream = ASAPNetsLRStream(num_inputs_lr, num_params, norm_layer, width=opt.lr_width,
                                              max_width=opt.lr_max_width, depth=opt.lr_depth,
                                              learned_ds_factor=opt.learned_ds_factor,
                                              reflection_pad=opt.reflection_pad, **lr_stream)

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def get_lowres(self, im):
        """Creates a lowres version of the input."""
        device = self.use_gpu()
        if(self.learned_ds_factor != self.downsampling):
            myds = BilinearDownsample(int(self.downsampling//self.learned_ds_factor), self.num_inputs,device)
            return myds(im)
        else:
            return im

    def forward(self, highres, z=None):
        lowres = self.get_lowres(highres)
        lr_features = self.lowres_stream(lowres)
        output = self.highres_stream(highres, lr_features)
        return output, lr_features#, lowres


def _get_coords(bs, h, w, device, ds, coords_type):
    """Creates the position encoding for the pixel-wise MLPs"""
    if coords_type == 'cosine':
        f0 = ds
        f = f0
        while f > 1:
            x = th.arange(0, w).float()
            y = th.arange(0, h).float()
            xcos = th.cos((2 * pi * th.remainder(x, f).float() / f).float())
            xsin = th.sin((2 * pi * th.remainder(x, f).float() / f).float())
            ycos = th.cos((2 * pi * th.remainder(y, f).float() / f).float())
            ysin = th.sin((2 * pi * th.remainder(y, f).float() / f).float())
            xcos = xcos.view(1, 1, 1, w).repeat(bs, 1, h, 1)
            xsin = xsin.view(1, 1, 1, w).repeat(bs, 1, h, 1)
            ycos = ycos.view(1, 1, h, 1).repeat(bs, 1, 1, w)
            ysin = ysin.view(1, 1, h, 1).repeat(bs, 1, 1, w)
            coords_cur = th.cat([xcos, xsin, ycos, ysin], 1).to(device)
            if f < f0:
                coords = th.cat([coords, coords_cur], 1).to(device)
            else:
                coords = coords_cur
            f = f//2
    else:
        raise NotImplementedError()
    return coords.to(device)


class ASAPNetsLRStream(th.nn.Sequential):
    """Convolutional LR stream to estimate the pixel-wise MLPs parameters"""
    def __init__(self, num_in, num_out, norm_layer, width=64, max_width=1024, depth=7, learned_ds_factor=16,
                 reflection_pad=False, replicate_pad=False):
        super(ASAPNetsLRStream, self).__init__()

        model = []

        self.num_out = num_out
        padw = 1
        if reflection_pad:
            padw = 0
            model += [th.nn.ReflectionPad2d(1)]
        if replicate_pad:
            padw = 0
            model += [th.nn.ReplicationPad2d(1)]

        count_ly = 0

        model += [norm_layer(th.nn.Conv2d(num_in, width, 3, stride=1, padding=padw)),
                  th.nn.ReLU(inplace=True)]

        num_ds_layers = int(log2(learned_ds_factor))

        # strided conv layers for learning downsampled representation of the input"
        for i in range(num_ds_layers):
            if reflection_pad:
                model += [th.nn.ReflectionPad2d(1)]
            if replicate_pad:
                model += [th.nn.ReplicationPad2d(1)]
            if i == num_ds_layers-1:
                last_width = max_width
                model += [norm_layer(th.nn.Conv2d(width, last_width, 3, stride=2, padding=padw)),
                          th.nn.ReLU(inplace=True)]
                width = last_width
            else:
                model += [norm_layer(th.nn.Conv2d(width, width, 3, stride=2, padding=padw)),
                      th.nn.ReLU(inplace=True)]

        # ConvNet to estimate the MLPs parameters"
        for i in range(count_ly, count_ly+depth):
            model += [ASAPNetsBlock(width, norm_layer, reflection_pad=reflection_pad, replicate_pad=replicate_pad)]

        # Final parameter prediction layer, transfer conv channels into the per-pixel number of MLP parameters
        model += [th.nn.Conv2d(width, self.num_out, 1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ASAPNetsHRStream(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, downsampling,
                 num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine",
                 no_one_hot=False, lr_instance=False):
        super(ASAPNetsHRStream, self).__init__()

        self.lr_instance = lr_instance
        self.downsampling = downsampling
        self.num_inputs = num_inputs - (1 if self.lr_instance else 0)
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.no_one_hot = no_one_hot
        self.channels = []
        self._set_channels()

        self.num_params = 0
        self.splits = {}
        self._set_num_params()

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling


    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = self.num_inputs
        if self.coordinates == "cosine":
            in_ch += int(4*log2(self.downsampling))
        self.channels = [in_ch]
        for _ in range(self.depth - 1):  # intermediate layer -> cste size
            self.channels.append(self.width)
        # output layer
        self.channels.append(self.num_outputs)

    def _set_num_params(self):
        nparams = 0
        self.splits = {
            "biases": [],
            "weights": [],
        }

        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1]
            nparams += nco  # FC biases
            self.splits["biases"].append((idx, idx + nco))
            idx += nco

            nparams += nci * nco  # FC weights
            self.splits["weights"].append((idx, idx + nco * nci))
            idx += nco * nci

        self.num_params = nparams

    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]

    def forward(self, highres, lr_params):
        assert lr_params.shape[1] == self.num_params, "incorrect input params"

        if self.lr_instance:
            highres = highres[:, :-1, :, :]

        # Fetch sizes
        k = int(self.downsampling)
        bs, _, h, w = highres.shape
        bs, _, h_lr, w_lr = lr_params.shape

        # Spatial encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                self.xy_coords = _get_coords(bs, h, w, highres.device, self.ds, self.coordinates)
            highres = th.cat([highres, self.xy_coords], 1)


        # Split input in tiles of size kxk according to the NN interp factor (the total downsampling factor),
        # with channels last (for matmul)
        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = highres.shape[1]
        # bs, 5 rgbxy, h//k=h_lr, w//k=w_lr, k, k
        tiles = highres.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)
        out = tiles
        num_layers = len(self.channels) - 1

        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = lr_params[:, wstart:wstop]
            b_ = lr_params[:, bstart:bstop]

            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nci, nco)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nco)
            out = th.matmul(out, w_) + b_

            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = th.nn.functional.leaky_relu(out, 0.01, inplace=True)
            else:
                out = F.tanh(out)

        # reorder the tiles in their correct position, and put channels first
        out = out.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)

        return out
