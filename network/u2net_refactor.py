import numpy as np
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import ConvAlgo

import math

__all__ = ['U2NET_full', 'U2NET_lite']

from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv, ResContextBlock, UpBlock, ReconBlock, ResBlock


def conv3d(in_planes, out_planes, stride=1, padding=0,dilation=1,indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=stride,
                             padding=padding,dilation=dilation, bias=False, indice_key=indice_key,algo=ConvAlgo.Native)

def _upsample_like(x, size):
    return nn.Upsample(size=size, mode='bilinear', align_corners=False)(x)


def _size_map(x, height):
    # {height: size} for Upsample
    size = list(x.shape[-2:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes
class REBNCONV(nn.Module):
        def __init__(self,
                     output_shape,
                     use_norm=True,
                     num_input_features=128,
                     dialate=1,
                     nclasses=20, n_height=32, strict=False, init_size=16):
            super(REBNCONV, self).__init__()
            self.nclasses = nclasses
            self.nheight = n_height
            self.strict = False
            sparse_shape = np.array(output_shape)
            # sparse_shape[0] = 11
            print(sparse_shape)
            self.sparse_shape = sparse_shape
            self.downCntx = ResContextBlock(num_input_features, init_size,dialate=dialate, indice_key="pre")
            self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True,dialate=dialate, indice_key="down2")
            self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True,dialate=dialate ,indice_key="down3")
            self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True,dialate=dialate ,height_pooling=False,
                                      indice_key="down4")
            self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True,dialate=dialate ,height_pooling=False,
                                      indice_key="down5")
            self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5",dialate=dialate)
            self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4",dialate=dialate)
            self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3",dialate=dialate)
            self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2",dialate=dialate)
            self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, dialate=dialate,indice_key="recon")
            self.logits = spconv.SubMConv3d(4 * init_size, nclasses,dilation=dialate, indice_key="logit", kernel_size=3, stride=1,
                                            padding=1,
                                            bias=True, algo=ConvAlgo.Native)
        def forward(self, voxel_features, coors, batch_size):
            # x = x.contiguous()
            coors = coors.int()
            # import pdb
            # pdb.set_trace()
            ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                          batch_size)
            ret = self.downCntx(ret)
            down1c, down1b = self.resBlock2(ret)
            down2c, down2b = self.resBlock3(down1c)
            down3c, down3b = self.resBlock4(down2c)
            down4c, down4b = self.resBlock5(down3c)
            up4e = self.upBlock0(down4c, down4b)
            up3e = self.upBlock1(up4e, down3b)
            up2e = self.upBlock2(up3e, down2b)
            up1e = self.upBlock3(up2e, down1b)
            up0e = self.ReconNet(up1e)
            up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))
            logits = self.logits(up0e)
            y = logits.dense()
            return y  # , up0e.features, coors
        # U-Net like symmetric encoder-decoder structure
class RSU(nn.Module):
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated=False):
        super(RSU, self).__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)
    def forward(self, voxel_features, coors, batch_size):
        sizes = _size_map(voxel_features, self.height)
        x = self.rebnconvin(voxel_features, coors, batch_size)
        # U-Net like symmetric encoder-decoder structure
        def unet(voxel_features,coors,batch_size, height=1):
            if height < self.height:
                x1 = getattr(self, f'rebnconv{height}')(voxel_features)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)
                x = getattr(self, f'rebnconv{height}d')(torch.cat((x2, x1), 1))
                return _upsample_like(x, sizes[height - 1]) if not self.dilated and height > 1 else x
            else:
                return getattr(self, f'rebnconv{height}')(voxel_features)
        return x + unet(x)
    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False):
           self.add_module('rebnconvin', REBNCONV(num_input_features=in_ch, output_shape=out_ch))
           self.add_module('downsample', nn.MaxPool3d(2, stride=2, ceil_mode=True))
           self.add_module(f'rebnconv1', REBNCONV(num_input_features=out_ch, output_shape=mid_ch))
           self.add_module(f'rebnconv1d', REBNCONV(num_input_features=mid_ch * 2, output_shape=out_ch))
           for i in range(2, height):
             dilate = 1 if not dilated else 2 ** (i - 1)
             self.add_module(f'rebnconv{i}', REBNCONV(num_input_features=mid_ch,output_shape= mid_ch, dialate=dilated))
             self.add_module(f'rebnconv{i}d', REBNCONV(num_input_features=mid_ch * 2,output_shape= mid_ch, dialate=dilated))

           dilate = 2 if not dilated else 2 ** (height - 1)
           self.add_module(f'rebnconv{height}', REBNCONV(num_input_features=mid_ch, output_shape=mid_ch, dialate=dilated))
class U2NET(nn.Module):
    def __init__(self, cfgs, out_ch):
        super(U2NET, self).__init__()
        self.out_ch = out_ch
        self._make_layers(cfgs)

    def forward(self, voxel_features, coors, batch_size):
        sizes = _size_map(voxel_features, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(self, voxel_features, coors, batch_size, height=1):
            if height < 6:
                x1 = getattr(self, f'stage{height}')(voxel_features,coors, batch_size)
                x2 = unet(getattr(self, 'downsample')(x1),  coors,batch_size,height+1)
                x = getattr(self, f'stage{height}d')(torch.cat((x2, x1), 1))
                side(x, height)
                return _upsample_like(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f'stage{height}')(voxel_features)
                side(x, height)
                return _upsample_like(x, sizes[height - 1])

        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f'side{h}')(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, 'outconv')(x)
            maps.insert(0, x)
            return [torch.sigmoid(x) for x in maps]

        unet(self,voxel_features, coors, batch_size, height=1)
        maps = fuse()
        return maps

    def _make_layers(self, cfgs):
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module('downsample', nn.MaxPool3d(2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], *v[1]))
            if v[2] > 0:
                # build side layer
                self.add_module(f'side{v[0][-1]}', conv3d(v[2], self.out_ch, 3, padding=1))
        # build fuse layer
        self.add_module('outconv', conv3d(int(self.height * self.out_ch), self.out_ch, 1))


def U2NET_full():
    full = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, 64, 64, 128), -1],
        'stage2': ['En_2', (6, 128, 64, 128), -1],
        'stage3': ['En_3', (5, 256, 128, 512), -1],
        'stage4': ['En_4', (4, 512, 256, 1024), -1],
        'stage5': ['En_5', (4, 1024, 512, 1024, True), -1],
        'stage6': ['En_6', (4, 1024, 512, 1024, True), 1024],
        'stage5d': ['De_5', (4, 1024, 512, 1024, True), 1024],
        'stage4d': ['De_4', (4, 1024, 256, 512), 512],
        'stage3d': ['De_3', (5, 512, 128, 256), 256],
        'stage2d': ['De_2', (6, 256, 64, 128), 128],
        'stage1d': ['De_1', (7, 128, 32, 128), 128],
    }
    return U2NET(cfgs=full, out_ch=1)


def U2NET_lite():
    lite = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, 3, 16, 64), -1],
        'stage2': ['En_2', (6, 64, 16, 64), -1],
        'stage3': ['En_3', (5, 64, 16, 64), -1],
        'stage4': ['En_4', (4, 64, 16, 64), -1],
        'stage5': ['En_5', (4, 64, 16, 64, True), -1],
        'stage6': ['En_6', (4, 64, 16, 64, True), 64],
        'stage5d': ['De_5', (4, 128, 16, 64, True), 64],
        'stage4d': ['De_4', (4, 128, 16, 64), 64],
        'stage3d': ['De_3', (5, 128, 16, 64), 64],
        'stage2d': ['De_2', (6, 128, 16, 64), 64],
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    }
    return U2NET(cfgs=lite, out_ch=1)
