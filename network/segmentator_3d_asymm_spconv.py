# -*- coding:utf-8 -*-
# author: Xinge
# @file: segmentator_3d_asymm_spconv.py

import numpy as np
import spconv.pytorch as spconv
import torch
from spconv.pytorch import ConvAlgo

from torch import nn


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key,algo=ConvAlgo.Native)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key,algo=ConvAlgo.Native)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key,algo=ConvAlgo.Native)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key,algo=ConvAlgo.Native)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key,algo=ConvAlgo.Native)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes,kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key,algo=ConvAlgo.Native)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes,kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key,algo=ConvAlgo.Native)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut =shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))
        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False,algo=ConvAlgo.Native)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False,algo=ConvAlgo.Native)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        resA = resA.replace_feature(resA.features + shortcut.features)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False,algo=ConvAlgo.Native)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA =upA.replace_feature(self.trans_bn(upA.features))

        ## upsample
        upA = self.up_subm(upA)

        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))

        upE = self.conv2(upE)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))

        upE = self.conv3(upE)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 =shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut3.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut =shortcut.replace_feature( shortcut.features + shortcut2.features + shortcut3.features)
        shortcut = shortcut.replace_feature(shortcut.features * x.features)
        return shortcut

class encoder1(nn.Module):
    def __init__(self,
                 num_input_features=128,
                 init_size=16):
        super(encoder1, self).__init__()
        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre1")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down12")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down13")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down14")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down15")
        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up10", up_key="down15")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up11", up_key="down14")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up12", up_key="down13")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up13", up_key="down12")
        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon1")
    def forward(self, features):

        ret = self.downCntx(features)
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
        return up0e #, up0e.features, coors

class decoder5(nn.Module):
    def __init__(self,
                 num_input_features=128,
                 init_size=16):
        super(decoder5, self).__init__()
        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre01d")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down012d")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down013d")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down014d")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down015d")
        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up010d", up_key="down015d")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up011d", up_key="down014d")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up012d", up_key="down013d")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up013d", up_key="down012d")
        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon01d")
    def forward(self, features):

        ret = self.downCntx(features)
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
        return up0e #, up0e.features, coors
class decoder4(nn.Module):
    def __init__(self,
                 num_input_features=128,
                 init_size=16):
        super(decoder4, self).__init__()
        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre1d")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down12d")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down13d")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down14d")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down15d")
        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up10d", up_key="down15d")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up11d", up_key="down14d")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up12d", up_key="down13d")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up13d", up_key="down12d")
        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon1d")
    def forward(self, features):

        ret = self.downCntx(features)
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
        return up0e #, up0e.features, coors
class encoder2(nn.Module):
    def __init__(self,
                 num_input_features=128,
                 init_size=16):
        super(encoder2, self).__init__()
        self.downCntx = ResContextBlock(num_input_features,2 * init_size, indice_key="pre2")
        self.resBlock2 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down23")
        self.resBlock3 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down24")
        self.resBlock4 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down25")
        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up20", up_key="down25")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up21", up_key="down24")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up22", up_key="down23")
        self.ReconNet = ReconBlock(4 * init_size, 4 * init_size, indice_key="recon2")
    def forward(self, features):
        ret = self.downCntx(features)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        up3e = self.upBlock0(down3c, down3b)
        up2e = self.upBlock1(up3e, down2b)
        up1e = self.upBlock2(up2e, down1b)
        up0e = self.ReconNet(up1e)
        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))
        return up0e #, up0e.features, coors

class decoder3(nn.Module):
        def __init__(self,
                     num_input_features=128,
                     init_size=16):
            super(decoder3, self).__init__()
            self.downCntx = ResContextBlock(num_input_features, 2 * init_size, indice_key="pre2d")
            self.resBlock2 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down23d")
            self.resBlock3 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                      indice_key="down24d")
            self.resBlock4 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                      indice_key="down25d")

            self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up20d", up_key="down25d")
            self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up21d", up_key="down24d")
            self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up22d", up_key="down23d")
            self.ReconNet = ReconBlock(4 * init_size, 4 * init_size, indice_key="recon2d")
        def forward(self, features):
            ret = self.downCntx(features)
            down1c, down1b = self.resBlock2(ret)
            down2c, down2b = self.resBlock3(down1c)
            down3c, down3b = self.resBlock4(down2c)
            up3e = self.upBlock0(down3c, down3b)
            up2e = self.upBlock1(up3e, down2b)
            up1e = self.upBlock2(up2e, down1b)
            up0e = self.ReconNet(up1e)
            up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))
            return up0e  # , up0e.featur
class encoder3(nn.Module):
    def __init__(self,
                 num_input_features=128,
                 init_size=16):
        super(encoder3, self).__init__()
        self.downCntx = ResContextBlock(num_input_features,4 * init_size, indice_key="pre3")
        self.resBlock2 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down34")
        self.resBlock3 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down35")
        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up30", up_key="down35")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up31", up_key="down34")
        self.ReconNet = ReconBlock(8 * init_size, 8 * init_size, indice_key="recon3")
    def forward(self, features):
        ret = self.downCntx(features)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        up2e = self.upBlock0(down2c, down2b)
        up1e = self.upBlock1(up2e, down1b)
        up0e = self.ReconNet(up1e)
        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))
        return up0e #, up0e.features, coors
class decoder2(nn.Module):
    def __init__(self,
                 num_input_features=128,
                 init_size=16):
        super(decoder2, self).__init__()
        self.downCntx = ResContextBlock(num_input_features,4 * init_size, indice_key="pre3d")
        self.resBlock2 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down34d")
        self.resBlock3 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down35d")
        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up30d", up_key="down35d")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up31d", up_key="down34d")
        self.ReconNet = ReconBlock(8 * init_size, 8 * init_size, indice_key="recon3d")
    def forward(self, features):
        ret = self.downCntx(features)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        up2e = self.upBlock0(down2c, down2b)
        up1e = self.upBlock1(up2e, down1b)
        up0e = self.ReconNet(up1e)
        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))
        return up0e #, up0e.features, coors

class encoder4(nn.Module):
    def __init__(self,
                 num_input_features=128,
                 init_size=16):
        super(encoder4, self).__init__()
        self.downCntx = ResContextBlock(num_input_features,8 * init_size, indice_key="pre4")
        self.resBlock2 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down45")
        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up40", up_key="down45")
        self.ReconNet = ReconBlock(16 * init_size, 16 * init_size, indice_key="recon4")
    def forward(self,features):
        ret = self.downCntx(features)
        down1c, down1b = self.resBlock2(ret)
        up1e = self.upBlock0(down1c, down1b)
        up0e = self.ReconNet(up1e)
        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))
        return up0e
class decoder1(nn.Module):
    def __init__(self,
                 num_input_features=128,
                 init_size=16):
        super(decoder1, self).__init__()
        self.downCntx = ResContextBlock(num_input_features, 8 * init_size, indice_key="pre4d")
        self.resBlock2 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down45d")
        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up40d", up_key="down45d")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up31d", up_key="down34d")
        self.ReconNet = ReconBlock(16 * init_size, 16 * init_size, indice_key="recon4d")
    def forward(self, features):
        ret = self.downCntx(features)
        down1c, down1b = self.resBlock2(ret)
        up1e = self.upBlock0(down1c, down1b)
        up0e = self.ReconNet(up1e)
        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))
        return up0e  # , u
class encoder5(nn.Module):
    def __init__(self,
                 num_input_features=128,
                 init_size=16):
        super(encoder5, self).__init__()
        self.downCntx = ResContextBlock(num_input_features, 16* init_size, indice_key="pre5")
        self.ReconNet = ReconBlock(16 * init_size, 16 * init_size, indice_key="recon5")
    def forward(self, features):
      
        ret = self.downCntx(features)

        up0e = self.ReconNet(ret)

        up0e = up0e.replace_feature(torch.cat((up0e.features, ret.features), 1))
        return up0e#, up0e.features, coors

class U2NET(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=128,
                 nclasses=20,  init_size=16):
        super(U2NET, self).__init__()
        sparse_shape = np.array(output_shape)
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.nclasses = nclasses
        self.encoder1 = encoder1(num_input_features, init_size)
        self.encoder2 = encoder2(4*init_size , init_size)
        self.encoder3 = encoder3(8*init_size , init_size)
        self.encoder4 = encoder4(16*init_size, init_size)
        self.decoder1 = decoder3(48*init_size, init_size)
        self.decoder2 = decoder4(16*init_size, init_size)
        self.decoder3 = decoder5(8*init_size, init_size)
        self.interEncoder1 = spconv.SparseMaxPool3d(indice_key="inter1", kernel_size=3,algo=ConvAlgo.Native)
        self.interEncoder2 = spconv.SparseMaxPool3d(indice_key="inter2", kernel_size=3,algo=ConvAlgo.Native)
        self.interEncoder3 = spconv.SparseMaxPool3d(indice_key="inter3", kernel_size=3,algo=ConvAlgo.Native)
        self.interDecoder1 = spconv.SparseInverseConv3d (32 * init_size, 32 * init_size, indice_key="inter3", kernel_size=3,bias=False,algo=ConvAlgo.Native)
        self.interDecoder2 = spconv.SparseInverseConv3d(8 * init_size, 8 * init_size, indice_key="inter2", kernel_size=3,bias=False,algo=ConvAlgo.Native)
        self.interDecoder3 = spconv.SparseInverseConv3d(4 * init_size, 4 * init_size, indice_key="inter1", kernel_size=3,bias=False,algo=ConvAlgo.Native)

        self.sidea2 = spconv.SparseInverseConv3d(8 * init_size, 8*init_size, indice_key="inter1", kernel_size=3,bias=False,algo=ConvAlgo.Native)



        self.logits1 = spconv.SubMConv3d(8 * init_size, nclasses, indice_key="logit1", kernel_size=1, stride=1,
                                         padding=1,
                                         bias=True, algo=ConvAlgo.Native)
        self.logits2 = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit2", kernel_size=1, stride=1,
                                         padding=1,
                                         bias=True, algo=ConvAlgo.Native)
        self.logits3 = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit3", kernel_size=1, stride=1,
                                         padding=1,
                                         bias=True, algo=ConvAlgo.Native)
        self.logits = spconv.SubMConv3d(16 * init_size, nclasses, indice_key="logit", kernel_size=1, stride=1, padding=1,
                                        bias=True,algo=ConvAlgo.Native)
    def forward(self, voxel_features, coors, batch_size):
       coors = coors.int()
       ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
       x1=self.encoder1.forward(ret)
       x2i=self.interEncoder1.forward(x1)
       x2o=self.encoder2.forward(x2i)
       x3i=self.interEncoder2.forward(x2o)
       x3o=self.encoder3.forward(x3i)
       x4i=self.interEncoder3.forward(x3o)
       x4o=self.encoder4.forward(x4i)
       y1i=self.interDecoder1.forward(x4o)
       y1i = y1i.replace_feature(torch.cat((y1i.features, x3o.features), 1))
       y1o=self.decoder1.forward(y1i)
       zy11 = self.interDecoder2.forward(y1o)
       zy12 = self.sidea2.forward(zy11)
       y2i=self.interDecoder2.forward(y1o)
       y2i = y2i.replace_feature(torch.cat((y2i.features, x2o.features), 1))
       y2o=self.decoder2.forward(y2i)
       zy21 = self.interDecoder3.forward(y2o)
       y3i= self.interDecoder3.forward(y2o)
       y3i = y3i.replace_feature(torch.cat((y3i.features, x1.features), 1))
       y3o=self.decoder3.forward(y3i)
       z = torch.cat((y3o.features, zy21.features, zy12.features), 1)
       logits1= self.logits1(zy12)
       logits2 = self.logits2(zy21)
       logits3 = self.logits3(y3o)
       ret = ret.replace_feature(z)
       logits = self.logits(ret)

       y= logits.dense(),logits3.dense(),logits2.dense(),logits1.dense()
       return y
class U2NETL(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=128,
                 nclasses=20,  init_size=16):
        super(U2NETL, self).__init__()
        sparse_shape = np.array(output_shape)
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.nclasses = nclasses
        self.encoder1 = encoder1(num_input_features, init_size)
        self.encoder2 = encoder2(4*init_size , init_size)
        self.encoder3 = encoder3(8*init_size , init_size)
        self.decoder1 = decoder4(24*init_size, init_size)
        self.decoder2 = decoder5(8*init_size,init_size)
        self.interEncoder1 = spconv.SparseMaxPool3d(indice_key="inter1", kernel_size=3,algo=ConvAlgo.Native)
        self.interEncoder2 = spconv.SparseMaxPool3d(indice_key="inter2", kernel_size=3,algo=ConvAlgo.Native)
        self.interDecoder1 = spconv.SparseInverseConv3d(16 * init_size, 16 * init_size, indice_key="inter2", kernel_size=3,bias=False,algo=ConvAlgo.Native)
        self.interDecoder2 = spconv.SparseInverseConv3d(4 * init_size, 4 * init_size, indice_key="inter1", kernel_size=3,bias=False,algo=ConvAlgo.Native)

        self.logits1 = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit1", kernel_size=1, stride=1,
                                         padding=1,
                                         bias=True, algo=ConvAlgo.Native)
        self.logits2 = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit2", kernel_size=1, stride=1,
                                         padding=1,
                                         bias=True, algo=ConvAlgo.Native)
        self.logits = spconv.SubMConv3d(8 * init_size, nclasses, indice_key="logit", kernel_size=1, stride=1, padding=1,
                                        bias=True,algo=ConvAlgo.Native)
    def forward(self, voxel_features, coors, batch_size):
       coors = coors.int()
       ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
       x1=self.encoder1.forward(ret)
       x2i=self.interEncoder1.forward(x1)
       x2o=self.encoder2.forward(x2i)
       x3i=self.interEncoder2.forward(x2o)
       x3o=self.encoder3.forward(x3i)
       y1i=self.interDecoder1.forward(x3o)
       y1i = y1i.replace_feature(torch.cat((y1i.features, x2o.features), 1))
       y1o=self.decoder1.forward(y1i)
       zy11 = self.interDecoder2.forward(y1o)
       y2i=self.interDecoder2.forward(y1o)
       y2i = y2i.replace_feature(torch.cat((y2i.features, x1.features), 1))
       y2o=self.decoder2.forward(y2i)
       z = torch.cat((y2o.features, zy11.features), 1)
       logits1= self.logits1(y2o)
       logits2 = self.logits2(zy11)
       ret = ret.replace_feature(z)
       logits = self.logits(ret)

       y= logits.dense(),logits2.dense(),logits1.dense()
       return y
class U2NETLPLUS(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=128,
                 nclasses=20,  init_size=16):
        super(U2NETLPLUS, self).__init__()
        sparse_shape = np.array(output_shape)
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.nclasses = nclasses
        self.encoder1 = encoder1(num_input_features, init_size)
        self.encoder2 = encoder2(4*init_size , init_size)
        self.decoder1 = decoder5(12*init_size, init_size)
        self.interEncoder1 = spconv.SparseMaxPool3d(indice_key="inter1", kernel_size=3,algo=ConvAlgo.Native)
        self.interDecoder1 = spconv.SparseInverseConv3d(8 * init_size, 8 * init_size, indice_key="inter1", kernel_size=3,bias=False,algo=ConvAlgo.Native)
        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=1, stride=1, padding=1,
                                        bias=True,algo=ConvAlgo.Native)
    def forward(self, voxel_features, coors, batch_size):
       coors = coors.int()
       ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
       x1=self.encoder1.forward(ret)
       x2i=self.interEncoder1.forward(x1)
       x2o=self.encoder2.forward(x2i)
       y1i=self.interDecoder1.forward(x2o)
       y1i = y1i.replace_feature(torch.cat((y1i.features, x1.features), 1))
       y1o=self.decoder1.forward(y1i)
       logits = self.logits(y1o)

       y= logits.dense()
       return [y]
class U2NETP2(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=128,
                 nclasses=20, init_size=16):
        super(U2NET, self).__init__()
        sparse_shape = np.array(output_shape)
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.nclasses = nclasses
        self.encoder1 = encoder1(num_input_features, init_size)
        self.encoder2 = encoder2(4 * init_size, init_size)
        self.encoder3 = encoder3(8 * init_size, init_size)
        self.encoder4 = encoder4(16 * init_size, init_size)
        self.decoder1 = decoder2(48 * init_size, init_size)
        self.decoder2 = decoder3(24 * init_size, init_size)
        self.decoder3 = decoder4(12 * init_size, init_size)
        self.interEncoder1 = spconv.SparseMaxPool3d(indice_key="inter1", kernel_size=3, algo=ConvAlgo.Native)
        self.interEncoder2 = spconv.SparseMaxPool3d(indice_key="inter2", kernel_size=3, algo=ConvAlgo.Native)
        self.interEncoder3 = spconv.SparseMaxPool3d(indice_key="inter3", kernel_size=3, algo=ConvAlgo.Native)
        self.interDecoder1 = spconv.SparseInverseConv3d(32 * init_size, 32 * init_size, indice_key="inter3",
                                                        kernel_size=3, bias=False, algo=ConvAlgo.Native)
        self.interDecoder2 = spconv.SparseInverseConv3d(16 * init_size, 16 * init_size, indice_key="inter2",
                                                        kernel_size=3, bias=False, algo=ConvAlgo.Native)
        self.interDecoder3 = spconv.SparseInverseConv3d(8 * init_size, 8 * init_size, indice_key="inter1",
                                                        kernel_size=3, bias=False, algo=ConvAlgo.Native)

        self.sidea1 = spconv.SparseInverseConv3d(16 * init_size, 16 * init_size, indice_key="inter2", kernel_size=3,
                                                 bias=False, algo=ConvAlgo.Native)
        self.sidea2 = spconv.SparseInverseConv3d(16 * init_size, 16 * init_size, indice_key="inter1", kernel_size=3,
                                                 bias=False, algo=ConvAlgo.Native)

        self.sideb1 = spconv.SparseInverseConv3d(8 * init_size, 8 * init_size, indice_key="inter1", kernel_size=3,
                                                 bias=False, algo=ConvAlgo.Native)

        self.logits = spconv.SubMConv3d(28 * init_size, nclasses, indice_key="logit", kernel_size=1, stride=1,
                                        padding=1,
                                        bias=True, algo=ConvAlgo.Native)

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        x1 = self.encoder1.forward(ret)
        x2i = self.interEncoder1.forward(x1)
        x2o = self.encoder2.forward(x2i)
        x3i = self.interEncoder2.forward(x2o)
        x3o = self.encoder3.forward(x3i)
        x4i = self.interEncoder3.forward(x3o)
        x4o = self.encoder4.forward(x4i)
        y1i = self.interDecoder1.forward(x4o)
        y1i = y1i.replace_feature(torch.cat((y1i.features, x3o.features), 1))
        y1o = self.decoder1.forward(y1i)
        zy11 = self.sidea1.forward(y1o)
        zy12 = self.sidea2.forward(zy11)
        y2i = self.interDecoder2.forward(y1o)
        y2i = y2i.replace_feature(torch.cat((y2i.features, x2o.features), 1))
        y2o = self.decoder2.forward(y2i)
        zy21 = self.sideb1.forward(y2o)
        y3i = self.interDecoder3.forward(y2o)
        y3i = y3i.replace_feature(torch.cat((y3i.features, x1.features), 1))
        y3o = self.decoder3.forward(y3i)
        z = torch.cat((y3o.features, zy21.features, zy12.features), 1)
        ret = ret.replace_feature(z)
        logits = self.logits(ret)

        return [logits]

class U2NETP(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=128,
                 nclasses=20, init_size=16):
        super(U2NET, self).__init__()
        sparse_shape = np.array(output_shape)
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.nclasses = nclasses
        self.encoder1 = encoder1(num_input_features, init_size)
        self.encoder2 = encoder2(4 * init_size, init_size)
        self.encoder3 = encoder3(8 * init_size, init_size)
        self.encoder4 = encoder4(16 * init_size, init_size)
        self.decoder1 = decoder2(48 * init_size, init_size)
        self.decoder2 = decoder3(24 * init_size, init_size)
        self.decoder3 = decoder4(12 * init_size, init_size)
        self.interEncoder1 = spconv.SparseMaxPool3d(indice_key="inter1", kernel_size=3, algo=ConvAlgo.Native)
        self.interEncoder2 = spconv.SparseMaxPool3d(indice_key="inter2", kernel_size=3, algo=ConvAlgo.Native)
        self.interEncoder3 = spconv.SparseMaxPool3d(indice_key="inter3", kernel_size=3, algo=ConvAlgo.Native)
        self.interDecoder1 = spconv.SparseInverseConv3d(32 * init_size, 32 * init_size, indice_key="inter3",
                                                        kernel_size=3, bias=False, algo=ConvAlgo.Native)
        self.interDecoder2 = spconv.SparseInverseConv3d(16 * init_size, 16 * init_size, indice_key="inter2",
                                                        kernel_size=3, bias=False, algo=ConvAlgo.Native)
        self.interDecoder3 = spconv.SparseInverseConv3d(8 * init_size, 8 * init_size, indice_key="inter1",
                                                        kernel_size=3, bias=False, algo=ConvAlgo.Native)

        self.sidea1 = spconv.SparseInverseConv3d(16 * init_size, 16 * init_size, indice_key="inter2", kernel_size=3,
                                                 bias=False, algo=ConvAlgo.Native)
        self.sidea2 = spconv.SparseInverseConv3d(16 * init_size, 16 * init_size, indice_key="inter1", kernel_size=3,
                                                 bias=False, algo=ConvAlgo.Native)

        self.sideb1 = spconv.SparseInverseConv3d(8 * init_size, 8 * init_size, indice_key="inter1", kernel_size=3,
                                                 bias=False, algo=ConvAlgo.Native)

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=1, stride=1,
                                        padding=1,
                                        bias=True, algo=ConvAlgo.Native)

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        x1 = self.encoder1.forward(ret)
        x2i = self.interEncoder1.forward(x1)
        x2o = self.encoder2.forward(x2i)
        x3i = self.interEncoder2.forward(x2o)
        x3o = self.encoder3.forward(x3i)
        x4i = self.interEncoder3.forward(x3o)
        x4o = self.encoder4.forward(x4i)
        y1i = self.interDecoder1.forward(x4o)
        y1i = y1i.replace_feature(torch.cat((y1i.features, x3o.features), 1))
        y1o = self.decoder1.forward(y1i)
        y2i = self.interDecoder2.forward(y1o)
        y2i = y2i.replace_feature(torch.cat((y2i.features, x2o.features), 1))
        y2o = self.decoder2.forward(y2i)
        y3i = self.interDecoder3.forward(y2o)
        y3i = y3i.replace_feature(torch.cat((y3i.features, x1.features), 1))
        y3o = self.decoder3.forward(y3i)
        logits = self.logits(y3o)

        return [logits]



class UNET(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=128,
                 nclasses=20,  init_size=16):
        super(UNET, self).__init__()
        sparse_shape = np.array(output_shape)
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.nclasses = nclasses
        self.encoder1 = encoder1(num_input_features, init_size)

        self.logits = spconv.SubMConv3d(4*init_size, nclasses, indice_key="logit", kernel_size=1, stride=1,
                                        padding=1,
                                        bias=True, algo=ConvAlgo.Native)
    def forward(self, voxel_features, coors, batch_size):
       coors = coors.int()
       ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
       x1=self.encoder1.forward(ret)
       logits= self.logits(x1)

       y= [logits.dense()]
       return y