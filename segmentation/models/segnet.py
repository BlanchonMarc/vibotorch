import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd, optim

def _conv2DBatchNormRelu(in_channels, n_filters, k_size, stride, padding,
                         dilation=1, bias=True):
    """Private function to use in the decoder and encoder

    Parameters
    ----------
    in_channels : int
        The input size.

    n_filters : int
        The output size.

    k_size : int
        The size parameter ( convolution size ).

    stride : int
        The stride parameter ( convolution stride ).

    padding : int
        The padding parameter ( convolution padding ).

    dilation : int
        The dilation parameter.

    bias : bool
        The toggle parameter for the bias activation.

    Returns
    -------
    nn.Sequential unit composed of Cond2D, BatchNorm and Relu

    """
    conv_mod = nn.Conv2d(int(in_channels), int(n_filters),
                         kernel_size=k_size, padding=padding,
                         stride=stride, bias=bias, dilation=1)
    return nn.Sequential(conv_mod,
                         nn.BatchNorm2d(int(n_filters)),
                         nn.ReLU(inplace=True))

class decoder(nn.Module):
    """Derived Class to define a Decoder Layer of Segnet Architecture

    Attributes
    ----------
    in_size : int
        The input size of the network.

    out_size : int
        The output size of the network.

    layer_size : int
        The parameter defining the depth of the layer.

    References
    ----------
    SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image
            Segmentation
    Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla, Senior Member, IEEE,
    """
    def __init__(self, in_size, out_size, layer_size):
        if layer_size == 2:
            self.unpool = nn.MaxUnpool2d(2, 2)
            self.conv1 = _conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv2 = _conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        else:
            self.unpool = nn.MaxUnpool2d(2, 2)
            self.conv1 = _conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv2 = _conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv3 = _conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape, layer_size):
        """Processing in Sequential - See PyTorch Doc"""
        if layer_size == 2:
            outputs = self.unpool(input=inputs, indices=indices,
                                  output_size=output_shape)
            outputs = self.conv1(outputs)
            outputs = self.conv2(outputs)

        else:
            outputs = self.unpool(input=inputs, indices=indices,
                                  output_size=output_shape)
            outputs = self.conv1(outputs)
            outputs = self.conv2(outputs)
            outputs = self.conv3(outputs)

        return outputs


class encoder(nn.Module):
    """Derived Class to define an Encoder Layer of Segnet Architecture

    Attributes
    ----------
    in_size : int
        The input size of the network.

    out_size : int
        The output size of the network.

    layer_size : int
        The parameter defining the depth of the layer.

    References
    ----------
    SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image
            Segmentation
    Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla, Senior Member, IEEE,
    """
    def __init__(self, in_size, out_size, layer_size):

        if layer_size == 2:
            self.conv1 = _conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
            self.conv2 = _conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

        else:
            self.conv1 = _conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
            self.conv2 = _conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.conv3 = _conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs, layer_size):
        """Processing in Sequential - See PyTorch Doc"""
        if layer_size == 2:
            outputs = self.conv1(inputs)
            outputs = self.conv2(outputs)
            unpooled_shape = outputs.size()
            outputs, indices = self.maxpool_with_argmax(outputs)
        else:
            outputs = self.conv1(inputs)
            outputs = self.conv2(outputs)
            outputs = self.conv3(outputs)
            unpooled_shape = outputs.size()
            outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnet(nn.Module):
    """Derived Class to define a SegNet architecture of nn

    Attributes
    ----------
    in_channels : int
        The input size of the network.

    n_classes : int
        The output size of the network.

    References
    ----------
    SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image
            Segmentation
    Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla, Senior Member, IEEE,
    """
    def __init__(self, in_channels, n_classes):
        """Sequential Instanciation of the different Layers"""

        self.layer_1 = encoder(in_channels, 64, 2)
        self.layer_2 = encoder(64, 128, 2)
        self.layer_3 = encoder(128, 256, 3)
        self.layer_4 = encoder(256, 512, 3)
        self.layer_5 = encoder(512, 512, 3)

        self.layer_6 = decoder(512, 512, 3)
        self.layer_7 = decoder(512, 256, 3)
        self.layer_8 = decoder(256, 128, 3)
        self.layer_9 = decoder(128, 64, 2)
        self.layer_10 = decoder(64, n_classes, 2)

    def forward(self, inputs):
        """Sequential Computation, see nn.Module.forward methods PyTorch"""

        down1, indices_1, unpool_shape1 = self.layer_1(inputs=inputs,
                                                       layer_size=2)
        down2, indices_2, unpool_shape2 = self.layer_2(inputs=down1,
                                                       layer_size=2)
        down3, indices_3, unpool_shape3 = self.layer_3(inputs=down2,
                                                       layer_size=3)
        down4, indices_4, unpool_shape4 = self.layer_4(inputs=down3,
                                                       layer_size=3)
        down5, indices_5, unpool_shape5 = self.layer_5(inputs=down4,
                                                       layer_size=3)

        up5 = self.layer_6(inputs=down5, indices=indices_5,
                           output_shape=unpool_shape5, layer_size=3)
        up4 = self.layer_7(inputs=up5, indices=indices_4,
                           output_shape=unpool_shape4, layer_size=3)
        up3 = self.layer_8(inputs=up4, indices=indices_3,
                           output_shape=unpool_shape3, layer_size=3)
        up2 = self.layer_9(inputs=up3, indices=indices_2,
                           output_shape=unpool_shape2, layer_size=2)
        output = self.layer_10(inputs=up2, indices=indices_1,
                               output_shape=unpool_shape1, layer_size=2)

        return output
