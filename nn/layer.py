"""Hyper Class to define a Layer component of Neural Network

This module contains the creation of different layers.

The module structure is the following:

- The ``Layer`` abstract base class is the main definition of
  the necessary functions in order to properly define a layer

---------------------------------------------------------------------
                              GENERAL

- ``conv2DBatchNormRelu`` definition of the generic ReLu activation
  layer for 2D convolution architecture of Neural Network

---------------------------------------------------------------------
                              SEGNET
            SegNet: A Deep Convolutional Encoder-Decoder
                Architecture for Image Segmentation

Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla, Senior Member, IEEE

- ``SegnetLayer_Decoder`` definition of the Decoder Layer from SegNet
  Neural Network Architecture
- ``SegnetLayer_Encoder`` definition of the Encoder Layer from SegNet
  Neural Network Architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn


class Layer(nn.Module):
    """Abstract Base Class to ensure the optimal quantity of functions."""
    def __init__(self):
        super(Layer, self).__init__()
        pass

    def forward(self):
        pass


"""GENERAL"""


class conv2DBatchNormRelu(nn.Module):
    """Derived Class to define an Encoder Layer of Segnet Architecture

    Attributes
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

    bias : bool
        The toggle parameter for the bias activation.

    dilation : int
        The dilation parameter.

    """
    def __init__(self, in_channels,
                 n_filters, k_size,
                 stride, padding, bias=True,
                 dilation=1):
        """Preprocessing the Sequence"""
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters),
                             kernel_size=k_size, padding=padding,
                             stride=stride, bias=bias, dilation=1)

        self.cbr_unit = nn.Sequential(conv_mod,
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        """Processing the initialied sequence - See PyTorch Doc"""
        outputs = self.cbr_unit(inputs)
        return outputs


"""SEGNET"""


class SegnetLayer_Decoder(nn.Module):
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
    SegNet: A Deep Convolutional Encoder-Decoder Architecture
    for Image Segmentation
    Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla, Senior Member, IEEE,
    """
    def __init__(self, in_size, out_size, layer_size):
        super(SegnetLayer_Decoder, self).__init__()
        if layer_size == 2:
            self.unpool = nn.MaxUnpool2d(2, 2)
            self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        else:
            self.unpool = nn.MaxUnpool2d(2, 2)
            self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

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


class SegnetLayer_Encoder(nn.Module):
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
    SegNet: A Deep Convolutional Encoder-Decoder Architecture
    for Image Segmentation
    Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla, Senior Member, IEEE,
    """
    def __init__(self, in_size, out_size, layer_size):
        super(SegnetLayer_Encoder, self).__init__()

        if layer_size == 2:
            self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

        else:
            self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
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


"""UPNET"""


class UpNetLayer_Encoder(nn.Module):
    """Derived Class to define an Encoder Layer of UpNet Architecture

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
    Efficient Deep Models for Monocular Road Segmentation
    Gabriel L. Oliveira, Wolfram Burgard and Thomas Brox
    """
    def __init__(self, in_size, out_size, layer_size):
        super(UpNetLayer_Encoder, self).__init__()

        if layer_size == 2:
            self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=2,
                                                    padding=1,
                                                    stride=2,
                                                    return_indices=True)

        else:
            self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=2,
                                                    padding=1,
                                                    stride=2,
                                                    return_indices=True)

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


class UpNetLayer_ParticularEncoder(nn.Module):
    """Derived Class to define an Encoder Layer of UpNet Architecture

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
    Efficient Deep Models for Monocular Road Segmentation
    Gabriel L. Oliveira, Wolfram Burgard and Thomas Brox
    """
    def __init__(self, in_size, out_size, layer_size):
        super(UpNetLayer_ParticularEncoder, self).__init__()

        if layer_size == 2:
            self.conv1 = conv2DBatchNormRelu(in_size, out_size, 1, 1, 0)
            self.conv2 = conv2DBatchNormRelu(out_size, out_size, 1, 1, 0)
            self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=1,
                                                    padding=0,
                                                    stride=2,
                                                    return_indices=True)
        else:
            self.conv1 = conv2DBatchNormRelu(in_size, out_size, 1, 1, 0)
            self.conv2 = conv2DBatchNormRelu(out_size, out_size, 1, 1, 0)
            self.conv3 = conv2DBatchNormRelu(out_size, out_size, 1, 1, 0)
            self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=1,
                                                    padding=0,
                                                    stride=2,
                                                    return_indices=True)

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


class UpNetLayer_ParticularEncoder_2(nn.Module):
    """Derived Class to define an Encoder Layer of UpNet Architecture

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
    Efficient Deep Models for Monocular Road Segmentation
    Gabriel L. Oliveira, Wolfram Burgard and Thomas Brox
    """
    def __init__(self, in_size, out_size, layer_size):
        super(UpNetLayer_ParticularEncoder_2, self).__init__()

        if layer_size == 2:
            self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 10)
            self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=2,
                                                    padding=0,
                                                    stride=2,
                                                    return_indices=True)
        else:
            self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 10)
            self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=2,
                                                    padding=0,
                                                    stride=2,
                                                    return_indices=True)

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


class UpNetLayer_Dropout(nn.Module):
    """Derived Class to define a Dropout Layer of UpNet Architecture

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
    Efficient Deep Models for Monocular Road Segmentation
    Gabriel L. Oliveira, Wolfram Burgard and Thomas Brox
    """
    def __init__(self):
        super(UpNetLayer_Dropout, self).__init__()

        self.dropout = nn.Dropout2d(p=0.25, inplace=False)

    def forward(self, inputs):
        """Processing in Sequential - See PyTorch Doc"""
        outputs = self.dropout(inputs)

        return outputs


class UpNetLayer_Decoder_Particular(nn.Module):
    """Derived Class to define a Decoder Layer of UpNet Architecture

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
    Efficient Deep Models for Monocular Road Segmentation
    Gabriel L. Oliveira, Wolfram Burgard and Thomas Brox
    """
    def __init__(self, in_size, out_size, layer_size):
        super(UpNetLayer_Decoder_Particular, self).__init__()
        if layer_size == 2:
            self.unpool = nn.MaxUnpool2d(2, padding=0, stride=2)
            self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        else:
            self.unpool = nn.MaxUnpool2d(2, padding=0, stride=2)
            self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, layer_size):
        """Processing in Sequential - See PyTorch Doc"""
        if layer_size == 2:
            outputs = self.unpool(input=inputs, indices=indices)
            outputs = self.conv1(outputs)
            outputs = self.conv2(outputs)

        else:
            outputs = self.unpool(input=inputs, indices=indices)
            outputs = self.conv1(outputs)
            outputs = self.conv2(outputs)
            outputs = self.conv3(outputs)

        return outputs


class UpNetLayer_Decoder(nn.Module):
    """Derived Class to define a Decoder Layer of UpNet Architecture

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
    Efficient Deep Models for Monocular Road Segmentation
    Gabriel L. Oliveira, Wolfram Burgard and Thomas Brox
    """
    def __init__(self, in_size, out_size, layer_size):
        super(UpNetLayer_Decoder, self).__init__()
        if layer_size == 2:
            self.unpool = nn.MaxUnpool2d(1, padding=0, stride=2)
            self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        else:
            self.unpool = nn.MaxUnpool2d(1, padding=0, stride=2)
            self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, layer_size):
        """Processing in Sequential - See PyTorch Doc"""
        if layer_size == 2:
            outputs = self.unpool(input=inputs, indices=indices)
            outputs = self.conv1(outputs)
            outputs = self.conv2(outputs)

        else:
            outputs = self.unpool(input=inputs, indices=indices)
            outputs = self.conv1(outputs)
            outputs = self.conv2(outputs)
            outputs = self.conv3(outputs)

        return outputs


class UpNetLayer_Decoder_Particular_2(nn.Module):
    """Derived Class to define a Decoder Layer of UpNet Architecture

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
    Efficient Deep Models for Monocular Road Segmentation
    Gabriel L. Oliveira, Wolfram Burgard and Thomas Brox
    """
    def __init__(self, in_size, out_size, layer_size):
        super(UpNetLayer_Decoder_Particular_2, self).__init__()
        if layer_size == 2:
            self.unpool = nn.MaxUnpool2d(2, padding=0, stride=2)
            self.conv1 = conv2DBatchNormRelu(in_size, in_size, 10, 1, 0)
            self.conv2 = conv2DBatchNormRelu(in_size, out_size, 10, 1, 0)
        else:
            self.unpool = nn.MaxUnpool2d(2, padding=0, stride=2)
            self.conv1 = conv2DBatchNormRelu(in_size, in_size, 4, 1, 1)
            self.conv2 = conv2DBatchNormRelu(in_size, in_size, 4, 1, 1)
            self.conv3 = conv2DBatchNormRelu(in_size, out_size, 4, 1, 1)

    def forward(self, inputs, indices, layer_size):
        """Processing in Sequential - See PyTorch Doc"""
        if layer_size == 2:
            outputs = self.unpool(input=inputs, indices=indices)
            outputs = self.conv1(outputs)
            outputs = self.conv2(outputs)

        else:
            outputs = self.unpool(input=inputs, indices=indices)
            outputs = self.conv1(outputs)
            outputs = self.conv2(outputs)
            outputs = self.conv3(outputs)

        return outputs


"""UNET"""


class UNet_Encoder(nn.Module):
    """Derived Class to define a Encoder Layer of UNet Architecture

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
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    """
    def __init__(self, in_size, out_size):
        super(UNet_Encoder, self).__init__()

        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

        self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=2)

    def forward(self, inputs):
        """Processing in Sequential - See PyTorch Doc"""

        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.maxpool_with_argmax(outputs)

        return outputs


class UNet_Encoder_Particular(nn.Module):
    """Derived Class to define a Encoder Layer of UNet Architecture

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
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    """
    def __init__(self, in_size, out_size):
        super(UNet_Encoder_Particular, self).__init__()

        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        """Processing in Sequential - See PyTorch Doc"""

        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        return outputs


class UNet_Decoder(nn.Module):
    """Derived Class to define a Decoder Layer of Unet Architecture

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
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    """
    def __init__(self, in_size, out_size):
        super(UNet_Decoder, self).__init__()

        self.unpool = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

    def forward(self, x1, x2):
        x1 = self.unpool(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        outputs = torch.cat([x2, x1], dim=1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class UNet_Decoder_Particular(nn.Module):
    """Derived Class to define a Decoder Layer of Unet Architecture

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
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    """
    def __init__(self, in_size, out_size):
        super(UNet_Decoder_Particular, self).__init__()

        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 1, 1, 0)

    def forward(self, x):
        outputs = self.conv1(x)
        return outputs
