# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from IPython.display import display
from PIL import Image
from torch.cuda import amp

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version, colorstr,
                           increment_path, is_notebook, make_divisible, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, yaml_load)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, smart_inference_mode


def autopad(k, p=None, d=1):  # 定义一个名为 autopad 的函数，接收三个参数：kernel（k），padding（p），和 dilation（d）
    # 用于计算在神经网络中使卷积核输出形状保持不变的填充大小
    if d > 1:
        # 如果dilation（d）大于1，计算实际的kernel大小
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 计算实际的卷积核大小
    if p is None:
        # 如果没有指定padding（p），则自动计算padding大小
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动计算padding大小，使得输出形状保持不变
    return p  # 返回计算得到的padding大小
'''
在卷积神经网络（CNN）中，`dilation`（扩张）是一个参数，用于定义卷积核（kernel）中元素的间隔。
它是一种增加卷积层感受野（即卷积层可以观察到的输入数据的区域大小）的技术，而不增加卷积核中的参数数量。

简单来说，`dilation`决定了在执行卷积操作时，卷积核内元素之间跳过的输入元素的数量。例如：

- 当 `dilation=1` 时（标准卷积），卷积核的每个元素都紧密相邻，不跳过任何输入元素。
- 当 `dilation=2` 时，卷积核的元素之间会跳过一个输入元素，这样卷积核覆盖的输入区域更广，但卷积核本身的大小不变。

使用扩张卷积可以帮助模型更好地捕捉输入数据中的空间层次结构，尤其是在图像和音频处理任务中。
这种方法尤其在处理较大的输入或需要更大感受野的场景中非常有效。

当使用一个 3x3 的卷积核，并且 `dilation=2` 时，卷积核覆盖的实际范围会变大。
在标准卷积（`dilation=1`）中，3x3 的卷积核覆盖 3x3 的区域。
但是，当 `dilation=2` 时，卷积核的元素之间会有一个元素的间隔。

为了具体说明，我们可以用以下方式可视化卷积核的覆盖范围：

- 在 `dilation=1` 的情况下（标准卷积），3x3 卷积核的布局（`*` 表示卷积核的元素）：
  ```
  * * *
  * * *
  * * *
  ```
- 在 `dilation=2` 的情况下，3x3 卷积核的布局变为（`*` 表示卷积核的元素，`.` 表示跳过的元素）：
  ```
  * . * . *
  . . . . .
  * . * . *
  . . . . .
  * . * . *
  ```
在这个扩张卷积的例子中，卷积核实际上覆盖了一个 5x5 的区域，但只在 9（3x3）个位置上有卷积核的权重（即计算）。这使得卷积核能够在不增加权重的情况下覆盖更大的输入区域。
'''


class Conv(nn.Module):  # 定义一个名为 Conv 的类，继承自 PyTorch 的 nn.Module
    # 标准卷积层，参数包括输入通道数(ch_in)，输出通道数(ch_out)，卷积核大小(kernel)，步长(stride)，填充(padding)，分组(groups)，空洞(dilation)，激活函数(activation)
    default_act = nn.SiLU()  # 默认激活函数为 SiLU

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        # 初始化函数，设置卷积层的参数
        super().__init__()  # 调用父类的初始化函数
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # 创建一个2D卷积层，包括输入通道数、输出通道数、卷积核大小、步长、自动填充、分组、空洞以及不使用偏置项
        self.bn = nn.BatchNorm2d(c2)  # 创建一个批量归一化层
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # 设置激活函数，如果act为True则使用默认激活函数，如果act是nn.Module的实例则使用act，否则不使用激活函数（恒等映射）

    def forward(self, x):
        # 前向传播函数
        return self.act(self.bn(self.conv(x)))  # 将输入x通过卷积层、批量归一化层，然后应用激活函数

    def forward_fuse(self, x):
        # 另一个前向传播函数，用于融合卷积层和激活函数
        return self.act(self.conv(x))  # 将输入x通过卷积层，然后应用激活函数



class DWConv(Conv):  # 定义一个名为 DWConv 的类，继承自之前定义的 Conv 类
    # 深度可分离卷积层
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # 构造函数，参数包括输入通道数（c1），输出通道数（c2），卷积核大小（k），步长（s），空洞（d），激活函数（act）
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        # 调用父类 Conv 的构造函数
        # 使用 math.gcd(c1, c2) 来确定分组数量（g），这里 g 是输入通道数和输出通道数的最大公约数
        # 其他参数传递给父类构造函数
'''
`DWConv`（深度可分离卷积，Depthwise Separable Convolution）和 `Conv`（标准卷积）是两种不同的卷积方式，它们在结构和计算效率上有显著的区别：

1. **标准卷积（`Conv`）**：
   - 在标准卷积中，每个输出通道是由所有输入通道上的卷积核生成的。
   - 例如，如果有 32 个输入通道和 64 个输出通道，每个输出通道的卷积核将在所有 32 个输入通道上应用。
   - 这种方法在特征提取方面非常有效，但它涉及大量的计算，因为每个输出通道都需要与每个输入通道进行卷积运算。

2. **深度可分离卷积（`DWConv`）**：
   - 深度可分离卷积将标准卷积分解为两个步骤：深度卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution）。
   - **深度卷积**：在深度卷积步骤中，对每个输入通道分别应用一个卷积核。这意味着每个输入通道只和自己的卷积核进行运算，而不与其他输入通道的卷积核交互。
   - **逐点卷积**：接着，逐点卷积使用 1x1 的卷积核来组合深度卷积的输出，生成最终的输出通道。
   - 这种方法显著减少了计算量和参数数量，因为深度卷积不需要在所有输入通道上进行全连接的卷积运算。

总结来说，`DWConv` 通过分解卷积过程，有效地降低了计算复杂性和模型大小，同时保持了足够的特征提取能力。
这使得它在设计轻量级和高效的卷积神经网络架构时非常有用，特别是在资源受限的环境（如移动设备）中。
相比之下，`Conv` 提供了更全面的特征提取能力，但代价是更高的计算和参数成本。
'''


class DWConvTranspose2d(nn.ConvTranspose2d):  # 定义一个名为 DWConvTranspose2d 的类，继承自 PyTorch 的 nn.ConvTranspose2d
    # 深度可分离的转置卷积层
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # 构造函数，参数包括输入通道数（c1），输出通道数（c2），卷积核大小（k），步长（s），输入填充（p1），输出填充（p2）
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))
        # 调用父类 nn.ConvTranspose2d 的构造函数
        # 设置分组数量为输入通道数和输出通道数的最大公约数，实现深度卷积的效果
'''
转置卷积（有时也被称为逆卷积或反卷积）通常用于卷积神经网络中的上采样操作，它的作用是将输入的特征图放大到更高的空间分辨率。
'''


class TransformerLayer(nn.Module):  # 定义一个名为 TransformerLayer 的类，继承自 PyTorch 的 nn.Module
    # Transformer 层的实现（移除了 LayerNorm 层以提高性能）

    def __init__(self, c, num_heads):
        # 初始化函数，参数包括通道数（c）和多头注意力的头数（num_heads）
        super().__init__()  # 调用父类的初始化函数
        self.q = nn.Linear(c, c, bias=False)  # 定义查询（Query）的线性变换层
        self.k = nn.Linear(c, c, bias=False)  # 定义键（Key）的线性变换层
        self.v = nn.Linear(c, c, bias=False)  # 定义值（Value）的线性变换层
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)  # 定义多头自注意力层
        self.fc1 = nn.Linear(c, c, bias=False)  # 定义前馈网络中的第一个线性层
        self.fc2 = nn.Linear(c, c, bias=False)  # 定义前馈网络中的第二个线性层

    def forward(self, x):
        # 前向传播函数
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x  # 将输入x通过q、k、v线性层，然后传入多头自注意力层，并与原始输入相加（残差连接）
        x = self.fc2(self.fc1(x)) + x  # 将自注意力的输出通过前馈网络，并与自注意力层的输出相加（残差连接）
        return x  # 返回最终的输出
'''
这些层的结合提供了 Transformer 架构的两个关键能力：自注意力和前馈网络处理。
自注意力使模型能够关注输入中的不同部分并学习它们之间的关系，而前馈网络则进一步处理这些信息，允许更复杂的数据表示。
残差连接在整个过程中保持信息流动，并帮助缓解深层网络中的梯度消失问题。
'''


class TransformerBlock(nn.Module):  # 定义一个名为 TransformerBlock 的类，继承自 PyTorch 的 nn.Module
    # Vision Transformer 实现，参考论文：https://arxiv.org/abs/2010.11929

    def __init__(self, c1, c2, num_heads, num_layers):
        # 初始化函数，参数包括输入通道数（c1），输出通道数（c2），多头注意力的头数（num_heads）和 Transformer 层的层数（num_layers）
        super().__init__()  # 调用父类的初始化函数
        self.conv = None  # 初始化卷积层为 None
        if c1 != c2:
            self.conv = Conv(c1, c2)  # 如果输入和输出通道数不同，则使用卷积层调整通道数
        self.linear = nn.Linear(c2, c2)  # 定义一个线性层，用于学习位置嵌入
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        # 创建多个 Transformer 层的序列
        self.c2 = c2  # 保存输出通道数

    def forward(self, x):
        # 前向传播函数
        if self.conv is not None:
            x = self.conv(x)  # 如果存在卷积层，先通过卷积层调整通道数
        b, _, w, h = x.shape  # 获取输入的批大小（b），宽度（w）和高度（h）
        p = x.flatten(2).permute(2, 0, 1)  # 将输入扁平化并变换维度，准备输入到 Transformer 层
        # 将位置嵌入加到扁平化后的输入上，并通过 Transformer 层序列
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)
        # 最后将输出的维度变换回原始的维度格式并返回

'''
'''

class Bottleneck(nn.Module):  # 定义一个名为 Bottleneck 的类，继承自 PyTorch 的 nn.Module
    # 标准的瓶颈结构

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # 构造函数，参数包括输入通道数（c1），输出通道数（c2），是否使用shortcut，分组数（g），扩展因子（e）
        super().__init__()  # 调用父类的初始化函数
        c_ = int(c2 * e)  # 计算隐藏层的通道数，这是输出通道数与扩展因子的乘积
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层，使用1x1卷积核，用于降低维度
        self.cv2 = Conv(c_, c2, 3, 1, g=g)  # 第二个卷积层，使用3x3卷积核，恢复维度，可以有分组
        self.add = shortcut and c1 == c2  # 判断是否添加shortcut连接（残差连接），条件是c1和c2相等且shortcut为True

    def forward(self, x):
        # 前向传播函数
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        # 如果启用shortcut且输入和输出通道数相等，则将输入和cv2的输出相加，否则只返回cv2的输出



class BottleneckCSP(nn.Module):  # 定义一个名为 BottleneckCSP 的类，继承自 PyTorch 的 nn.Module
    # CSP瓶颈结构，来源于 Cross Stage Partial Networks

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # 构造函数，参数包括输入通道数（c1），输出通道数（c2），Bottleneck层数（n），是否使用shortcut，分组数（g），扩展因子（e）
        super().__init__()  # 调用父类的初始化函数
        c_ = int(c2 * e)  # 计算隐藏层的通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层，1x1卷积，用于降低维度
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)  # 第二个卷积层，1x1卷积，同样用于降低维度
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)  # 第三个卷积层，1x1卷积，用于处理通过Bottleneck的特征
        self.cv4 = Conv(2 * c_, c2, 1, 1)  # 第四个卷积层，1x1卷积，用于合并特征后的降维
        self.bn = nn.BatchNorm2d(2 * c_)  # 批量归一化层，应用于合并后的特征
        self.act = nn.SiLU()  # 激活函数 SiLU
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # 创建多个Bottleneck层，形成序列

    def forward(self, x):
        # 前向传播函数
        y1 = self.cv3(self.m(self.cv1(x)))  # 将输入通过cv1、Bottleneck序列和cv3
        y2 = self.cv2(x)  # 将输入通过cv2
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))
        # 将y1和y2合并（通道维度上连接），然后通过批量归一化和激活函数，最后通过cv4输出



class CrossConv(nn.Module):  # 定义一个名为 CrossConv 的类，继承自 PyTorch 的 nn.Module
    # 交叉卷积下采样

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # 构造函数，参数包括输入通道数（c1），输出通道数（c2），卷积核大小（k），步长（s），分组数（g），扩展因子（e），是否使用shortcut
        super().__init__()  # 调用父类的初始化函数
        c_ = int(c2 * e)  # 计算隐藏层的通道数
        self.cv1 = Conv(c1, c_, (1, k), (1, s))  # 第一个卷积层，使用(1, k)大小的卷积核，主要沿着一个方向卷积
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)  # 第二个卷积层，使用(k, 1)大小的卷积核，沿着垂直于第一个卷积层的方向卷积
        self.add = shortcut and c1 == c2  # 判断是否添加shortcut连接（残差连接），条件是c1和c2相等且shortcut为True

    def forward(self, x):
        # 前向传播函数
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        # 如果启用shortcut且输入和输出通道数相等，则将输入和cv2的输出相加，否则只返回cv2的输出



class C3(nn.Module):  # 定义一个名为 C3 的类，继承自 PyTorch 的 nn.Module
    # 带有三个卷积层的 CSP 瓶颈结构

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # 构造函数，参数包括输入通道数（c1），输出通道数（c2），Bottleneck层数（n），是否使用shortcut，分组数（g），扩展因子（e）
        super().__init__()  # 调用父类的初始化函数
        c_ = int(c2 * e)  # 计算隐藏层的通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层，1x1卷积，用于降低维度
        self.cv2 = Conv(c1, c_, 1, 1)  # 第二个卷积层，1x1卷积，同样用于降低维度
        self.cv3 = Conv(2 * c_, c2, 1)  # 第三个卷积层，1x1卷积，用于合并特征后的降维
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # 创建多个Bottleneck层，形成序列

    def forward(self, x):
        # 前向传播函数
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        # 将输入x分别通过cv1和Bottleneck序列，以及cv2，然后在通道维度上合并，最后通过cv3输出



class C3x(C3):  # 定义一个名为 C3x 的类，它继承自 C3 类
    # 带有交叉卷积的 C3 模块

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        # 构造函数，参数与 C3 类相同
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类 C3 的构造函数
        c_ = int(c2 * e)  # 计算隐藏层的通道数，这里使用了扩展因子 e
        # 使用 CrossConv 替换 C3 类中的 Bottleneck 层，创建一个新的 nn.Sequential
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))



class C3TR(C3):  # 定义一个名为 C3TR 的类，它继承自 C3 类
    # 带有 TransformerBlock 的 C3 模块

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        # 构造函数，参数包括输入通道数（c1），输出通道数（c2），TransformerBlock 的层数（n），是否使用 shortcut 连接，分组数（g），扩展因子（e）
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类 C3 的构造函数
        c_ = int(c2 * e)  # 计算隐藏层的通道数，这是输出通道数乘以扩展因子 e
        # 使用 TransformerBlock 替换 C3 类中的 Bottleneck 层，创建一个 TransformerBlock 实例
        self.m = TransformerBlock(c_, c_, 4, n)  # 参数4表示 TransformerBlock 中多头注意力的头数



class C3SPP(C3):  # 定义一个名为 C3SPP 的类，它继承自 C3 类
    # 带有 SPP (空间金字塔池化) 模块的 C3 模块

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        # 构造函数，参数包括输入通道数（c1），输出通道数（c2），SPP内核大小列表（k），层数（n），是否使用shortcut连接，分组数（g），以及扩展因子（e）
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类 C3 的构造函数
        c_ = int(c2 * e)  # 计算隐藏层的通道数，这是输出通道数乘以扩展因子 e
        # 替换 C3 类中的 Bottleneck 层，使用 SPP 模块来增强特征提取
        self.m = SPP(c_, c_, k)  # 使用给定的核大小列表 k 初始化 SPP 模块



class C3Ghost(C3):  # 定义一个名为 C3Ghost 的类，它继承自 C3 类
    # 带有 GhostBottleneck 的 C3 模块

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        # 构造函数，参数包括输入通道数（c1），输出通道数（c2），GhostBottleneck 层数（n），是否使用 shortcut 连接，分组数（g），扩展因子（e）
        super().__init__(c1, c2, n, shortcut, g, e)  # 调用父类 C3 的构造函数
        c_ = int(c2 * e)  # 计算隐藏层的通道数，这是输出通道数乘以扩展因子 e
        # 使用 nn.Sequential 创建一个序列，包含 n 个 GhostBottleneck 模块
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))



class SPP(nn.Module):  # 定义一个名为 SPP 的类，继承自 PyTorch 的 nn.Module
    # 空间金字塔池化（SPP）层

    def __init__(self, c1, c2, k=(5, 9, 13)):
        # 构造函数，参数包括输入通道数（c1），输出通道数（c2），以及一系列的池化核大小（k）
        super().__init__()  # 调用父类的初始化函数
        c_ = c1 // 2  # 计算隐藏层的通道数，为输入通道数的一半
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层，使用1x1卷积核，用于降低维度
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        # 第二个卷积层，用于合并 SPP 层后的特征，并调整通道数
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        # 创建一个模块列表，包含不同大小的最大池化层

    def forward(self, x):
        # 前向传播函数
        x = self.cv1(x)  # 通过第一个卷积层
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # 忽略 torch 1.9.0 max_pool2d() 的警告
            # 通过一系列最大池化层，然后将结果与原始特征图拼接
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
            # 通过第二个卷积层，合并特征并调整通道数
'''
SPP 层能够从输入特征图中捕获不同尺度的空间信息，这在处理尺度变化较大的视觉任务中非常有用。
通过在不同尺度上池化，SPP 层能够增强模型对尺度变化的适应性。
'''


class SPPF(nn.Module):  # 定义一个名为 SPPF 的类，继承自 PyTorch 的 nn.Module
    # 快速空间金字塔池化（SPPF）层，用于 YOLOv5

    def __init__(self, c1, c2, k=5):  # 构造函数，参数包括输入通道数（c1），输出通道数（c2），以及池化核大小（k）
        super().__init__()  # 调用父类的初始化函数
        c_ = c1 // 2  # 计算隐藏层的通道数，为输入通道数的一半
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层，使用1x1卷积核，用于降低维度
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # 第二个卷积层，用于合并特征并调整通道数
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # 定义一个最大池化层，使用给定的核大小 k

    def forward(self, x):
        # 前向传播函数
        x = self.cv1(x)  # 通过第一个卷积层
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # 忽略 torch 1.9.0 max_pool2d() 的警告
            y1 = self.m(x)  # 对x应用一次最大池化
            y2 = self.m(y1)  # 对y1再次应用最大池化
            # 将原始特征图x与两次池化的结果y1, y2以及对y2再次池化的结果进行拼接
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
            # 通过第二个卷积层合并特征并调整通道数
'''
这种方法相比于传统的 SPP 层（使用多个不同大小的池化层）更加高效，因为它仅使用单个池化层多次应用于输入。
这种设计在提升效率的同时保留了捕获多尺度空间信息的能力，特别适合于实时物体检测任务。
'''


class Focus(nn.Module):  # 定义一个名为 Focus 的类，继承自 PyTorch 的 nn.Module
    # 将图像的宽度和高度信息聚焦到通道空间

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # 构造函数，参数包括输入通道数（c1），输出通道数（c2），卷积核大小（k），步长（s），填充（p），分组数（g）和是否使用激活函数（act）
        super().__init__()  # 调用父类的初始化函数
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)  # 初始化卷积层，将输入通道数乘以4

    def forward(self, x):  # 前向传播函数，x 的形状为 (b, c, w, h)
        # 将输入张量的子像素重新排列，然后在通道维度上拼接
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # 对拼接后的张量应用卷积
'''
Focus 层的设计旨在有效地聚焦图像的空间信息到通道维度，同时减少输入数据的空间维度（宽度和高度减半），这在提高网络的计算效率和性能方面是有益的。
通过一个具体的例子来说明 `Focus` 类的作用。假设我们有一个简单的输入张量 `x`，其形状为 `[1, 2, 4, 4]`，这里 1 是批量大小（batch size），2 是通道数，4x4 是宽度和高度。我们将用 `Focus` 层处理这个张量，并观察输出。

假设输入张量 `x` 如下所示（为了简化，我使用了较小的数字）：

```
张量 x 的形状：[1, 2, 4, 4]
内容（假设）：
第一个通道：  第二个通道：
1 2 3 4       5 6 7 8
5 6 7 8       1 2 3 4
9 0 1 2       9 0 1 2
3 4 5 6       3 4 5 6
```

`Focus` 层将执行以下操作：

1. 将输入张量 `x` 的每个通道分成2x2的块，并将每个块的元素重排到通道维度上。这将扩展通道的数量，同时减少宽度和高度的尺寸。

   - 从每个 2x2 区域提取左上角的元素（`::2, ::2`）。
   - 从每个 2x2 区域提取右上角的元素（`1::2, ::2`）。
   - 从每个 2x2 区域提取左下角的元素（`::2, 1::2`）。
   - 从每个 2x2 区域提取右下角的元素（`1::2, 1::2`）。

2. 将这些提取出的元素在通道维度上拼接，形成一个新的张量。

结果是，通道数增加了4倍（在这个例子中，从2增加到8），而宽度和高度各减半（从4x4变为2x2）。

最终，`Focus` 层输出的张量可能如下所示（假设没有进一步的卷积处理）：

```
输出张量的形状：[1, 8, 2, 2]
内容（按上述步骤重排）：
第1-4通道：       第5-8通道：
1 3   2 4        5 7   6 8
9 1   0 2        9 1   0 2

5 7   6 8        1 3   2 4
3 5   4 6        3 5   4 6
```

这个操作有效地将空间信息“聚焦”到通道维度，同时减少了数据的空间尺寸，这在许多深度学习模型中有助于减少计算量并提高效率。
特别是在处理图像数据时，这种方法可以帮助模型更有效地捕获和利用空间信息。
'''


class GhostConv(nn.Module):  # 定义一个名为 GhostConv 的类，继承自 PyTorch 的 nn.Module
    # Ghost 卷积

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # 构造函数，参数包括输入通道数（c1），输出通道数（c2），卷积核大小（k），步长（s），分组数（g），是否使用激活函数（act）
        super().__init__()  # 调用父类的初始化函数
        c_ = c2 // 2  # 计算隐藏层的通道数，为输出通道数的一半
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)  # 第一个卷积层，负责生成一半的特征图
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)  # 第二个卷积层，使用更大的卷积核生成剩余的特征图

    def forward(self, x):
        # 前向传播函数
        y = self.cv1(x)  # 通过第一个卷积层生成一半的特征图
        return torch.cat((y, self.cv2(y)), 1)  # 将第一个卷积层的输出与第二个卷积层的输出在通道维度上拼接
'''
Ghost 卷积（Ghost Convolution）是一种高效的卷积方法，旨在减少传统卷积操作的计算量而不牺牲网络的性能。
这种方法最初在 [GhostNet](https://github.com/huawei-noah/ghostnet) 研究中被提出，主要用于移动和计算效率敏感的应用。
以下是 Ghost 卷积的核心作用和原理：

### 核心作用：

1. **减少计算量**：Ghost 卷积通过减少卷积核的数量来降低传统卷积操作的计算量。
2. **维持性能**：尽管减少了计算量，但通过巧妙的设计，Ghost 卷积能够维持与传统卷积相似的性能水平。

### 原理：

Ghost 卷积的基本思想是将传统的卷积操作分成两个步骤：

1. **原始卷积**：首先使用较少的卷积核对输入特征图进行卷积，生成一部分特征图。
这一步类似于常规卷积，但使用的卷积核数量远少于传统卷积。

2. **廉价操作**：然后对上一步生成的特征图应用廉价的线性操作（如线性卷积、深度卷积或其他廉价变换），以生成剩余的特征图。
这一步不涉及大量乘法操作，因此计算成本较低。

### 结果：

这种方法生成的总特征图数量与传统卷积相同，但由于第二步使用的是计算成本较低的操作，整体上减少了计算量。Ghost 卷积特别适用于需要轻量级网络设计的场景，如移动设备、边缘计算和实时应用。

总的来说，Ghost 卷积的目的是在保持网络性能的同时，减少模型的计算复杂度和参数量，从而提高效率和速度。
'''


class GhostBottleneck(nn.Module):  # 定义一个名为 GhostBottleneck 的类，继承自 PyTorch 的 nn.Module
    # Ghost 瓶颈结构

    def __init__(self, c1, c2, k=3, s=1):  # 构造函数，参数包括输入通道数（c1），输出通道数（c2），卷积核大小（k），步长（s）
        super().__init__()  # 调用父类的初始化函数
        c_ = c2 // 2  # 计算隐藏层的通道数，为输出通道数的一半
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # 第一个 Ghost 卷积层，用于点卷积（pw）
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # 深度可分离卷积层（dw），当步长为2时使用
            GhostConv(c_, c2, 1, 1, act=False))  # 第二个 Ghost 卷积层，用于线性变换
        self.shortcut = nn.Sequential(
            DWConv(c1, c1, k, s, act=False),
            Conv(c1, c2, 1, 1, act=False)
        ) if s == 2 else nn.Identity()  # shortcut 路径，当步长为2时包含深度可分离卷积和1x1卷积

    def forward(self, x):
        # 前向传播函数
        return self.conv(x) + self.shortcut(x)  # 将 conv 路径的输出和 shortcut 路径的输出相加



class Contract(nn.Module):  # 定义一个名为 Contract 的类，继承自 PyTorch 的 nn.Module
    # 将宽度和高度的信息压缩到通道维度

    def __init__(self, gain=2):
        # 构造函数，参数 gain 表示空间维度减小的倍数
        super().__init__()  # 调用父类的初始化函数
        self.gain = gain  # 设置空间维度减小的倍数

    def forward(self, x):
        # 前向传播函数
        b, c, h, w = x.size()  # 获取输入张量的维度：批量大小（b），通道数（c），高度（h）和宽度（w）
        s = self.gain  # 获得空间压缩的倍数
        x = x.view(b, c, h // s, s, w // s, s)  # 重新排列张量，准备进行压缩
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # 改变张量的维度顺序，使其变为(b, s, s, c, h//s, w//s)
        return x.view(b, c * s * s, h // s, w // s)  # 将张量压缩成新的形状(b, c * s * s, h//s, w//s)



class Expand(nn.Module):  # 定义一个名为 Expand 的类，继承自 PyTorch 的 nn.Module
    # 将通道扩展到宽度和高度

    def __init__(self, gain=2):
        # 构造函数，参数 gain 表示空间维度增加的倍数
        super().__init__()  # 调用父类的初始化函数
        self.gain = gain  # 设置空间扩展的倍数

    def forward(self, x):
        # 前向传播函数
        b, c, h, w = x.size()  # 获取输入张量的维度：批量大小（b），通道数（c），高度（h）和宽度（w）
        s = self.gain  # 获得空间扩展的倍数
        x = x.view(b, s, s, c // s ** 2, h, w)  # 重新排列张量，准备进行扩展
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # 改变张量的维度顺序，使其变为(b, c // s ** 2, h, s, w, s)
        return x.view(b, c // s ** 2, h * s, w * s)  # 将张量扩展成新的形状(b, c // s ** 2, h * s, w * s)



class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


'''
DetectMultiBackend

1. **多后端检测**：名称中的 "Detect" 暗示了这个函数与检测任务相关，可能是涉及图像或视频中的对象检测。"MultiBackend" 表示它能够支持多种后端。
在机器学习和计算机视觉领域，"后端" 通常指的是执行计算的库或框架，如 PyTorch、TensorFlow、OpenCV 等。

2. **框架兼容性**：这个函数可能设计用来在不同的计算框架或库中执行相似的检测任务，自动选择或兼容多种后端。

3. **模型和硬件适配**：它可能包含逻辑来处理不同类型的模型（例如不同的神经网络架构），并且可能考虑到了运行模型的硬件环境（如 CPU、GPU 或特定的硬件加速器）。

4. **自动化和优化**：函数可能包括自动选择最优后端的功能，基于当前环境和可用资源（如内存和计算能力）来优化性能。

5. **接口统一**：为了处理多种后端，这个函数可能提供一个统一的接口，允许用户无需关心底层细节就能执行检测任务。

总的来说，`DetectMultiBackend` 是一个为了在不同的技术栈和硬件环境中执行对象检测任务而设计的通用、灵活的函数。
这样的设计使得它能够适应多种应用场景和性能需求。
'''
class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:  # load metadata dict
                d = json.loads(extra_files['config.txt'],
                               object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                      for k, v in d.items()})
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements('opencv-python>=4.5.4')
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements('openvino')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            executable_network = ie.compile_model(network, device_name="CPU")  # device_name="MYRIAD" for Intel NCS2
            stride, names = self._load_metadata(Path(w).with_suffix('.yaml'))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
            import tensorflow as tf
            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, 'rb') as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                delegate = {
                    'Linux': 'libedgetpu.so.1',
                    'Darwin': 'libedgetpu.1.dylib',
                    'Windows': 'edgetpu.dll'}[platform.system()]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta['stride']), meta['names']
        elif tfjs:  # TF.js
            raise NotImplementedError('ERROR: YOLOv5 TF.js inference is not supported')
        elif paddle:  # PaddlePaddle
            LOGGER.info(f'Loading {w} for PaddlePaddle inference...')
            check_requirements('paddlepaddle-gpu' if cuda else 'paddlepaddle')
            import paddle.inference as pdi
            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).rglob('*.pdmodel'))  # get *.pdmodel file from *_paddle_model dir
            weights = Path(w).with_suffix('.pdiparams')
            config = pdi.Config(str(w), str(weights))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f'Using {w} as Triton Inference Server...')
            check_requirements('tritonclient[all]')
            from utils.triton import TritonRemoteModel
            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            raise NotImplementedError(f'ERROR: {w} is not a supported format')

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.executable_network([im]).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output['index'])
                    if int8:
                        scale, zero_point = output['quantization']
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None

'''
AutoShape 类是一个为 YOLOv5 模型提供灵活输入处理和后处理的工具，使其能够在不同的数据源和环境中进行对象检测。
'''
class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    '''
    conf：设置用于 NMS 的置信度阈值。只有当检测框的置信度高于此阈值时，才会被考虑为有效检测。
    iou：设置用于 NMS 的交并比阈值。用于判断两个检测框是否重叠的程度，重叠超过此阈值的检测框会被抑制。
    agnostic：当设置为 True 时，NMS 会忽略检测框的类别，即类别不可知的 NMS。这意味着所有类别的检测框都会被一视同仁地处理。
    multi_label：是否允许每个检测框有多个标签。通常，在一个检测框中只保留置信度最高的标签。
    classes：可以指定要检测的类别列表。如果设置，NMS 将只保留列表中指定类别的检测结果。
    max_det：每张图像的最大检测数量限制。这有助于控制输出的检测框数量，特别是在图像中可能有大量检测结果的情况下。
    amp：指示是否启用自动混合精度推理。当设置为 True 时，会在可能的情况下使用半精度浮点数（float16），以加速模型推理并减少内存占用。
    '''
    conf = 0.25  # NMS置信度阈值
    iou = 0.45  # NMS交并比（IoU）阈值
    agnostic = False  # NMS是否不考虑类别（类别不可知）
    multi_label = False  # NMS是否允许每个框有多个标签
    classes = None  # （可选列表）按类别过滤，例如 [0, 15, 16] 代表 COCO 数据集中的人、猫和狗
    max_det = 1000  # 每张图像的最大检测数量
    amp = False  # 是否启用自动混合精度（AMP）推理

    def __init__(self, model, verbose=True):
        super().__init__()  # 调用父类 nn.Module 的构造函数
        if verbose:
            LOGGER.info('Adding AutoShape... ')  # 如果启用了详细模式，记录日志信息
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())
        # 将特定的属性从提供的模型复制到 AutoShape 实例

        self.dmb = isinstance(model, DetectMultiBackend)  # 检查提供的模型是否是 DetectMultiBackend 类型
        self.pt = not self.dmb or model.pt  # 确定模型是否为 PyTorch 模型
        self.model = model.eval()  # 将模型设置为评估模式

        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # 获取模型的最后一层
            m.inplace = False  # 设置 inplace 属性为 False，确保多线程推理时的安全性
            m.export = True  # 设置 export 属性为 True，表示在推理时不输出损失值
    '''
     _apply 方法确保了当对 AutoShape 实例执行如 .to(device)、.cpu()、.cuda() 或 .half() 等操作时，
     不仅模型的参数和缓存张量得到适当的处理，其内部特定的非参数属性也得到相应的更新，以保证模型在不同设备或数据类型下的一致性和正确性。
    '''
    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)  # 调用父类 nn.Module 的 _apply 方法

        if self.pt:
            # 如果模型是 PyTorch 模型
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # 获取模型的最后一层
            m.stride = fn(m.stride)  # 应用函数 fn 到 stride 属性
            m.grid = list(map(fn, m.grid))  # 应用函数 fn 到 grid 属性

            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))  # 如果 anchor_grid 是列表，则对其每个元素应用函数 fn
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        # Inference from various sources. For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())  # 创建三个性能分析对象

        with dt[0]:
            # 使用第一个性能分析对象监控以下代码块
            if isinstance(size, int):  # 检查是否提供了单一整数作为尺寸
                size = (size, size)  # 将单一尺寸扩展为二元组，表示宽度和高度

            p = next(self.model.parameters()) if self.pt else torch.empty(1,device=self.model.device)  # 获取一个模型参数或创建一个空的张量
            autocast = self.amp and (p.device.type != 'cpu')  # 根据设备类型确定是否启用自动混合精度（AMP）推理

            if isinstance(ims, torch.Tensor):  # 检查输入是否为 PyTorch 张量
                with amp.autocast(autocast):  # 使用自动混合精度上下文管理器
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # 在自动混合精度环境下进行模型推理

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # 将输入转换为列表形式，计算图像数量

            shape0, shape1, files = [], [], []  # 初始化用于存储原始图像尺寸、调整后尺寸和文件名的列表

            for i, im in enumerate(ims):
                f = f'image{i}'  # 生成图像的默认文件名

                if isinstance(im, (str, Path)):  # 如果图像是字符串或路径
                    # 如果是URL，通过HTTP请求读取图像，否则直接打开文件
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                    im = np.asarray(exif_transpose(im))  # 转换图像为Numpy数组，并处理图像的方向

                elif isinstance(im, Image.Image):  # 如果图像是PIL图像
                    im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f  # 同样转换为Numpy数组，并获取文件名

                files.append(Path(f).with_suffix('.jpg').name)  # 将处理后的文件名添加到列表

                if im.shape[0] < 5:  # 如果图像是CHW格式（通常在PyTorch DataLoader中）
                    im = im.transpose((1, 2, 0))  # 将其转换为HWC格式

                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # 确保图像是三通道RGB

                s = im.shape[:2]  # 获取图像的高度和宽度
                shape0.append(s)  # 添加原始尺寸到shape0列表

                g = max(size) / max(s)  # 计算缩放因子
                shape1.append([int(y * g) for y in s])  # 计算并添加调整后的尺寸到shape1列表

                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # 确保图像数据在内存中连续

            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # 调整shape1的尺寸使其可被模型的stride整除
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # 对每个图像应用letterbox函数进行缩放和填充
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # 转换图像数组为BCHW格式并确保数据连续
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # 将图像转换为PyTorch张量，移动到合适的设备，转换数据类型，并归一化到0-1范围

        with amp.autocast(autocast):
            # 使用自动混合精度（Automatic Mixed Precision, AMP）上下文管理器
            # 当 'autocast' 为 True 时，AMP 会在可能的情况下使用半精度来加速推理

            # Inference
            with dt[1]:
                # 使用第二个性能分析对象监控模型的推理过程
                y = self.model(x, augment=augment)  # 对预处理后的图像执行前向推理

            # Post-process
            with dt[2]:
                # 使用第三个性能分析对象监控后处理过程
                # 应用非极大值抑制（Non-Maximum Suppression, NMS）来过滤重叠的检测框
                y = non_max_suppression(y if self.dmb else y[0],
                                        self.conf,
                                        self.iou,
                                        self.classes,
                                        self.agnostic,
                                        self.multi_label,
                                        max_det=self.max_det)

                # 调整检测框的尺寸以匹配原始图像的尺寸
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            # 返回包含检测结果的 Detections 对象
            return Detections(ims, y, files, dt, self.names, x.shape)

'''
Detections
用于处理 YOLOv5 推理结果的类。类中包含了一系列方法，用于管理和展示对象检测的结果。
'''
class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        # 初始化函数
        # ims：图像列表，pred：预测结果列表，files：文件名列表
        # times：性能分析时间，names：类别名称列表，shape：输入形状
        super().__init__()
        d = pred[0].device  # 获取设备信息
        # 计算归一化因子，用于坐标转换
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]
        self.ims = ims  # 保存图像列表
        self.pred = pred  # 保存预测结果
        self.names = names  # 保存类别名称
        self.files = files  # 保存文件名
        self.times = times  # 保存性能分析时间
        self.xyxy = pred  # 保存原始预测结果
        self.xywh = [xyxy2xywh(x) for x in pred]  # 转换为xywh格式
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # 转换为归一化的xyxy格式
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # 转换为归一化的xywh格式
        self.n = len(self.pred)  # 保存图像数量
        self.t = tuple(x.t / self.n * 1E3 for x in times)  # 计算平均处理时间
        self.s = tuple(shape)  # 保存输入形状

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
        # 执行方法，根据参数显示、保存、裁剪或渲染检测结果
        # 参数包括控制是否打印、显示、保存、裁剪、渲染结果，以及标签和保存目录的设置
        s, crops = '', []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(', ')
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                display(im) if is_notebook() else im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip('\n')
            return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    @TryExcept('Showing images is not supported in this environment')
    def show(self, labels=True):
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['ims', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def print(self):
        LOGGER.info(self.__str__())

    def __len__(self):  # override len(results)
        return self.n

    def __str__(self):  # override print(results)
        return self._run(pprint=True)  # print results

    def __repr__(self):
        return f'YOLOv5 {self.__class__} instance\n' + self.__str__()

'''
Proto 类是为了在 YOLOv5 模型中实现掩码原型（Proto）模块，用于分割模型。
'''
class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # 构造函数，参数包括输入通道数（c1），中间层通道数（c_），输出通道数（c2）
        super().__init__()  # 调用 nn.Module 的构造函数
        self.cv1 = Conv(c1, c_, k=3)  # 第一个卷积层，3x3 卷积核
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 上采样层，放大因子为2
        self.cv2 = Conv(c_, c_, k=3)  # 第二个卷积层，3x3 卷积核
        self.cv3 = Conv(c_, c2)  # 第三个卷积层，输出层

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

'''
Classify 类是 YOLOv5 模型中的分类头部分，用于将特征图转换为类别预测。
'''
class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0):
        # 初始化函数
        # c1: 输入通道数, c2: 输出通道数, k: 卷积核大小, s: 步长, p: 填充, g: 分组数, dropout_p: dropout概率
        super().__init__()  # 调用 nn.Module 的构造函数
        c_ = 1280  # 定义一个中间通道数，例如 efficientnet_b0 的特征图尺寸
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)  # 创建一个卷积层
        self.pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，将特征图尺寸转换为 1x1
        self.drop = nn.Dropout(p=dropout_p, inplace=True)  # dropout层，防止过拟合
        self.linear = nn.Linear(c_, c2)  # 线性层，用于类别预测

    def forward(self, x):
        # 前向传播函数
        if isinstance(x, list):
            x = torch.cat(x, 1)  # 如果输入是列表，将其在通道维度上拼接
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))  # 依次通过卷积、池化、dropout和线性层

import math
from functools import partial
from timm.models._efficientnet_blocks import  SqueezeExcite as SE
from einops import rearrange, reduce

from timm.models.layers import *
from timm.models.layers import DropPath
inplace = True


# SE
class SE(nn.Module):
    def __init__(self, c1, ratio=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'in_1d': partial(nn.InstanceNorm1d, eps=eps),
        'in_2d': partial(nn.InstanceNorm2d, eps=eps),
        'in_3d': partial(nn.InstanceNorm3d, eps=eps),
        'bn_1d': partial(nn.BatchNorm1d, eps=eps),
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        # 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm2d, eps=eps),
    }
    return norm_dict[norm_layer]


def get_act(act_layer='relu'):
    act_dict = {
        'none': nn.Identity,
        'sigmoid': Sigmoid,
        'swish': Swish,
        'mish': Mish,
        'hsigmoid': HardSigmoid,
        'hswish': HardSwish,
        'hmish': HardMish,
        'tanh': Tanh,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'prelu': PReLU,
        'gelu': GELU,
        'silu': nn.SiLU
    }
    return act_dict[act_layer]


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(1, 1, dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerScale2D(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ConvNormAct(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 skip=False, norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


# ========== Multi-Scale Populations, for down-sampling and inductive bias ==========
class MSPatchEmb(nn.Module):

    def __init__(self, dim_in, emb_dim, kernel_size=2, c_group=-1, stride=1, dilations=[1, 2, 3],
                 norm_layer='bn_2d', act_layer='silu'):
        super().__init__()
        self.dilation_num = len(dilations)
        assert dim_in % c_group == 0
        c_group = math.gcd(dim_in, emb_dim) if c_group == -1 else c_group
        self.convs = nn.ModuleList()
        for i in range(len(dilations)):
            padding = math.ceil(((kernel_size - 1) * dilations[i] + 1 - stride) / 2)
            self.convs.append(nn.Sequential(
                nn.Conv2d(dim_in, emb_dim, kernel_size, stride, padding, dilations[i], groups=c_group),
                get_norm(norm_layer)(emb_dim),
                get_act(act_layer)(emb_dim)))

    def forward(self, x):
        if self.dilation_num == 1:
            x = self.convs[0](x)
        else:
            x = torch.cat([self.convs[i](x).unsqueeze(dim=-1) for i in range(self.dilation_num)], dim=-1)
            x = reduce(x, 'b c h w n -> b c h w', 'mean').contiguous()
        return x



class iRMB(nn.Module):

    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
                 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=64, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none',
                                  act_layer='none')
            self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias,
                                 norm_layer='none', act_layer=act_layer, inplace=inplace)
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none',
                                     act_layer=act_layer, inplace=inplace)
            else:
                self.v = nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation,
                                      groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
        self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            # padding
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            # attention
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
                           dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
            # unpadding
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x

"""
https://arxiv.org/abs/2303.03667
<<Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks>>
"""
# --------------------------FasterNet----------------------------
from timm.models.layers import DropPath


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()  # 调用父类的初始化方法
        self.dim_conv3 = dim // n_div  # 计算卷积操作的维度
        self.dim_untouched = dim - self.dim_conv3  # 计算未触及的维度
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)  # 定义部分卷积层

        if forward == 'slicing':  # 如果前向传播方法为slicing
            self.forward = self.forward_slicing  # 设置前向传播方法为forward_slicing
        elif forward == 'split_cat':  # 如果前向传播方法为split_cat
            self.forward = self.forward_split_cat  # 设置前向传播方法为forward_split_cat
        else:  # 如果前向传播方法既不是slicing也不是split_cat
            raise NotImplementedError  # 抛出未实现的错误

    def forward_slicing(self, x):
        # 仅用于推理
        x = x.clone()  # 克隆输入x，保持原始输入不变，用于后续的残差连接
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])  # 对输入的一部分应用部分卷积

        return x  # 返回处理后的x

    def forward_split_cat(self, x):
        # 用于训练/推理
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)  # 将输入x分割成两部分
        x1 = self.partial_conv3(x1)  # 对x1部分应用部分卷积
        x = torch.cat((x1, x2), 1)  # 将卷积后的x1和未处理的x2拼接起来
        return x  # 返回处理后的x



class MLPBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()  # 调用父类的初始化方法
        self.dim = dim  # 输入维度
        self.mlp_ratio = mlp_ratio  # MLP块的隐藏层维度倍数
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # Dropout路径或恒等映射
        self.n_div = n_div  # 分割参数，用于部分卷积

        mlp_hidden_dim = int(dim * mlp_ratio)  # 计算MLP隐藏层的维度
        mlp_layer = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),  # 1x1卷积，扩展维度
            norm_layer(mlp_hidden_dim),  # 正则化层
            act_layer(),  # 激活层
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)  # 1x1卷积，恢复维度
        ]
        self.mlp = nn.Sequential(*mlp_layer)  # 将MLP层序列化
        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )  # 部分卷积用于空间混合
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)  # 层缩放参数
            self.forward = self.forward_layer_scale  # 如果有层缩放值，则使用带层缩放的前向传播
        else:
            self.forward = self.forward  # 否则使用标准前向传播

    def forward(self, x):
        shortcut = x  # 残差连接
        x = self.spatial_mixing(x)  # 应用空间混合
        x = shortcut + self.drop_path(self.mlp(x))  # 应用MLP并加上残差
        return x  # 返回结果

    def forward_layer_scale(self, x):
        shortcut = x  # 残差连接
        x = self.spatial_mixing(x)  # 应用空间混合
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))  # 应用带层缩放的MLP并加上残差
        return x  # 返回结果



class BasicStage(nn.Module):
    def __init__(self,
                 dim,
                 depth=1,
                 n_div=4,
                 mlp_ratio=2,
                 layer_scale_init_value=0,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        dpr = [x.item()
               for x in torch.linspace(0, 0.0, sum([1, 2, 8, 2]))]
        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.blocks(x)
        return x


class PatchEmbed_FasterNet(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, patch_stride, norm_layer=nn.BatchNorm2d):
        super().__init__()  # 调用父类的初始化方法
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)  # 定义一个卷积层，用于将输入的图像切分为多个嵌入
        if norm_layer is not None:  # 如果指定了正则化层
            self.norm = norm_layer(embed_dim)  # 创建一个正则化层
        else:  # 如果没有指定正则化层
            self.norm = nn.Identity()  # 使用恒等映射，即不进行任何操作

    def forward(self, x):
        x = self.norm(self.proj(x))  # 首先使用卷积层进行嵌入，然后对嵌入结果应用正则化
        return x  # 返回处理后的结果

    def fuseforward(self, x):
        x = self.proj(x)  # 直接使用卷积层进行嵌入，不应用正则化
        return x  # 返回处理后的结果


class PatchMerging_FasterNet(nn.Module):
    def __init__(self, dim, out_dim, k, patch_stride2, norm_layer=nn.BatchNorm2d):
        super().__init__()  # 调用父类的初始化方法
        self.reduction = nn.Conv2d(dim, out_dim, kernel_size=k, stride=patch_stride2, bias=False)  # 定义一个卷积层，用于降维和合并相邻的嵌入
        if norm_layer is not None:  # 如果指定了正则化层
            self.norm = norm_layer(out_dim)  # 创建一个正则化层
        else:  # 如果没有指定正则化层
            self.norm = nn.Identity()  # 使用恒等映射，即不进行任何操作

    def forward(self, x):
        x = self.norm(self.reduction(x))  # 首先使用卷积层进行降维和合并，然后对结果应用正则化
        return x  # 返回处理后的结果

    def fuseforward(self, x):
        x = self.reduction(x)  # 直接使用卷积层进行降维和合并，不应用正则化
        return x  # 返回处理后的结果


# -----------------------------------
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class CA_Bottleneck(nn.Module):
    #  Bottleneck with 1 Attention
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.ca = CoordAtt(c2, c2, 32)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.ca(self.cv2(self.cv1(x))) if self.add else self.ca(self.cv2(self.cv1(x)))


class C3_CA(nn.Module):
    # CSP Bottleneck with 3 convolutions and 1 CA.
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(CA_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

# ---------------------------CA End---------------------------
# BiFPN
# 两个特征图add操作
class BiFPN_Add2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add2, self).__init__()
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))


# 三个特征图add操作
class BiFPN_Add3(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add3, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        # Fast normalized fusion
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))

import torch
from torch import nn


class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        self.Excitation = nn.Sequential()
        self.Excitation.add_module(
            "FC1", nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1)
        )
        self.Excitation.add_module("ReLU", nn.ReLU())
        self.Excitation.add_module(
            "FC2", nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1)
        )
        self.Excitation.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))


class Conv_BN_HSwish(nn.Module):
    def __init__(self, c1, c2, stride):
        super(Conv_BN_HSwish, self).__init__()
        self.conv = nn.Conv2d(c1, c2, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MobileNetV3_InvertedResidual(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(MobileNetV3_InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish() if use_hs else nn.ReLU(),
                # Squeeze-and-Excite
                SeBlock(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish() if use_hs else nn.ReLU(),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SeBlock(hidden_dim) if use_se else nn.Sequential(),
                nn.Hardswish() if use_hs else nn.ReLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y


import torch
from torch import nn


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x


class CBRM(nn.Module):  # conv BN ReLU Maxpool2d
    def __init__(self, c1, c2):
        super(CBRM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

    def forward(self, x):
        return self.maxpool(self.conv(x))


class Shuffle_Block(nn.Module):
    def __init__(self, ch_in, ch_out, stride):
        super(Shuffle_Block, self).__init__()

        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = ch_out // 2
        assert (self.stride != 1) or (ch_in == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    ch_in, ch_in, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(
                    ch_in,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                ch_in if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

import torch
from torch import nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(
        self, c1, c2, k=1, s=1, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        self.Excitation = nn.Sequential()
        self.Excitation.add_module(
            "FC1", nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1)
        )
        self.Excitation.add_module("ReLU", nn.ReLU())
        self.Excitation.add_module(
            "FC2", nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1)
        )
        self.Excitation.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))


class G_bneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(
        self, c1, c2, midc, k=5, s=1, use_se=False
    ):  # ch_in, ch_mid, ch_out, kernel, stride, use_se
        super().__init__()
        assert s in [1, 2]
        c_ = midc
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # Expansion
            (
                Conv(c_, c_, 3, s=2, p=1, g=c_, act=False) if s == 2 else nn.Identity()
            ),  # dw
            # Squeeze-and-Excite
            SeBlock(c_) if use_se else nn.Sequential(),
            GhostConv(c_, c2, 1, 1, act=False),
        )  # Squeeze pw-linear

        self.shortcut = (
            nn.Identity()
            if (c1 == c2 and s == 1)
            else nn.Sequential(
                Conv(c1, c1, 3, s=s, p=1, g=c1, act=False),
                Conv(c1, c2, 1, 1, act=False),
            )
        )

    def forward(self, x):
        # print(self.conv(x).shape)
        # print(self.shortcut(x).shape)
        return self.conv(x) + self.shortcut(x)

import torch
from torch import nn


class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        self.Excitation = nn.Sequential()
        self.Excitation.add_module(
            "FC1", nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1)
        )
        self.Excitation.add_module("ReLU", nn.ReLU())
        self.Excitation.add_module(
            "FC2", nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1)
        )
        self.Excitation.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))


class drop_connect:
    def __init__(self, drop_connect_rate):
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x, training):
        if not training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.shape[0]
        random_tensor = keep_prob
        random_tensor += torch.rand(
            [batch_size, 1, 1, 1], dtype=x.dtype, device=x.device
        )
        binary_mask = torch.floor(random_tensor)  # 1
        x = (x / keep_prob) * binary_mask
        return x


class stem(nn.Module):
    def __init__(self, c1, c2, act="ReLU6"):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=c2)
        if act == "ReLU6":
            self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MBConvBlock(nn.Module):
    def __init__(
        self, inp, final_oup, k, s, expand_ratio, drop_connect_rate, has_se=False
    ):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.has_se = has_se
        self.id_skip = True  # skip connection and drop connect
        se_ratio = 0.25

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._momentum, eps=self._epsilon
            )

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            padding=(k - 1) // 2,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._momentum, eps=self._epsilon
        )

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(inp * se_ratio))
            self.se = SeBlock(oup, 4)

        # Output phase
        self._project_conv = nn.Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._momentum, eps=self._epsilon
        )
        self._relu = nn.ReLU6(inplace=True)

        self.drop_connect = drop_connect(drop_connect_rate)

    def forward(self, x, drop_connect_rate=None):
        """
        :param x: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            x = self._relu(self._bn0(self._expand_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x = self.se(x)

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if (
            self.id_skip
            and self.stride == 1
            and self.input_filters == self.output_filters
        ):
            if drop_connect_rate:
                x = self.drop_connect(x, training=self.training)
            x += identity  # skip connection
        return x




