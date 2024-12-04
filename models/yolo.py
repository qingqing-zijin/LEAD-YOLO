# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
from models.rfa import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# YOLOv5的检测头
class Detect(nn.Module):
    # YOLOv5的检测头部分，用于构建目标检测模型
    stride = None  # 在构建过程中计算的步长
    dynamic = False  # 强制重构网格
    export = False  # 导出模式
    '''
    在 YOLOv5 或类似的目标检测模型中，`anchors` 是一个包含多个锚点尺寸的列表。这些锚点尺寸通常以多个子列表的形式组织，其中每个子列表包含特定检测层的锚点尺寸。
    `anchors[1]` 就是指这些子列表中的第二个，代表第二个检测层的锚点尺寸。

    举例来说，如果 `anchors` 定义如下：
    
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    
    这里，`anchors` 包含三个子列表，每个列表对应一个检测层。在 YOLOv5 中，这些检测层通常是为了检测不同大小的物体。例如：
    
    - `anchors[0]` -> `[10, 13, 16, 30, 33, 23]` 可能用于检测小尺寸的物体。
    - `anchors[1]` -> `[30, 61, 62, 45, 59, 119]` 用于中等尺寸的物体。
    - `anchors[2]` -> `[116, 90, 156, 198, 373, 326]` 用于大尺寸的物体。
    
    在 `anchors[1]` 这个列表中，每对数字代表一个锚点的宽度和高度。例如，`30` 和 `61` 是一对，表示一个锚点的宽度为 30，高度为 61。
    这些尺寸是相对于网络输入尺寸的，通常是基于训练数据集中对象尺寸的统计分析得出的。
    
    '''
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # 检测层的初始化方法

        super().__init__()
        self.nc = nc  # 类别数量
        self.no = nc + 5  # 每个锚点的输出数量
        self.nl = len(anchors)  # 检测层的数量
        self.na = len(anchors[0]) // 2  # 锚点数量
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化网格
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化锚点网格
        # 初始化锚点
        '''
        self.register_buffer('anchors', ...) 将这个张量注册为模型的一个缓冲区。在 PyTorch 中，缓冲区是指那些你希望与模型一起保存和加载，但不是模型参数的张量。
        注册为缓冲区的张量不会被视为模型参数，因此在训练过程中不会被优化器更新。
        这对于像锚点这样的不需要在训练过程中更新，但是模型的固有部分的值很重要。
        '''
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # 形状为(nl,na,2)
        # 初始化输出头的卷积层
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 输出卷积层
        #self.m = nn.ModuleList(DecoupledHead(x, nc, 1, anchors) for x in ch)

        self.inplace = inplace  # 使用原地操作（例如，切片赋值）

    def forward(self, x):
        z = []  # 存储每个特征层的输出

        # 遍历网络层
        for i in range(self.nl):
            # 对Neck输出的特征使用输出头的卷积层进行处理
            x[i] = self.m[i](x[i])  # 对特征图x[i]应用卷积层m[i]

            # 获取特征图的维度，bs为批大小，ny和nx为特征图的高度和宽度
            bs, _, ny, nx = x[i].shape
            # 调整特征图的形状，使其适应锚点数量和输出数量
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # 非训练模式，即推理模式
            if not self.training:
                # 如果需要动态创建网格或者网格尺寸与当前特征图尺寸不匹配
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # 创建网格
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # 如果模型是用于分割任务
                if isinstance(self, Segment):
                    # 分割xy, wh, conf和mask
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # 计算xy坐标
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # 计算宽高
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)  # 合并结果

                else:  # 用于检测任务
                    # 分割xy, wh和conf
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # 计算xy坐标
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # 计算宽高
                    y = torch.cat((xy, wh, conf), 4)  # 合并结果

                # 添加处理后的特征图到z列表
                z.append(y.view(bs, self.na * nx * ny, self.no))

        # 根据模式返回不同的结果
        # 训练模式：返回原始特征图
        # 非训练模式：根据是否导出模型返回不同的结果
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    '''
    _make_grid
    这个函数主要用于生成两个网格：一个是普通的坐标网格（grid），另一个是锚点网格（anchor_grid）。
    普通网格用于表示特征图上每个单元的位置，而锚点网格则是根据特征图上的锚点位置进行缩放的网格，通常用于计算目标检测中的边界框。
    '''
    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        # 获取当前锚点的设备和数据类型
        d = self.anchors[i].device  # 获取锚点所在的设备（比如CPU或GPU）
        t = self.anchors[i].dtype  # 获取锚点的数据类型

        # 网格的形状，其中na是每个网格的锚点数量
        shape = 1, self.na, ny, nx, 2  # 网格形状，其中2表示每个网格单元的坐标(x, y)

        # 根据下采样的特征图的宽nx和高ny创建网格单元
        # torch.arange(n) 生成一个从0到n-1的一维张量
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)  # 分别创建y和x轴的坐标序列
        # 使用meshgrid创建网格，这里考虑了PyTorch版本兼容性
        # torch.meshgrid函数接收两个一维张量（在这个例子中是y和x）并生成两个二维张量。
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # 创建网格坐标点
        # 添加网格偏移，并扩展到指定形状
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # 将xv和yv堆叠形成网格，并减去0.5偏移

        # 生成锚点网格，这里锚点是相对于特征图的尺寸进行缩放的
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)  # 调整锚点尺寸并扩展到网格形状

        # 返回生成的网格和锚点网格
        return grid, anchor_grid  # 返回网格和锚点网格


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5的基础模型
    def forward(self, x, profile=False, visualize=False):
        # 定义前向传播，单尺度推理，训练时使用
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # 初始化输出和时间记录列表
        # 对模型逐层进行推理
        for m in self.model:
            if m.f != -1:  # 如果不是来自前一层
                # 从早期层获取输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                # 如果开启性能分析，则记录每层的推理时间
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 执行当前层运算
            # 保存输出结果
            y.append(x if m.i in self.save else None)
            if visualize:
                # 如果开启可视化，对特征进行可视化
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        # 评估每层的推理时间
        c = m == self.model[-1]  # 判断是否为最后一层
        # 计算FLOPs
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        # 日志输出层的信息
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        # 融合模型的Conv2d()和BatchNorm2d()层
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                # 更新卷积层
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                # 移除批标准化层
                delattr(m, 'bn')
                # 更新前向传播函数
                m.forward = m.forward_fuse
            if type(m) is PatchEmbed_FasterNet:
                m.proj = fuse_conv_and_bn(m.proj, m.norm)
                delattr(m, 'norm')  # remove BN
                m.forward = m.fuseforward
            if type(m) is PatchMerging_FasterNet:
                m.reduction = fuse_conv_and_bn(m.reduction, m.norm)
                delattr(m, 'norm')  # remove BN
                m.forward = m.fuseforward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        # 打印模型信息
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # 对模型应用to(), cpu(), cuda(), half()等操作
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        # 对Detect和Segment的头部进行权重初始化
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self



class DetectionModel(BaseModel):
    # YOLOv5检测模型
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        # 初始化，输入为模型配置文件、输入通道数、类别数、锚点
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # 如果cfg是字典，直接作为模型配置
        else:  # 如果是*.yaml文件
            import yaml  # 导入yaml模块
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # 从yaml文件加载模型配置

        # 定义模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # 获取输入通道数
        if nc and nc != self.yaml['nc']:
            # 如果提供了类别数并且与yaml文件中不同，则覆盖
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # 覆盖yaml中的类别数
        if anchors:
            # 如果提供了锚点，则覆盖yaml文件中的锚点
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # 覆盖yaml中的锚点
        # 解析并构建模型
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # 解析模型，获取保存列表
        self.names = [str(i) for i in range(self.yaml['nc'])]  # 默认类别名称
        self.inplace = self.yaml.get('inplace', True)

        # 构建strides和anchors
        m = self.model[-1]  # 获取模型的最后一个模块，通常是Detect
        if isinstance(m, (Detect, Segment)):
            s = 256  # 最小步长的两倍
            m.inplace = self.inplace
            # 实例化forward方法
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            # 计算stride
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)  # 检查锚点顺序
            m.anchors /= m.stride.view(-1, 1, 1)  # 调整锚点大小
            self.stride = m.stride
            # 初始化检测头的偏置
            self._initialize_biases()  # 只运行一次

        # 初始化权重和偏置
        initialize_weights(self)
        self.info()  # 打印模型信息
        LOGGER.info('')


    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        # 增强前向传播
        img_size = x.shape[-2:]  # 获取输入图像的高度和宽度
        s = [1, 0.83, 0.67]  # 定义缩放比例
        f = [None, 3, None]  # 定义翻转操作（2-上下翻转，3-左右翻转）
        y = []  # 初始化输出列表

        for si, fi in zip(s, f):
            # 对每种缩放和翻转组合进行操作
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))  # 缩放并翻转图像
            yi = self._forward_once(xi)[0]  # 对处理后的图像进行一次前向传播
            yi = self._descale_pred(yi, fi, si, img_size)  # 反缩放预测结果
            y.append(yi)  # 添加到输出列表

        y = self._clip_augmented(y)  # 裁剪增强后的结果
        return torch.cat(y, 1), None  # 返回增强后的推理结果，用于训练

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # 模型Detect Head的初始化方法
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None

# parse_model函数，用于解析YOLOv5模型的配置字典并构建模型。
def parse_model(d, ch):  # model_dict, input_channels(3)
    # 解析YOLOv5 model.yaml字典
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 从字典中解析anchors, 类别数nc, depth和width的倍数gd和gw, 以及激活函数
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # 重新定义默认激活函数，例如Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # 打印激活函数

    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 锚点的数量
    no = na * (nc + 5)  # 输出的数量 = 锚点数 * (类别数 + 5)

    layers, save, c2 = [], [], ch[-1]  # 初始化层列表，保存列表和输出通道数

    # 遍历模型的backbone和head部分，进行解析和构建
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # 实例化模块
        '''
        在这段代码中，`eval` 函数被用于两个主要目的：
    
        1. **将字符串转换为模块**：当 `m` 是一个字符串时，`eval(m)` 用于将这个字符串转换成对应的 Python 对象或函数。
        在这个上下文中，`m` 很可能是表示模块或类名的字符串（例如 `"Conv"`、`"Bottleneck"` 等），`eval(m)` 就会把这些字符串转换成实际的 Python 类或函数。
        这是一种动态地根据字符串内容创建相应对象的方法。

        2. **解析参数列表中的字符串**：对于 `args` 列表中的每个元素，如果它是一个字符串，`eval(a)` 会尝试计算这个字符串表达式的值。
        这在配置文件中使用字符串来表示表达式或变量值时非常有用。例如，如果 `args` 中的某个元素是 `"2 * 16"`，`eval` 会计算这个表达式的结果，即 32。

        总的来说，这里的 `eval` 函数用于动态地解释和执行由字符串表示的代码片段。
        这使得代码可以基于文本配置（如 YAML 文件）灵活地构建模型
        '''
        m = eval(m) if isinstance(m, str) else m  # 将字符串转换为模块
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # 将字符串转换为相应的值

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # 应用深度倍数
        # 解析模块
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x,
                C3_CA,
                iRMB,
                RFAConv, RFCAConv, RFCBAMConv,
                BasicStage, PatchEmbed_FasterNet, PatchMerging_FasterNet,
                Conv_BN_HSwish, MobileNetV3_InvertedResidual,
                Shuffle_Block, CBRM, G_bneck,
                stem, MBConvBlock
        }:
            c1, c2 = ch[f], args[0]  # 输入和输出通道数
            if c2 != no:  # 如果不是输出层
                '''
                make_divisible 函数通常用于确保某个数值能够被另一个数（在这种情况下是 8）整除。
                这在深度学习，尤其是在构建卷积神经网络时非常有用，因为某些硬件或者软件框架对于层的输入和输出通道数有特定的整除性要求，以保证计算的效率。
                '''
                c2 = make_divisible(c2 * gw, 8)  # 应用宽度倍数

            args = [c1, c2, *args[1:]]  # 更新参数
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x, C3_CA}:
                args.insert(2, n)  # 插入重复次数
                n = 1
            elif m in [BasicStage]:
                args.pop(1)
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)  # 计算拼接后的通道数
        # 添加bifpn_add结构
        elif m in [BiFPN_Add2, BiFPN_Add3]:
            c2 = max([ch[x] for x in f])
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])  # 添加输入通道数
            if isinstance(args[1], int):  # 锚点数量
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)  # 应用宽度倍数
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        # 创建模块
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 构造模块
        t = str(m)[8:-2].replace('__main__.', '')  # 获取模块类型
        np = sum(x.numel() for x in m_.parameters())  # 计算参数数量
        # 附加索引，'from'索引，类型，参数数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # 打印模块信息
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 添加到保存列表
        layers.append(m_)  # 添加到层列表
        if i == 0:
            ch = []
        ch.append(c2)  # 更新通道列表

    return nn.Sequential(*layers), sorted(save)

class DecoupledHead(nn.Module):
	#代码是参考啥都会一点的老程大佬的 https://blog.csdn.net/weixin_44119362
    def __init__(self, ch=256, nc=80, width=1.0, anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.merge = Conv(ch, 256 * width, 1, 1)
        self.cls_convs1 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.cls_convs2 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.reg_convs1 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.reg_convs2 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.cls_preds = nn.Conv2d(256 * width, self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256 * width, 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256 * width, 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        # 分类=3x3conv + 3x3conv + 1x1convpred
        x1 = self.cls_convs1(x)
        x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        # 回归=3x3conv（共享） + 3x3conv（共享） + 1x1pred
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        # 置信度=3x3conv（共享）+ 3x3conv（共享） + 1x1pred
        x22 = self.obj_preds(x2)
        out = torch.cat([x21, x22, x1], 1)
        return out



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
