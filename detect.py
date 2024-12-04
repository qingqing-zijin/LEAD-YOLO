# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # 模型路径或Triton URL
        source=ROOT / 'data/images',  # 文件/目录/URL/glob/screen/0（摄像头）
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml路径
        imgsz=(640, 640),  # 推理大小（高度，宽度）
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # NMS IOU阈值
        max_det=1000,  # 每张图片最大检测量
        device='',  # CUDA设备，例如 0 或 0,1,2,3 或 cpu
        view_img=False,  # 展示结果
        save_txt=False,  # 将结果保存为*.txt
        save_conf=False,  # 在--save-txt标签中保存置信度
        save_crop=False,  # 保存裁剪后的预测框
        nosave=False,  # 不保存图片/视频
        classes=None,  # 按类别过滤：--class 0，或者 --class 0 2 3
        agnostic_nms=False,  # 类别不可知的NMS
        augment=False,  # 增强推理
        visualize=False,  # 可视化特征
        update=False,  # 更新所有模型
        project=ROOT / 'runs/detect',  # 将结果保存到project/name
        name='exp',  # 将结果保存到project/name
        exist_ok=False,  # 存在的project/name可接受，不自动递增
        line_thickness=10,  # 边框厚度（像素）
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # 使用FP16半精度推理
        dnn=False,  # 使用OpenCV DNN进行ONNX推理
        vid_stride=1,  # 视频帧率步进
):
    # 处理输入源
    source = str(source)
    # 决定是否保存推理图像
    save_img = not nosave and not source.endswith('.txt')  # 保存推理图像
    # 检查输入源是否为文件
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 检查输入源是否为URL
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 检查输入源是否为摄像头
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    # 检查输入源是否为屏幕截图
    screenshot = source.lower().startswith('screen')
    # 如果输入源是URL且为文件，则下载
    if is_url and is_file:
        source = check_file(source)  # 下载

    # 处理保存目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增加运行编号
    # 创建保存目录
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

    # 加载模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    # 检查图像大小
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸

    # 数据加载器
    bs = 1  # 批处理大小
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 运行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 预热
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # 对数据集中的每个元素进行处理
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # 将图像数据从NumPy数组转换为PyTorch张量，并移动到指定的设备（CPU或GPU）
            im = torch.from_numpy(im).to(model.device)
            # 根据模型是否使用FP16半精度，将图像数据转换为相应的格式
            im = im.half() if model.fp16 else im.float()  # uint8到fp16/32
            # 将图像数据从0-255范围标准化到0.0-1.0范围
            im /= 255
            # 如果图像数据是3维的，则扩展为4维（增加批处理维度）
            if len(im.shape) == 3:
                im = im[None]

                # 推理过程
        with dt[1]:
            # 如果启用可视化，则为每个路径创建一个新的保存路径
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 使用模型进行推理，获取预测结果
            pred = model(im, augment=augment, visualize=visualize)

        # 非最大抑制(NMS)过程
        with dt[2]:
            # 应用NMS处理预测结果，以消除冗余和重叠的检测框
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 处理预测结果
        for i, det in enumerate(pred):  # 对每个图像的预测结果进行处理
            seen += 1
            if webcam:  # 如果是摄像头输入，处理批量数据
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:  # 其他输入类型
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # 创建保存路径和文本文件路径
            p = Path(p)  # 将路径转换为Path对象
            save_path = str(save_dir / p.name)  # 图片保存路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # 文本文件路径
            s += '%gx%g ' % im.shape[2:]  # 打印图像尺寸信息
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化因子
            imc = im0.copy() if save_crop else im0  # 处理裁剪保存
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 从推理尺寸调整检测框到原始图像尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印检测结果
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 每类的检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到输出字符串

                # 写入检测结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # 将检测结果写入文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 转换为归一化的xywh格式
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 将边界框添加到图像
                    if save_img or save_crop or view_img:
                        c = int(cls)  # 类别转换为整数
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.box_label(xyxy, label, color=(0, 0, 0))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # 显示结果
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许Linux下窗口大小调整
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 等待1毫秒

            # 保存检测结果（带有检测框的图像）
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 视频或流
                    if vid_path[i] != save_path:  # 处理新视频
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # 释放之前的视频写入器
                        if vid_cap:  # 处理视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # 处理流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # 设置保存视频的路径和格式
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # 强制结果视频使用.mp4后缀
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # 打印推理时间
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # 打印整体结果
    t = tuple(x.t / seen * 1E3 for x in dt)  # 每张图像的处理速度
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # 更新模型（以修复SourceChangeWarning）


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/YOLOv5s/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/lead_yolo(ssdd).pt',help='model path or triton URL')
    parser.add_argument('--source', type=str, default='.\\source', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/SSDD.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
