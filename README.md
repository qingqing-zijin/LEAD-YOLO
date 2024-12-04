

# LEAD-YOLO

------

## Introduction

This project is based on YOLOv5 by Glenn Jocher and the Ultralytics team. The following modifications have been made to adapt the model for Synthetic Aperture Radar (SAR) ship detection, optimizing it for resource-constrained edge devices:

1. **FasterNet Backbone**:
   - Replaced the original backbone with FasterNet to reduce model complexity while maintaining performance.
   - For more details, refer to:
     - Chen, J.; Kao, S.; He, H.; Zhuo, W.; Wen, S.; Lee, C.-H.; Chan, S.-H.G. 2023. *“Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks.”* arXiv preprint arXiv:2303.03667. [Link](https://arxiv.org/abs/2303.03667).
2. **RFCBAMConv Module**:
   - Integrated a Receptive Field-aware Channel and Spatial Attention (RFCBAM) module to enhance feature representation, improving the model's ability to detect ships in SAR imagery.
   - For more details, refer to:
     - Zhang, X.; Liu, C.; Yang, D.; Song, T.; Ye, Y.; Li, K.; Song, Y. 2023. *“RFAConv: Innovating Spatial Attention and Standard Convolutional Operation.”* arXiv preprint arXiv:2304.03198. [Link](http://arxiv.org/abs/2304.03198).
3. **C3 CA Module**:
   - Incorporated the Coordinate Attention (CA) mechanism into the C3 module to better encode spatial features relevant to SAR ship detection.
   - For more details, refer to:
     - Hou, Q.; Zhou, D.; Feng, J. 2021. *“Coordinate Attention for Efficient Mobile Network Design.”* arXiv preprint arXiv:2103.02907. [Link](https://doi.org/10.48550/arXiv.2103.02907).

These enhancements aim to strike a balance between detection accuracy and computational efficiency, making the model suitable for edge device deployment.

For more details, please see the original project at [Ultralytics YOLOv5 repository](https://github.com/ultralytics/yolov5).

## Datasets

LEAD-YOLO has been evaluated on three publicly available datasets:

1. **SSDD**: Synthetic Aperture Radar Ship Detection Dataset  
   - **Description**: Provides SAR images of ships for detection tasks, collected from Sentinel-1, TerraSAR, and RadarSat-2.  
   - **Source**: Zhang, T. et al., "SAR Ship Detection Dataset (SSDD): Official Release and Comprehensive Data Analysis," Remote Sensing, vol. 13, no. 18, p. 3690, September 2021.  
     [DOI: 10.3390/rs13183690](https://doi.org/10.3390/rs13183690)  
   - **GitHub**: [https://github.com/TianwenZhang0825/Official-SSDD](https://github.com/TianwenZhang0825/Official-SSDD)

2. **HRSID**: High-Resolution SAR Images Dataset  
   - **Description**: Focuses on high-resolution SAR images for ship detection and segmentation.  
   - **Source**: Wei, S. et al., "HRSID: A High-Resolution SAR Images Dataset for Ship Detection and Instance Segmentation," IEEE Access, vol. 8, pp. 120234-120254, 2020.  
     [DOI: 10.1109/ACCESS.2020.3005861](https://doi.org/10.1109/ACCESS.2020.3005861)  
   - **GitHub**: [https://github.com/chaozhong2010/HRSID](https://github.com/chaozhong2010/HRSID)

3. **SAR-ship**: Annotated SAR Ship Imagery  
   - **Description**: Includes a large number of annotated SAR images for ship detection under complex backgrounds.  
   - **Source**: Wang, Y. et al., "A SAR Dataset of Ship Detection for Deep Learning under Complex Backgrounds," Remote Sensing, vol. 11, no. 765, 2019.  
     [DOI: 10.3390/rs11070765](https://doi.org/10.3390/rs11070765)  
   - **GitHub**: [https://github.com/CAESAR-Radi/SAR-Ship-Dataset](https://github.com/CAESAR-Radi/SAR-Ship-Dataset)

------

## Quick Start Guide

### 1. Install Dependencies

Ensure your system meets the following requirements:
- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.3 (for GPU support)

### 2. Clone the Repository

```bash
git clone https://github.com/your-repo/LEAD-YOLO.git
cd LEAD-YOLO
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train.py --cfg ./models/LEAD-YOLO.yaml --data ./data/SSDD.yaml
```

### 4. Evaluate the Model

```bash
python val.py --data ./data/SSDD.yaml --task speed --batch 1 --weights
./weights/lead_yolo(ssdd).pt
```

### 5. Inference

Run inference on a single image or batch:

```bash
python detect.py --weights ./weights/lead_yolo(ssdd).pt --data ./data/SSDD.yaml --source ./source
```

The results for training, evaluation, and inference will be saved in the `runs/` directory under the project folder. Specifically, training outputs (e.g., logs, weights, and plots) are stored in `runs/train/`, evaluation metrics and related files in `runs/val/`, and inference outputs (e.g., annotated images or videos) in `runs/detect/`. Each run creates a unique subdirectory for its results.

### Note

This Quick Start Guide provides a simplified overview tailored to the modifications introduced in LEAD-YOLO. For more detailed instructions and advanced usage options, please refer to the original [YOLOv5 README](https://github.com/ultralytics/yolov5/blob/master/README.md) by Ultralytics.

##  Experiment Results

### Backbone Comparsion

![output (4)](.\README-LEADYOLO.assets\backbone comparision.png)

|   Backbone    |  P (%)   |  R (%)   |  F1 (%)  |  AP (%)  |   FPS    | Para (M) |
| :-----------: | :------: | :------: | :------: | :------: | :------: | :------: |
| CSPDarknet53  |   98.1   |   94.3   |   96.2   |   98.0   |   22.4   |   7.01   |
| EfficientNet  |   95.0   |   93.2   |   93.8   |   96.2   |   47.3   |   1.08   |
|  MobileNetV3  |   97.9   |   94.6   |   96.2   |   97.8   |   16.6   |   5.02   |
| ShuffleNetV2  |   94.6   |   92.8   |   93.7   |   96.5   |   57.4   |   0.84   |
|  GhostNetV2   |   96.6   |   94.8   |   95.7   |   97.6   |   27.9   |   4.53   |
| **FasterNet** | **96.8** | **94.7** | **95.7** | **97.7** | **43.8** | **3.05** |

### Ablation Studies

![output](.\README-LEADYOLO.assets\output.png)

| Model       | Dataset  | FasterNet | RFCBAMConv | C3_CA | AP(%)       | Params(M)     | FPS            |
| ----------- | -------- | --------- | ---------- | ----- | ----------- | ------------- | -------------- |
| YOLOv5s     | SSDD     |           |            |       | 98.0        |               |                |
|             | HRSID    |           |            |       | 92.7        | 7.01          | 22.4           |
|             | SAR-ship |           |            |       | 98.1        |               |                |
|             | SSDD     | ✓         |            |       | 97.7 (-0.3) | 3.05 (-56.5%) | 43.8 (+95.54%) |
|             | HRSID    | ✓         |            |       | 90.6 (-2.1) | 3.05 (-56.5%) | 43.8 (+95.54%) |
|             | SAR-ship | ✓         |            |       | 96.5 (-1.6) | 3.05 (-56.5%) | 43.8 (+95.54%) |
|             | SSDD     | ✓         | ✓          |       | 97.8 (-0.2) | 3.11 (-55.6%) | 36.4 (+60.7%)  |
|             | HRSID    | ✓         | ✓          |       | 90.9 (-1.8) | 3.11 (-55.6%) | 36.4 (+60.7%)  |
|             | SAR-ship | ✓         | ✓          |       | 96.9 (-1.2) | 3.11 (-55.6%) | 36.4 (+60.7%)  |
| LEAD-YOLOv5 | SSDD     | ✓         | ✓          | ✓     | 98.1 (+0.1) | 3.13 (-55.4%) | 35.3 (+57.6%)  |
|             | HRSID    | ✓         | ✓          | ✓     | 91.2 (-1.5) | 3.13 (-55.4%) | 35.3 (+57.6%)  |
|             | SAR-ship | ✓         | ✓          | ✓     | 97.5 (-0.6) | 3.13 (-55.4%) | 35.3 (+57.6%)  |

------

### Detection methods Comparsion

![output (5)](.\README-LEADYOLO.assets\Detection methods.png)

|     Method     | **P(%)** | **R(%)** | **F1(%)** |  AP(%)   | Para (M) |
| :------------: | :------: | :------: | :-------: | :------: | :------: |
|      SSD       |  93.73   |  43.86   |   59.61   |  85.17   |   23.8   |
| Efficient-YOLO |  96.09   |  85.69   |   90.57   |  93.56   |   8.2    |
|   BiFA-YOLO    |  94.85   |  93.97   |   94.16   |  93.90   |   19.5   |
|     YOLOv4     |  94.88   |  93.43   |   94.05   |  94.59   |   64.4   |
|    YOLOv11     |   96.6   |   96.5   |   96.5    |   98.5   |   20.1   |
|    YOLOv5s     |    95    |   91.2   |   93.02   |   96.3   |   7.0    |
| **LEAD-YOLO**  | **96.5** | **92.0** | **94.23** | **96.7** | **3.13** |

------
All experimental results, including figures and tables, are stored in the `experiment_results` folder for easy reference and access.

## Deployment on Cambricon MLU220

This project has been successfully deployed on the Cambricon MLU220 platform. Due to the specific requirements and tools involved in the deployment process, we recommend referring to the official Cambricon documentation for detailed instructions.

For more information, please visit the [Cambricon Official Website](https://www.cambricon.com/) and search for the relevant deployment guides and tools for MLU220.

## License

This project is an improvement based on YOLOv5, originally created by Glenn Jocher and the Ultralytics team. It is licensed under the GNU General Public License v3.0 (GPL-3.0). The full license terms are included in the `LICENSE` file.

## Citation

If you find this work helpful, please consider referencing the following paper, which is currently under review:

```
@article{LEAD-YOLO2024,
  title={A Lightweight, Efficient, Adaptive Design of YOLOv5 for Enhanced SAR Ship Detection},
  author={Hao Mo, Jiwen Wu, Hong Xia, Xubang Yu, and Erhu Zhao},
  journal={Remote Sensing Letters},
  year={2024},
  publisher={Taylor & Francis}
}
```

## Contact

For questions or collaboration, please contact:

- **HaoMo**: [mohao@ncepu.edu.cn]

