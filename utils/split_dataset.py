from pathlib import Path
import shutil
import random

# 设置原始数据集路径及目标路径
src_dataset_path = Path('D:\BaiduNetdiskWorkspace\postgraduate\pythonProject\yolov5\data\SAR-ship\SAR-Ship-Dataset\ship_dataset_v0\ship_dataset_v0') # 原始图像和标签的存放路径
dst_dataset_path = Path('D:\\BaiduNetdiskWorkspace\\postgraduate\\pythonProject\\yolov5\\data\\SAR-ship')

# 目标文件夹路径
train_images_dst_path = dst_dataset_path / 'images/train'
val_images_dst_path = dst_dataset_path / 'images/val'
train_labels_dst_path = dst_dataset_path / 'labels/train'
val_labels_dst_path = dst_dataset_path / 'labels/val'

# 确保目录存在
for path in [train_images_dst_path, val_images_dst_path, train_labels_dst_path, val_labels_dst_path]:
    path.mkdir(parents=True, exist_ok=True)

# 列出所有原始图像和标签
images = list(src_dataset_path.glob('*.jpg'))  # 假设图像文件是jpg格式
labels = [img.with_suffix('.txt') for img in images]  # 假设标签文件和图像文件名相同，后缀不同

# 划分数据集
data = list(zip(images, labels))
random.shuffle(data)
train_data = data[:int(0.7 * len(data))]
val_data = data[int(0.7 * len(data)):]

# 复制文件到目标目录
for img, label in train_data:
    shutil.copy(img, train_images_dst_path)
    shutil.copy(label, train_labels_dst_path)

for img, label in val_data:
    shutil.copy(img, val_images_dst_path)
    shutil.copy(label, val_labels_dst_path)
