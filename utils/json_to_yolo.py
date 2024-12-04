import os
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

# 设定路径
src_images_path = Path('D:\\BaiduNetdiskWorkspace\\postgraduate\\pythonProject\\yolov5\\data\\SAR-ship\\JPEGImages')
annotations_path =  Path("D:\\BaiduNetdiskWorkspace\\postgraduate\\pythonProject\\yolov5\\data\\SAR-ship\\Annotations_new")
dst_dataset_path =  Path('D:\\BaiduNetdiskWorkspace\\postgraduate\\pythonProject\\yolov5\\data\\SAR-ship')
train_images_dst_path = dst_dataset_path / 'images/train'
val_images_dst_path = dst_dataset_path / 'images/val'
train_labels_dst_path = dst_dataset_path / 'labels/train'
val_labels_dst_path = dst_dataset_path / 'labels/val'

# 创建所需的文件夹
for path in [train_images_dst_path, val_images_dst_path, train_labels_dst_path, val_labels_dst_path]:
    os.makedirs(path, exist_ok=True)


# 读取并处理标签文件
def process_annotations(json_file, images_dst_path, labels_dst_path):
    with open(json_file) as f:
        data = json.load(f)

    for image_info in data['images']:
        image_filename = image_info['file_name']
        image_path = src_images_path / image_filename
        shutil.copy(image_path, images_dst_path / image_filename)

        image_width = image_info['width']
        image_height = image_info['height']

        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_info['id']]
        with open(labels_dst_path / (image_filename.rsplit('.', 1)[0] + '.txt'), 'w') as label_file:
            for ann in annotations:
                bbox = ann['bbox']
                class_id = ann['category_id'] - 1
                x_center = (bbox[0] + bbox[2] / 2) / image_width
                y_center = (bbox[1] + bbox[3] / 2) / image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height
                label_file.write(f'{class_id} {x_center} {y_center} {width} {height}\n')


# 读取并处理标签文件
def process_annotations(xml_file, images_dst_path, labels_dst_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find('filename').text + '.jpg'
    image_path = src_images_path / filename
    # 检查图像文件是否存在
    if not image_path.exists():
        print(f"Warning: The image file {filename} does not exist. Skipping...")
        return  # 跳过当前文件，继续执行下一个文件
    shutil.copy(image_path, images_dst_path / filename)

    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)

    for obj in root.iter('object'):
        class_name = obj.find('name').text
        # 假设所有对象都属于同一类别，这里以类别"ship"为例，并将其索引设置为0
        class_id = 0 if class_name == 'ship' else -1  # 如果有多个类别，需要相应地修改
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        with open(labels_dst_path / (filename.rsplit('.', 1)[0] + '.txt'), 'a') as label_file:
            label_file.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

# 遍历Annotations_new目录下的所有xml文件，并处理
for xml_file in annotations_path.glob('*.xml'):
    process_annotations(xml_file, train_images_dst_path, train_labels_dst_path)  # 假设所有图像都用于训练
    # 如果需要区分训练集和验证集，可以在这里添加逻辑
#处理训练集和验证集数据
# process_annotations(annotations_path / 'train2017.json', train_images_dst_path, train_labels_dst_path)
# process_annotations(annotations_path / 'test2017.json', val_images_dst_path, val_labels_dst_path)
