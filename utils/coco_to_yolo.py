import os
import json
from tqdm import tqdm
import shutil
from pycocotools.coco import COCO

def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format:
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format:
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """

    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]


def make_folders(path="output"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def convert_coco_json_to_yolo_txt(output_path, json_file):

    path = make_folders(output_path)

    coco = COCO(json_file)

    with open(json_file) as f:
        json_data = json.load(f)

    # write _darknet.labels, which holds names of all classes (one class per line)
    label_file = os.path.join(output_path, "_darknet.labels")
    with open(label_file, "w") as f:
        for category in tqdm(json_data["categories"], desc="Categories"):
            category_name = category["name"]
            f.write(f"{category_name}\n")

    # get img ids
    img_ids = coco.getImgIds()

    # process a file for each image
    for id in tqdm(img_ids, desc="Annotation txt for each image"):
        image = coco.loadImgs(ids=[id])[0]
        # print(image)
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        # anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        ann_ids = coco.getAnnIds(imgIds=[id])
        anno_in_image = coco.loadAnns(ids=ann_ids)
        anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
        with open(anno_txt, "w") as f:
            for anno in anno_in_image:
                category = anno["category_id"] - 1   # MAP ONE LOWER CLASS
                bbox_COCO = anno["bbox"]
                x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("Converting COCO Json to YOLO txt finished!")


if __name__ == "__main__":
    output_path = '/tmp/pycharm_project_33/data/HRSID_JPG/annotations/test'
    json_file = '/tmp/pycharm_project_33/data/HRSID_JPG/annotations/test2017.json'
    convert_coco_json_to_yolo_txt(output_path, json_file)