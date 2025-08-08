import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import jittor as jt
from jittor.dataset import Dataset





def resize_img(img, boxes, min_size=600, max_size=1000):
    """按短边缩放至 min_size，长边不超过 max_size，返回缩放后图像与 box"""
    orig_w, orig_h = img.size

    # 缩放比例
    scale = min(min_size / min(orig_w, orig_h), max_size / max(orig_w, orig_h))
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # 缩放 boxes
    boxes = boxes.astype(np.float32)
    boxes[:, [0, 2]] *= scale
    boxes[:, [1, 3]] *= scale

    # 返回图像宽高四元素
    size_info = [orig_h, orig_w, new_h, new_w]
    return img_resized, boxes, size_info



class VOCDetection(Dataset):
    def __init__(self, image_dir, ann_path,training=True):
        super().__init__()
        self.training=training
        self.image_dir = image_dir
        self.ann_path = ann_path  # 可以是 .json 或 XML 路径
        self._load_annotations()

    def __len__(self):
        return len(self.image_infos)

    def _load_annotations(self):
        self.image_infos = []
        self.image_id_to_annotations = {}

        # trainval和test都用COCO格式json加载
        with open(self.ann_path, 'r') as f:
            coco = json.load(f)

        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        for img_info in coco['images']:
            img_id = img_info['id']
            self.image_infos.append({
                'id': img_id,
                'file_name': img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height']
            })

    def __getitem__(self, index):
        info = self.image_infos[index]
        img_id = info['id']
        file_name = info['file_name']
        img_path = os.path.join(self.image_dir, file_name)

        # 取出对应 annotation
        ann_list = self.image_id_to_annotations.get(img_id, [])
        boxes, labels = [], []
        for ann in ann_list:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+ w, y+ h])
            labels.append(ann['category_id'] )  

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        # 加载并预处理图像
        image = Image.open(img_path).convert("RGB")
        image, boxes, size = resize_img(image, boxes)
        image = np.array(image) / 255.0

        return image, boxes, labels, img_id, size
    '''
    def collate_batch(self, batch):
        images = [jt.array(x[0]).permute(2, 0, 1) for x in batch]
        boxes = [jt.array(x[1]) for x in batch]
        labels = [jt.array(x[2]) for x in batch]
        img_ids = [x[3] for x in batch]
        size = [x[4] for x in batch]  #新增，收集所有size
        
        images = jt.stack(images)
        return images, boxes, labels, img_ids, size
    '''
    def collate_batch(self, batch):
        images = [jt.array(x[0]).permute(2, 0, 1) for x in batch]
        boxes = [jt.array(x[1]) for x in batch]
        labels = [jt.array(x[2]) for x in batch]
        img_ids = [x[3] for x in batch]
        sizes = [x[4] for x in batch]

        if self.training:
            # 找出 batch 内最大高和最大宽
            max_h = max([img.shape[1] for img in images])
            max_w = max([img.shape[2] for img in images])

            padded_images = []
            for img in images:
                c, h, w = img.shape
                pad_h = max_h - h
                pad_w = max_w - w
                # jt.nn.pad的pad参数是 (pad_left, pad_right, pad_top, pad_bottom)
                # 这里我们在右和下pad
                padded_img = jt.nn.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
                padded_images.append(padded_img)

            images = jt.stack(padded_images)
            return images, boxes, labels, img_ids, sizes
        else:
            images = jt.stack(images)
            # 验证时不做pad，直接返回list形式
            return images, boxes, labels, img_ids, sizes


# VOC类别（21类）
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
