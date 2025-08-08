import json
import random
from collections import defaultdict
from tqdm import tqdm
import os



with open("./VOCdevkit/VOC2012/voc_total_coco.json", "r") as f:#加载COCO标注文件
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = coco['categories'] 

#image_id->file_name 映射
id2filename = {img['id']: img['file_name'] for img in images}

# 改成voc标注[x1,y1,x2,y2,class_id]
image_annos = defaultdict(list)
for ann in annotations:
    image_id = ann['image_id']
    bbox = ann['bbox']  #COCO格式: [x, y, w, h]
    class_id = ann['category_id'] -1 #voc 从0开始
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    image_annos[image_id].append([int(x1), int(y1), int(x2), int(y2), class_id])

#用于类别均衡划分
image_classes = defaultdict(set)
for image_id, annos in image_annos.items():
    for _, _, _, _, cls in annos:
        image_classes[image_id].add(cls)

#按类别均匀划分图像
class_to_images = defaultdict(set)
for image_id, cls_set in image_classes.items():
    for cls in cls_set:
        class_to_images[cls].add(image_id)

train_ids, test_ids = set(), set()


for cls, img_ids in class_to_images.items():
    img_ids = list(img_ids)
    random.shuffle(img_ids)
    split_idx = int(0.8 * len(img_ids))
    train_ids.update(img_ids[:split_idx])
    test_ids.update(img_ids[split_idx:])

# 防止重叠：从测试集中移除训练集的图片
test_ids = test_ids - train_ids

train_ids = list(train_ids)
test_ids = list(test_ids)

print(f"Train images: {len(train_ids)}, Test images: {len(test_ids)}")



def save_split(image_ids, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for image_id in tqdm(image_ids):
            file_name = id2filename[image_id]
            annos = image_annos[image_id] 
            line = file_name 
            for x1, y1, x2, y2, cls_id in annos:
                line += f" {x1},{y1},{x2},{y2},{cls_id}"
            f.write(line + "\n")



save_dir = "./pascal_anno"
save_split(train_ids, os.path.join(save_dir, "train.txt"))
save_split(test_ids, os.path.join(save_dir, "val.txt"))
