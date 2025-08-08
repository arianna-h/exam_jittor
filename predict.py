import jittor as jt

from model.Faster_RCNN import Faster_RCNN
from utils.util import get_classes
from utils.eval import detect_image


num_classes=21 #总类别 + 1 背景类

img_path="./example/2007_000032.jpg" #图像路径


class_names, _ = get_classes("./model_data/voc_classes.txt")

#模型和优化器

model_path=''

model = Faster_RCNN(num_classes=num_classes, 
                    backbone="vgg",
                    anchor_scales   = [8, 16, 32], 
                    pretrained=False,
                    n_test_pre_nms=3000, 
                    n_test_post_nms=600 
                    )

#根据预训练权重的Key和模型的Key进行加载
params = jt.load(model_path)       # 加载参数字典
model.load_state_dict(params)  # 将参数加载进模型



r=detect_image(net=model,image_path=img_path,class_name=class_names,num_classes=num_classes,confidence=0.6)


save_path = './example/result.jpg'
r.save(save_path)  # 保存图像到指定路径