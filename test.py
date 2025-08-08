import os
import datetime
import jittor as jt
import numpy as np

from model.Faster_RCNN import Faster_RCNN
from utils.util import get_classes
from utils.callback import EvalCallback

#环境类
jt.flags.use_cuda = 1  #jt的CUDA训练
jt.set_global_seed(42) #种子







#数据集类
num_classes=21 #总类别 + 1 背景类

img_path="./VOCdevkit/VOC2012/JPEGImages" #数据文件夹


val_annotation_path="./pascal_anno/val.txt"  #验证划分


with open(val_annotation_path, encoding='utf-8') as f:
    val_lines   = f.readlines()
val_lines = [img_path + line.strip() + "\n" for line in val_lines]



#自动生成log路径并重定向所有输出
base_dir = os.path.dirname(os.path.abspath(__file__))
record_path = os.path.join(base_dir,"checkpoints/voc/",f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
os.makedirs(record_path, exist_ok=True)
log_file=os.path.join(record_path,'log.txt')
save = os.path.join(record_path,"checkpoints")
class_names, num_classes = get_classes("./model_data/voc_classes.txt")



model_path=''  #设置模型路径

model = Faster_RCNN(num_classes=num_classes, 
                    backbone="vgg",
                    anchor_scales   = [8, 16, 32], #anchor 在特征图的尺寸 ，如果缩放回原始图像会 是16倍大小
                    pretrained=False,
                    n_train_pre_nms=6000, #训练时 nms 前保留多少 proposal
                    n_train_post_nms=1000, # 训练时 nms后保留多少
                    nms_iou=0.7, # nms处理的iou
                    n_test_pre_nms=3000, # 预测时nms前保留多少
                    n_test_post_nms=600 #预测时nms后保留多少
                    )

#根据预训练权重的Key和模型的Key进行加载
if model_path != '':
    print('Load weights {}.'.format(model_path))
    model_dict      = model.state_dict()
    pretrained_dict = jt.load(model_path)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))




eval_callback   = EvalCallback(model, [600, 600], class_names, 21, val_lines, record_path, 
                                        eval_flag=1, period=1)
model.eval()
eval_callback.on_epoch_end(0,draw=False)

